"""Panels & LEDs tab and the floating *Selected* inspector, both driven by :class:`LayoutState`.

Panel list (tab): add panels from templates / the designer / a single LED, one folder per panel
with the essentials (select, enabled, mirror, remove). Inspector (floating panel): everything about
the selected panel — pose, LED on/off + roles, duct guide, designer, template export.
Templates are v2 layout files in ``templates/`` whose ``panels`` get appended to the scene.
"""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

from lighting_simulator.domain.beam_profile import LAMBERTIAN, get_profile, profile_names
from lighting_simulator.domain.led import ROLES
from lighting_simulator.scene.layout import Layout, Panel, load_layout, save_json

_ROLE_COLORS = {'vio': "#1E90FF", 'flash': "#FF8C00", 'both': "#FFFFFF"}
_ACTIONS = {"Toggle on / off": 'toggle', "Set role: VIO": 'vio', "Set role: Flash": 'flash', "Set role: Both": 'both'}


def build(ctx):
    server = ctx.server
    state = ctx.state
    tab_panels = ctx.tab_panels
    inspector_tab = ctx.inspector_tab
    templates_dir = ctx.templates_dir
    enter_designer = ctx.enter_designer          # (panel_index | None) -> None, set later by scene_view
    led_lumens_slider = ctx.led_lumens_slider
    flash_lumens = ctx.flash_lumens

    click_action = ['toggle']
    inspector_handles = []
    list_handles = []
    syncing = [False]

    # ------------------------------------------------------------------ templates
    def template_names():
        if not os.path.isdir(templates_dir):
            return []
        return sorted((f[:-5] for f in os.listdir(templates_dir) if f.lower().endswith(".json")), key=str.lower)

    def load_template_panels(name):
        lay = load_layout(os.path.join(templates_dir, f"{name}.json"))
        return [copy.deepcopy(p) for p in lay.panels]

    def save_panel_as_template(panel: Panel, name):
        tpl = copy.deepcopy(panel)
        tpl.name = name
        tpl.position, tpl.rotation, tpl.mirror = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False
        tpl.source = None
        path = os.path.join(templates_dir, f"{name.lower().replace(' ', '_')}.json")
        save_json(Layout(name=name, panels=[tpl]), path)
        refresh_templates()
        print(f"✓ Template saved: {path}")
        return path

    # ------------------------------------------------------------------ tab: panel list
    with tab_panels:
        add_folder = server.gui.add_folder("Add panels")
    with add_folder:
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:4px;'>A panel is a rigid group of "
                            "LEDs with its own pose. Tick <b>Mirror</b> to get its left/right twin for free. "
                            "Click a panel in 3-D to edit it in the <b>Selected</b> window.</div>")
        template_dropdown = server.gui.add_dropdown("Template", options=["(none)"] + template_names(),
                                                    initial_value="(none)")
        add_template_btn = server.gui.add_button("➕ Add panel from template", color="green")
        new_panel_btn = server.gui.add_button("✏️ New panel in the Designer")
        add_led_btn = server.gui.add_button("➕ Add single LED (1-LED panel)")
        refresh_tpl_btn = server.gui.add_button("🔄 Refresh templates")

    with tab_panels:
        list_folder = server.gui.add_folder("Panels in the scene")
    list_summary = None

    def refresh_templates():
        cur = template_dropdown.value
        template_dropdown.options = ["(none)"] + template_names()
        template_dropdown.value = cur if cur in template_dropdown.options else "(none)"

    @add_template_btn.on_click
    def _(_):
        name = template_dropdown.value
        if not name or name == "(none)":
            print("Pick a template first.")
            return
        panels = load_template_panels(name)
        for p in panels:
            p.source = {'template': name}
        state.add_panels(panels)
        print(f"✓ Added {len(panels)} panel(s) from template '{name}'")

    new_panel_btn.on_click(lambda _: enter_designer[0](None))

    @add_led_btn.on_click
    def _(_):
        state.add_panel(state.new_single_led_panel())

    refresh_tpl_btn.on_click(lambda _: refresh_templates())

    def rebuild_list():
        nonlocal list_summary
        for h in list_handles:
            try:
                h.remove()
            except Exception:
                pass
        list_handles.clear()
        n_led = sum(len(p.leds) * (2 if p.mirror else 1) for p in state.panels if p.enabled)
        with list_folder:
            list_summary = server.gui.add_html(
                f"<div style='color:#888;font-size:11px;'>{len(state.panels)} panel(s), {n_led} LED(s) in the scene"
                + (" — none yet" if not state.panels else "") + "</div>")
            list_handles.append(list_summary)
            for i, panel in enumerate(state.panels):
                mark = "▶ " if i == state.selected else ""
                f = server.gui.add_folder(f"{mark}{i}: {panel.name}  ({len(panel.leds)} LED{'s' if len(panel.leds) != 1 else ''}"
                                          f"{', mirrored' if panel.mirror else ''})", expand_by_default=False)
                list_handles.append(f)
                with f:
                    sel = server.gui.add_button("Select")
                    en = server.gui.add_checkbox("Enabled", initial_value=panel.enabled)
                    mi = server.gui.add_checkbox("Mirror (left/right twin)", initial_value=panel.mirror)
                    rm = server.gui.add_button("Remove", color="red")
                sel.on_click(lambda _, i=i: state.select(i))
                en.on_update(lambda _, p=panel, h=en: (setattr(p, 'enabled', bool(h.value)), state.notify()))
                mi.on_update(lambda _, p=panel, h=mi: (setattr(p, 'mirror', bool(h.value)), state.notify()))
                rm.on_click(lambda _, i=i: state.remove_panel(i))

    # ------------------------------------------------------------------ inspector
    def _clear_inspector():
        for h in inspector_handles:
            try:
                h.remove()
            except Exception:
                pass
        inspector_handles.clear()

    def _add(h):
        inspector_handles.append(h)
        return h

    def _led_matrix(panel: Panel, index):
        rows = panel.rows or [list(range(len(panel.leds)))]

        def _apply(indices):
            idx = [i for i in indices if i < len(panel.leds)]
            action = click_action[0]
            if action == 'toggle':
                new_state = not all(panel.leds[i].on for i in idx)
                for i in idx:
                    panel.leds[i].on = new_state
            else:
                for i in idx:
                    panel.leds[i].role = action
                    panel.leds[i].on = True
            state.notify()
            populate_inspector(index)

        _add(server.gui.add_html("<hr style='margin:6px 0;'><b>LEDs</b>"))
        action_dd = _add(server.gui.add_dropdown(
            "LED button action", options=list(_ACTIONS), initial_value=next(k for k, v in _ACTIONS.items() if v == click_action[0]),
            hint="Roles: VIO = on in flight only, Flash = photogrammetry pulse only, Both = always on."))
        action_dd.on_update(lambda _: click_action.__setitem__(0, _ACTIONS[action_dd.value]))
        n_on = sum(1 for l in panel.leds if l.on)
        counts = {r: sum(1 for l in panel.leds if l.on and l.role == r) for r in ROLES}
        _add(server.gui.add_html(
            "<div style='font-size:11px;color:#bbb;margin:-2px 0 4px;'>"
            f"{n_on}/{len(panel.leds)} on: <span style='color:{_ROLE_COLORS['vio']};'>■ VIO {counts['vio']}</span> &nbsp;"
            f"<span style='color:{_ROLE_COLORS['flash']};'>■ Flash {counts['flash']}</span> &nbsp;"
            f"<span style='color:{_ROLE_COLORS['both']};'>■ Both {counts['both']}</span><br>"
            f"Flight flux <b>{float(led_lumens_slider.value):,.0f} lm</b>/LED · Flash flux <b>{flash_lumens():,.0f} lm</b>/LED "
            "(Display tab)</div>"))
        all_btn = _add(server.gui.add_button("ALL LEDs", color="#BBBBBB" if n_on else "#666666"))
        all_btn.on_click(lambda _: _apply(range(len(panel.leds))))
        if len(rows) > 1:
            for r, members in enumerate(rows):
                any_on = any(panel.leds[i].on for i in members if i < len(panel.leds))
                b = _add(server.gui.add_button(f"Row {r + 1}", color="#BBBBBB" if any_on else "#666666"))
                b.on_click(lambda _, idx=list(members): _apply(idx))
        for i, led in enumerate(panel.leds):
            b = _add(server.gui.add_button(f"L{i + 1}", color=_ROLE_COLORS.get(led.role, "#FFFFFF") if led.on else "#444444"))
            b.on_click(lambda _, i=i: _apply([i]))

        angles = sorted({round(float(l.beam_angle), 1) for l in panel.leds})
        beam = _add(server.gui.add_number(
            "Beam angle, all LEDs (°)", float(angles[0]) if len(angles) == 1 else 0.0, min=0.0, max=180.0, step=1.0,
            hint="Full beam angle of every LED in this panel (0 = leave as is). Per-LED values: Designer."
                 + (f" Currently mixed: {', '.join(f'{a:g}' for a in angles)}°." if len(angles) > 1 else "")))

        @beam.on_update
        def _(_):
            v = float(beam.value)
            if v <= 0:
                return
            for l in panel.leds:
                l.beam_angle = v
            state.notify()

        names = profile_names()
        used = {l.profile or LAMBERTIAN for l in panel.leds}
        cur = next(iter(used)) if len(used) == 1 else LAMBERTIAN
        prof = _add(server.gui.add_dropdown(
            "Beam profile, all LEDs", options=names, initial_value=cur if cur in names else LAMBERTIAN,
            hint="Measured intensity-vs-angle curve of the emitter (datasheet polar plot) instead of the cosⁿ model. "
                 "With a profile the beam angle above is ignored. Add your own as beam_profiles/<name>.json."
                 + (f" Currently mixed: {', '.join(sorted(used))}." if len(used) > 1 else "")))

        @prof.on_update
        def _(_):
            name = None if prof.value == LAMBERTIAN else prof.value
            for l in panel.leds:
                l.profile = name
            state.notify()
            populate_inspector(index)

        p_obj = get_profile(cur)
        if p_obj is not None:
            _add(server.gui.add_html(
                f"<div style='font-size:10px;color:#888;margin:-4px 0 4px;'>{p_obj.name}: 50 % at "
                f"{p_obj.half_intensity_angle_deg():.0f}° full angle, emits to ±{p_obj.max_angle_deg:.0f}°.</div>"))

    def _guide_controls(panel: Panel, index):
        guide = state.guides.get(index)
        chk = _add(server.gui.add_checkbox("Anchor to duct axis (fit from LED rows)", initial_value=guide is not None,
                                           hint="Fits a cylinder through the LED rows; the slider then orbits the "
                                                "panel around that axis (pose sliders are hidden meanwhile)."))
        if guide is not None:
            if guide.get('warning'):
                _add(server.gui.add_markdown(f"*{guide['warning']}*"))
            sl = _add(server.gui.add_slider("Slide around the duct (°)", min=-180, max=180, step=0.5,
                                            initial_value=float(guide.get('theta_deg', 0.0))))
            sl.on_update(lambda _: state.set_guide_theta(index, sl.value))

        @chk.on_update
        def _(_):
            if chk.value:
                err = state.fit_guide(index)
                if err:
                    print(f"Guide: {err}")
            else:
                state.release_guide(index)
            populate_inspector(index)

    def populate_inspector(index):
        _clear_inspector()
        panel = state.panel(index)
        with inspector_tab:
            if panel is None:
                _add(server.gui.add_markdown("Click a panel in the 3-D view, or pick one in *Panels & LEDs*."))
                return
            name_in = _add(server.gui.add_text("Name", initial_value=panel.name))

            def _rename(_):
                new = name_in.value.strip()
                if new and new != panel.name:
                    panel.name = state.unique_name(new)
                    state.notify("structure")
            name_in.on_update(_rename)
            en = _add(server.gui.add_checkbox("Enabled", initial_value=panel.enabled))
            en.on_update(lambda _: (setattr(panel, 'enabled', bool(en.value)), state.notify("structure")))
            mi = _add(server.gui.add_checkbox("Mirror to the other side (left/right twin)", initial_value=panel.mirror))
            mi.on_update(lambda _: (setattr(panel, 'mirror', bool(mi.value)), state.notify("structure")))
            if index not in state.guides:
                pos = _add(server.gui.add_vector3("Position (cm)", tuple(panel.position), step=0.1))
                rot = _add(server.gui.add_vector3("Rotation X/Y/Z (°)", tuple(panel.rotation), step=0.5,
                                                  min=(-180.0, -180.0, -180.0), max=(180.0, 180.0, 180.0)))

                def _pose(_):
                    if syncing[0]:
                        return
                    panel.position = tuple(float(v) for v in pos.value)
                    panel.rotation = tuple(float(v) for v in rot.value)
                    state.notify()
                pos.on_update(_pose)
                rot.on_update(_pose)
            _guide_controls(panel, index)
            _led_matrix(panel, index)
            _add(server.gui.add_html("<hr style='margin:6px 0;'>"))
            edit_btn = _add(server.gui.add_button("✏️ Edit LEDs in the Designer", color="green"))
            edit_btn.on_click(lambda _: enter_designer[0](index))
            dup_btn = _add(server.gui.add_button("Duplicate panel"))
            dup_btn.on_click(lambda _: state.duplicate_panel(index))
            tpl_name = _add(server.gui.add_text("Template name", initial_value=panel.name))
            tpl_btn = _add(server.gui.add_button("💾 Save panel as template"))
            tpl_btn.on_click(lambda _: save_panel_as_template(panel, tpl_name.value.strip() or panel.name))
            rm_btn = _add(server.gui.add_button("Remove panel", color="red"))
            rm_btn.on_click(lambda _: state.remove_panel(index))

    # ------------------------------------------------------------------ react to state changes
    def _on_state_change(what):
        if what in ("structure", "selection"):
            rebuild_list()
            populate_inspector(state.selected)
        elif what == "guide":
            populate_inspector(state.selected)

    state.on_change(_on_state_change)
    rebuild_list()
    populate_inspector(None)

    return SimpleNamespace(populate_inspector=populate_inspector, rebuild_list=rebuild_list,
                           refresh_templates=refresh_templates, template_names=template_names,
                           save_panel_as_template=save_panel_as_template, clear_inspector=_clear_inspector)
