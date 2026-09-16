"""Optimize tab: design modes, spec builders, background runner, live plot and report hooks.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import copy
import json
import os
import threading as _threading
import time
import traceback as _traceback
import webbrowser as _wb
import numpy as np
from lighting_simulator.domain.guides import circle_line_segments_m as _circle_line_segments_m
from lighting_simulator.simulation import gpu_backend as _gpu_backend
from lighting_simulator.optimization import (
    OptimizerSpec as _OptimizerSpec,
    problem_from_spec as _problem_from_spec,
    run as _run_optimization,
)


def build(ctx):
    _project_root = ctx._project_root
    apply_config = ctx.apply_config
    camera_fov_h = ctx.camera_fov_h
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    tilt_fov_deg = ctx.tilt_fov_deg
    flash_current_input = ctx.flash_current_input
    led_voltage_input = ctx.led_voltage_input
    led_efficacy_input = ctx.led_efficacy_input
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    config_dir = ctx.config_dir
    config_dropdown = ctx.config_dropdown
    current_config_name = ctx.current_config_name
    custom_groups = ctx.custom_groups
    diffuser_angle_slider = ctx.diffuser_angle_slider
    diffuser_enable_chk = ctx.diffuser_enable_chk
    diffuser_transmission_slider = ctx.diffuser_transmission_slider
    get_available_configs = ctx.get_available_configs
    get_current_config = ctx.get_current_config
    led_lumens_slider = ctx.led_lumens_slider
    project_loaded = ctx.project_loaded
    ray_uniformity_slider = ctx.ray_uniformity_slider
    save_name_input = ctx.save_name_input
    server = ctx.server
    show_intensity_map = ctx.show_intensity_map
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_mesh_data = ctx.stl_mesh_data
    tab_optim = ctx.tab_optim
    uniformity_percentile_slider = ctx.uniformity_percentile_slider
    update_intensity_map = ctx.update_intensity_map
    vio_cam1_pitch = ctx.vio_cam1_pitch
    vio_cam1_yaw = ctx.vio_cam1_yaw
    vio_cam2_pitch = ctx.vio_cam2_pitch
    vio_cam2_yaw = ctx.vio_cam2_yaw
    vio_landscape = ctx.vio_landscape
    vio_long_fov = ctx.vio_long_fov
    vio_pos_x = ctx.vio_pos_x
    vio_pos_y = ctx.vio_pos_y
    vio_pos_z = ctx.vio_pos_z
    wall_dist_slider = ctx.wall_dist_slider
    wall_view_size = ctx.wall_view_size

    # =====================================================================
    # Optimize tab — runs lighting_simulator.optimization in a background thread
    # =====================================================================
    optim_specs_dir = ("optimization_specs" if os.path.isdir("optimization_specs")
                       else os.path.join(_project_root, "optimization_specs"))
    optim_output_dir = os.path.join("exports", "optim") if os.path.isdir("exports") \
        else os.path.join(_project_root, "exports", "optim")
    _OPTIM_NO_SPEC = "— none (build from UI) —"
    _OPTIM_NO_GROUP = "(no custom groups)"
    _optim_state = {
        'thread': None, 'stop': _threading.Event(), 'budget': 1,
        'best_cfg': None, 'evals': [], 'scores': [], 'bests': [], 'last_ui': 0.0,
    }

    def _optim_spec_names():
        names = [_OPTIM_NO_SPEC]
        if os.path.isdir(optim_specs_dir):
            names += sorted((f[:-5] for f in os.listdir(optim_specs_dir) if f.lower().endswith(".json")),
                            key=str.lower)
        return names

    def _optim_group_labels():
        labels = []
        for idx, g in enumerate(custom_groups):
            name = (g.get('panel_slot_name') or g.get('template_name')
                    or getattr(g.get('folder'), 'label', None) or f"Custom Group {g['id']}")
            labels.append(f"{idx}: {name}")
        return labels or [_OPTIM_NO_GROUP]

    _MODE_REFINE = "1 · Refine the current panels"
    _MODE_DUCTS = "2 · Design LEDs on the ducts (from scratch)"
    _MODE_PRESET = "3 · Run a preset spec file"
    _VIO_GEOM_ROOM = "Room (6 walls around the rig)"
    _VIO_GEOM_WALL = "Single far wall"
    _OBJ_BOTH = "Flight + flash (weighted)"
    _OBJ_FLIGHT = "Flight image only"
    _OBJ_FLASH = "Flash image only"

    with tab_optim:
        server.gui.add_html(
            "<div style='color:#bbb;font-size:12px;line-height:1.4;margin-bottom:4px;'>"
            "<b>How it works</b> — the optimiser repeatedly ray-traces candidate LED layouts and keeps the one "
            "with the most uniform light inside the main-camera FOV (plus your constraints). Pick a design mode:"
            "<ul style='margin:4px 0 0 14px;padding:0;'>"
            "<li><b>1 Refine</b>: keep the scene, nudge one existing panel (position, tilt, beam, on/off, current).</li>"
            "<li><b>2 Ducts</b>: remove all LEDs and place a new symmetric lattice on the duct rings within tolerances.</li>"
            "<li><b>3 Preset</b>: run a JSON spec from <code>optimization_specs/</code> as-is (optionally overriding parts with the UI).</li>"
            "</ul>Camera, emission and VIO poses always come from the FOV / Intensity tabs. "
            "Results go to <code>exports/optim/&lt;run name&gt;/</code> (best_config.json + report.pdf).</div>",
            order=1,
        )
        optim_mode = server.gui.add_dropdown("Design mode", options=[_MODE_REFINE, _MODE_DUCTS, _MODE_PRESET],
                                             initial_value=_MODE_REFINE, order=2)

    # -- ① Refine ---------------------------------------------------------
    with tab_optim:
        _optim_refine_folder = server.gui.add_folder("① Panel to refine", order=10)
    with _optim_refine_folder:
        optim_group_dropdown = server.gui.add_dropdown("Group", options=_optim_group_labels())
        optim_refresh_groups_btn = server.gui.add_button("🔄 Refresh group list")
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>What may change</div>")
        optim_var_pose = server.gui.add_checkbox("Move / rotate the panel", initial_value=True)
        optim_pos_delta = server.gui.add_vector3("± position (cm)", (2.0, 2.0, 2.0),
                                                 min=(0.0, 0.0, 0.0), max=(50.0, 50.0, 50.0), step=0.5)
        optim_rot_delta = server.gui.add_vector3("± rotation (°)", (15.0, 15.0, 20.0),
                                                 min=(0.0, 0.0, 0.0), max=(180.0, 180.0, 180.0), step=1.0)
        optim_var_tilts = server.gui.add_checkbox("Per-LED beam tilt", initial_value=False,
                                                  hint="Dynamic (designer / template) groups only")
        optim_tilt_range = server.gui.add_slider("± beam tilt (°)", min=5, max=90, step=5, initial_value=45)
        optim_var_beam = server.gui.add_checkbox("Shared beam angle", initial_value=False)
        optim_beam_range = server.gui.add_multi_slider("Beam angle range (°)", min=30, max=180, step=5,
                                                       initial_value=(90, 130))
        optim_var_states = server.gui.add_checkbox("LED on / off", initial_value=False)
        optim_var_roles = server.gui.add_checkbox(
            "LED roles (off / VIO / flash / both)", initial_value=False,
            hint="Lets the optimiser decide which LEDs flash and which stay on for VIO; needs flash mode below. "
                 "Replaces 'LED on / off'.",
        )
        optim_var_current = server.gui.add_checkbox("Drive current (→ lumens)", initial_value=False,
                                                    hint="Shared continuous per-LED current; lumens = I · V · efficacy")
        optim_current_range = server.gui.add_multi_slider("Current range (A)", min=0.1, max=13.0, step=0.1,
                                                          initial_value=(0.5, 3.0))

    # -- ③ Preset ---------------------------------------------------------
    with tab_optim:
        _optim_preset_folder = server.gui.add_folder("③ Preset spec", order=30)
    with _optim_preset_folder:
        optim_spec_dropdown = server.gui.add_dropdown(
            "Spec file", options=_optim_spec_names(), initial_value=_OPTIM_NO_SPEC,
            hint="optimization_specs/*.json",
        )
        optim_refresh_specs_btn = server.gui.add_button("🔄 Refresh spec list")
        optim_spec_info_html = server.gui.add_html("<div style='color:#888;font-size:12px;'>No spec selected.</div>")
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Overrides (off = use the spec as written)</div>")
        optim_preset_use_scene = server.gui.add_checkbox(
            "Start from the current scene", initial_value=False,
            hint="Replaces the spec's base_config with the loaded scene",
        )
        optim_preset_use_wall = server.gui.add_checkbox(
            "Use the Evaluation folder (walls, camera, emission)", initial_value=False,
        )
        optim_preset_use_obj = server.gui.add_checkbox(
            "Use the Objective / Electrical folders", initial_value=False,
        )
        optim_copy_preset_btn = server.gui.add_button(
            "📋 Copy spec into the controls & switch mode",
            hint="Fills every folder from the spec, then selects mode 1 or 2 so you can edit and run it",
        )

    # -- Evaluation (modes 1 & 2, or preset override) -------------------------
    with tab_optim:
        _optim_eval_folder = server.gui.add_folder("Evaluation walls", order=40)
    with _optim_eval_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:4px;'>Where uniformity is measured. Camera FOV "
            "and LED lumens are taken from the FOV / Intensity tabs.</div>"
        )
        optim_wall_dists = server.gui.add_text(
            "Wall distances (cm)", initial_value="",
            hint="Comma-separated; score is averaged over them. Empty = Intensity Map wall distance",
        )
        optim_grid = server.gui.add_slider(
            "Wall grid resolution", min=5, max=200, step=5, initial_value=60,
            hint="Cells per side. Aim for ~1 cm cells inside the FOV (auto wall size at 50 cm ≈ 80 cm → 60–80); "
                 "coarser blurs beam overlaps, finer needs more rays",
        )
        _WS_AUTO, _WS_FIXED, _WS_LIST = "Auto: fit camera FOV at each distance", "Fixed size", "Custom list"
        optim_wall_size_mode = server.gui.add_dropdown("Wall size", options=[_WS_AUTO, _WS_FIXED, _WS_LIST],
                                                       initial_value=_WS_AUTO)
        optim_wall_size = server.gui.add_slider(
            "Fixed wall size (cm)", min=100, max=2000, step=10, initial_value=int(wall_view_size.value),
            visible=False,
        )
        optim_wall_sizes = server.gui.add_text(
            "Wall sizes per distance (cm)", initial_value="", visible=False,
            hint="Comma list matching 'Wall distances'",
        )
        optim_rpp = server.gui.add_number("Rays per pixel", 1500, min=10, max=100000, step=10,
                                          hint="Rays are random: aim for ≥ 1000 hits per FOV cell (≈ 3 % noise). "
                                               "Every new best is re-checked with fresh rays")

        @optim_wall_size_mode.on_update
        def _(_):
            optim_wall_size.visible = optim_wall_size_mode.value == _WS_FIXED
            optim_wall_sizes.visible = optim_wall_size_mode.value == _WS_LIST

    with tab_optim:
        _optim_duct_folder = server.gui.add_folder("② Ducts", order=20)
    with _optim_duct_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:6px;'>Enter the nominal duct geometry and the "
            "mechanical tolerances; the optimiser places an LED lattice on the duct surface (and its "
            "left/right mirror) anywhere inside those tolerances. Existing LEDs are removed.</div>"
        )
        duct_show = server.gui.add_checkbox("Show ducts in 3D", initial_value=True)
        duct_center = server.gui.add_vector3("Duct centre (cm)", (6.3, 12.0, 0.0), step=0.1)
        duct_radius = server.gui.add_number("Duct radius (cm)", 8.5, min=1.0, max=50.0, step=0.1)
        duct_axis = server.gui.add_dropdown("Duct axis", options=["Z (vertical)", "Y (lateral)", "X (forward)"],
                                            initial_value="Z (vertical)")
        duct_rot = server.gui.add_vector3("Duct rotation X/Y/Z (°)", (0.0, 0.0, 0.0),
                                          min=(-90.0, -90.0, -180.0), max=(90.0, 90.0, 180.0), step=0.5,
                                          hint="Tilt of the duct about its centre (applied X, then Y, then Z); "
                                               "the mirrored duct is tilted symmetrically")
        duct_mirror = server.gui.add_checkbox("Mirror across XZ (symmetric pair)", initial_value=True)
        duct_theta0 = server.gui.add_slider("Nominal LED position on duct (°, 0 = +X)", min=-180, max=180,
                                            step=5, initial_value=30)
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>Tolerances</div>")
        duct_tol_radius = server.gui.add_number("± radius / stand-off (cm)", 2.0, min=0.0, max=20.0, step=0.5)
        duct_tol_arc = server.gui.add_number("± around the duct (cm along circumference)", 20.0,
                                             min=0.0, max=100.0, step=1.0)
        duct_tol_axial = server.gui.add_number("± along duct axis (cm)", 2.0, min=0.0, max=20.0, step=0.5)
        duct_tol_center = server.gui.add_vector3("± duct centre shift (cm)", (0.0, 0.0, 0.0),
                                                 min=(0.0, 0.0, 0.0), max=(50.0, 50.0, 50.0), step=0.5)
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>LED lattice</div>")
        duct_rows = server.gui.add_number("Max rows (along axis)", 3, min=1, max=8, step=1)
        duct_cols = server.gui.add_number("Max columns (around duct)", 4, min=1, max=12, step=1)
        duct_var_counts = server.gui.add_checkbox("Optimise row / column count", initial_value=True)
        duct_span_cm = server.gui.add_multi_slider("Lattice arc span (cm)", min=1, max=60, step=1,
                                                   initial_value=(3, 25))
        duct_pitch_cm = server.gui.add_multi_slider("Row pitch (cm)", min=0.5, max=5.0, step=0.1,
                                                    initial_value=(0.8, 2.0))
        duct_tilt = server.gui.add_slider("± beam tilt toward axis (°)", min=0, max=90, step=5, initial_value=60)
        duct_tilt_shared = server.gui.add_checkbox("Shared tilt for all LEDs", initial_value=False)
        duct_tilt_symmetric = server.gui.add_checkbox(
            "Mirror tilt top / bottom", initial_value=False,
            hint="Arc grid: rows above the panel middle look up by +t, rows below look down by −t "
                 "(one tilt per row pair, middle row straight). Overrides 'Shared tilt'.",
        )
        duct_beam = server.gui.add_multi_slider("Beam angle range (°)", min=30, max=180, step=5,
                                                initial_value=(90, 130))
        duct_current = server.gui.add_multi_slider("Drive current range (A)", min=0.1, max=13.0, step=0.1,
                                                   initial_value=(0.3, 3.0))
        duct_on_off = server.gui.add_checkbox("Optimise per-LED on/off", initial_value=False)
        duct_roles = server.gui.add_checkbox("Optimise LED roles (VIO / flash / both)", initial_value=False,
                                             hint="Needs flash mode enabled below; replaces on/off")
        duct_led_size = server.gui.add_number("LED size (cm)", 1.0, min=0.2, max=3.0, step=0.1)

    _DUCT_AXES = {"Z (vertical)": ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
                  "Y (lateral)": ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
                  "X (forward)": ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])}

    def _duct_rotation():
        rot = [float(v) for v in duct_rot.value]
        return rot if any(abs(r) > 1e-9 for r in rot) else None

    def _duct_frame_vectors(mirror=False):
        """(axis, reference) after the duct rotation; ``mirror`` reflects them across XZ."""
        from lighting_simulator.optimization.variables import Duct
        axis, ref = _DUCT_AXES[duct_axis.value]
        a, u, _v = Duct(axis=tuple(axis), reference=tuple(ref), rotation_deg=_duct_rotation()).frame()
        if mirror:
            flip = np.array([1.0, -1.0, 1.0])
            a, u = a * flip, u * flip
        return a, u

    def _optim_duct_variables():
        from lighting_simulator.optimization.variables import arc_cm_to_deg
        axis, ref = _DUCT_AXES[duct_axis.value]
        r = float(duct_radius.value)
        d_theta = arc_cm_to_deg(duct_tol_arc.value, r)
        span_lo, span_hi = (arc_cm_to_deg(v, r) for v in duct_span_cm.value)
        rows, cols = int(duct_rows.value), int(duct_cols.value)
        t = float(duct_tilt.value)
        var = {
            'type': 'duct_ring', 'name': 'duct',
            'duct': {'center': [float(v) for v in duct_center.value], 'axis': axis, 'radius': r,
                     'reference': ref, 'mount_offset': 0.0, 'rotation_deg': _duct_rotation()},
            'placement': 'arc',
            'n_leds': rows * cols, 'n_rows': rows,
            'theta_range': [float(duct_theta0.value) - d_theta, float(duct_theta0.value) + d_theta],
            'arc_span_range': [max(0.5, span_lo), max(span_lo + 0.5, span_hi)],
            'axial_range': [-float(duct_tol_axial.value), float(duct_tol_axial.value)],
            'row_pitch_range': [float(v) for v in duct_pitch_cm.value],
            'radial_range': [-float(duct_tol_radius.value), float(duct_tol_radius.value)],
            'tilt_axial_range': [-t, t] if t > 0 else None,
            'shared_tilt': bool(duct_tilt_shared.value),
            'symmetric_tilt': bool(duct_tilt_symmetric.value),
            'beam_angle_range': [float(v) for v in duct_beam.value], 'shared_beam_angle': True,
            'current_range': [float(v) for v in duct_current.value],
            'optimize_enabled': bool(duct_on_off.value),
            'optimize_roles': bool(duct_roles.value and optim_flash_enable.value),
            'led_size': float(duct_led_size.value),
            'mirror_xz': bool(duct_mirror.value),
        }
        if duct_var_counts.value:
            var['n_rows_range'] = [1, rows]
            var['n_cols_range'] = [1, cols]
        if any(v > 0 for v in duct_tol_center.value):
            var['center_delta'] = [float(v) for v in duct_tol_center.value]
        return [var]

    _duct_preview_handles = []

    def _duct_sector_wireframe_m(center, axis, ref, theta_lo, theta_hi, r_lo, r_hi, t_lo, t_hi, n_arc=32):
        """Edges (metres) of a cylindrical-shell sector: a box 'wrapped' around the duct."""
        from lighting_simulator.optimization.variables import Duct
        a, u, v = Duct(center=tuple(center), axis=tuple(axis), radius=1.0, reference=tuple(ref)).frame()
        c = np.asarray(center, float)

        def p(theta_deg, r, t):
            th = np.radians(theta_deg)
            return c + r * (np.cos(th) * u + np.sin(th) * v) + t * a

        segs = []
        thetas = np.linspace(theta_lo, theta_hi, n_arc + 1)
        for r in (r_lo, r_hi):
            for t in (t_lo, t_hi):
                pts = np.array([p(th, r, t) for th in thetas])
                segs += [[pts[k], pts[k + 1]] for k in range(n_arc)]
        for th in (theta_lo, theta_hi):
            for t in (t_lo, t_hi):
                segs.append([p(th, r_lo, t), p(th, r_hi, t)])
            for r in (r_lo, r_hi):
                segs.append([p(th, r, t_lo), p(th, r, t_hi)])
        return np.asarray(segs, float) / 100.0

    def _draw_duct_preview(_=None):
        from lighting_simulator.optimization.variables import arc_cm_to_deg
        for h in _duct_preview_handles:
            try:
                h.remove()
            except Exception:
                pass
        _duct_preview_handles.clear()
        if not duct_show.value or optim_mode.value != _MODE_DUCTS:
            return
        centers = [np.asarray(duct_center.value, float)]
        if duct_mirror.value:
            centers.append(centers[0] * np.array([1.0, -1.0, 1.0]))
        r = float(duct_radius.value)
        tol_r, tol_t = float(duct_tol_radius.value), float(duct_tol_axial.value)
        theta0 = float(duct_theta0.value)
        d_theta = arc_cm_to_deg(duct_tol_arc.value, r)
        half_span = arc_cm_to_deg(duct_span_cm.value[1], r) / 2.0
        shift = float(max(duct_tol_center.value))
        half = tol_t + 2.0
        from lighting_simulator.optimization.variables import Duct
        _a, _u, _v = Duct(center=(0, 0, 0), axis=tuple(_DUCT_AXES[duct_axis.value][0]), radius=1.0,
                          reference=tuple(_DUCT_AXES[duct_axis.value][1])).frame()
        # XZ mirroring flips theta only if the circumferential direction runs along Y
        mirror_sign = -1.0 if abs(_v[1]) > 0.5 else 1.0
        for i, c in enumerate(centers):
            axis, ref = _duct_frame_vectors(mirror=(i == 1))
            th0 = theta0 * (mirror_sign if i == 1 else 1.0)
            for k, t in enumerate((-half, 0.0, half)):  # the duct itself
                segs = _circle_line_segments_m(c + axis * t, r, axis, n_seg=64)
                _duct_preview_handles.append(server.scene.add_line_segments(
                    f"/optim_ducts/{i}/ring_{k}", points=segs, colors=(1.0, 0.55, 0.1), line_width=2.0))
            # tolerance box for the lattice centre: ± arc, ± radius, ± axial (+ centre shift on r/t)
            segs = _duct_sector_wireframe_m(c, axis, ref, th0 - d_theta, th0 + d_theta,
                                            max(0.1, r - tol_r - shift), r + tol_r + shift,
                                            -tol_t - shift, tol_t + shift)
            _duct_preview_handles.append(server.scene.add_line_segments(
                f"/optim_ducts/{i}/tol_box", points=segs, colors=(1.0, 0.85, 0.2), line_width=2.5))
            # envelope reachable by any LED of the lattice (adds half the max arc span each side)
            if half_span > 0:
                segs = _duct_sector_wireframe_m(c, axis, ref, th0 - d_theta - half_span, th0 + d_theta + half_span,
                                                max(0.1, r - tol_r - shift), r + tol_r + shift,
                                                -tol_t - shift, tol_t + shift)
                _duct_preview_handles.append(server.scene.add_line_segments(
                    f"/optim_ducts/{i}/envelope", points=segs, colors=(1.0, 0.95, 0.6), line_width=1.0))

    for _h in (duct_show, duct_center, duct_radius, duct_axis, duct_rot, duct_mirror, duct_tol_radius,
               duct_tol_axial, duct_theta0, duct_tol_arc, duct_span_cm, duct_tol_center):
        _h.on_update(_draw_duct_preview)

    def _optim_mode_changed(_=None):
        mode = optim_mode.value
        preset = mode == _MODE_PRESET
        _optim_refine_folder.visible = mode == _MODE_REFINE
        _optim_duct_folder.visible = mode == _MODE_DUCTS
        _optim_preset_folder.visible = preset
        _optim_eval_folder.visible = (not preset) or optim_preset_use_wall.value
        _optim_obj_folder.visible = (not preset) or optim_preset_use_obj.value
        _optim_elec_folder.visible = (not preset) or optim_preset_use_obj.value
        _draw_duct_preview()

    optim_mode.on_update(_optim_mode_changed)
    optim_preset_use_wall.on_update(_optim_mode_changed)
    optim_preset_use_obj.on_update(_optim_mode_changed)

    with tab_optim:
        _optim_elec_folder = server.gui.add_folder("Electrical & operating modes", order=60)
    with _optim_elec_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:6px;'>LED flux is linear in current. "
            "Two operating points share the geometry: <b>flight</b> = VIO + Both LEDs at their continuous current, "
            "<b>flash</b> = Flash + Both LEDs at the pulse current (VIO LEDs stay continuous).</div>"
        )
        optim_drv_voltage = led_voltage_input  # shared with the Display tab ("Electrical")
        optim_drv_efficacy = led_efficacy_input
        server.gui.add_html("<div style='color:#888;font-size:11px;'>Forward voltage / efficacy / flash current are "
                            "taken from the Display tab → Electrical.</div>")
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Pulse driver (Flash / Both LEDs)</div>")
        optim_drv_max_current = server.gui.add_number("Max current per LED (A)", 13.0, min=0.1, max=50.0, step=0.1)
        optim_leds_per_driver = server.gui.add_number("LEDs per pulse driver", 4, min=1, max=64, step=1)
        optim_max_pulse_drivers = server.gui.add_number("Max pulse drivers (0 = no limit)", 0, min=0, step=1)
        optim_pulse_driver_cost = server.gui.add_slider("Cost per pulse driver", min=0.0, max=0.2, step=0.005,
                                                        initial_value=0.0)
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Continuous driver (VIO LEDs)</div>")
        optim_cont_max_current = server.gui.add_number("Max current per LED (A)", 3.0, min=0.1, max=50.0, step=0.1)
        optim_cont_leds_per_driver = server.gui.add_number("LEDs per continuous driver", 8, min=1, max=64, step=1)
        optim_max_cont_drivers = server.gui.add_number("Max continuous drivers (0 = no limit)", 0, min=0, step=1)
        optim_cont_driver_cost = server.gui.add_slider("Cost per continuous driver", min=0.0, max=0.2, step=0.005,
                                                       initial_value=0.0)
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Budgets (all drivers)</div>")
        optim_max_drivers = server.gui.add_number("Max drivers total (0 = no limit)", 0, min=0, step=1)
        optim_driver_cost = server.gui.add_slider("Cost per driver (any class)", min=0.0, max=0.2, step=0.005,
                                                  initial_value=0.0)
        optim_max_current = server.gui.add_number("Max continuous current (A, 0 = off)", 0.0, min=0.0, step=1.0,
                                                  hint="Sum over the LEDs lit in flight")
        optim_max_peak_current = server.gui.add_number("Max peak current in flash (A, 0 = off)", 0.0, min=0.0,
                                                       step=5.0, hint="VIO continuous + (Flash + Both) × flash current")
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>Flash mode (photogrammetry pulse)</div>")
        optim_flash_enable = server.gui.add_checkbox(
            "Enable flash mode", initial_value=False,
            hint="Off: every LED is continuous and roles are ignored (single operating point). "
                 "On: LED roles decide which LEDs pulse; the flash image is scored too.",
        )
        optim_objective_mode = server.gui.add_dropdown(
            "Optimise for", options=[_OBJ_BOTH, _OBJ_FLIGHT, _OBJ_FLASH], initial_value=_OBJ_BOTH,
            hint="Which operating point's uniformity drives the score (constraints always apply to both)",
        )
        optim_flash_current = flash_current_input  # shared with the Display tab
        optim_flash_uni_w = server.gui.add_slider("Flash uniformity weight", min=0.0, max=5.0, step=0.1,
                                                  initial_value=1.0)
        optim_flash_lux = server.gui.add_number("Flash avg lux in FOV (0 = off)", 41000, min=0, step=1000)
        optim_flash_dist = server.gui.add_number("… at wall distance (cm)", 50, min=10, max=1500, step=5,
                                                 hint="Must be one of the wall distances above (nearest is used)")
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>VIO coverage (flight mode)</div>")
        optim_vio_enable = server.gui.add_checkbox("Require VIO FOV coverage", initial_value=False,
                                                   hint="Uses the VIO camera poses from the FOV tab")
        optim_vio_lux = server.gui.add_number("Min lux on VIO surfaces", 120, min=0, step=10)
        optim_vio_fraction = server.gui.add_slider("Min share of VIO FOV lit (%)", min=0, max=100, step=5,
                                                   initial_value=50)
        optim_vio_geometry = server.gui.add_dropdown(
            "Evaluate VIO on", options=[_VIO_GEOM_ROOM, _VIO_GEOM_WALL], initial_value=_VIO_GEOM_ROOM,
            hint="Room: six walls around the rig (what the fisheyes really see). Wall: a single far plane.")
        optim_vio_room_dist = server.gui.add_number("Room wall distance (cm)", 300, min=50, max=2000, step=10,
                                                    hint="Opposite walls are twice this apart (300 → 6 m room)")
        optim_vio_room_grid = server.gui.add_number("Room grid per wall", 20, min=5, max=80, step=5,
                                                    hint="Coarse on purpose: 20 → 30 cm cells in a 6 m room")
        optim_vio_dist = server.gui.add_number("VIO wall distance (cm)", 300, min=50, max=2000, step=10)
        optim_vio_wall_size = server.gui.add_number("VIO wall size (cm)", 1200, min=100, max=5000, step=50)
        optim_vio_grid = server.gui.add_number("VIO wall grid resolution", 40, min=5, max=200, step=5)

    def _optim_vio_geometry_changed(_=None):
        room = optim_vio_geometry.value == _VIO_GEOM_ROOM
        for h in (optim_vio_room_dist, optim_vio_room_grid):
            h.visible = room
        for h in (optim_vio_dist, optim_vio_wall_size, optim_vio_grid):
            h.visible = not room

    def _optim_flash_changed(_=None):
        on = bool(optim_flash_enable.value)
        for h in (optim_objective_mode, optim_flash_uni_w, optim_flash_lux, optim_flash_dist,
                  optim_cont_max_current, optim_cont_leds_per_driver, optim_max_cont_drivers, optim_cont_driver_cost,
                  optim_max_pulse_drivers, optim_pulse_driver_cost, optim_max_peak_current, optim_var_roles, duct_roles):
            h.visible = on

    optim_flash_enable.on_update(_optim_flash_changed)
    _optim_flash_changed()

    optim_vio_geometry.on_update(_optim_vio_geometry_changed)
    _optim_vio_geometry_changed()

    with tab_optim:
        _optim_obj_folder = server.gui.add_folder("Objective & constraints", order=50)
    with _optim_obj_folder:
        optim_metric = server.gui.add_dropdown(
            "Metric", options=["u0", "u1", "cv"], initial_value="u0",
            hint="u0 = Emin/Eavg (Emin at the percentile below), u1 = Emin/Emax, cv = σ/Eavg (uses all cells; "
                 "least sensitive to ray noise)",
        )
        optim_min_pct = server.gui.add_slider("Emin percentile (%)", min=0.0, max=10.0, step=0.5, initial_value=2.0,
                                              hint="0 = single darkest cell (very noisy on fine grids); 2–5 recommended")
        optim_cov_w = server.gui.add_slider("Coverage penalty weight", min=0.0, max=5.0, step=0.1, initial_value=1.0)
        optim_min_lux = server.gui.add_number("Min average lux, normal mode (0 = off)", 0, min=0, step=10)
        optim_lux_w = server.gui.add_slider("Lux penalty weight", min=0.0, max=5.0, step=0.1, initial_value=1.0)
        optim_tilt_enable = server.gui.add_checkbox(
            "Add ±tilt FOV uniformity", initial_value=False,
            hint="Also score the camera pitched up and down by the FOV tab's 'Tilt FOV angle' (mean 1−U of both, "
                 "penalty 'tilt_uniformity'). With VIO geometry = Room the tilted footprints are measured on the "
                 "room ceiling / floor / walls; otherwise on the (enlarged) flat wall.",
        )
        optim_tilt_w = server.gui.add_slider("Tilt uniformity weight", min=0.0, max=5.0, step=0.1, initial_value=1.0)
        optim_max_leds = server.gui.add_number("Max active LEDs (0 = no limit)", 0, min=0, step=1)
        optim_max_leds_w = server.gui.add_slider("Penalty per LED over limit", min=0.0, max=1.0, step=0.01,
                                                 initial_value=0.05)
        optim_led_cost = server.gui.add_slider("Cost per active LED", min=0.0, max=0.1, step=0.001, initial_value=0.0)
        optim_spacing = server.gui.add_slider("Min LED spacing (cm, 0 = off)", min=0.0, max=10.0, step=0.1,
                                              initial_value=0.0)
        optim_spacing_w = server.gui.add_slider("Spacing penalty weight", min=0.0, max=5.0, step=0.1, initial_value=1.0)
        optim_min_beam_angle = server.gui.add_slider("Min beam angle off camera axis (°, 0 = off)", min=0, max=90,
                                                     step=5, initial_value=0,
                                                     hint="Prefer LEDs tilted at least this far from +X")
        optim_beam_angle_w = server.gui.add_slider("Beam angle penalty weight", min=0.0, max=5.0, step=0.1,
                                                   initial_value=0.5)
        optim_symmetry_w = server.gui.add_slider("Symmetry penalty weight (0 = off)", min=0.0, max=5.0, step=0.1,
                                                 initial_value=0.0,
                                                 hint="Share of LEDs without a left/right mirror partner")
        optim_keepout_html = server.gui.add_html(
            "<div style='color:#888;font-size:12px;'>Keep-out boxes: none (defined in preset specs)</div>"
        )

    with tab_optim:
        _optim_opt_folder = server.gui.add_folder("Optimizer", order=70)
    with _optim_opt_folder:
        optim_method = server.gui.add_dropdown(
            "Method", options=["differential_evolution", "nelder_mead", "random_search"],
            initial_value="differential_evolution",
            hint="differential_evolution: global, handles integer variables (rows/cols, on-off, roles) — use this "
                 "for layouts. nelder_mead: local polish of a continuous design only. random_search: quick scan.",
        )
        optim_max_evals = server.gui.add_number("Max evaluations", 3000, min=10, max=200000, step=10)
        optim_population = server.gui.add_number("Population", 60, min=4, max=2000, step=1,
                                                 hint="≈ 3× the number of variables")
        optim_seed = server.gui.add_number("Seed", 0, min=0, step=1)
        optim_workers = server.gui.add_number("CPU workers (-1 = all cores)", 1, min=-1, max=128, step=1,
                                              hint="Differential evolution only; ignored when the GPU is used")
        optim_use_gpu = server.gui.add_checkbox(
            "Use GPU", initial_value=_gpu_backend.HAS_GPU_MODULE,
            hint="Single-process GPU tracing; falls back to CPU if no backend passes the self-test",
        )
        optim_polish = server.gui.add_checkbox("Polish with Nelder-Mead", initial_value=False)
        optim_name = server.gui.add_text("Run name", initial_value="", hint="Output folder name; empty = auto")

    with tab_optim:
        _optim_run_folder = server.gui.add_folder("Run", order=80)
    with _optim_run_folder:
        optim_run_btn = server.gui.add_button("▶ Run optimization", color="green")
        optim_stop_btn = server.gui.add_button("■ Stop", color="red", disabled=True)
        optim_progress = server.gui.add_progress_bar(0.0)
        optim_status_html = server.gui.add_html(
            "<div style='color:#888;font-size:12px;'>Idle</div>"
        )
        _OPTIM_EMPTY_PLOT = tuple(np.array([1.0]) for _ in range(4))  # > 0 so the log axis is valid
        optim_plot = server.gui.add_uplot(
            data=_OPTIM_EMPTY_PLOT,
            series=(
                {"label": "eval"},
                {"label": "score", "stroke": "transparent", "width": 0,
                 "points": {"show": True, "size": 4, "fill": "#9e9e9e", "stroke": "#9e9e9e"}},
                {"label": "trend (moving avg)", "stroke": "#ff9800", "width": 2, "dash": [6, 4]},
                {"label": "best", "stroke": "#4CAF50", "width": 2},
            ),
            scales={"x": {"time": False}},
            aspect=1.6,
        )
        optim_plot_log = server.gui.add_checkbox("Logarithmic score axis", initial_value=True,
                                                 hint="Can be toggled while an optimisation is running")

        def _optim_plot_scale(_=None):
            # uPlot distr: 1 = linear, 3 = logarithmic (scores are always > 0)
            optim_plot.scales = {"x": {"time": False},
                                 "y": {"distr": 3, "log": 10} if optim_plot_log.value else {"distr": 1}}

        optim_plot_log.on_update(_optim_plot_scale)
        _optim_plot_scale()
        optim_autoload = server.gui.add_checkbox("Load best into scene when finished", initial_value=True)
        optim_load_btn = server.gui.add_button("📥 Load best into scene")
        optim_report_btn = server.gui.add_button("📄 Open PDF report", disabled=True)
        optim_save_name = server.gui.add_text("Save best as", initial_value="")
        optim_save_btn = server.gui.add_button("💾 Save best to configs/")

    _optim_mode_changed()  # initial folder visibility for the default mode

    def _optim_status(text, color="#ccc"):
        optim_status_html.content = (
            f"<div style='font-family:sans-serif;font-size:12px;color:{color};white-space:pre-wrap;'>{text}</div>"
        )

    def _optim_current_spec():
        name = optim_spec_dropdown.value
        if not name or name == _OPTIM_NO_SPEC:
            return None, None
        path = os.path.join(optim_specs_dir, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), os.path.dirname(os.path.abspath(path))

    def _optim_apply_spec_to_controls(spec):
        """Mirror a preset spec into the editable controls."""
        wall = spec.get('wall', {})
        dist = wall.get('wall_dist', 100)
        optim_wall_dists.value = ", ".join(f"{d:g}" for d in (dist if isinstance(dist, list) else [dist]))
        optim_grid.value = int(wall.get('grid_size', optim_grid.value))
        ws = wall.get('wall_size')
        if isinstance(ws, list):
            optim_wall_sizes.value = ", ".join(f"{s:g}" for s in ws)
            optim_wall_size_mode.value = _WS_LIST
        elif ws == "auto" or ws is None:
            optim_wall_size_mode.value = _WS_AUTO
        else:
            optim_wall_size_mode.value = _WS_FIXED
            optim_wall_size.value = int(ws)
        optim_wall_size.visible = optim_wall_size_mode.value == _WS_FIXED
        optim_wall_sizes.visible = optim_wall_size_mode.value == _WS_LIST
        optim_rpp.value = int(wall.get('rays_per_pixel', optim_rpp.value))
        obj = spec.get('objective', {})
        optim_metric.value = obj.get('metric', 'u0')
        optim_min_pct.value = float(obj.get('min_percentile', 0.0))
        optim_cov_w.value = float(obj.get('coverage_weight', 1.0))
        optim_min_lux.value = int(obj.get('min_avg_lux') or 0)
        optim_lux_w.value = float(obj.get('lux_weight', 1.0))
        optim_tilt_enable.value = bool(obj.get('tilt_fov_deg'))
        if obj.get('tilt_fov_deg'):
            tilt_fov_deg.value = int(round(float(obj['tilt_fov_deg']) / 5) * 5)
        optim_tilt_w.value = float(obj.get('tilt_fov_weight', 1.0))
        con = spec.get('constraints', {})
        optim_max_leds.value = int(con.get('max_leds') or 0)
        optim_max_leds_w.value = float(con.get('max_leds_weight', 0.05))
        optim_led_cost.value = float(con.get('led_cost', 0.0))
        optim_spacing.value = float(con.get('min_led_spacing_cm') or 0.0)
        optim_spacing_w.value = float(con.get('spacing_weight', 1.0))
        optim_min_beam_angle.value = int(con.get('min_beam_angle_deg') or 0)
        optim_beam_angle_w.value = float(con.get('beam_angle_weight', 0.5))
        optim_symmetry_w.value = float(con.get('symmetry_weight', 0.0))
        optim_max_drivers.value = int(con.get('max_drivers') or 0)
        optim_driver_cost.value = float(con.get('driver_cost', 0.0))
        optim_max_current.value = float(con.get('max_total_current_a') or 0.0)
        optim_max_peak_current.value = float(con.get('max_peak_current_a') or 0.0)
        optim_max_pulse_drivers.value = int(con.get('max_pulse_drivers') or 0)
        optim_pulse_driver_cost.value = float(con.get('pulse_driver_cost', 0.0))
        optim_max_cont_drivers.value = int(con.get('max_cont_drivers') or 0)
        optim_cont_driver_cost.value = float(con.get('cont_driver_cost', 0.0))
        drv = spec.get('driver', {})
        optim_drv_voltage.value = float(drv.get('voltage_v', 6.0))
        optim_drv_efficacy.value = float(drv.get('efficacy_lm_per_w', 180.0))
        optim_drv_max_current.value = float(drv.get('max_current_a', 13.0))
        optim_leds_per_driver.value = int(drv.get('leds_per_driver', 1))
        cdrv = spec.get('cont_driver') or {}
        optim_cont_max_current.value = float(cdrv.get('max_current_a', 3.0))
        optim_cont_leds_per_driver.value = int(cdrv.get('leds_per_driver', 8))
        optim_objective_mode.value = (_OBJ_FLASH if float(obj.get('flight_weight', 1.0)) == 0.0 else _OBJ_BOTH)
        vio = spec.get('vio') or {}
        optim_vio_geometry.value = _VIO_GEOM_ROOM if vio.get('geometry') == 'room' else _VIO_GEOM_WALL
        optim_vio_room_dist.value = int(vio.get('room_dist', 300))
        optim_vio_room_grid.value = int(vio.get('room_grid_size', 20))
        optim_vio_dist.value = int(vio.get('wall_dist', 300))
        optim_vio_wall_size.value = int(vio.get('wall_size', 1200))
        optim_vio_grid.value = int(vio.get('grid_size', 40))
        optim_flash_enable.value = False
        optim_vio_enable.value = False
        for m in spec.get('modes', []):
            is_flash = m.get('flash') if m.get('flash') is not None else m.get('current_a') is not None
            if is_flash:
                optim_flash_enable.value = True
                optim_flash_current.value = float(m.get('current_a') or optim_flash_current.value)
                optim_flash_lux.value = int(m.get('min_avg_lux') or 0)
                optim_flash_dist.value = int(m.get('min_avg_lux_dist') or optim_flash_dist.value)
                optim_flash_uni_w.value = float(m.get('uniformity_weight', 0.0))
                if float(obj.get('flight_weight', 1.0)) > 0 and not m.get('uniformity_weight'):
                    optim_objective_mode.value = _OBJ_FLIGHT
            if m.get('vio_min_lux'):
                optim_vio_enable.value = True
                optim_vio_lux.value = int(m['vio_min_lux'])
                optim_vio_fraction.value = int(round(100 * float(m.get('vio_min_fraction', 0.5))))
        _optim_flash_changed()
        n_keep = len(con.get('keep_out', []))
        optim_keepout_html.content = (
            f"<div style='color:#888;font-size:12px;'>Keep-out boxes: {n_keep} (from preset spec)</div>"
        )
        opt = spec.get('optimizer', {})
        optim_method.value = opt.get('method', 'differential_evolution')
        optim_max_evals.value = int(opt.get('max_evals', 300))
        optim_population.value = int(opt.get('population', 20))
        optim_seed.value = int(opt.get('seed', 0))
        optim_workers.value = int(opt.get('workers', 1))
        optim_polish.value = bool(opt.get('polish', False))
        optim_name.value = spec.get('name', '')
        _optim_apply_spec_variables(spec)

    def _optim_apply_spec_variables(spec):
        """Fill the ①/② folders from the spec's variables and select the matching mode."""
        variables = spec.get('variables', [])
        ducts = [v for v in variables if v.get('type') == 'duct_ring']
        if ducts:
            import math as _math
            v = ducts[0]
            d = v.get('duct', {})
            r = float(d.get('radius', duct_radius.value))
            duct_center.value = tuple(float(c) for c in d.get('center', duct_center.value))
            duct_radius.value = r
            ax = np.asarray(d.get('axis', [0, 0, 1]), float)
            duct_axis.value = max(_DUCT_AXES, key=lambda k: abs(np.dot(_DUCT_AXES[k][0], ax)))
            duct_rot.value = tuple(float(c) for c in (d.get('rotation_deg') or (0.0, 0.0, 0.0)))
            duct_mirror.value = bool(v.get('mirror_xz', False))
            th = v.get('theta_range', [-60, 60])
            duct_theta0.value = int(round((th[0] + th[1]) / 2 / 5) * 5)
            duct_tol_arc.value = round(r * _math.radians((th[1] - th[0]) / 2), 1)
            span = v.get('arc_span_range', [10, 120])
            duct_span_cm.value = (max(1, int(round(r * _math.radians(span[0])))),
                                  max(2, int(round(r * _math.radians(span[1])))))
            axial = v.get('axial_range', [-1.5, 1.5])
            duct_tol_axial.value = max(abs(axial[0]), abs(axial[1]))
            rad = v.get('radial_range')
            duct_tol_radius.value = max(abs(rad[0]), abs(rad[1])) if rad else 0.0
            duct_tol_center.value = tuple(float(c) for c in v.get('center_delta', (0.0, 0.0, 0.0)))
            n_rows = int(v.get('n_rows', 1))
            n_cols = max(1, int(v.get('n_leds', n_rows)) // n_rows)
            rr, cc = v.get('n_rows_range'), v.get('n_cols_range')
            duct_var_counts.value = bool(rr or cc)
            duct_rows.value = int(rr[1]) if rr else n_rows
            duct_cols.value = int(cc[1]) if cc else n_cols
            duct_pitch_cm.value = tuple(float(p) for p in v.get('row_pitch_range', (0.8, 2.0)))
            tilt = v.get('tilt_axial_range')
            duct_tilt.value = int(max(abs(tilt[0]), abs(tilt[1]))) if tilt else 0
            duct_tilt_shared.value = bool(v.get('shared_tilt', True))
            duct_tilt_symmetric.value = bool(v.get('symmetric_tilt', False))
            duct_beam.value = tuple(float(b) for b in v.get('beam_angle_range', (90, 130)))
            duct_current.value = tuple(float(c) for c in v.get('current_range', (0.3, 3.0)))
            duct_on_off.value = bool(v.get('optimize_enabled', False))
            duct_roles.value = bool(v.get('optimize_roles', False))
            duct_led_size.value = float(v.get('led_size', 1.0))
            optim_mode.value = _MODE_DUCTS
        else:
            types = {v.get('type') for v in variables}
            gi = next((int(v['group_index']) for v in variables if 'group_index' in v), None)
            if gi is not None:
                label = next((l for l in optim_group_dropdown.options if l.startswith(f"{gi}:")), None)
                if label:
                    optim_group_dropdown.value = label
            optim_var_pose.value = 'panel_pose' in types
            optim_var_tilts.value = 'beam_tilts' in types
            optim_var_beam.value = 'beam_angle' in types
            optim_var_states.value = 'led_states' in types
            optim_var_roles.value = 'led_roles' in types
            optim_var_current.value = 'group_current' in types
            for v in variables:
                if v.get('type') == 'panel_pose':
                    optim_pos_delta.value = tuple(float(x) for x in v.get('pos_delta', (2, 2, 2)))
                    optim_rot_delta.value = tuple(float(x) for x in v.get('rot_delta', (10, 10, 10)))
                elif v.get('type') == 'beam_tilts':
                    t = v.get('tilt_range', (-20, 20))
                    optim_tilt_range.value = int(max(abs(t[0]), abs(t[1])))
                elif v.get('type') == 'beam_angle':
                    optim_beam_range.value = tuple(float(x) for x in v.get('angle_range', (60, 130)))
                elif v.get('type') == 'group_current':
                    optim_current_range.value = tuple(float(x) for x in v.get('current_range', (0.5, 3.0)))
            optim_mode.value = _MODE_REFINE
        _optim_mode_changed()

    def _optim_describe_spec(spec):
        kinds = {}
        for v in spec.get('variables', []):
            kinds[v.get('type')] = kinds.get(v.get('type'), 0) + 1
        wall = spec.get('wall', {})
        dist = wall.get('wall_dist', '?')
        modes = ", ".join(m.get('name', '?') for m in spec.get('modes', [])) or "none"
        return ("<div style='color:#bbb;font-size:12px;line-height:1.4;'>"
                f"<b>{spec.get('name', '')}</b><br>{spec.get('description', '')}<br>"
                f"base: <code>{spec.get('base_config', '?')}</code>"
                f"{' (cleared)' if spec.get('clear_base') else ''}<br>"
                f"walls: {dist} cm · grid {wall.get('grid_size', '?')} · {wall.get('rays_per_pixel', '?')} rpp<br>"
                f"variables: {', '.join(f'{k} ×{n}' for k, n in kinds.items()) or 'none'}<br>"
                f"modes: {modes} · optimizer: {spec.get('optimizer', {}).get('method', '?')}, "
                f"{spec.get('optimizer', {}).get('max_evals', '?')} evals</div>")

    @optim_spec_dropdown.on_update
    def _(_):
        try:
            spec, _dir = _optim_current_spec()
        except Exception as exc:
            optim_spec_info_html.content = f"<div style='color:#ff6666;font-size:12px;'>Could not read spec: {exc}</div>"
            return
        optim_spec_info_html.content = (_optim_describe_spec(spec) if spec
                                        else "<div style='color:#888;font-size:12px;'>No spec selected.</div>")

    @optim_copy_preset_btn.on_click
    def _(_):
        try:
            spec, _dir = _optim_current_spec()
        except Exception as exc:
            _optim_status(f"Could not read spec: {exc}", "#ff6666")
            return
        if spec is None:
            _optim_status("Select a spec file first.", "#ffaa00")
            return
        _optim_apply_spec_to_controls(spec)
        _optim_status(f"Copied '{optim_spec_dropdown.value}' into the controls — now in mode "
                      f"'{optim_mode.value}'. Edit anything and press Run.")

    @optim_refresh_specs_btn.on_click
    def _(_):
        cur = optim_spec_dropdown.value
        optim_spec_dropdown.options = _optim_spec_names()
        if cur in optim_spec_dropdown.options:
            optim_spec_dropdown.value = cur

    @optim_refresh_groups_btn.on_click
    def _(_):
        cur_g = optim_group_dropdown.value
        optim_group_dropdown.options = _optim_group_labels()
        if cur_g in optim_group_dropdown.options:
            optim_group_dropdown.value = cur_g

    def _optim_selected_group_index():
        label = optim_group_dropdown.value or ""
        if not label or label == _OPTIM_NO_GROUP or ':' not in label:
            raise ValueError("Select a custom group (click 'Refresh group list' after adding panels).")
        idx = int(label.split(':', 1)[0])
        if idx >= len(custom_groups):
            raise ValueError("Group list is stale — click 'Refresh group list'.")
        return idx

    def _optim_group_variables():
        gi = _optim_selected_group_index()
        group = custom_groups[gi]
        variables = []
        if optim_var_pose.value:
            variables.append({'type': 'panel_pose', 'group_index': gi,
                              'pos_delta': [float(v) for v in optim_pos_delta.value],
                              'rot_delta': [float(v) for v in optim_rot_delta.value]})
        if optim_var_tilts.value:
            if not group.get('is_dynamic'):
                raise ValueError("Per-LED beam tilt needs a dynamic (designer / template) group.")
            t = float(optim_tilt_range.value)
            variables.append({'type': 'beam_tilts', 'group_index': gi, 'tilt_range': [-t, t]})
        if optim_var_beam.value:
            lo, hi = optim_beam_range.value
            variables.append({'type': 'beam_angle', 'group_index': gi, 'angle_range': [float(lo), float(hi)]})
        if optim_var_roles.value and optim_flash_enable.value:
            variables.append({'type': 'led_roles', 'group_index': gi})
        elif optim_var_states.value:
            variables.append({'type': 'led_states', 'group_index': gi})
        if optim_var_current.value:
            lo, hi = optim_current_range.value
            variables.append({'type': 'group_current', 'group_index': gi, 'current_range': [float(lo), float(hi)]})
        if not variables:
            raise ValueError("Enable at least one variable checkbox.")
        return variables

    def _optim_ui_wall_sections(work):
        """Walls / camera / emission from the UI into ``work``."""
        work['camera'] = {'pos_x': float(camera_pos_x.value), 'pos_y': float(camera_pos_y.value),
                          'pitch': float(camera_pitch.value), 'fov_h': float(camera_fov_h.value),
                          'fov_v': float(camera_fov_v.value)}
        work['emission'] = {'default_lumens': float(led_lumens_slider.value),
                            'ray_uniformity': float(ray_uniformity_slider.value)}
        dists_txt = optim_wall_dists.value.strip()
        dists = ([float(t) for t in dists_txt.replace(';', ',').split(',') if t.strip()] if dists_txt
                 else [float(wall_dist_slider.value)])
        wall = {'wall_dist': dists, 'grid_size': int(optim_grid.value), 'rays_per_pixel': int(optim_rpp.value)}
        if optim_wall_size_mode.value == _WS_LIST:
            sizes_txt = optim_wall_sizes.value.strip()
            if not sizes_txt:
                raise ValueError("Wall size is 'Custom list' but the list is empty.")
            wall['wall_size'] = [float(t) for t in sizes_txt.replace(';', ',').split(',') if t.strip()]
        elif optim_wall_size_mode.value == _WS_FIXED:
            wall['wall_size'] = float(optim_wall_size.value)
        else:
            wall['wall_size'] = "auto"
        work['wall'] = wall

    def _optim_ui_objective_sections(work, keep_out=None, keep_out_weight=1.0):
        """Objective / constraints / driver / modes / VIO from the UI into ``work``."""
        work['objective'] = {
            'metric': optim_metric.value,
            'min_percentile': float(optim_min_pct.value),
            'flight_weight': 0.0 if (optim_flash_enable.value and optim_objective_mode.value == _OBJ_FLASH) else 1.0,
            'coverage_weight': float(optim_cov_w.value),
            'min_avg_lux': float(optim_min_lux.value) or None,
            'lux_weight': float(optim_lux_w.value),
            'tilt_fov_deg': float(tilt_fov_deg.value) if optim_tilt_enable.value else None,
            'tilt_fov_weight': float(optim_tilt_w.value),
        }
        work['constraints'] = {
            'max_leds': int(optim_max_leds.value) or None,
            'max_leds_weight': float(optim_max_leds_w.value),
            'led_cost': float(optim_led_cost.value),
            'max_drivers': int(optim_max_drivers.value) or None,
            'driver_cost': float(optim_driver_cost.value),
            'max_total_current_a': float(optim_max_current.value) or None,
            'max_peak_current_a': (float(optim_max_peak_current.value) or None) if optim_flash_enable.value else None,
            'max_pulse_drivers': (int(optim_max_pulse_drivers.value) or None) if optim_flash_enable.value else None,
            'pulse_driver_cost': float(optim_pulse_driver_cost.value) if optim_flash_enable.value else 0.0,
            'max_cont_drivers': (int(optim_max_cont_drivers.value) or None) if optim_flash_enable.value else None,
            'cont_driver_cost': float(optim_cont_driver_cost.value) if optim_flash_enable.value else 0.0,
            'min_led_spacing_cm': float(optim_spacing.value) or None,
            'spacing_weight': float(optim_spacing_w.value),
            'min_beam_angle_deg': float(optim_min_beam_angle.value) or None,
            'beam_angle_weight': float(optim_beam_angle_w.value),
            'symmetry_weight': float(optim_symmetry_w.value),
            'keep_out': list(keep_out or []),
            'keep_out_weight': float(keep_out_weight),
        }
        work['driver'] = {
            'voltage_v': float(optim_drv_voltage.value),
            'efficacy_lm_per_w': float(optim_drv_efficacy.value),
            'max_current_a': float(optim_drv_max_current.value),
            'leds_per_driver': int(optim_leds_per_driver.value),
        }
        if optim_flash_enable.value:
            work['cont_driver'] = {'max_current_a': float(optim_cont_max_current.value),
                                   'leds_per_driver': int(optim_cont_leds_per_driver.value)}
        else:
            work.pop('cont_driver', None)
        modes = [{'name': 'flight'}]
        if optim_vio_enable.value:
            modes[0].update({'vio_min_lux': float(optim_vio_lux.value),
                             'vio_min_fraction': float(optim_vio_fraction.value) / 100.0})
            work['vio'] = {
                'position': [float(vio_pos_x.value), float(vio_pos_y.value), float(vio_pos_z.value)],
                'cam1_pitch': float(vio_cam1_pitch.value), 'cam1_yaw': float(vio_cam1_yaw.value),
                'cam2_pitch': float(vio_cam2_pitch.value), 'cam2_yaw': float(vio_cam2_yaw.value),
                'long_fov': float(vio_long_fov.value), 'landscape': bool(vio_landscape.value),
                'geometry': 'room' if optim_vio_geometry.value == _VIO_GEOM_ROOM else 'wall',
                'room_dist': float(optim_vio_room_dist.value), 'room_grid_size': int(optim_vio_room_grid.value),
                'wall_dist': float(optim_vio_dist.value), 'wall_size': float(optim_vio_wall_size.value),
                'grid_size': int(optim_vio_grid.value),
            }
        else:
            work.pop('vio', None)
        if optim_flash_enable.value:
            flash_only = optim_objective_mode.value == _OBJ_FLASH
            flight_only = optim_objective_mode.value == _OBJ_FLIGHT
            modes.append({'name': 'flash', 'current_a': float(optim_flash_current.value),
                          'min_avg_lux': float(optim_flash_lux.value) or None,
                          'min_avg_lux_dist': float(optim_flash_dist.value),
                          'uniformity_weight': 0.0 if flight_only else (1.0 if flash_only else float(optim_flash_uni_w.value))})
        work['modes'] = modes if (optim_vio_enable.value or optim_flash_enable.value) else []

    def _optim_build():
        """Assemble (Problem, OptimizerSpec) for the selected design mode."""
        mode = optim_mode.value
        spec_dir = None
        base_cfg = None
        if mode == _MODE_PRESET:
            spec, spec_dir = _optim_current_spec()
            if spec is None:
                raise ValueError("Mode 3 needs a spec file — pick one in the '③ Preset spec' folder.")
            work = copy.deepcopy(spec)
            if optim_preset_use_scene.value:
                base_cfg = get_current_config()
                base_cfg['name'] = current_config_name[0] or 'scene'
                work.pop('base_config', None)
            if optim_preset_use_wall.value:
                _optim_ui_wall_sections(work)
            if optim_preset_use_obj.value:
                con = spec.get('constraints', {})
                _optim_ui_objective_sections(work, con.get('keep_out'), con.get('keep_out_weight', 1.0))
        else:
            work = {}
            base_cfg = get_current_config()
            base_cfg['name'] = current_config_name[0] or 'scene'
            _optim_ui_wall_sections(work)
            _optim_ui_objective_sections(work)
            if mode == _MODE_DUCTS:
                work['variables'] = _optim_duct_variables()
                work['clear_base'] = True  # the ducts carry the whole rig
            else:
                work['variables'] = _optim_group_variables()
                work['clear_base'] = False

        run_name = optim_name.value.strip()
        if run_name:
            work['name'] = run_name
        elif base_cfg is not None:
            stem = str(base_cfg['name']).lower().replace(' ', '_')
            work['name'] = stem if stem.endswith('_optim') else f"{stem}_optim"

        # Reproduce the live scene exactly: STL occluder and diffuser are UI-only state.
        extra = {}
        if base_cfg is not None:
            if stl_absorber_enable.value and stl_mesh_data[0] is not None:
                extra['stl_mesh'] = stl_mesh_data[0]
            if diffuser_enable_chk.value:
                extra['diffuser'] = (float(diffuser_angle_slider.value),
                                     float(diffuser_transmission_slider.value) / 100.0)

        problem = _problem_from_spec(work, spec_dir, base_cfg=base_cfg, use_gpu=optim_use_gpu.value, **extra)
        opt = _OptimizerSpec(
            method=optim_method.value, max_evals=int(optim_max_evals.value), seed=int(optim_seed.value),
            population=int(optim_population.value), workers=int(optim_workers.value),
            polish=bool(optim_polish.value), log_every=0,
        )
        return problem, opt

    def _optim_refresh_ui(logger, final=False):
        st = _optim_state
        n_done = max(logger.n, logger.n_external)
        optim_progress.value = float(min(100.0, 100.0 * n_done / max(1, st['budget'])))
        best = logger.best
        elapsed = time.perf_counter() - logger.t0
        rate = n_done / elapsed if elapsed > 0 else 0.0
        head = "Finished" if final else "Running"
        _optim_status(
            f"{head}: {n_done}/{st['budget']} evals, {elapsed:.0f}s ({rate:.1f} eval/s)"
            + (" [GPU]" if logger.problem.use_gpu else " [CPU]")
            + f"\nBest: {best.summary() if best else '—'}",
            "#4CAF50" if final else "#ccc",
        )
        xs, ys, bs = st['evals'], st['scores'], st['bests']
        if len(xs) > 1:
            y = np.asarray(ys, float)
            win = max(5, len(y) // 20)  # ~5 % of the run; centred moving average
            kernel = np.ones(win) / win
            trend = np.convolve(np.pad(y, (win // 2, win - 1 - win // 2), mode='edge'), kernel, mode='valid')
            step = max(1, len(xs) // 1500)
            optim_plot.data = (np.asarray(xs[::step], float), y[::step], trend[::step],
                               np.asarray(bs[::step], float))

    def _optim_on_eval(logger, ev, x):
        st = _optim_state
        st['evals'].append(max(logger.n, logger.n_external))
        st['scores'].append(ev.score)
        st['bests'].append(logger.best.score)
        now = time.perf_counter()
        if now - st['last_ui'] >= 0.3:
            st['last_ui'] = now
            _optim_refresh_ui(logger)

    def _optim_load_best():
        cfg = _optim_state['best_cfg']
        if cfg is None:
            print("[optim] No optimised configuration yet.")
            return
        project_loaded[0] = True
        current_config_name[0] = cfg.get('name', 'optim')
        apply_config(cfg)
        save_name_input.value = cfg.get('name', '')
        optim_group_dropdown.options = _optim_group_labels()
        # Show the same robust U0 the optimiser scored with.
        uniformity_percentile_slider.value = float(_optim_state.get('min_percentile', uniformity_percentile_slider.value))
        if show_intensity_map.value:
            update_intensity_map()
        print(f"[optim] Loaded best configuration into the scene: {cfg.get('description', '')}")

    def _optim_worker(problem, opt):
        st = _optim_state
        logger_ref = [None]

        def on_eval(logger, ev, x):
            logger_ref[0] = logger
            _optim_on_eval(logger, ev, x)

        try:
            summary, _best = _run_optimization(problem, opt, output_dir=optim_output_dir,
                                               on_eval=on_eval, stop_event=st['stop'])
            best_path = os.path.join(summary['run_dir'], "best_config.json")
            with open(best_path, "r", encoding="utf-8") as f:
                st['best_cfg'] = json.load(f)
            if logger_ref[0] is not None:
                _optim_refresh_ui(logger_ref[0], final=True)
            st['report'] = summary.get('report')
            optim_report_btn.disabled = not st['report']
            report_line = f"\nReport: {st['report']}" if st['report'] else "\n(report generation failed — see console)"
            if summary.get('stopped'):
                _optim_status(f"Stopped after {summary['evaluations']} evals.\nBest: {_best.summary()}\n"
                              f"Saved: {best_path}{report_line}", "#ffaa00")
            else:
                _optim_status(f"Finished: {summary['evaluations']} evals in {summary['elapsed_s']}s\n"
                              f"Best: {_best.summary()}\nSaved: {best_path}{report_line}", "#4CAF50")
            if not optim_save_name.value.strip():
                optim_save_name.value = os.path.basename(summary['run_dir'])
            if optim_autoload.value:
                _optim_load_best()
        except Exception as exc:
            _traceback.print_exc()
            _optim_status(f"Optimisation failed: {exc}", "#ff6666")
        finally:
            st['thread'] = None
            optim_run_btn.disabled = False
            optim_stop_btn.disabled = True

    @optim_run_btn.on_click
    def _(_):
        st = _optim_state
        if st['thread'] is not None:
            _optim_status("An optimisation is already running — press Stop first.", "#ffaa00")
            return
        try:
            problem, opt = _optim_build()
        except Exception as exc:
            _traceback.print_exc()
            _optim_status(f"Cannot build problem: {exc}", "#ff6666")
            return
        if problem.use_gpu:
            _optim_status("Checking GPU backend…")
            if not _gpu_backend.gpu_available():
                problem.use_gpu = False
                print("[optim] GPU requested but unavailable — using CPU.")
        st['stop'].clear()
        st['budget'] = opt.max_evals
        st['best_cfg'] = None
        st['min_percentile'] = float(problem.objective.min_percentile)
        st['report'] = None
        optim_report_btn.disabled = True
        st['evals'], st['scores'], st['bests'], st['last_ui'] = [], [], [], 0.0
        optim_progress.value = 0.0
        optim_plot.data = _OPTIM_EMPTY_PLOT
        optim_run_btn.disabled = True
        optim_stop_btn.disabled = False
        backend = _gpu_backend.gpu_backend_label() if problem.use_gpu else "CPU"
        _optim_status(f"Starting '{problem.name}': {problem.dim} variables, {opt.max_evals} evals, "
                      f"{len(problem.walls)} wall distance(s), {backend}…")
        t = _threading.Thread(target=_optim_worker, args=(problem, opt), daemon=True, name="optimizer")
        st['thread'] = t
        t.start()

    @optim_stop_btn.on_click
    def _(_):
        if _optim_state['thread'] is not None:
            _optim_state['stop'].set()
            _optim_status("Stopping after the current evaluation…", "#ffaa00")

    optim_load_btn.on_click(lambda _: _optim_load_best())

    @optim_report_btn.on_click
    def _(_):
        path = _optim_state.get('report')
        if path and os.path.exists(path):
            _wb.open("file://" + os.path.abspath(path))
        else:
            _optim_status("No report available for the last run.", "#ffaa00")

    @optim_save_btn.on_click
    def _(_):
        cfg = _optim_state['best_cfg']
        if cfg is None:
            _optim_status("Nothing to save yet — run an optimisation first.", "#ffaa00")
            return
        name = optim_save_name.value.strip()
        if not name:
            _optim_status("Enter a name in 'Save best as'.", "#ffaa00")
            return
        cfg = dict(cfg)
        stem = name.lower().replace(' ', '_')
        path = os.path.join(config_dir, f"{stem}.json")
        n = 2
        while os.path.exists(path):  # never overwrite an existing config
            path = os.path.join(config_dir, f"{stem}_{n:03d}.json")
            n += 1
        cfg['name'] = os.path.splitext(os.path.basename(path))[0]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        config_dropdown.options = get_available_configs()
        _optim_status(f"Saved best configuration to {path}"
                      + ("  (name already existed → suffixed)" if n > 2 else ""), "#4CAF50")


    return SimpleNamespace()
