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
from lighting_simulator.domain.beam_profile import LAMBERTIAN as _LAMBERTIAN, get_profile as _get_profile, profile_names as _profile_names
from lighting_simulator.domain.guides import circle_line_segments_m as _circle_line_segments_m
from lighting_simulator.simulation import gpu_backend as _gpu_backend
from lighting_simulator.optimization import (
    OptimizerSpec as _OptimizerSpec,
    problem_from_spec as _problem_from_spec,
    run as _run_optimization,
)
from lighting_simulator.scene.layout import convert_v1 as _convert_v1, save_json as _save_json


def build(ctx):
    _project_root = ctx._project_root
    apply_config = ctx.apply_config
    camera_fov_h = ctx.camera_fov_h
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    tilt_fov_deg = ctx.tilt_fov_deg
    flash_lumens_input = ctx.flash_lumens_input
    led_voltage_input = ctx.led_voltage_input
    led_efficacy_input = ctx.led_efficacy_input
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    config_dir = ctx.config_dir
    config_dropdown = ctx.config_dropdown
    current_config_name = ctx.current_config_name
    state = ctx.state
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
    _OPTIM_NO_GROUP = "(no panels)"
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
        for idx, p in enumerate(state.panels):
            labels.append(f"{idx}: {p.name}")
        return labels or [_OPTIM_NO_GROUP]

    _MODE_REFINE = "1 · Refine the current panels"
    _MODE_DUCTS = "2 · Design LEDs on the ducts (from scratch)"
    _MODE_PRESET = "3 · Run a preset spec file"

    with tab_optim:
        server.gui.add_html(
            "<div style='color:#bbb;font-size:12px;line-height:1.4;margin-bottom:4px;'>"
            "<b>How it works</b> — the optimiser repeatedly ray-traces candidate LED layouts and keeps the one "
            "with the best score on the test cases (③) under your constraints (④). Folders top to bottom: "
            "<b>①</b> what may change, <b>②</b> how bright the LEDs are, <b>③</b> what is measured, "
            "<b>④</b> hard limits, <b>⑤</b> search settings, <b>⑥</b> run. Design modes:"
            "<ul style='margin:4px 0 0 14px;padding:0;'>"
            "<li><b>1 Refine</b>: keep the scene, nudge one existing panel (free deltas or sliding along a duct "
            "surface; tilt, beam angle, on/off, roles).</li>"
            "<li><b>2 Ducts</b>: remove all LEDs and place a new symmetric lattice on the duct rings within tolerances.</li>"
            "<li><b>3 Preset</b>: run a JSON spec from <code>optimization_specs/</code> as-is (optionally overriding it with ②–④).</li>"
            "</ul>Camera, tilt angle and VIO poses always come from the FOV tab. "
            "Results go to <code>exports/optim/&lt;run name&gt;/</code> (best_config.json + report.pdf).</div>",
            order=1,
        )
        optim_mode = server.gui.add_dropdown("Design mode", options=[_MODE_REFINE, _MODE_DUCTS, _MODE_PRESET],
                                             initial_value=_MODE_REFINE, order=2)

    _BEAM_KEEP, _BEAM_FIXED, _BEAM_OPT = "Keep scene values", "Fixed value", "Optimise in range"

    # -- ① Refine ---------------------------------------------------------
    with tab_optim:
        _optim_refine_folder = server.gui.add_folder("① Layout & variables — refine a panel", order=10)
    with _optim_refine_folder:
        optim_group_dropdown = server.gui.add_dropdown("Group", options=_optim_group_labels())
        optim_refresh_groups_btn = server.gui.add_button("🔄 Refresh group list")
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Position</div>")
        _MOVE_FREE, _MOVE_DUCT, _MOVE_NONE = "Free (± position / rotation)", "On a duct surface", "Fixed"
        optim_move_mode = server.gui.add_dropdown(
            "Panel movement", options=[_MOVE_FREE, _MOVE_DUCT, _MOVE_NONE], initial_value=_MOVE_FREE,
            hint="Free: rigid ± deltas. On a duct: the existing layout is anchored to a cylinder (where its LED "
                 "centroid projects onto it) and slides / swivels along the surface, LEDs turning with the wall.")
        optim_pos_delta = server.gui.add_vector3("± position (cm)", (2.0, 2.0, 2.0),
                                                 min=(0.0, 0.0, 0.0), max=(50.0, 50.0, 50.0), step=0.5)
        optim_rot_delta = server.gui.add_vector3("± rotation (°)", (15.0, 15.0, 20.0),
                                                 min=(0.0, 0.0, 0.0), max=(180.0, 180.0, 180.0), step=1.0)
        rd_center = server.gui.add_vector3("Duct centre (cm)", (6.3, 12.0, 0.0), step=0.1, visible=False)
        rd_radius = server.gui.add_number("Duct radius (cm)", 8.5, min=1.0, max=50.0, step=0.1, visible=False)
        rd_axis = server.gui.add_dropdown("Duct axis", options=["Z (vertical)", "Y (lateral)", "X (forward)"],
                                          initial_value="Z (vertical)", visible=False)
        rd_rot = server.gui.add_vector3("Duct rotation X/Y/Z (°)", (0.0, 0.0, 0.0), min=(-90.0, -90.0, -180.0),
                                        max=(90.0, 90.0, 180.0), step=0.5, visible=False)
        rd_tol_arc = server.gui.add_number("± around the duct (cm along circumference)", 10.0, min=0.0, max=100.0,
                                           step=1.0, visible=False)
        rd_tol_axial = server.gui.add_number("± along duct axis (cm)", 2.0, min=0.0, max=20.0, step=0.5, visible=False)
        rd_tol_radial = server.gui.add_number("± stand-off from surface (cm)", 0.0, min=0.0, max=10.0, step=0.5,
                                              visible=False)
        rd_tilt = server.gui.add_slider("± swivel toward axis (°)", min=0, max=90, step=5, initial_value=0,
                                        visible=False, hint="Whole panel pitches about its tangent line")
        rd_spin = server.gui.add_slider("± spin about panel normal (°)", min=0, max=180, step=5, initial_value=0,
                                        visible=False)
        rd_led_arc = server.gui.add_number("Per-LED: ± slide around duct (cm)", 0.0, min=0.0, max=30.0, step=0.5,
                                           visible=False,
                                           hint="On top of the panel move, every LED may slide on the surface by its own "
                                                "amount (2 variables per LED). 0 = LEDs stay rigid to the panel.")
        rd_led_axial = server.gui.add_number("Per-LED: ± slide along axis (cm)", 0.0, min=0.0, max=10.0, step=0.5,
                                             visible=False)
        _RD_NO_MIRROR = "(none)"
        rd_mirror = server.gui.add_dropdown("Mirror partner group", options=[_RD_NO_MIRROR] + _optim_group_labels(),
                                            initial_value=_RD_NO_MIRROR, visible=False,
                                            hint="This group is replaced by the XZ mirror of the moved panel")
        rd_info = server.gui.add_html("", visible=False)
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Beams</div>")
        optim_beam_mode = server.gui.add_dropdown("Beam angle", options=[_BEAM_KEEP, _BEAM_FIXED, _BEAM_OPT],
                                                  initial_value=_BEAM_KEEP,
                                                  hint="Viewing angle of every LED in the group")
        optim_beam_fixed = server.gui.add_number("Beam angle (°)", 120.0, min=10.0, max=180.0, step=5.0, visible=False)
        optim_beam_range = server.gui.add_multi_slider("Beam angle range (°)", min=30, max=180, step=5,
                                                       initial_value=(90, 130), visible=False)
        optim_beam_profile = server.gui.add_dropdown(
            "Beam profile", options=[_BEAM_KEEP] + _profile_names(), initial_value=_BEAM_KEEP,
            hint="Measured intensity-vs-angle curve for every LED of the group (see Selected → Beam profile). "
                 "'Keep scene values' uses whatever the panel has; with a measured profile the beam angle is ignored.")
        optim_var_tilts = server.gui.add_checkbox("Optimise per-LED beam tilt", initial_value=False,
                                                  hint="Dynamic (designer / template) groups only")
        optim_tilt_range = server.gui.add_slider("± beam tilt (°)", min=5, max=90, step=5, initial_value=45,
                                                 visible=False)
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>LEDs</div>")
        optim_var_states = server.gui.add_checkbox("Optimise LED on / off", initial_value=False)
        optim_var_roles = server.gui.add_checkbox(
            "Optimise LED roles (off / VIO / flash / both)", initial_value=False, visible=False,
            hint="Lets the optimiser decide which LEDs flash and which stay on for VIO (flash mode in ②). "
                 "Replaces 'LED on / off'.",
        )
        optim_var_current = server.gui.add_checkbox("Optimise drive current (→ flux)", initial_value=False, visible=False,
                                                    hint="Electrical model in ②: shared continuous per-LED current, "
                                                         "lumens = I · V · efficacy")
        optim_current_range = server.gui.add_multi_slider("Current range (A)", min=0.1, max=13.0, step=0.1,
                                                          initial_value=(0.5, 3.0), visible=False)

    # -- ① Preset ---------------------------------------------------------
    with tab_optim:
        _optim_preset_folder = server.gui.add_folder("① Preset spec", order=10)
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
        optim_preset_use_ui = server.gui.add_checkbox(
            "Use the UI folders ② ③ ④ instead of the spec's", initial_value=False,
            hint="Light sources, test cases (walls, camera, VIO, targets) and constraints come from the folders below",
        )
        optim_copy_preset_btn = server.gui.add_button(
            "📋 Copy spec into the controls & switch mode",
            hint="Fills every folder from the spec, then selects mode 1 or 2 so you can edit and run it",
        )

    # -- ① Ducts ----------------------------------------------------------
    with tab_optim:
        _optim_duct_folder = server.gui.add_folder("① Layout & variables — LEDs on the ducts", order=10)
    with _optim_duct_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:6px;'>Enter the nominal duct geometry and the "
            "mechanical tolerances; the optimiser places an LED lattice on the duct surface (and its "
            "left/right mirror) anywhere inside those tolerances. Existing LEDs are removed.</div>"
        )
        server.gui.add_html("<div style='font-weight:600;'>Duct</div>")
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
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Position tolerances</div>")
        duct_tol_radius = server.gui.add_number("± radius / stand-off (cm)", 2.0, min=0.0, max=20.0, step=0.5)
        duct_tol_arc = server.gui.add_number("± around the duct (cm along circumference)", 20.0,
                                             min=0.0, max=100.0, step=1.0)
        duct_tol_axial = server.gui.add_number("± along duct axis (cm)", 2.0, min=0.0, max=20.0, step=0.5)
        duct_tol_center = server.gui.add_vector3("± duct centre shift (cm)", (0.0, 0.0, 0.0),
                                                 min=(0.0, 0.0, 0.0), max=(50.0, 50.0, 50.0), step=0.5)
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>LED lattice</div>")
        duct_rows = server.gui.add_number("Max rows (along axis)", 3, min=1, max=8, step=1)
        duct_cols = server.gui.add_number("Max columns (around duct)", 4, min=1, max=12, step=1)
        duct_var_counts = server.gui.add_checkbox("Optimise row / column count", initial_value=True)
        duct_span_cm = server.gui.add_multi_slider("Lattice arc span (cm)", min=1, max=60, step=1,
                                                   initial_value=(3, 25))
        duct_pitch_cm = server.gui.add_multi_slider("Row pitch (cm)", min=0.5, max=5.0, step=0.1,
                                                    initial_value=(0.8, 2.0))
        duct_led_size = server.gui.add_number("LED size (cm)", 1.0, min=0.2, max=3.0, step=0.1)
        duct_on_off = server.gui.add_checkbox("Optimise per-LED on/off", initial_value=False,
                                              hint="Only when the row / column count is fixed: the counts already decide how many LEDs exist")
        duct_roles = server.gui.add_checkbox("Optimise LED roles (VIO / flash / both)", initial_value=False, visible=False,
                                             hint="Needs the flash mode (②); replaces on/off. With variable row / column counts "
                                                  "every LED keeps a role (no 'off'); with fixed counts a role may also be 'off'.")
        server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Beams</div>")
        duct_beam_mode = server.gui.add_dropdown("Lattice beam angle", options=[_BEAM_FIXED, _BEAM_OPT], initial_value=_BEAM_FIXED,
                                                 hint="Viewing angle shared by all LEDs of the lattice")
        duct_beam_fixed = server.gui.add_number("Lattice beam angle (°)", 120.0, min=10.0, max=180.0, step=5.0)
        duct_beam = server.gui.add_multi_slider("Lattice beam angle range (°)", min=30, max=180, step=5,
                                                initial_value=(90, 130), visible=False)
        duct_beam_profile = server.gui.add_dropdown(
            "Lattice beam profile", options=_profile_names(), initial_value=_LAMBERTIAN,
            hint="Measured intensity-vs-angle curve for the lattice LEDs; with a measured profile the beam angle "
                 "controls are ignored.")
        duct_tilt = server.gui.add_slider("± beam tilt toward axis (°)", min=0, max=90, step=5, initial_value=60,
                                          hint="0 = beams stay normal to the duct surface")
        duct_tilt_shared = server.gui.add_checkbox("Shared tilt for all LEDs", initial_value=False)
        duct_tilt_symmetric = server.gui.add_checkbox(
            "Mirror tilt top / bottom", initial_value=False,
            hint="Arc grid: rows above the panel middle look up by +t, rows below look down by −t "
                 "(one tilt per row pair, middle row straight). Overrides 'Shared tilt'.",
        )
        duct_var_current = server.gui.add_checkbox("Optimise drive current (→ flux)", initial_value=False, visible=False,
                                                   hint="Electrical model in ②")
        duct_current = server.gui.add_multi_slider("Drive current range (A)", min=0.1, max=13.0, step=0.1,
                                                   initial_value=(0.3, 3.0), visible=False)

    def _optim_beam_mode_changed(_=None):
        profiled = duct_beam_profile.value != _LAMBERTIAN
        optim_beam_fixed.visible = optim_beam_mode.value == _BEAM_FIXED
        optim_beam_range.visible = optim_beam_mode.value == _BEAM_OPT
        duct_beam_mode.visible = not profiled
        duct_beam_fixed.visible = duct_beam_mode.value == _BEAM_FIXED and not profiled
        duct_beam.visible = duct_beam_mode.value == _BEAM_OPT and not profiled
        optim_tilt_range.visible = bool(optim_var_tilts.value)

    for _h in (optim_beam_mode, duct_beam_mode, optim_var_tilts, duct_beam_profile):
        _h.on_update(_optim_beam_mode_changed)
    _optim_beam_mode_changed()

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
            'shared_beam_angle': True,
            'optimize_enabled': bool(duct_on_off.value),
            'optimize_roles': bool(duct_roles.value and optim_flash_enable.value),
            'led_size': float(duct_led_size.value),
            'mirror_xz': bool(duct_mirror.value),
        }
        if duct_beam_profile.value != _LAMBERTIAN:
            var['beam_profile'] = duct_beam_profile.value
            var['default_beam_angle'] = 2.0 * _get_profile(duct_beam_profile.value).max_angle_deg
        elif duct_beam_mode.value == _BEAM_OPT:
            var['beam_angle_range'] = [float(v) for v in duct_beam.value]
        else:
            var['default_beam_angle'] = float(duct_beam_fixed.value)
        if optim_electrical.value and duct_var_current.value:
            var['current_range'] = [float(v) for v in duct_current.value]
        else:
            var['lumens'] = float(optim_vio_lumens.value)  # so the saved design shows the same flux in the UI
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

    # -- ① refine on a duct: variable spec + preview ------------------------------
    def _refine_duct_dict():
        axis, ref = _DUCT_AXES[rd_axis.value]
        rot = [float(v) for v in rd_rot.value]
        return {'center': [float(v) for v in rd_center.value], 'axis': axis, 'radius': float(rd_radius.value),
                'reference': ref, 'mount_offset': 0.0,
                'rotation_deg': rot if any(abs(r) > 1e-9 for r in rot) else None}

    def _refine_duct_variable(gi):
        from lighting_simulator.optimization.variables import arc_cm_to_deg
        d_theta = arc_cm_to_deg(rd_tol_arc.value, float(rd_radius.value))
        var = {'type': 'duct_panel_pose', 'group_index': gi, 'duct': _refine_duct_dict(),
               'theta_range': [-d_theta, d_theta] if d_theta > 0 else None,
               'axial_range': [-float(rd_tol_axial.value), float(rd_tol_axial.value)] if rd_tol_axial.value > 0 else None,
               'radial_range': [-float(rd_tol_radial.value), float(rd_tol_radial.value)] if rd_tol_radial.value > 0 else None,
               'tilt_axial_range': [-float(rd_tilt.value), float(rd_tilt.value)] if rd_tilt.value > 0 else None,
               'spin_range': [-float(rd_spin.value), float(rd_spin.value)] if rd_spin.value > 0 else None}
        d_led = arc_cm_to_deg(rd_led_arc.value, float(rd_radius.value))
        if d_led > 0:
            var['led_theta_range'] = [-d_led, d_led]
        if rd_led_axial.value > 0:
            var['led_axial_range'] = [-float(rd_led_axial.value), float(rd_led_axial.value)]
        if rd_mirror.value != _RD_NO_MIRROR and ':' in rd_mirror.value:
            mi = int(rd_mirror.value.split(':', 1)[0])
            if mi == gi:
                raise ValueError("The mirror partner must be a different group.")
            var['mirror_group_index'] = mi
        if all(var.get(k) is None for k in ('theta_range', 'axial_range', 'radial_range', 'tilt_axial_range', 'spin_range',
                                            'led_theta_range', 'led_axial_range')):
            raise ValueError("Every duct range is 0: nothing can move.")
        return var

    _refine_duct_handles = []

    def _draw_refine_duct_preview(_=None):
        from lighting_simulator.optimization.variables import DuctPanelPose, arc_cm_to_deg
        for h in _refine_duct_handles:
            try:
                h.remove()
            except Exception:
                pass
        _refine_duct_handles.clear()
        on = optim_mode.value == _MODE_REFINE and optim_move_mode.value == _MOVE_DUCT
        for h in (rd_center, rd_radius, rd_axis, rd_rot, rd_tol_arc, rd_tol_axial, rd_tol_radial, rd_tilt, rd_spin,
                  rd_led_arc, rd_led_axial, rd_mirror, rd_info):
            h.visible = on
        for h in (optim_pos_delta, optim_rot_delta):
            h.visible = optim_mode.value == _MODE_REFINE and optim_move_mode.value == _MOVE_FREE
        if not on:
            return
        try:
            gi = _optim_selected_group_index()
            var = DuctPanelPose(group_index=gi, duct=_refine_duct_dict()).bind(get_current_config())
        except Exception as exc:
            rd_info.content = f"<div style='color:#ffaa00;font-size:12px;'>{exc}</div>"
            return
        r = float(rd_radius.value)
        rd_info.content = ("<div style='color:#8bc34a;font-size:12px;'>Anchor: θ = %.1f° around the axis, %.1f cm along "
                           "it, LED centroid %.1f cm %s the surface.</div>"
                           % (var.theta0, var.axial0, abs(var.radial0), "outside" if var.radial0 >= 0 else "inside"))
        d = _refine_duct_dict()
        from lighting_simulator.optimization.variables import Duct
        a, u, _v = Duct(**d).frame()
        c = np.asarray(d['center'], float)
        d_theta = arc_cm_to_deg(rd_tol_arc.value, r)
        d_led = arc_cm_to_deg(rd_led_arc.value, r)
        half = float(rd_tol_axial.value) + float(rd_led_axial.value) + 2.0
        for k, t in enumerate((-half, 0.0, half)):
            _refine_duct_handles.append(server.scene.add_line_segments(
                f"/optim_refine_duct/ring_{k}", points=_circle_line_segments_m(c + a * t, r, a, n_seg=64),
                colors=(1.0, 0.55, 0.1), line_width=2.0))
        r_lo = max(0.1, r + var.radial0 - float(rd_tol_radial.value))
        r_hi = r + var.radial0 + float(rd_tol_radial.value) + 0.2
        segs = _duct_sector_wireframe_m(c, a, u, var.theta0 - d_theta, var.theta0 + d_theta, r_lo, r_hi,
                                        var.axial0 - float(rd_tol_axial.value), var.axial0 + float(rd_tol_axial.value))
        _refine_duct_handles.append(server.scene.add_line_segments(
            "/optim_refine_duct/tol_box", points=segs, colors=(1.0, 0.85, 0.2), line_width=2.5))
        if d_led > 0 or rd_led_axial.value > 0:  # envelope any single LED may reach
            th_lo = float(var._led_theta.min()) - d_theta - d_led
            th_hi = float(var._led_theta.max()) + d_theta + d_led
            ax_lo = float(var._led_axial.min()) - float(rd_tol_axial.value) - float(rd_led_axial.value)
            ax_hi = float(var._led_axial.max()) + float(rd_tol_axial.value) + float(rd_led_axial.value)
            segs = _duct_sector_wireframe_m(c, a, u, th_lo, th_hi, r_lo, r_hi, ax_lo, ax_hi)
            _refine_duct_handles.append(server.scene.add_line_segments(
                "/optim_refine_duct/envelope", points=segs, colors=(1.0, 0.95, 0.6), line_width=1.0))

    for _h in (optim_move_mode, optim_group_dropdown, rd_center, rd_radius, rd_axis, rd_rot, rd_tol_arc,
               rd_tol_axial, rd_tol_radial, rd_led_arc, rd_led_axial):
        _h.on_update(_draw_refine_duct_preview)

    def _optim_mode_changed(_=None):
        mode = optim_mode.value
        preset = mode == _MODE_PRESET
        _optim_refine_folder.visible = mode == _MODE_REFINE
        _optim_duct_folder.visible = mode == _MODE_DUCTS
        _optim_preset_folder.visible = preset
        ui_folders = (not preset) or optim_preset_use_ui.value
        _optim_light_folder.visible = ui_folders
        _optim_tests_folder.visible = ui_folders
        _optim_con_folder.visible = ui_folders
        _draw_duct_preview()
        _draw_refine_duct_preview()

    optim_mode.on_update(_optim_mode_changed)
    optim_preset_use_ui.on_update(_optim_mode_changed)

    with tab_optim:
        _optim_light_folder = server.gui.add_folder("② Light sources", order=20)
    with _optim_light_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:6px;'>Per-LED flux in each operating point. "
            "<b>Flight</b> = VIO + Both LEDs on continuously (seen by the VIO cameras, T2). "
            "<b>Flash</b> = Flash + Both LEDs during the photogrammetry pulse (seen by the main camera, T1 / T3); "
            "VIO LEDs keep their flight flux.</div>"
        )
        optim_vio_lumens = server.gui.add_number("Flight / VIO flux per LED (lm)", float(led_lumens_slider.value),
                                                 min=1.0, max=100000.0, step=10.0,
                                                 hint="Forced on every LED (panel lumens overrides are ignored)")
        optim_flash_enable = server.gui.add_checkbox(
            "Flash mode (photogrammetry pulse)", initial_value=False,
            hint="Off: one continuous image, roles are ignored. On: LED roles decide which LEDs pulse and T1 / T3 "
                 "judge the FLASH image; the flight image is still traced for T2 and the report.",
        )
        optim_flash_lumens = server.gui.add_number(
            "Flash flux per LED (lm)",
            float(flash_lumens_input.value),
            min=1.0, max=1000000.0, step=100.0, visible=False)
        server.gui.add_html("<hr style='margin:8px 0;'>")
        optim_electrical = server.gui.add_checkbox(
            "Electrical model (currents & drivers)", initial_value=False,
            hint="Adds current = flux / (V · efficacy) per LED, driver-IC counts and the current / driver limits "
                 "in ④. Also unlocks 'drive current' as a variable in ①. Off: fluxes above are all that matters.",
        )
        optim_drv_voltage = led_voltage_input  # shared with the Display tab ("Electrical")
        optim_drv_efficacy = led_efficacy_input
        optim_elec_info = server.gui.add_html("", visible=False)
        optim_drv_max_current = server.gui.add_number("Pulse driver: max current per LED (A)", 13.0, min=0.1, max=50.0,
                                                      step=0.1, visible=False)
        optim_leds_per_driver = server.gui.add_number("Pulse driver: LEDs per driver", 4, min=1, max=64, step=1,
                                                      visible=False)
        optim_cont_max_current = server.gui.add_number("Continuous driver: max current per LED (A)", 3.0, min=0.1,
                                                       max=50.0, step=0.1, visible=False)
        optim_cont_leds_per_driver = server.gui.add_number("Continuous driver: LEDs per driver", 8, min=1, max=64,
                                                           step=1, visible=False)

    def _optim_elec_info(_=None):
        v, eff = float(optim_drv_voltage.value), float(optim_drv_efficacy.value)
        lm_a = v * eff
        i_fl, i_fx = float(optim_vio_lumens.value) / lm_a, float(optim_flash_lumens.value) / lm_a
        optim_elec_info.content = (
            "<div style='color:#888;font-size:11px;line-height:1.5;'>"
            f"{v:g} V × {eff:g} lm/W = {lm_a:,.0f} lm/A (Display tab → Electrical). "
            f"Flight {i_fl:.2f} A/LED" + (f", flash {i_fx:.2f} A/LED" if optim_flash_enable.value else "") + ".</div>")

    with tab_optim:
        _optim_tests_folder = server.gui.add_folder("③ Test cases", order=30)
    with _optim_tests_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:12px;margin-bottom:4px;'>Score = T1 + T2 + T3 + penalties (④). "
            "Main camera, tilt angle and VIO poses come from the FOV tab.</div>"
            "<div style='font-weight:600;'>T1 · Inspection image — flat wall, main camera (base score)</div>"
        )
        optim_wall_dists = server.gui.add_text(
            "Wall distances (cm)", initial_value="",
            hint="Comma-separated; the score is averaged over them. Empty = Intensity Map wall distance",
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
        optim_metric = server.gui.add_dropdown(
            "Metric", options=["u0", "u1", "cv"], initial_value="u0",
            hint="u0 = Emin/Eavg (Emin at the percentile below), u1 = Emin/Emax, cv = σ/Eavg (uses all cells; "
                 "least sensitive to ray noise)",
        )
        optim_min_pct = server.gui.add_slider("Emin percentile (%)", min=0.0, max=10.0, step=0.5, initial_value=2.0,
                                              hint="0 = single darkest cell (very noisy on fine grids); 2–5 recommended")
        optim_cov_w = server.gui.add_slider("Coverage penalty weight", min=0.0, max=5.0, step=0.1, initial_value=1.0,
                                            hint="Share of FOV cells left dark (T1 and T3)")
        optim_min_lux = server.gui.add_number("Min average lux in FOV (0 = off)", 0, min=0, step=100,
                                              hint="Target for the judged image (flash when flash mode is on)")
        optim_lux_dist = server.gui.add_number("… at wall distance (cm, 0 = worst distance)", 0, min=0, max=1500, step=5,
                                               hint="Nearest entry of 'Wall distances' is used")
        optim_lux_w = server.gui.add_slider("Lux penalty weight", min=0.0, max=5.0, step=0.1, initial_value=1.0)

        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>T2 · VIO coverage — 5-wall room, flight image, fisheye cameras</div>")
        optim_vio_enable = server.gui.add_checkbox("Add VIO coverage test case", initial_value=False,
                                                   hint="Uses the VIO camera poses from the FOV tab. Flight image only: "
                                                        "the VIO cameras never see the flash.")
        optim_vio_lux = server.gui.add_number("Min lux on VIO surfaces", 120, min=0, step=10, visible=False)
        optim_vio_fraction = server.gui.add_slider("Min share of VIO FOV lit (%)", min=0, max=100, step=5,
                                                   initial_value=50, visible=False)
        optim_vio_w = server.gui.add_slider("T2 weight", min=0.0, max=5.0, step=0.1, initial_value=1.0, visible=False)
        optim_vio_room_dist = server.gui.add_number("Room wall distance (cm)", 300, min=50, max=2000, step=10, visible=False,
                                                    hint="Five walls (front, sides, top, bottom — no back wall) this far "
                                                         "from the rig; opposite walls are twice this apart")
        optim_vio_room_grid = server.gui.add_number("Room grid per wall", 20, min=5, max=80, step=5, visible=False,
                                                    hint="Coarse on purpose: 20 → 30 cm cells in a 6 m room")

        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;'>T3 · Tilted view — 5-wall rooms, main camera ±tilt</div>")
        optim_tilt_enable = server.gui.add_checkbox(
            "Add ±tilt FOV test case", initial_value=False,
            hint="A 5-wall room is placed at every wall distance and the main camera is pitched up / down by the FOV "
                 "tab's 'Tilt FOV angle' (same image as T1). Uniformity + coverage of the cells inside each tilted "
                 "footprint, computed analytically (no ray noise). Penalty 'tilt_uniformity'.",
        )
        optim_tilt_info = server.gui.add_html("", visible=False)
        optim_tilt_w = server.gui.add_slider("T3 weight", min=0.0, max=2.0, step=0.05, initial_value=0.3, visible=False,
                                             hint="Keep well below 1: a 45° footprint spans wall + ceiling and can never "
                                                  "be as uniform as the straight view")
        optim_tilt_grid = server.gui.add_number("T3 room grid per wall", 32, min=8, max=128, step=4, visible=False,
                                                hint="Independent of the T2 room grid; only the footprint cells are computed")

        @optim_wall_size_mode.on_update
        def _(_):
            optim_wall_size.visible = optim_wall_size_mode.value == _WS_FIXED
            optim_wall_sizes.visible = optim_wall_size_mode.value == _WS_LIST

    with tab_optim:
        _optim_con_folder = server.gui.add_folder("④ Constraints", order=40)
    with _optim_con_folder:
        server.gui.add_html("<div style='color:#888;font-size:12px;margin-bottom:4px;'>Soft penalties added to the score.</div>"
                            "<div style='font-weight:600;'>LEDs & geometry</div>")
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
        optim_elec_con_html = server.gui.add_html("<div style='font-weight:600;margin-top:6px;'>Electrical (② model on)</div>",
                                                  visible=False)
        optim_max_drivers = server.gui.add_number("Max drivers total (0 = no limit)", 0, min=0, step=1, visible=False)
        optim_driver_cost = server.gui.add_slider("Cost per driver (any class)", min=0.0, max=0.2, step=0.005,
                                                  initial_value=0.0, visible=False)
        optim_max_current = server.gui.add_number("Max continuous current (A, 0 = off)", 0.0, min=0.0, step=1.0,
                                                  visible=False, hint="Sum over the LEDs lit in flight")
        optim_max_pulse_drivers = server.gui.add_number("Max pulse drivers (0 = no limit)", 0, min=0, step=1, visible=False)
        optim_pulse_driver_cost = server.gui.add_slider("Cost per pulse driver", min=0.0, max=0.2, step=0.005,
                                                        initial_value=0.0, visible=False)
        optim_max_cont_drivers = server.gui.add_number("Max continuous drivers (0 = no limit)", 0, min=0, step=1,
                                                       visible=False)
        optim_cont_driver_cost = server.gui.add_slider("Cost per continuous driver", min=0.0, max=0.2, step=0.005,
                                                       initial_value=0.0, visible=False)
        optim_max_peak_current = server.gui.add_number("Max peak current in flash (A, 0 = off)", 0.0, min=0.0,
                                                       step=5.0, visible=False,
                                                       hint="VIO continuous + (Flash + Both) × flash current")

    def _optim_sources_changed(_=None):
        flash, elec = bool(optim_flash_enable.value), bool(optim_electrical.value)
        optim_flash_lumens.visible = flash
        for h in (optim_var_roles, duct_roles):
            h.visible = flash
        for h in (optim_elec_info, optim_drv_max_current, optim_leds_per_driver, optim_elec_con_html,
                  optim_max_drivers, optim_driver_cost, optim_max_current, optim_var_current, duct_var_current):
            h.visible = elec
        optim_current_range.visible = elec and bool(optim_var_current.value)
        duct_current.visible = elec and bool(duct_var_current.value)
        for h in (optim_cont_max_current, optim_cont_leds_per_driver, optim_max_pulse_drivers, optim_pulse_driver_cost,
                  optim_max_cont_drivers, optim_cont_driver_cost, optim_max_peak_current):
            h.visible = elec and flash
        vio = bool(optim_vio_enable.value)
        for h in (optim_vio_lux, optim_vio_fraction, optim_vio_w, optim_vio_room_dist, optim_vio_room_grid):
            h.visible = vio
        tilt = bool(optim_tilt_enable.value)
        for h in (optim_tilt_info, optim_tilt_w, optim_tilt_grid):
            h.visible = tilt
        optim_tilt_info.content = (f"<div style='color:#888;font-size:11px;'>Camera pitched ±{float(tilt_fov_deg.value):g}° "
                                   "(FOV tab → 'Tilt FOV angle').</div>")
        if elec:
            _optim_elec_info()

    for _h in (optim_flash_enable, optim_electrical, optim_var_current, duct_var_current, optim_vio_enable,
               optim_tilt_enable, optim_vio_lumens, optim_flash_lumens, optim_drv_voltage, optim_drv_efficacy, tilt_fov_deg):
        _h.on_update(_optim_sources_changed)
    _optim_sources_changed()

    def _duct_var_counts_changed(_=None):
        duct_on_off.visible = not bool(duct_var_counts.value)

    duct_var_counts.on_update(_duct_var_counts_changed)
    _duct_var_counts_changed()
    with tab_optim:
        _optim_opt_folder = server.gui.add_folder("⑤ Optimizer", order=50)
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
        optim_analytic = server.gui.add_checkbox(
            "Analytic direct light (no ray tracing)", initial_value=False,
            hint="Score every candidate with the closed-form direct illuminance instead of Monte-Carlo rays: "
                 "exact expected value, zero noise, ~100× faster; 'Rays per pixel' and the GPU are ignored. "
                 "Frame shadows kept, wall reflections not modelled (the optimiser never traces bounces anyway).",
        )
        optim_polish = server.gui.add_checkbox("Polish with Nelder-Mead", initial_value=False)
        optim_name = server.gui.add_text("Run name", initial_value="", hint="Output folder name; empty = auto")

    with tab_optim:
        _optim_run_folder = server.gui.add_folder("⑥ Run", order=60)
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
        optim_save_btn = server.gui.add_button("💾 Save best to layouts/")

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
        optim_lux_dist.value = 0
        optim_lux_w.value = float(obj.get('lux_weight', 1.0))
        optim_tilt_enable.value = bool(obj.get('tilt_fov_deg'))
        if obj.get('tilt_fov_deg'):
            tilt_fov_deg.value = int(round(float(obj['tilt_fov_deg']) / 5) * 5)
        optim_tilt_w.value = float(obj.get('tilt_fov_weight', 0.3))
        optim_tilt_grid.value = int(obj.get('tilt_room_grid_size', 32))
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
        optim_electrical.value = bool(spec.get('electrical', False))
        optim_drv_voltage.value = float(drv.get('voltage_v', 6.0))
        optim_drv_efficacy.value = float(drv.get('efficacy_lm_per_w', 180.0))
        optim_drv_max_current.value = float(drv.get('max_current_a', 13.0))
        optim_leds_per_driver.value = int(drv.get('leds_per_driver', 1))
        cdrv = spec.get('cont_driver') or {}
        optim_cont_max_current.value = float(cdrv.get('max_current_a', 3.0))
        optim_cont_leds_per_driver.value = int(cdrv.get('leds_per_driver', 8))
        lm_per_a = optim_drv_voltage.value * optim_drv_efficacy.value
        vio = spec.get('vio') or {}
        optim_vio_room_dist.value = int(vio.get('room_dist', 300))
        optim_vio_room_grid.value = int(vio.get('room_grid_size', 20))
        optim_flash_enable.value = False
        optim_vio_enable.value = False
        optim_vio_lumens.value = float(spec.get('emission', {}).get('default_lumens', optim_vio_lumens.value))
        for m in spec.get('modes', []):
            is_flash = m.get('flash') if m.get('flash') is not None else m.get('current_a') is not None
            if is_flash:
                optim_flash_enable.value = True
                if m.get('lumens') is not None:
                    optim_flash_lumens.value = float(m['lumens'])
                elif m.get('current_a') is not None:
                    optim_flash_lumens.value = float(m['current_a']) * lm_per_a
            elif m.get('lumens') is not None:
                optim_vio_lumens.value = float(m['lumens'])
            if m.get('min_avg_lux'):
                optim_min_lux.value = int(m['min_avg_lux'])
                optim_lux_dist.value = int(m.get('min_avg_lux_dist') or 0)
                optim_lux_w.value = float(m.get('lux_weight', 1.0))
            if m.get('vio_min_lux'):
                optim_vio_enable.value = True
                optim_vio_lux.value = int(m['vio_min_lux'])
                optim_vio_fraction.value = int(round(100 * float(m.get('vio_min_fraction', 0.5))))
                optim_vio_w.value = float(m.get('vio_weight', 1.0))
        _optim_sources_changed()
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
            if v.get('beam_angle_range'):
                duct_beam_mode.value = _BEAM_OPT
                duct_beam.value = tuple(float(b) for b in v['beam_angle_range'])
            else:
                duct_beam_mode.value = _BEAM_FIXED
                duct_beam_fixed.value = float(v.get('default_beam_angle', 120.0))
            _bp = v.get('beam_profile') or _LAMBERTIAN
            duct_beam_profile.value = _bp if _bp in duct_beam_profile.options else _LAMBERTIAN
            duct_var_current.value = bool(v.get('current_range'))
            if v.get('current_range'):
                duct_current.value = tuple(float(c) for c in v['current_range'])
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
            optim_move_mode.value = (_MOVE_DUCT if 'duct_panel_pose' in types
                                     else _MOVE_FREE if 'panel_pose' in types else _MOVE_NONE)
            optim_var_tilts.value = 'beam_tilts' in types
            optim_beam_mode.value = _BEAM_OPT if 'beam_angle' in types else _BEAM_KEEP
            optim_var_states.value = 'led_states' in types
            optim_var_roles.value = 'led_roles' in types
            optim_var_current.value = 'group_current' in types
            for v in variables:
                if v.get('type') == 'panel_pose':
                    optim_pos_delta.value = tuple(float(x) for x in v.get('pos_delta', (2, 2, 2)))
                    optim_rot_delta.value = tuple(float(x) for x in v.get('rot_delta', (10, 10, 10)))
                elif v.get('type') == 'duct_panel_pose':
                    import math as _math
                    d = v.get('duct', {})
                    r = float(d.get('radius', rd_radius.value))
                    rd_center.value = tuple(float(c) for c in d.get('center', rd_center.value))
                    rd_radius.value = r
                    ax = np.asarray(d.get('axis', [0, 0, 1]), float)
                    rd_axis.value = max(_DUCT_AXES, key=lambda k: abs(np.dot(_DUCT_AXES[k][0], ax)))
                    rd_rot.value = tuple(float(c) for c in (d.get('rotation_deg') or (0.0, 0.0, 0.0)))

                    def _half(rng):
                        return max(abs(rng[0]), abs(rng[1])) if rng else 0.0
                    rd_tol_arc.value = round(r * _math.radians(_half(v.get('theta_range'))), 1)
                    rd_tol_axial.value = _half(v.get('axial_range'))
                    rd_tol_radial.value = _half(v.get('radial_range'))
                    rd_tilt.value = int(_half(v.get('tilt_axial_range')))
                    rd_spin.value = int(_half(v.get('spin_range')))
                    rd_led_arc.value = round(r * _math.radians(_half(v.get('led_theta_range'))), 1)
                    rd_led_axial.value = _half(v.get('led_axial_range'))
                    mi = v.get('mirror_group_index')
                    label = next((l for l in rd_mirror.options if l.startswith(f"{mi}:")), None) if mi is not None else None
                    rd_mirror.value = label or _RD_NO_MIRROR
                elif v.get('type') == 'beam_tilts':
                    t = v.get('tilt_range', (-20, 20))
                    optim_tilt_range.value = int(max(abs(t[0]), abs(t[1])))
                elif v.get('type') == 'beam_angle':
                    optim_beam_range.value = tuple(float(x) for x in v.get('angle_range', (60, 130)))
                elif v.get('type') == 'group_current':
                    optim_current_range.value = tuple(float(x) for x in v.get('current_range', (0.5, 3.0)))
            optim_mode.value = _MODE_REFINE
        _optim_beam_mode_changed()
        _optim_sources_changed()
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
        cur_m = rd_mirror.value
        rd_mirror.options = [_RD_NO_MIRROR] + _optim_group_labels()
        rd_mirror.value = cur_m if cur_m in rd_mirror.options else _RD_NO_MIRROR
        _draw_refine_duct_preview()

    def _optim_selected_group_index():
        label = optim_group_dropdown.value or ""
        if not label or label == _OPTIM_NO_GROUP or ':' not in label:
            raise ValueError("Select a panel (click 'Refresh group list' after adding panels).")
        idx = int(label.split(':', 1)[0])
        if idx >= len(state.panels):
            raise ValueError("Group list is stale — click 'Refresh group list'.")
        return idx

    def _optim_group_variables(base_cfg):
        """Variables for the selected group; a fixed beam angle is written into ``base_cfg`` instead."""
        gi = _optim_selected_group_index()
        variables = []
        if optim_move_mode.value == _MOVE_FREE:
            variables.append({'type': 'panel_pose', 'group_index': gi,
                              'pos_delta': [float(v) for v in optim_pos_delta.value],
                              'rot_delta': [float(v) for v in optim_rot_delta.value]})
        elif optim_move_mode.value == _MOVE_DUCT:
            variables.append(_refine_duct_variable(gi))
        if optim_var_tilts.value:
            t = float(optim_tilt_range.value)
            variables.append({'type': 'beam_tilts', 'group_index': gi, 'tilt_range': [-t, t]})
        if optim_beam_mode.value == _BEAM_OPT:
            lo, hi = optim_beam_range.value
            variables.append({'type': 'beam_angle', 'group_index': gi, 'angle_range': [float(lo), float(hi)]})
        elif optim_beam_mode.value == _BEAM_FIXED:
            g = base_cfg['custom_groups'][gi]
            n = g.get('num_leds', len(g.get('led_positions', [])))
            g['led_viewing_angles'] = [float(optim_beam_fixed.value)] * n
        if optim_beam_profile.value != _BEAM_KEEP:
            if optim_beam_profile.value != _LAMBERTIAN and optim_beam_mode.value == _BEAM_OPT:
                raise ValueError("A measured beam profile fixes the beam shape: set 'Beam angle' to Keep or Fixed.")
            g = base_cfg['custom_groups'][gi]
            n = g.get('num_leds', len(g.get('led_positions', [])))
            g['led_profiles'] = [None if optim_beam_profile.value == _LAMBERTIAN else optim_beam_profile.value] * n
        if optim_var_roles.value and optim_flash_enable.value:
            variables.append({'type': 'led_roles', 'group_index': gi})
        elif optim_var_states.value:
            variables.append({'type': 'led_states', 'group_index': gi})
        if optim_var_current.value and optim_electrical.value:
            lo, hi = optim_current_range.value
            variables.append({'type': 'group_current', 'group_index': gi, 'current_range': [float(lo), float(hi)]})
        if not variables:
            raise ValueError("Nothing to optimise: enable a movement, beam or LED variable in ①.")
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
        """Light sources / test-case targets / constraints from the UI into ``work``."""
        flash, elec = bool(optim_flash_enable.value), bool(optim_electrical.value)
        work['objective'] = {
            'metric': optim_metric.value,
            'min_percentile': float(optim_min_pct.value),
            'coverage_weight': float(optim_cov_w.value),
            'tilt_fov_deg': float(tilt_fov_deg.value) if optim_tilt_enable.value else None,
            'tilt_fov_weight': float(optim_tilt_w.value),
            'tilt_room_grid_size': int(optim_tilt_grid.value),
        }
        work['constraints'] = {
            'max_leds': int(optim_max_leds.value) or None,
            'max_leds_weight': float(optim_max_leds_w.value),
            'led_cost': float(optim_led_cost.value),
            'min_led_spacing_cm': float(optim_spacing.value) or None,
            'spacing_weight': float(optim_spacing_w.value),
            'min_beam_angle_deg': float(optim_min_beam_angle.value) or None,
            'beam_angle_weight': float(optim_beam_angle_w.value),
            'symmetry_weight': float(optim_symmetry_w.value),
            'keep_out': list(keep_out or []),
            'keep_out_weight': float(keep_out_weight),
        }
        work['electrical'] = elec
        work.pop('driver', None)
        work.pop('cont_driver', None)
        if elec:
            work['constraints'].update({
                'max_drivers': int(optim_max_drivers.value) or None,
                'driver_cost': float(optim_driver_cost.value),
                'max_total_current_a': float(optim_max_current.value) or None,
                'max_peak_current_a': (float(optim_max_peak_current.value) or None) if flash else None,
                'max_pulse_drivers': (int(optim_max_pulse_drivers.value) or None) if flash else None,
                'pulse_driver_cost': float(optim_pulse_driver_cost.value) if flash else 0.0,
                'max_cont_drivers': (int(optim_max_cont_drivers.value) or None) if flash else None,
                'cont_driver_cost': float(optim_cont_driver_cost.value) if flash else 0.0,
            })
            work['driver'] = {
                'voltage_v': float(optim_drv_voltage.value),
                'efficacy_lm_per_w': float(optim_drv_efficacy.value),
                'max_current_a': float(optim_drv_max_current.value),
                'leds_per_driver': int(optim_leds_per_driver.value),
            }
            if flash:
                work['cont_driver'] = {'max_current_a': float(optim_cont_max_current.value),
                                       'leds_per_driver': int(optim_cont_leds_per_driver.value)}
        # The lux target applies to the judged image: the flash mode when it exists, flight otherwise.
        lux = {'min_avg_lux': float(optim_min_lux.value) or None,
               'min_avg_lux_dist': float(optim_lux_dist.value) or None,
               'lux_weight': float(optim_lux_w.value)}
        flight = {'name': 'flight', 'lumens': float(optim_vio_lumens.value)}
        if optim_vio_enable.value:
            flight.update({'vio_min_lux': float(optim_vio_lux.value),
                           'vio_min_fraction': float(optim_vio_fraction.value) / 100.0,
                           'vio_weight': float(optim_vio_w.value)})
            work['vio'] = {
                'position': [float(vio_pos_x.value), float(vio_pos_y.value), float(vio_pos_z.value)],
                'cam1_pitch': float(vio_cam1_pitch.value), 'cam1_yaw': float(vio_cam1_yaw.value),
                'cam2_pitch': float(vio_cam2_pitch.value), 'cam2_yaw': float(vio_cam2_yaw.value),
                'long_fov': float(vio_long_fov.value), 'landscape': bool(vio_landscape.value),
                'room_dist': float(optim_vio_room_dist.value), 'room_grid_size': int(optim_vio_room_grid.value),
            }
        else:
            work.pop('vio', None)
        modes = [flight]
        if flash:
            modes.append({'name': 'flash', 'flash': True, 'lumens': float(optim_flash_lumens.value), **lux})
        else:
            flight.update(lux)
        work['modes'] = modes

    def _optim_build():
        """Assemble (Problem, OptimizerSpec) for the selected design mode."""
        mode = optim_mode.value
        spec_dir = None
        base_cfg = None
        if mode == _MODE_PRESET:
            spec, spec_dir = _optim_current_spec()
            if spec is None:
                raise ValueError("Mode 3 needs a spec file — pick one in the '① Preset spec' folder.")
            work = copy.deepcopy(spec)
            if optim_preset_use_scene.value:
                base_cfg = get_current_config()
                base_cfg['name'] = current_config_name[0] or 'scene'
                work.pop('base_config', None)
            if optim_preset_use_ui.value:
                _optim_ui_wall_sections(work)
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
                work['variables'] = _optim_group_variables(base_cfg)
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

        problem = _problem_from_spec(work, spec_dir, base_cfg=base_cfg, use_gpu=optim_use_gpu.value,
                                     analytic=optim_analytic.value, **extra)
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
            + (" [analytic]" if logger.problem.analytic else (" [GPU]" if logger.problem.use_gpu else " [CPU]"))
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
        if problem.use_gpu and not problem.analytic:
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
        backend = "analytic" if problem.analytic else (_gpu_backend.gpu_backend_label() if problem.use_gpu else "CPU")
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
        stem = name.lower().replace(' ', '_')
        path = os.path.join(config_dir, f"{stem}.json")
        n = 2
        while os.path.exists(path):  # never overwrite an existing layout
            path = os.path.join(config_dir, f"{stem}_{n:03d}.json")
            n += 1
        lay = _convert_v1(cfg, default_lumens=float(optim_vio_lumens.value))
        lay.name = os.path.splitext(os.path.basename(path))[0]
        lay.flux.flash_lumens = float(optim_flash_lumens.value) if optim_flash_enable.value else None
        _save_json(lay, path)
        config_dropdown.options = get_available_configs()
        _optim_status(f"Saved best layout to {path}"
                      + ("  (name already existed → suffixed)" if n > 2 else ""), "#4CAF50")


    return SimpleNamespace()
