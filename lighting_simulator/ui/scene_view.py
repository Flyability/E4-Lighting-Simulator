"""3-D scene: update_scene, wall/grid, selected-panel inspector, Panel Designer, and the scene-level GUI wiring.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import json
import os
import time
import numpy as np
import trimesh
from lighting_simulator.camera.fov import (
    camera_fov_wall_trapezoid as _camera_fov_wall_trapezoid, fov_plane_mask_to_quads_and_contour,
    rasterize_fisheye_fov_on_plane, vio_hfov_vfov_deg, vio_optical_axis,
)
from lighting_simulator.domain.geometry import as_vec3 as _as_vec3
from lighting_simulator.domain.guides import (
    bake_and_disable_guide, circle_line_segments_m as _circle_line_segments_m,
    dynamic_group_world_geometry as _dynamic_group_world_geometry, enable_circular_guide,
    guide_is_enabled as _guide_is_enabled,
)
from lighting_simulator.domain.led_factory import create_leds
from lighting_simulator.domain.optics import effective_lambertian_exponent as _get_effective_n
from lighting_simulator.raytracing.mesh import ray_mesh_intersection as _ray_mesh_intersection
from lighting_simulator.ui.mesh_lighting import _build_stl_transform


def build(ctx):
    _ELIOS3_SLOTS = ctx._ELIOS3_SLOTS
    _clear_mirror_state = ctx._clear_mirror_state
    _clear_panel_slot = ctx._clear_panel_slot
    _enable_mirror_for = ctx._enable_mirror_for
    _expand_mirror_configs = ctx._expand_mirror_configs
    _inspector_handles = ctx._inspector_handles
    _inspector_syncing = ctx._inspector_syncing
    _just_clicked_mesh = ctx._just_clicked_mesh
    _last_intensity_cache = ctx._last_intensity_cache
    _last_room_cache = ctx._last_room_cache
    _mirror_counterpart_name = ctx._mirror_counterpart_name
    _mirror_primary = ctx._mirror_primary
    _mode_toggle_syncing = ctx._mode_toggle_syncing
    _owner_display_name = ctx._owner_display_name
    _owner_groups = ctx._owner_groups
    _owner_individual_leds = ctx._owner_individual_leds
    _panel_dropdowns = ctx._panel_dropdowns
    _panel_slot_data = ctx._panel_slot_data
    _refresh_uniformity = ctx._refresh_uniformity
    _refresh_vio_fov_label = ctx._refresh_vio_fov_label
    _select_panel_impl = ctx._select_panel_impl
    abs0_off_x = ctx.abs0_off_x
    abs0_off_y = ctx.abs0_off_y
    abs0_off_z = ctx.abs0_off_z
    abs1_off_x = ctx.abs1_off_x
    abs1_off_y = ctx.abs1_off_y
    abs1_off_z = ctx.abs1_off_z
    abs2_off_x = ctx.abs2_off_x
    abs2_off_y = ctx.abs2_off_y
    abs2_off_z = ctx.abs2_off_z
    abs2_rot_z = ctx.abs2_rot_z
    abs3_off_x = ctx.abs3_off_x
    abs3_off_y = ctx.abs3_off_y
    abs3_off_z = ctx.abs3_off_z
    abs3_rot_z = ctx.abs3_rot_z
    absorber_handles = ctx.absorber_handles
    absorbers_enable = ctx.absorbers_enable
    camera_fov_h = ctx.camera_fov_h
    camera_fov_handles = ctx.camera_fov_handles
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    capture_camera_fov_image = ctx.capture_camera_fov_image
    capture_fov_btn = ctx.capture_fov_btn
    cell_area_html = ctx.cell_area_html
    circle_center_slider = ctx.circle_center_slider
    clear_csv_pattern = ctx.clear_csv_pattern
    create_custom_group = ctx.create_custom_group
    csv_clear_btn = ctx.csv_clear_btn
    csv_import_btn = ctx.csv_import_btn
    current_leds = ctx.current_leds
    custom_groups = ctx.custom_groups
    designer_gizmo = ctx.designer_gizmo
    designer_led_nodes = ctx.designer_led_nodes
    designer_mode = ctx.designer_mode
    designer_scene_handles = ctx.designer_scene_handles
    designer_state = ctx.designer_state
    designer_syncing = ctx.designer_syncing
    designer_ui_handles = ctx.designer_ui_handles
    designer_widget_refs = ctx.designer_widget_refs
    diffuser_angle_slider = ctx.diffuser_angle_slider
    diffuser_enable_chk = ctx.diffuser_enable_chk
    draw_room_walls = ctx.draw_room_walls
    export_lux_matrix = ctx.export_lux_matrix
    export_lux_matrix_button = ctx.export_lux_matrix_button
    get_available_templates = ctx.get_available_templates
    global_pos_x_slider = ctx.global_pos_x_slider
    global_pos_y_slider = ctx.global_pos_y_slider
    global_pos_z_slider = ctx.global_pos_z_slider
    global_rotation_z_slider = ctx.global_rotation_z_slider
    group_buttons = ctx.group_buttons
    group_colors_hex = ctx.group_colors_hex
    guide_handles = ctx.guide_handles
    import_csv_pattern = ctx.import_csv_pattern
    imported_csv_handles = ctx.imported_csv_handles
    individual_leds = ctx.individual_leds
    inspector_tab = ctx.inspector_tab
    intensity_grid_size = ctx.intensity_grid_size
    intensity_handles = ctx.intensity_handles
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_threshold_slider = ctx.intensity_threshold_slider
    led_buttons = ctx.led_buttons
    led_handles = ctx.led_handles
    led_lumens_slider = ctx.led_lumens_slider
    led_states = ctx.led_states
    legend_html = ctx.legend_html
    loading_in_progress = ctx.loading_in_progress
    offset_front_neg_x = ctx.offset_front_neg_x
    offset_front_neg_y = ctx.offset_front_neg_y
    offset_front_neg_z = ctx.offset_front_neg_z
    offset_front_pos_x = ctx.offset_front_pos_x
    offset_front_pos_y = ctx.offset_front_pos_y
    offset_front_pos_z = ctx.offset_front_pos_z
    offset_side_neg_x = ctx.offset_side_neg_x
    offset_side_neg_y = ctx.offset_side_neg_y
    offset_side_neg_z = ctx.offset_side_neg_z
    offset_side_pos_x = ctx.offset_side_pos_x
    offset_side_pos_y = ctx.offset_side_pos_y
    offset_side_pos_z = ctx.offset_side_pos_z
    open_panel_designer_btn = ctx.open_panel_designer_btn
    radius_slider = ctx.radius_slider
    ray_handles = ctx.ray_handles
    ray_length_slider = ctx.ray_length_slider
    read_cell_at_ray = ctx.read_cell_at_ray
    ray_uniformity_slider = ctx.ray_uniformity_slider
    room_back_dist = ctx.room_back_dist
    room_front_dist = ctx.room_front_dist
    room_intensity_handles = ctx.room_intensity_handles
    room_mode_enable = ctx.room_mode_enable
    room_side_dist = ctx.room_side_dist
    room_top_bottom_dist = ctx.room_top_bottom_dist
    room_wall_handles = ctx.room_wall_handles
    rot_front_neg = ctx.rot_front_neg
    rot_front_pos = ctx.rot_front_pos
    rot_side_neg = ctx.rot_side_neg
    rot_side_pos = ctx.rot_side_pos
    rot_y_front_neg = ctx.rot_y_front_neg
    rot_y_front_pos = ctx.rot_y_front_pos
    rot_y_side_neg = ctx.rot_y_side_neg
    rot_y_side_pos = ctx.rot_y_side_pos
    row1_chk = ctx.row1_chk
    row2_chk = ctx.row2_chk
    row3_chk = ctx.row3_chk
    row4_chk = ctx.row4_chk
    row_buttons = ctx.row_buttons
    run_benchmark = ctx.run_benchmark
    run_benchmark_button = ctx.run_benchmark_button
    save_custom_group_template = ctx.save_custom_group_template
    select_panel = ctx.select_panel
    selected_owner = ctx.selected_owner
    server = ctx.server
    show_back_wall = ctx.show_back_wall
    show_camera_fov = ctx.show_camera_fov
    show_intensity_map = ctx.show_intensity_map
    show_led_markers = ctx.show_led_markers
    show_random_rays = ctx.show_random_rays
    show_rays_output = ctx.show_rays_output
    show_room_intensity = ctx.show_room_intensity
    show_room_walls = ctx.show_room_walls
    show_vio_fov = ctx.show_vio_fov
    static_scene_handles = ctx.static_scene_handles
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_mesh_data = ctx.stl_mesh_data
    stl_mesh_handle = ctx.stl_mesh_handle
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale
    template_dropdown = ctx.template_dropdown
    uniformity_percentile_slider = ctx.uniformity_percentile_slider
    update_all_led_buttons = ctx.update_all_led_buttons
    update_intensity_button = ctx.update_intensity_button
    update_intensity_map = ctx.update_intensity_map
    update_room_button = ctx.update_room_button
    update_room_intensity_map = ctx.update_room_intensity_map
    update_stl_mesh = ctx.update_stl_mesh
    update_ui_visibility = ctx.update_ui_visibility
    viewing_angle_slider = ctx.viewing_angle_slider
    vio_cam1_pitch = ctx.vio_cam1_pitch
    vio_cam1_yaw = ctx.vio_cam1_yaw
    vio_cam2_pitch = ctx.vio_cam2_pitch
    vio_cam2_yaw = ctx.vio_cam2_yaw
    vio_fill_fov = ctx.vio_fill_fov
    vio_fov_handles = ctx.vio_fov_handles
    vio_landscape = ctx.vio_landscape
    vio_long_fov = ctx.vio_long_fov
    vio_occupancy_lux = ctx.vio_occupancy_lux
    vio_pos_x = ctx.vio_pos_x
    vio_pos_y = ctx.vio_pos_y
    vio_pos_z = ctx.vio_pos_z
    wall_dist_slider = ctx.wall_dist_slider
    wall_view_size = ctx.wall_view_size

    def _clear_inspector():
        for h in _inspector_handles:
            try:
                h.remove()
            except Exception:
                pass
        _inspector_handles.clear()

    def _inspector_add(handle):
        _inspector_handles.append(handle)
        return handle

    def _mirror_slider(label, src, minv, maxv, step):
        sl = server.gui.add_slider(label, min=minv, max=maxv, step=step, initial_value=src.value)
        _inspector_add(sl)

        def _on(_):
            if _inspector_syncing[0]:
                return
            _inspector_syncing[0] = True
            try:
                src.value = sl.value
            finally:
                _inspector_syncing[0] = False

        sl.on_update(_on)
        return sl

    def _mirror_checkbox(label, src):
        chk = server.gui.add_checkbox(label, initial_value=bool(src.value))
        _inspector_add(chk)

        def _on(_):
            if _inspector_syncing[0]:
                return
            _inspector_syncing[0] = True
            try:
                src.value = chk.value
            finally:
                _inspector_syncing[0] = False

        chk.on_update(_on)
        return chk

    def _inspector_led_matrix(group):
        """ALL / Row / LED on-off buttons that write into group['led_states']."""
        led_states_g = group['led_states']
        led_rows = group.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        num_leds = group.get('num_leds', len(led_states_g))

        def _refresh_hidden_colors():
            if callable(group.get('update_button_colors')):
                group['update_button_colors']()
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and group in _pdata.get('groups', []):
                    uf = _pdata.get('update_btn_colors')
                    if callable(uf):
                        uf()

        _inspector_add(server.gui.add_html("<hr style='margin:6px 0;'><b>LED Controls:</b>"))
        all_btn = _inspector_add(server.gui.add_button(
            "ALL LEDs", color="#FF00FF" if any(led_states_g) else "#666666"
        ))

        def _on_all(_):
            new_state = not all(led_states_g)
            for i in range(len(led_states_g)):
                led_states_g[i] = new_state
            _refresh_hidden_colors()
            update_scene()
            populate_inspector(selected_owner[0])

        all_btn.on_click(_on_all)

        for row_idx, led_indices in enumerate(led_rows):
            any_on = any(led_states_g[i] for i in led_indices if i < len(led_states_g))
            row_btn = _inspector_add(server.gui.add_button(
                f"Row {row_idx + 1}", color="#FF00FF" if any_on else "#666666"
            ))

            def _make_row(indices):
                def _on(_):
                    new_state = not all(led_states_g[i] for i in indices if i < len(led_states_g))
                    for i in indices:
                        if i < len(led_states_g):
                            led_states_g[i] = new_state
                    _refresh_hidden_colors()
                    update_scene()
                    populate_inspector(selected_owner[0])
                return _on

            row_btn.on_click(_make_row(list(led_indices)))

        for led_idx in range(num_leds):
            if led_idx >= len(led_states_g):
                break
            color = "#FF00FF" if led_states_g[led_idx] else "#444444"
            led_btn = _inspector_add(server.gui.add_button(f"L{led_idx + 1}", color=color))

            def _make_led(idx):
                def _on(_):
                    led_states_g[idx] = not led_states_g[idx]
                    _refresh_hidden_colors()
                    update_scene()
                    populate_inspector(selected_owner[0])
                return _on

            led_btn.on_click(_make_led(led_idx))

    def _inspector_mirror_checkbox(owner):
        """Checkbox making `owner` the mirror primary (XZ reflection).

        Enabling it disables the panel on the opposite side and renders a live
        mirrored copy of this panel there instead."""
        is_primary = _mirror_primary[0] == owner
        chk = _inspector_add(server.gui.add_checkbox(
            "Mirror to other side (XZ)", initial_value=is_primary
        ))
        if is_primary:
            cp_name = _mirror_counterpart_name[0]
            note = (
                f"Mirroring active — **{cp_name}** is disabled."
                if cp_name else
                "Mirroring active — no opposite-side panel found to disable."
            )
            _inspector_add(server.gui.add_markdown(f"*{note}*"))

        @chk.on_update
        def _(_):
            if loading_in_progress[0]:
                return
            if chk.value:
                _enable_mirror_for(owner)
            elif _mirror_primary[0] == owner:
                _clear_mirror_state()
            update_scene()
            if selected_owner[0] == owner:
                populate_inspector(owner)

    def _inspector_group_pose_sliders(group):
        """Free XYZ/Euler sliders; hidden while the circular guide is anchoring the panel."""
        if _guide_is_enabled(group):
            return
        _mirror_slider("Position X (cm)", group['pos_x'], -100, 100, 0.1)
        _mirror_slider("Position Y (cm)", group['pos_y'], -50, 50, 0.1)
        _mirror_slider("Position Z (cm)", group['pos_z'], -50, 50, 0.1)
        if group.get('rot_tilt_lr') is not None:
            _mirror_slider("Tilt Left/Right (°)", group['rot_tilt_lr'], -180, 180, 1)
        if group.get('rot_tilt_ud') is not None:
            _mirror_slider("Tilt Up/Down (°)", group['rot_tilt_ud'], -180, 180, 1)
        if group.get('rot_roll') is not None:
            _mirror_slider("Rotate on axis (°)", group['rot_roll'], -180, 180, 1)

    def _inspector_guide_controls(group):
        """Checkbox to fit/anchor a circular guide and the slide slider while on."""
        enabled = _guide_is_enabled(group)
        chk = _inspector_add(server.gui.add_checkbox(
            "Anchor to circular guide", initial_value=enabled
        ))
        if group.get('guide_error'):
            _inspector_add(server.gui.add_markdown(f"**Guide:** {group['guide_error']}"))
        if enabled:
            guide = group.get('guide') or {}
            if guide.get('warning'):
                _inspector_add(server.gui.add_markdown(f"*{guide['warning']}*"))
            n_circ = len(guide.get('circles') or [])
            _inspector_add(server.gui.add_markdown(
                f"Pose locked — {n_circ} construction circle(s). "
                "Uncheck to restore free XYZ/Euler."
            ))
            sl = _inspector_add(server.gui.add_slider(
                "Slide along guide (°)",
                min=-180, max=180, step=0.5,
                initial_value=float(guide.get('theta_deg', 0.0)),
            ))

            @sl.on_update
            def _(_):
                if loading_in_progress[0] or _inspector_syncing[0]:
                    return
                if not isinstance(group.get('guide'), dict):
                    return
                group['guide']['theta_deg'] = float(sl.value)
                update_scene()

        @chk.on_update
        def _(_):
            if loading_in_progress[0] or _inspector_syncing[0]:
                return
            if chk.value:
                ok, err = enable_circular_guide(group)
                if not ok:
                    group['guide'] = None
                    group['guide_error'] = err or "Could not fit a circular guide."
                else:
                    group['guide_error'] = None
            else:
                bake_and_disable_guide(group)
            update_scene()
            if selected_owner[0] is not None:
                populate_inspector(selected_owner[0])

    def _designer_rotation_matrix(rx_deg, ry_deg, rz_deg):
        """LED-local Euler convention used by the designer: Rx @ Ry @ Rz."""
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def _designer_euler_from_axes(direction, row_direction=None):
        """Recover designer Euler values from an LED forward and row axis."""
        forward = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        if row_direction is None:
            row = np.array([0.0, 1.0, 0.0])
        else:
            row = np.asarray(row_direction, dtype=float)
            row -= forward * np.dot(row, forward)
            if np.linalg.norm(row) < 1e-9:
                row = np.array([0.0, 1.0, 0.0])
        row /= np.linalg.norm(row)
        third = np.cross(forward, row)
        R = np.column_stack([forward, row, third])
        ry = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
        if abs(np.cos(ry)) > 1e-6:
            rx = np.arctan2(-R[1, 2], R[2, 2])
            rz = np.arctan2(-R[0, 1], R[0, 0])
        else:
            rx, rz = np.arctan2(R[2, 1], R[1, 1]), 0.0
        return tuple(float(v) for v in np.degrees([rx, ry, rz]))

    def _designer_quaternion(R):
        """Return a wxyz quaternion for an orthonormal 3x3 rotation matrix."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return (0.25 / s, (R[2, 1] - R[1, 2]) * s,
                    (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s)
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                    (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
        if i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                    0.25 * s, (R[1, 2] + R[2, 1]) / s)
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s, 0.25 * s)

    def _designer_matrix_from_wxyz(wxyz):
        """Convert a wxyz quaternion to a 3x3 rotation matrix."""
        w, x, y, z = [float(v) for v in wxyz]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=float)

    def _clear_designer_gizmo():
        if designer_gizmo[0] is not None:
            try:
                designer_gizmo[0].remove()
            except (KeyError, AttributeError):
                pass
            designer_gizmo[0] = None

    def _clear_designer_scene():
        _clear_designer_gizmo()
        for nodes in designer_led_nodes:
            for key in ('box', 'frame'):
                try:
                    nodes[key].remove()
                except (KeyError, AttributeError):
                    pass
        designer_led_nodes.clear()
        for handle in designer_scene_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_scene_handles.clear()

    def _designer_led_pose(led):
        """Return (position_m, wxyz) for an LED in designer state."""
        R = _designer_rotation_matrix(led['rx'], led['ry'], led['rz'])
        pos = (led['x'] / 100.0, led['y'] / 100.0, led['z'] / 100.0)
        return pos, _designer_quaternion(R)

    def _update_designer_led_pose(index):
        """In-place pose update for one LED box + frame (no recreate)."""
        if index is None or index < 0 or index >= len(designer_led_nodes):
            return
        led = designer_state[0]['leds'][index]
        pos, wxyz = _designer_led_pose(led)
        nodes = designer_led_nodes[index]
        nodes['box'].position = pos
        nodes['box'].wxyz = wxyz
        nodes['frame'].position = pos
        nodes['frame'].wxyz = wxyz

    def _update_designer_selection_colors():
        selected = designer_state[0]['selected_led']
        for index, nodes in enumerate(designer_led_nodes):
            try:
                # Viser expects 0-255 when assigning .color (floats stay near-black).
                nodes['box'].color = (38, 255, 255) if index == selected else (255, 255, 255)
            except Exception:
                pass

    def _sync_widgets_from_led(led):
        """Push LED state into slider/number widgets without feedback loops."""
        refs = designer_widget_refs[0]
        if not refs:
            return
        designer_syncing[0] = True
        try:
            for key, pair in refs.items():
                if key not in led:
                    continue
                slider, number = pair
                val = led[key]
                try:
                    slider.value = val
                except Exception:
                    pass
                try:
                    number.value = val
                except Exception:
                    pass
        finally:
            designer_syncing[0] = False

    def _place_gizmo():
        """Create/move the transform gizmo onto the selected LED (LED-local axes)."""
        index = designer_state[0]['selected_led']
        leds = designer_state[0]['leds']
        if index is None or index < 0 or index >= len(leds):
            _clear_designer_gizmo()
            return
        led = leds[index]
        pos, wxyz = _designer_led_pose(led)
        if designer_gizmo[0] is None:
            gizmo = server.scene.add_transform_controls(
                "/panel_designer/gizmo",
                scale=0.08,
                depth_test=False,
                position=pos,
                wxyz=wxyz,
            )

            @gizmo.on_update
            def _on_gizmo_update(_):
                if designer_syncing[0] or not designer_mode[0]:
                    return
                i = designer_state[0]['selected_led']
                if i is None or i >= len(designer_state[0]['leds']):
                    return
                cur = designer_state[0]['leds'][i]
                gx, gy, gz = designer_gizmo[0].position
                cur['x'], cur['y'], cur['z'] = gx * 100.0, gy * 100.0, gz * 100.0
                Rm = _designer_matrix_from_wxyz(designer_gizmo[0].wxyz)
                rx, ry, rz = _designer_euler_from_axes(Rm @ np.array([1.0, 0.0, 0.0]),
                                                      Rm @ np.array([0.0, 1.0, 0.0]))
                cur['rx'], cur['ry'], cur['rz'] = rx, ry, rz
                _update_designer_led_pose(i)
                _sync_widgets_from_led(cur)

            @gizmo.on_drag_end
            def _on_gizmo_drag_end(_):
                if not designer_mode[0]:
                    return
                i = designer_state[0]['selected_led']
                if i is None or i >= len(designer_state[0]['leds']):
                    return
                cur = designer_state[0]['leds'][i]
                # Snap to the same steps as the UI controls.
                cur['x'] = round(cur['x'] / 0.05) * 0.05
                cur['y'] = round(cur['y'] / 0.05) * 0.05
                cur['z'] = round(cur['z'] / 0.05) * 0.05
                cur['rx'] = round(cur['rx'] / 0.5) * 0.5
                cur['ry'] = round(cur['ry'] / 0.5) * 0.5
                cur['rz'] = round(cur['rz'] / 0.5) * 0.5
                _update_designer_led_pose(i)
                _sync_widgets_from_led(cur)
                designer_syncing[0] = True
                try:
                    pos2, wxyz2 = _designer_led_pose(cur)
                    designer_gizmo[0].position = pos2
                    designer_gizmo[0].wxyz = wxyz2
                finally:
                    designer_syncing[0] = False

            designer_gizmo[0] = gizmo
        else:
            designer_syncing[0] = True
            try:
                designer_gizmo[0].position = pos
                designer_gizmo[0].wxyz = wxyz
            finally:
                designer_syncing[0] = False

    def _select_designer_led_index(index, rebuild_ui=True):
        """Select an LED for editing; move gizmo and refresh inspector."""
        _just_clicked_mesh[0] = True
        designer_state[0]['selected_led'] = index
        _update_designer_selection_colors()
        _place_gizmo()
        if rebuild_ui:
            _build_designer_ui()

    def update_designer_scene(full_rebuild=True):
        """Render the isolated panel. full_rebuild recreates LED meshes + gizmo."""
        if not designer_mode[0]:
            _clear_designer_scene()
            return
        leds = designer_state[0]['leds']
        if full_rebuild:
            _clear_designer_scene()
            extent_cm = max(10.0, max(
                (max(abs(led[a]) for a in ('x', 'y', 'z')) + led['size']
                 for led in leds), default=5.0
            ))
            designer_scene_handles.append(server.scene.add_grid(
                "/panel_designer/grid", width=extent_cm * 2 / 100.0,
                height=extent_cm * 2 / 100.0, plane="yz", cell_size=0.01,
            ))
            designer_scene_handles.append(server.scene.add_frame(
                "/panel_designer/panel_reference_frame", axes_length=0.05,
                axes_radius=0.002, origin_radius=0.004,
            ))
            for index, led in enumerate(leds):
                pos, wxyz = _designer_led_pose(led)
                selected = index == designer_state[0]['selected_led']
                # Sibling paths (not child of the box) so poses are not double-applied.
                box = server.scene.add_box(
                    f"/panel_designer/led_{index}",
                    dimensions=(0.0005, led['size'] / 100.0, led['size'] / 100.0),
                    color=(38, 255, 255) if selected else (255, 255, 255),
                    position=pos,
                    wxyz=wxyz,
                )
                frame = server.scene.add_frame(
                    f"/panel_designer/frame_{index}",
                    axes_length=max(0.01, led['size'] / 150.0), axes_radius=0.001,
                    origin_radius=0.002,
                    position=pos,
                    wxyz=wxyz,
                )

                def _on_led_click(_event, selected_index=index):
                    _select_designer_led_index(selected_index)

                box.on_click(_on_led_click)
                designer_led_nodes.append({'box': box, 'frame': frame})
            _place_gizmo()
        else:
            for index in range(len(leds)):
                _update_designer_led_pose(index)
            _update_designer_selection_colors()
            _place_gizmo()

    def _designer_set_camera(view):
        """Place every connected client on an orthographic-style panel view."""
        leds = designer_state[0]['leds']
        extent = max(0.3, max(
            (max(abs(led[a]) for a in ('x', 'y', 'z')) / 100.0 + led['size'] / 100.0
             for led in leds), default=0.1
        ) * 3)
        views = {
            'XY': ((0, 0, extent), (0, 1, 0)),
            '-XY': ((0, 0, -extent), (0, 1, 0)),
            'XZ': ((0, extent, 0), (0, 0, 1)),
            '-XZ': ((0, -extent, 0), (0, 0, 1)),
            'YZ': ((extent, 0, 0), (0, 0, 1)),
            '-YZ': ((-extent, 0, 0), (0, 0, 1)),
        }
        position, up = views[view]
        for client in server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = (0, 0, 0)
            client.camera.up_direction = up

    def _designer_apply_to_group(group, leds):
        """Replace one dynamic group's local LED arrays from designer LEDs."""
        group['is_dynamic'] = True
        group['num_leds'] = len(leds)
        group['led_positions'] = [(led['x'], led['y'], led['z']) for led in leds]
        group['original_led_positions'] = list(group['led_positions'])
        rotations = []
        rows = []
        for led in leds:
            R = _designer_rotation_matrix(led['rx'], led['ry'], led['rz'])
            rotations.append(tuple(R @ np.array([1.0, 0.0, 0.0])))
            rows.append(tuple(R @ np.array([0.0, 1.0, 0.0])))
        group['led_rotations'] = rotations
        group['original_led_rotations'] = list(rotations)
        group['led_row_directions'] = rows
        group['original_led_row_directions'] = list(rows)
        group['led_euler_angles'] = [(led['rx'], led['ry'], led['rz']) for led in leds]
        group['led_beam_tilts'] = [0.0] * len(leds)
        group['led_sizes'] = [led['size'] for led in leds]
        group['led_viewing_angles'] = [led['view_angle'] for led in leds]
        group['led_lumens'] = [led['lumens'] if led['custom_lumens'] else None for led in leds]
        group['led_states'] = [True] * len(leds)
        group['led_rows'] = [list(range(len(leds)))] if leds else []

    def _designer_template_group():
        """Return a serializable dynamic group for Save As Template."""
        leds = designer_state[0]['leds']
        temp = {'is_dynamic': True, 'position': [0, 0, 0],
                'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0,
                'enabled': True}
        _designer_apply_to_group(temp, leds)
        return temp

    def _save_designer_template(_event=None):
        name = designer_state[0]['name'].strip() or 'Untitled Panel'
        save_custom_group_template(name, [_designer_template_group()], [])
        fresh = get_available_templates()
        template_dropdown.options = ["Empty"] + fresh
        for dropdown in _panel_dropdowns:
            current = dropdown.value
            dropdown.options = ["-- Nessuno --"] + fresh
            dropdown.value = current if current in dropdown.options else "-- Nessuno --"
        print(f"✓ Panel Designer template saved: {name}")

    def _save_designer(_event=None):
        state = designer_state[0]
        by_group = {}
        for led in state['leds']:
            group = led.get('_group')
            if group is not None:
                by_group.setdefault(id(group), [group, []])[1].append(led)
            elif led.get('_individual') is not None:
                item = led['_individual']
                item['pos_x'].value, item['pos_y'].value, item['pos_z'].value = led['x'], led['y'], led['z']
                item['rot_x'].value, item['rot_y'].value, item['rot_z'].value = led['rx'], led['ry'], led['rz']
                item['size'].value, item['viewing_angle'].value = led['size'], led['view_angle']
                item['lumens_override'].value = led['custom_lumens']
                item['lumens_value'].value = led['lumens']
        if state['editing_owner'] is None:
            group = create_custom_group(skip_update_scene=True, num_leds=max(1, len(state['leds'])),
                                        led_rows=[list(range(len(state['leds'])))] if state['leds'] else [[0]],
                                        group_name=state['name'])
            _designer_apply_to_group(group, state['leds'])
            group['template_name'] = state['name']
            selected_owner[0] = ('custom_group', group['id'])
        else:
            for group, group_leds in by_group.values():
                _designer_apply_to_group(group, group_leds)
        _exit_designer()

    def _collect_normal_scene_visibility_targets():
        """Handles that should be hidden while the panel designer is open."""
        targets = list(static_scene_handles)
        try:
            targets.append(wall_handle)
        except Exception:
            pass
        if stl_mesh_handle[0] is not None:
            targets.append(stl_mesh_handle[0])
        targets.extend(room_wall_handles)
        targets.extend(intensity_handles)
        targets.extend(room_intensity_handles)
        targets.extend(imported_csv_handles)
        return targets

    def _set_normal_scene_visible(visible):
        for handle in _collect_normal_scene_visibility_targets():
            try:
                handle.visible = visible
            except Exception:
                pass

    def _clear_normal_dynamic_scene():
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles, vio_fov_handles, guide_handles
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        led_handles.clear()
        ray_handles.clear()
        absorber_handles.clear()
        camera_fov_handles.clear()
        vio_fov_handles.clear()
        guide_handles.clear()
    def _exit_designer(_event=None):
        designer_mode[0] = False
        designer_widget_refs[0] = {}
        _clear_designer_scene()
        for handle in designer_ui_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_ui_handles.clear()
        _clear_inspector()
        _set_normal_scene_visible(True)
        update_scene()
        populate_inspector(selected_owner[0])

    def _build_designer_ui():
        """Rebuild designer controls inside the floating Selected inspector."""
        _clear_inspector()
        for handle in designer_ui_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_ui_handles.clear()
        designer_widget_refs[0] = {}
        state = designer_state[0]
        with inspector_tab:
            _inspector_add(server.gui.add_markdown("**Panel Designer**"))
            exit_btn = _inspector_add(server.gui.add_button("Exit Designer", color="red"))
            exit_btn.on_click(_exit_designer)

            name_input = _inspector_add(server.gui.add_text("Panel name", initial_value=state['name']))
            name_input.on_update(lambda _: state.__setitem__('name', name_input.value))

            _inspector_add(server.gui.add_markdown("**View plane**"))
            for view in ('XY', 'XZ', 'YZ', '-XY', '-XZ', '-YZ'):
                button = _inspector_add(server.gui.add_button(view))
                button.on_click(lambda _, v=view: _designer_set_camera(v))

            _inspector_add(server.gui.add_markdown("**LEDs** — click a mesh or button; drag the gizmo arrows/rings"))
            for index, _led in enumerate(state['leds']):
                if index == state['selected_led']:
                    button = _inspector_add(server.gui.add_button(f"LED {index + 1}", color="#00CCCC"))
                else:
                    button = _inspector_add(server.gui.add_button(f"LED {index + 1}"))

                def _choose(_event, selected_index=index):
                    _select_designer_led_index(selected_index)

                button.on_click(_choose)

            add_btn = _inspector_add(server.gui.add_button("Add LED", color="green"))
            duplicate_btn = _inspector_add(server.gui.add_button("Duplicate LED"))
            remove_btn = _inspector_add(server.gui.add_button("Remove LED", color="red"))

            def _add(_):
                state['leds'].append({
                    'x': 0.0, 'y': 0.0, 'z': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0,
                    'size': 0.5, 'view_angle': 120.0, 'custom_lumens': False,
                    'lumens': 100.0, '_group': state.get('target_group'),
                })
                state['selected_led'] = len(state['leds']) - 1
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            add_btn.on_click(_add)

            def _duplicate(_):
                i = state['selected_led']
                if i is None:
                    return
                new_led = dict(state['leds'][i])
                new_led['y'] += 0.5
                state['leds'].append(new_led)
                state['selected_led'] = len(state['leds']) - 1
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            duplicate_btn.on_click(_duplicate)

            def _remove(_):
                i = state['selected_led']
                if i is None:
                    return
                state['leds'].pop(i)
                state['selected_led'] = min(i, len(state['leds']) - 1) if state['leds'] else None
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            remove_btn.on_click(_remove)

            i = state['selected_led']
            if i is not None and i < len(state['leds']):
                led = state['leds'][i]
                _inspector_add(server.gui.add_markdown(
                    f"**LED {i + 1}** — drag gizmo (LED-local axes) or edit values"
                ))
                for label, key, low, high, step in (
                    ('Position X (cm)', 'x', -30, 30, 0.05),
                    ('Position Y (cm)', 'y', -30, 30, 0.05),
                    ('Position Z (cm)', 'z', -30, 30, 0.05),
                    ('Rotation X (°)', 'rx', -180, 180, 0.5),
                    ('Rotation Y (°)', 'ry', -180, 180, 0.5),
                    ('Rotation Z (°)', 'rz', -180, 180, 0.5),
                    ('Square side (cm)', 'size', 0.1, 10, 0.05),
                    ('Viewing angle (°)', 'view_angle', 1, 180, 1),
                ):
                    slider = _inspector_add(server.gui.add_slider(
                        label, min=low, max=high, step=step, initial_value=led[key]
                    ))
                    number = _inspector_add(server.gui.add_number(
                        f"{label} value", initial_value=led[key], step=step
                    ))
                    designer_widget_refs[0][key] = (slider, number)

                    def _bind(source, other, field, needs_rebuild):
                        def _update(_):
                            if designer_syncing[0]:
                                return
                            designer_syncing[0] = True
                            try:
                                led[field] = float(source.value)
                                other.value = source.value
                            finally:
                                designer_syncing[0] = False
                            if needs_rebuild:
                                update_designer_scene(full_rebuild=True)
                            else:
                                _update_designer_led_pose(state['selected_led'])
                                _place_gizmo()
                        source.on_update(_update)

                    needs_rebuild = key in ('size', 'view_angle')
                    _bind(slider, number, key, needs_rebuild)
                    _bind(number, slider, key, needs_rebuild)

                lumens_check = _inspector_add(server.gui.add_checkbox(
                    "Custom lumens", initial_value=led['custom_lumens']
                ))
                lumens_value = _inspector_add(server.gui.add_number(
                    "Lumens value", initial_value=led['lumens'], min=1, step=1
                ))
                lumens_check.on_update(lambda _: led.__setitem__('custom_lumens', lumens_check.value))
                lumens_value.on_update(lambda _: led.__setitem__('lumens', float(lumens_value.value)))

            _inspector_add(server.gui.add_html("<hr style='margin:8px 0;'>"))
            save_btn = _inspector_add(server.gui.add_button("Save", color="green"))
            template_btn = _inspector_add(server.gui.add_button("Save As Template"))
            cancel_btn = _inspector_add(server.gui.add_button("Cancel", color="red"))
            save_btn.on_click(_save_designer)
            template_btn.on_click(_save_designer_template)
            cancel_btn.on_click(_exit_designer)

    def _enter_panel_designer(owner=None):
        """Open an empty designer or extract a selected panel's local LEDs."""
        state = {'name': 'Untitled Panel', 'editing_owner': owner, 'leds': [],
                 'selected_led': None, 'target_group': None}
        groups = _owner_groups(owner) if owner is not None else []
        for group in groups:
            if not group.get('is_dynamic', False):
                continue
            state['name'] = group.get('template_name') or _owner_display_name(owner)
            positions = group.get('original_led_positions', group.get('led_positions', []))
            eulers = group.get('led_euler_angles', [])
            directions = group.get('original_led_rotations', group.get('led_rotations', []))
            rows = group.get('original_led_row_directions', group.get('led_row_directions', []))
            sizes = group.get('led_sizes', [])
            view_angles = group.get('led_viewing_angles', [])
            lumens_list = group.get('led_lumens', [])
            for i, pos in enumerate(positions):
                euler = eulers[i] if i < len(eulers) else _designer_euler_from_axes(
                    directions[i] if i < len(directions) else (1, 0, 0),
                    rows[i] if i < len(rows) else None,
                )
                state['leds'].append({
                    'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                    'rx': float(euler[0]), 'ry': float(euler[1]), 'rz': float(euler[2]),
                    'size': float(sizes[i]) if i < len(sizes) else 0.5,
                    'view_angle': float(view_angles[i]) if i < len(view_angles) else 120.0,
                    'custom_lumens': i < len(lumens_list) and lumens_list[i] is not None,
                    'lumens': float(lumens_list[i]) if i < len(lumens_list) and lumens_list[i] is not None else 100.0,
                    '_group': group,
                })
            state['target_group'] = state['target_group'] or group
        for item in _owner_individual_leds(owner):
            state['name'] = _owner_display_name(owner)
            state['leds'].append({
                'x': item['pos_x'].value, 'y': item['pos_y'].value, 'z': item['pos_z'].value,
                'rx': item['rot_x'].value, 'ry': item['rot_y'].value, 'rz': item['rot_z'].value,
                'size': item['size'].value, 'view_angle': item['viewing_angle'].value,
                'custom_lumens': item['lumens_override'].value,
                'lumens': item['lumens_value'].value, '_individual': item,
            })
        if owner is not None and not state['leds']:
            print("Panel Designer requires a dynamic panel or individual LEDs.")
            return
        if state['leds']:
            state['selected_led'] = 0
        designer_state[0] = state
        designer_mode[0] = True
        # Isolate the view without destroying static nodes (wall/grid/axes/STL).
        _clear_normal_dynamic_scene()
        _set_normal_scene_visible(False)
        _build_designer_ui()
        update_designer_scene(full_rebuild=True)
        _designer_set_camera('YZ')

    open_panel_designer_btn.on_click(lambda _: _enter_panel_designer())

    def populate_inspector(owner):
        """Rebuild the floating inspector for the given owner (or empty state)."""
        if designer_mode[0]:
            # Designer owns the Selected tab while active.
            return
        _clear_inspector()
        with inspector_tab:
            if owner is None:
                _inspector_add(server.gui.add_markdown(
                    "Click a panel or LED group in the 3D view to edit it."
                ))
                return

            kind, key = owner
            deselect_btn = _inspector_add(server.gui.add_button("Deselect"))

            @deselect_btn.on_click
            def _(_):
                select_panel(None)

            if kind == 'slot':
                data = _panel_slot_data[key] if 0 <= key < len(_panel_slot_data) else None
                slot_groups = [g for g in custom_groups if g.get('panel_slot') == key]
                if data is None and not slot_groups:
                    _inspector_add(server.gui.add_markdown("This slot is empty."))
                    return
                if data is None:
                    # Panel restored from a saved configuration (no slot UI):
                    # expose its group controls directly so it stays editable.
                    slot_name = _ELIOS3_SLOTS[key]['name']
                    _inspector_add(server.gui.add_markdown(f"**Slot {slot_name}**"))
                    _inspector_mirror_checkbox(owner)
                    for g_idx, group in enumerate(slot_groups):
                        if len(slot_groups) > 1:
                            _inspector_add(server.gui.add_html(
                                f"<hr style='margin:8px 0;'><b>Group {g_idx + 1}</b>"
                            ))
                        _mirror_checkbox("Enable", group['enable'])
                        _inspector_guide_controls(group)
                        _inspector_group_pose_sliders(group)
                        if group.get('lumens_override') is not None:
                            _mirror_checkbox("Enable custom lumens", group['lumens_override'])
                            _mirror_slider("Lumens per LED (lm)", group['lumens_value'], 1, 900000, 1)
                        _inspector_led_matrix(group)
                    return
                slot_name = data.get('slot_label', _ELIOS3_SLOTS[key]['name'])
                tmpl = data.get('template_display', data.get('template_name', ''))
                _inspector_add(server.gui.add_markdown(
                    f"**Slot {slot_name}**  \n{tmpl}"
                    + ("  \n*Loaded as individual LEDs*" if data.get('as_individual') else "")
                ))
                edit_designer_btn = _inspector_add(server.gui.add_button("Edit in Panel Designer"))
                edit_designer_btn.on_click(lambda _, o=owner: _enter_panel_designer(o))
                _inspector_mirror_checkbox(owner)
                ctrl = data.get('controls') or {}
                slot_groups_live = data.get('groups') or slot_groups
                slot_anchored = any(_guide_is_enabled(g) for g in slot_groups_live)
                if ctrl.get('enable') is not None:
                    _mirror_checkbox("Enable Slot", ctrl['enable'])
                    if slot_anchored:
                        _inspector_add(server.gui.add_markdown(
                            "Slot offset/rotation locked while a group is anchored to a circular guide."
                        ))
                    else:
                        _mirror_slider("Offset X (cm)", ctrl['pos_x'], -50, 50, 0.1)
                        _mirror_slider("Offset Y (cm)", ctrl['pos_y'], -50, 50, 0.1)
                        _mirror_slider("Offset Z (cm)", ctrl['pos_z'], -50, 50, 0.1)
                        _mirror_slider("Rotate on axis (°)", ctrl['rot_x'], -180, 180, 1)
                        _mirror_slider("Tilt Up/Down (°)", ctrl['rot_y'], -180, 180, 1)
                        _mirror_slider("Tilt Left/Right (°)", ctrl['rot_z'], -180, 180, 1)
                    _inspector_add(server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>"))
                    _mirror_checkbox("Enable custom lumens", ctrl['lumens_chk'])
                    _mirror_slider("Lumens per LED (lm)", ctrl['lumens_slider'], 1, 900000, 1)
                if data.get('as_individual'):
                    _inspector_add(server.gui.add_markdown(
                        "Per-LED edits are in the **Individual LEDs** folder."
                    ))
                else:
                    for g_idx, group in enumerate(slot_groups_live):
                        _inspector_add(server.gui.add_html(
                            f"<hr style='margin:8px 0;'><b>Group {g_idx + 1}</b>"
                        ))
                        _inspector_guide_controls(group)
                        _inspector_led_matrix(group)
                remove_btn = _inspector_add(server.gui.add_button("Remove Slot", color="red"))

                @remove_btn.on_click
                def _(_):
                    _clear_panel_slot(key)
                    if key < len(_panel_dropdowns):
                        _panel_dropdowns[key].value = "-- Nessuno --"
                    update_scene()

            elif kind == 'custom_group':
                group = next((g for g in custom_groups if g.get('id') == key), None)
                if group is None:
                    _inspector_add(server.gui.add_markdown("This group no longer exists."))
                    return
                title = group.get('template_name') or f"Custom Group {key}"
                _inspector_add(server.gui.add_markdown(f"**{title}**"))
                edit_designer_btn = _inspector_add(server.gui.add_button("Edit in Panel Designer"))
                edit_designer_btn.on_click(lambda _, o=owner: _enter_panel_designer(o))
                _inspector_mirror_checkbox(owner)
                _mirror_checkbox("Enable", group['enable'])
                _inspector_guide_controls(group)
                _inspector_group_pose_sliders(group)
                if group.get('lumens_override') is not None:
                    _inspector_add(server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>"))
                    _mirror_checkbox("Enable custom lumens", group['lumens_override'])
                    _mirror_slider("Lumens per LED (lm)", group['lumens_value'], 1, 900000, 1)
                _inspector_led_matrix(group)
                remove_btn = _inspector_add(server.gui.add_button("Remove Group", color="red"))

                @remove_btn.on_click
                def _(_):
                    if group in custom_groups:
                        custom_groups.remove(group)
                    try:
                        group['folder'].remove()
                    except Exception:
                        pass
                    select_panel(None)

            elif kind == 'base_group':
                names = ["Front+", "Front-", "Side+", "Side-"]
                rot_z = [rot_front_pos, rot_front_neg, rot_side_pos, rot_side_neg]
                rot_y = [rot_y_front_pos, rot_y_front_neg, rot_y_side_pos, rot_y_side_neg]
                off_x = [offset_front_pos_x, offset_front_neg_x, offset_side_pos_x, offset_side_neg_x]
                off_y = [offset_front_pos_y, offset_front_neg_y, offset_side_pos_y, offset_side_neg_y]
                off_z = [offset_front_pos_z, offset_front_neg_z, offset_side_pos_z, offset_side_neg_z]
                if key < 0 or key > 3:
                    _inspector_add(server.gui.add_markdown("Unknown base group."))
                    return
                _inspector_add(server.gui.add_markdown(f"**Base group: {names[key]}**"))
                _mirror_slider(f"Rotate {names[key]} Z (°)", rot_z[key], -180, 180, 1)
                _mirror_slider(f"Rotate {names[key]} local Y (tilt °)", rot_y[key], -180, 180, 1)
                ymin, ymax = ((-40, 40) if key == 2 else (-40, 50) if key == 3 else (-30, 30))
                _mirror_slider("Offset X (cm)", off_x[key], -30, 30, 0.1)
                _mirror_slider("Offset Y (cm)", off_y[key], ymin, ymax, 0.1)
                _mirror_slider("Offset Z (cm)", off_z[key], -30, 30, 0.1)
                _inspector_add(server.gui.add_html("<hr style='margin:6px 0;'><b>LED Controls:</b>"))
                color_hex = group_colors_hex[key]
                start = key * 12
                any_on = any(led_states[start:start + 12])
                all_btn = _inspector_add(server.gui.add_button("ALL", color=color_hex if any_on else "#444444"))

                def _on_all(_):
                    new_state = not all(led_states[start:start + 12])
                    for i in range(start, start + 12):
                        led_states[i] = new_state
                    update_all_led_buttons()
                    update_scene()
                    update_ui_visibility()
                    populate_inspector(selected_owner[0])

                all_btn.on_click(_on_all)
                for row_idx in range(4):
                    r0 = start + row_idx * 3
                    any_row = any(led_states[r0:r0 + 3])
                    row_btn = _inspector_add(server.gui.add_button(
                        f"Row {row_idx + 1}", color=color_hex if any_row else "#666666"
                    ))

                    def _make_row(s):
                        def _on(_):
                            new_state = not all(led_states[s:s + 3])
                            for i in range(s, s + 3):
                                led_states[i] = new_state
                            update_all_led_buttons()
                            update_scene()
                            update_ui_visibility()
                            populate_inspector(selected_owner[0])
                        return _on

                    row_btn.on_click(_make_row(r0))
                for li in range(12):
                    gi = start + li
                    on = led_states[gi]
                    led_btn = _inspector_add(server.gui.add_button(
                        f"L{li + 1}", color=color_hex if on else "#444444"
                    ))

                    def _make_led(idx):
                        def _on(_):
                            led_states[idx] = not led_states[idx]
                            update_all_led_buttons()
                            update_scene()
                            update_ui_visibility()
                            populate_inspector(selected_owner[0])
                        return _on

                    led_btn.on_click(_make_led(gi))

            else:
                _inspector_add(server.gui.add_markdown(f"Unknown selection: `{kind}`"))

    def _select_panel_real(owner):
        if designer_mode[0]:
            return
        if selected_owner[0] == owner:
            return
        selected_owner[0] = owner
        populate_inspector(owner)
        update_scene()

    _select_panel_impl[0] = _select_panel_real
    populate_inspector(None)
    
    def update_scene():
        """Redraw the scene based on current slider values (without intensity map)."""
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles, vio_fov_handles, guide_handles
        if designer_mode[0]:
            # Designer owns the 3D view; do not rebuild (would destroy the gizmo).
            return

        # Ray-box intersection helper for update_scene (positions in cm)
        def ray_box_intersection(pos, direction, box):
            center = np.array(box['center'], dtype=float)
            half = np.array(box['half_sizes'], dtype=float)
            rotation = box.get('rotation', None)
            
            # If box has rotation, transform ray to box's local space
            if rotation is not None:
                qw, qx, qy, qz = rotation
                # Convert quaternion to rotation matrix
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                    [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                    [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
                ])
                # Transform ray to local space (inverse rotation)
                R_inv = R.T
                local_pos = R_inv @ (pos - center)
                local_dir = R_inv @ direction
                pos = local_pos
                direction = local_dir
                center = np.array([0.0, 0.0, 0.0])
            
            tmin = -np.inf
            tmax = np.inf
            for k in range(3):
                if abs(direction[k]) < 1e-12:
                    if pos[k] < center[k] - half[k] or pos[k] > center[k] + half[k]:
                        return None
                else:
                    t1 = (center[k] - half[k] - pos[k]) / direction[k]
                    t2 = (center[k] + half[k] - pos[k]) / direction[k]
                    t_near = min(t1, t2)
                    t_far = max(t1, t2)
                    tmin = max(tmin, t_near)
                    tmax = min(tmax, t_far)
                    if tmin > tmax:
                       return None
            if tmax < 0:
                return None
            return tmin if tmin > 0 else (tmax if tmax > 0 else None)

        # Clear previous objects (safely ignore already-removed handles)
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles.clear()
        ray_handles.clear()
        absorber_handles.clear()
        camera_fov_handles.clear()
        vio_fov_handles.clear()
        guide_handles.clear()
        # Get current values (fixed angles: front=0°, side=90°)
        front_angle = 0.0  # Fixed front angle
        side_angle = 90.0  # Fixed side angle
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        wall_dist = wall_dist_slider.value
        circle_center_x = circle_center_slider.value
        ray_length = ray_length_slider.value

        # Create LEDs
        # Read per-group rotation slider values
        rotations = [
            rot_front_pos.value,
            rot_front_neg.value,
            rot_side_pos.value,
            rot_side_neg.value,
        ]
        
        rotations_y = [
            rot_y_front_pos.value,
            rot_y_front_neg.value,
            rot_y_side_pos.value,
            rot_y_side_neg.value,
        ]
        
        offsets = [
            (offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value),
            (offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value),
            (offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value),
            (offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value),
        ]

        # Build custom groups configs list
        custom_groups_configs = []
        for group in custom_groups:
            config = {
                'enabled': group['enable'].value,
                'position': (group['pos_x'].value, group['pos_y'].value, group['pos_z'].value),
                'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                'led_states': group['led_states'],
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                translated_positions, rotated_directions, rotated_row_dirs = _dynamic_group_world_geometry(group)
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_sizes'] = group.get('led_sizes', [])
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                config['led_lumens'] = group.get('led_lumens', [])
                if rotated_row_dirs:
                    config['led_row_directions'] = rotated_row_dirs
            # Pass lumens override for custom group
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
            if group.get('panel_slot') is not None:
                config['owner'] = ('slot', group['panel_slot'])
            else:
                config['owner'] = ('custom_group', group['id'])
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),  # Default to True if not set
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
                'beam_tilt': led['beam_tilt'].value,
            }
            # Pass lumens override for individual LED
            if led.get('lumens_override') and led['lumens_override'].value:
                config['lumens_override'] = float(led['lumens_value'].value)
            else:
                config['lumens_override'] = None
            # Pass external lens settings
            if led.get('ext_lens_enable') and led['ext_lens_enable'].value:
                config['ext_lens_angle'] = float(led['ext_lens_angle'].value)
                config['ext_lens_efficiency'] = float(led['ext_lens_efficiency'].value) / 100.0
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and led in _pdata.get('individual_leds', []):
                    config['owner'] = ('slot', _si)
                    break
            if 'owner' not in config and led.get('panel_slot') is not None:
                config['owner'] = ('slot', led['panel_slot'])
            individual_leds_configs.append(config)
        
        # Inject the live XZ-mirrored copy of the mirror-primary panel (if any)
        _expand_mirror_configs(custom_groups_configs, individual_leds_configs)

        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            default_lumens=float(led_lumens_slider.value),
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
        # ── Apply global Z rotation to all LEDs ──
        global_rot_z_deg = global_rotation_z_slider.value
        if abs(global_rot_z_deg) > 0.01:
            g_rad = np.radians(global_rot_z_deg)
            cg, sg = np.cos(g_rad), np.sin(g_rad)
            Rg = np.array([[cg, -sg, 0],
                           [sg,  cg, 0],
                           [0,   0,  1]], dtype=float)
            for led in leds:
                led.position = Rg @ led.position
                led.direction = Rg @ led.direction
                if hasattr(led, 'row_direction') and led.row_direction is not None:
                    led.row_direction = Rg @ np.asarray(led.row_direction)
                if hasattr(led, 'square_normal') and led.square_normal is not None:
                    led.square_normal = Rg @ np.asarray(led.square_normal)

        # ── Apply global position offset to all LEDs ──
        _gp_x = global_pos_x_slider.value
        _gp_y = global_pos_y_slider.value
        _gp_z = global_pos_z_slider.value
        if abs(_gp_x) > 0.001 or abs(_gp_y) > 0.001 or abs(_gp_z) > 0.001:
            _gp_offset = np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            for led in leds:
                led.position = led.position + _gp_offset

        # ── Apply diffuser lens effect ──
        # A diffuser lens scatters light, widening the viewing angle toward
        # a near-Lambertian distribution.  The transmission loss is applied
        # separately in the simulation functions (lumens_per_led *= transmission).
        if diffuser_enable_chk.value:
            diff_angle = float(diffuser_angle_slider.value)
            for led in leds:
                # Widen to diffuser output angle (only if wider than native)
                led.viewing_angle = max(led.viewing_angle, diff_angle)

        # Save LEDs for reuse in room intensity calculation (in place: ui.room_mode holds a reference)
        current_leds[:] = leds

        # Rest-pose construction circles + axis for the selected anchored panel
        def _globalize_cm(p):
            v = np.asarray(p, dtype=float).reshape(3)
            if abs(global_rot_z_deg) > 0.01:
                g_rad = np.radians(global_rot_z_deg)
                cg, sg = np.cos(g_rad), np.sin(g_rad)
                v = np.array([cg * v[0] - sg * v[1], sg * v[0] + cg * v[1], v[2]])
            v = v + np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            return v

        def _globalize_dir(d):
            v = np.asarray(d, dtype=float).reshape(3)
            if abs(global_rot_z_deg) > 0.01:
                g_rad = np.radians(global_rot_z_deg)
                cg, sg = np.cos(g_rad), np.sin(g_rad)
                v = np.array([cg * v[0] - sg * v[1], sg * v[0] + cg * v[1], v[2]])
            return v

        draw_groups = []
        owner = selected_owner[0]
        if owner is not None:
            draw_groups = list(_owner_groups(owner))
        for gi, group in enumerate(draw_groups):
            if not _guide_is_enabled(group):
                continue
            guide = group['guide']
            origin = _as_vec3(guide.get('origin', (0, 0, 0)))
            axis = _as_vec3(guide.get('axis', (0, 0, 1)))
            an = np.linalg.norm(axis)
            if an < 1e-12:
                continue
            axis = axis / an
            circles = guide.get('circles') or []
            for ci, circ in enumerate(circles):
                segs = _circle_line_segments_m(
                    _globalize_cm(circ.get('center', origin)),
                    float(circ.get('radius', 0.0)),
                    _globalize_dir(circ.get('normal', axis)),
                    n_seg=64,
                )
                if segs.shape[0] == 0:
                    continue
                handle = server.scene.add_line_segments(
                    f"/guides/group_{group.get('id', gi)}/circle_{ci}",
                    points=segs,
                    colors=(0.55, 0.75, 0.95),
                    line_width=1.5,
                )
                guide_handles.append(handle)
            ts = [float(np.dot(_as_vec3(c.get('center', origin)) - origin, axis)) for c in circles] or [0.0]
            t_lo, t_hi = min(ts) - 3.0, max(ts) + 3.0
            if abs(t_hi - t_lo) < 4.0:
                t_lo, t_hi = -8.0, 8.0
            p0 = _globalize_cm(origin + axis * t_lo) / 100.0
            p1 = _globalize_cm(origin + axis * t_hi) / 100.0
            axis_h = server.scene.add_line_segments(
                f"/guides/group_{group.get('id', gi)}/axis",
                points=np.array([[p0, p1]]),
                colors=(1.0, 0.85, 0.2),
                line_width=3.0,
            )
            guide_handles.append(axis_h)

        # Build absorbers
        absorbers = []
        angles_deg = [front_angle, -front_angle, side_angle, -side_angle]
        for i, angle_deg in enumerate(angles_deg):
            if i not in (0, 1):
                continue
            angle_rad = np.radians(angle_deg)
            gx = circle_center_x + radius * np.cos(angle_rad)
            gy = radius * np.sin(angle_rad)
            y_offset = 6.5 if i == 0 else -6.5
            gy = gy + y_offset
            
            radial = np.array((gx - circle_center_x, gy, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)
            
            base_abs_cx = gx + radial_unit[0] * 5.0 - 5.0
            y_base_offset = -4.2 if i == 0 else 4.2
            base_abs_cy = gy + radial_unit[1] * 5.0 + y_base_offset
            base_abs_cz = 0.0
            
            if not absorbers_enable.value:
                continue
            if i == 0:
                abs_cx = base_abs_cx + abs0_off_x.value
                abs_cy = base_abs_cy + abs0_off_y.value
                abs_cz = base_abs_cz + abs0_off_z.value
            else:
                abs_cx = base_abs_cx + abs1_off_x.value
                abs_cy = base_abs_cy + abs1_off_y.value
                abs_cz = base_abs_cz + abs1_off_z.value
            
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': None,
            })
        
        # Add abs2 and abs3 at origin with offsets
        if absorbers_enable.value:
            # Abs2 with rotation
            abs_cx = 0.0 + abs2_off_x.value
            abs_cy = 0.0 + abs2_off_y.value
            abs_cz = 0.0 + abs2_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs2_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
            
            # Abs3 with rotation
            abs_cx = 0.0 + abs3_off_x.value
            abs_cy = 0.0 + abs3_off_y.value
            abs_cz = 0.0 + abs3_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs3_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })

        # ── Apply global Z rotation to absorbers ──
        if abs(global_rot_z_deg) > 0.01:
            for a in absorbers:
                cx, cy, cz = a['center']
                new_cx = cg * cx - sg * cy
                new_cy = sg * cx + cg * cy
                a['center'] = (new_cx, new_cy, cz)
                # Compose global rotation with existing quaternion rotation
                if a.get('rotation') is not None:
                    qw0, qx0, qy0, qz0 = a['rotation']
                    # Quaternion for Rg around Z: (cos(a/2), 0, 0, sin(a/2))
                    half = g_rad / 2.0
                    gqw, gqx, gqy, gqz = np.cos(half), 0.0, 0.0, np.sin(half)
                    # q_new = q_global * q_existing
                    nw = gqw*qw0 - gqx*qx0 - gqy*qy0 - gqz*qz0
                    nx = gqw*qx0 + gqx*qw0 + gqy*qz0 - gqz*qy0
                    ny = gqw*qy0 - gqx*qz0 + gqy*qw0 + gqz*qx0
                    nz = gqw*qz0 + gqx*qy0 - gqy*qx0 + gqz*qw0
                    a['rotation'] = (nw, nx, ny, nz)
                else:
                    half = g_rad / 2.0
                    a['rotation'] = (np.cos(half), 0.0, 0.0, np.sin(half))

        # Draw absorber boxes (red) in the scene
        for idx, a in enumerate(absorbers):
            cx, cy, cz = a['center']
            hx, hy, hz = a['half_sizes']
            rot = a.get('rotation', None)
            # Viser add_box dimensions are in meters (x,y,z)
            dims = ((hx * 2) / 100.0, (hy * 2) / 100.0, (hz * 2) / 100.0)
            pos_m = (cx / 100.0, cy / 100.0, cz / 100.0)
            if rot is not None:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                    wxyz=rot,
                )
            else:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                )
            absorber_handles.append(handle)

        # Draw LEDs as squares with center source (if enabled)
        if show_led_markers.value:
            for i, led in enumerate(leds):
                led_idx = getattr(led, 'led_index', i)
                led_enabled = not (hasattr(led, 'enabled') and not led.enabled)
                
                # Build local coordinate system for LED
                # Use square_normal (original direction) for mesh if beam_tilt is applied
                square_dir = getattr(led, 'mesh_normal', led.direction)
                z_axis = square_dir / np.linalg.norm(square_dir)
                
                # Use row_direction as reference for consistent square orientation across all rows
                row_dir = getattr(led, 'row_direction', None)
                if row_dir is not None:
                    # y_axis aligned with row direction (direction along the row of LEDs)
                    y_axis = row_dir / np.linalg.norm(row_dir)
                    # Make y_axis perpendicular to z_axis (Gram-Schmidt)
                    y_axis = y_axis - z_axis * np.dot(y_axis, z_axis)
                    if np.linalg.norm(y_axis) < 0.01:  # Nearly parallel, use fallback
                        if abs(z_axis[2]) < 0.9:
                            x_axis = np.cross(z_axis, [0, 0, 1])
                        else:
                            x_axis = np.cross(z_axis, [0, 1, 0])
                        x_axis = x_axis / np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                    else:
                        y_axis = y_axis / np.linalg.norm(y_axis)
                        x_axis = np.cross(y_axis, z_axis)
                else:
                    # Fallback for backwards compatibility
                    if abs(z_axis[2]) < 0.9:
                        x_axis = np.cross(z_axis, [0, 0, 1])
                    else:
                        x_axis = np.cross(z_axis, [0, 1, 0])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                
                # Create rotation matrix from local axes to world axes
                # Square default orientation: thin in X, extends in Y and Z
                # We want: thin along z_axis (LED direction), extends along x_axis and y_axis
                # Rotation matrix: columns are the target axes in world coordinates
                rot_matrix = np.column_stack([z_axis, x_axis, y_axis])
                
                # Convert rotation matrix to quaternion (wxyz format)
                # Using Shepperd's method for numerical stability
                trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
                    y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
                    z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
                else:
                    if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
                        w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                        x = 0.25 * s
                        y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                        z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                    elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
                        w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                        x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                        y = 0.25 * s
                        z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
                        w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                        x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                        y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                        z = 0.25 * s
                quat_wxyz = np.array([w, x, y, z])
                
                # Square size from LED width (in cm, convert to meters)
                square_size = led.width / 100.0  # LED width in cm converted to meters
                square_thickness = 0.0002  # Very thin (0.2mm)
                dims = (square_thickness, square_size, square_size)
                
                # White color: bright if enabled, dim if disabled; cyan if this panel is selected
                square_color = (1.0, 1.0, 1.0) if led_enabled else (0.3, 0.3, 0.3)
                _owner = getattr(led, 'owner', None)
                if selected_owner[0] is not None and _owner == selected_owner[0]:
                    square_color = (0.15, 1.0, 1.0) if led_enabled else (0.08, 0.45, 0.45)
                
                # Draw square base with rotation
                handle = server.scene.add_box(
                    f"/leds/led_{led_idx}_base",
                    dimensions=dims,
                    color=square_color,
                    position=tuple(led.position / 100.0),  # Convert cm to m for viser
                    wxyz=tuple(quat_wxyz),
                )
                led_handles.append(handle)
                if _owner is not None:
                    def _make_owner_click(own):
                        def _on_click(_event):
                            _just_clicked_mesh[0] = True
                            select_panel(own)
                        return _on_click
                    handle.on_click(_make_owner_click(_owner))
                
                # Draw small center source sphere (only if enabled)
                if led_enabled:
                    handle = server.scene.add_icosphere(
                        f"/leds/led_{led_idx}_source",
                        radius=0.001,  # Very small 1mm source
                        color=led.color,
                        position=tuple(led.position / 100.0),
                    )
                    led_handles.append(handle)
                    if _owner is not None:
                        handle.on_click(_make_owner_click(_owner))

        # Prepare STL mesh data for ray tracing visualization if enabled
        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_ref = stl_mesh_data[0]
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            # Apply global Z rotation to STL transform
            if abs(global_rot_z_deg) > 0.01:
                T_global = np.eye(4)
                T_global[:3, :3] = Rg
                transform = T_global @ transform
            stl_mesh_for_raytracing = {
                'vertices': mesh_ref.vertices,
                'faces': mesh_ref.faces,
                'transform': transform
            }

        # Draw rays (toggleable)
        if show_rays_output.value:
            for i, led in enumerate(leds):
                if hasattr(led, 'enabled') and not led.enabled:
                    continue
                vis_rays = led.get_visualization_rays(ray_length)

                for j, (pos, direction) in enumerate(vis_rays):
                    # Calculate end point, clipping at absorbers and wall
                    # Rays have infinite length until they hit something
                    
                    # Check absorbers first via box intersection
                    t_abs_min = None
                    if absorbers is not None:
                        for a in absorbers:
                            t_hit = ray_box_intersection(pos, direction, a)
                            if t_hit is not None and t_hit > 0:
                                if t_abs_min is None or t_hit < t_abs_min:
                                    t_abs_min = t_hit
                    
                    # Check STL mesh intersection
                    if stl_mesh_for_raytracing is not None:
                        t_stl = _ray_mesh_intersection(pos, direction, stl_mesh_for_raytracing)
                        if t_stl is not None and t_stl > 0:
                            if t_abs_min is None or t_stl < t_abs_min:
                                t_abs_min = t_stl

                    # Clip at wall(s) - if room mode, check all 5 walls
                    t_wall = None
                    if room_mode_enable.value:
                        # Check all 5 room walls
                        front_dist = room_front_dist.value
                        side_dist = room_side_dist.value
                        top_bottom_dist = room_top_bottom_dist.value
                        
                        wall_intersections = []
                        # Front wall
                        if direction[0] != 0:
                            t = (front_dist - pos[0]) / direction[0]
                            if t > 0:
                                wall_intersections.append(t)
                        # Left wall
                        if direction[1] != 0:
                            t = (-side_dist - pos[1]) / direction[1]
                            if t > 0:
                                wall_intersections.append(t)
                        # Right wall
                        if direction[1] != 0:
                            t = (side_dist - pos[1]) / direction[1]
                            if t > 0:
                                wall_intersections.append(t)
                        # Top wall
                        if direction[2] != 0:
                            t = (top_bottom_dist - pos[2]) / direction[2]
                            if t > 0:
                                wall_intersections.append(t)
                        # Bottom wall
                        if direction[2] != 0:
                            t = (-top_bottom_dist - pos[2]) / direction[2]
                            if t > 0:
                                wall_intersections.append(t)
                        
                        if wall_intersections:
                            t_wall = min(wall_intersections)
                    else:
                        # Single front wall only
                        if direction[0] != 0:
                            t_wall = (wall_dist - pos[0]) / direction[0]

                    # Choose nearest positive intersection (absorber before wall)
                    t_clip = None
                    if t_abs_min is not None and t_abs_min > 0:
                        t_clip = t_abs_min
                    if t_wall is not None and t_wall > 0:
                        if t_clip is None or t_wall < t_clip:
                            t_clip = t_wall

                    # Use intersection point, or very far if no intersection
                    if t_clip is not None:
                        end = pos + direction * t_clip
                    else:
                        end = pos + direction * 1000.0  # 10 meters if no intersection

                    # Draw line (positions in meters)
                    points = np.array([pos / 100.0, end / 100.0])
                    led_idx = getattr(led, 'led_index', i)
                    handle = server.scene.add_line_segments(
                        f"/rays/led_{led_idx}/ray_{j}",
                        points=points.reshape(1, 2, 3),
                        colors=led.color,  # Single color tuple
                        line_width=2.0,
                    )
                    ray_handles.append(handle)

                # Add random rays if enabled
                if show_random_rays.value:
                    led_idx = getattr(led, 'led_index', i)
                    np.random.seed(42 + led_idx)  # Consistent random rays
                    num_random_rays = 50  # Fixed number of visualization rays
                    for k in range(num_random_rays):
                        # Random direction within viewing cone using cosine power distribution
                        u1, u2 = np.random.uniform(0, 1, 2)
                        
                        uniformity = float(ray_uniformity_slider.value)
                        n = _get_effective_n(led, uniformity)
                        
                        _vis_angle = getattr(led, 'ext_lens_angle', None) or led.viewing_angle
                        max_theta = np.radians(_vis_angle / 2.0)
                        cos_max = np.cos(max_theta)
                        cos_theta_sampled = 1.0 - u1 * (1.0 - cos_max)
                        cos_theta_sampled = np.clip(cos_theta_sampled, -1.0, 1.0)
                        theta = np.arccos(cos_theta_sampled)
                        phi = 2 * np.pi * u2

                        z_axis = led.direction
                        if abs(z_axis[2]) < 0.9:
                            x_axis = np.cross(z_axis, [0, 0, 1])
                        else:
                            x_axis = np.cross(z_axis, [0, 1, 0])
                        x_axis = x_axis / np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)

                        local_dir = np.array(
                            [
                                np.sin(theta) * np.cos(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(theta),
                            ]
                        )
                        world_dir = (
                            local_dir[0] * x_axis
                            + local_dir[1] * y_axis
                            + local_dir[2] * z_axis
                        )
                        world_dir = world_dir / np.linalg.norm(world_dir)

                        # Compute nearest intersection with absorbers or wall
                        # Rays have infinite length until they hit something
                        t_abs_min = None
                        if absorbers is not None:
                            for a in absorbers:
                                t_hit = ray_box_intersection(led.position, world_dir, a)
                                if t_hit is not None and t_hit > 0:
                                    if t_abs_min is None or t_hit < t_abs_min:
                                        t_abs_min = t_hit
                        
                        # Check STL mesh intersection
                        if stl_mesh_for_raytracing is not None:
                            t_stl = _ray_mesh_intersection(led.position, world_dir, stl_mesh_for_raytracing)
                            if t_stl is not None and t_stl > 0:
                                if t_abs_min is None or t_stl < t_abs_min:
                                    t_abs_min = t_stl

                        t_wall = None
                        if room_mode_enable.value:
                            # Check all 5 room walls
                            front_dist = room_front_dist.value
                            side_dist = room_side_dist.value
                            top_bottom_dist = room_top_bottom_dist.value
                            
                            wall_intersections = []
                            # Front wall
                            if world_dir[0] != 0:
                                t = (front_dist - led.position[0]) / world_dir[0]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Left wall
                            if world_dir[1] != 0:
                                t = (-side_dist - led.position[1]) / world_dir[1]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Right wall
                            if world_dir[1] != 0:
                                t = (side_dist - led.position[1]) / world_dir[1]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Top wall
                            if world_dir[2] != 0:
                                t = (top_bottom_dist - led.position[2]) / world_dir[2]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Bottom wall
                            if world_dir[2] != 0:
                                t = (-top_bottom_dist - led.position[2]) / world_dir[2]
                                if t > 0:
                                    wall_intersections.append(t)
                            
                            if wall_intersections:
                                t_wall = min(wall_intersections)
                        else:
                            # Single front wall only
                            if world_dir[0] != 0:
                                t_wall = (wall_dist - led.position[0]) / world_dir[0]

                        t_clip = None
                        if t_abs_min is not None and t_abs_min > 0:
                            t_clip = t_abs_min
                        if t_wall is not None and t_wall > 0:
                            if t_clip is None or t_wall < t_clip:
                                t_clip = t_wall

                        # Use intersection point, or very far if no intersection
                        if t_clip is not None:
                            end = led.position + world_dir * t_clip
                        else:
                            end = led.position + world_dir * 1000.0  # 10 meters if no intersection

                        points = np.array([led.position / 100.0, end / 100.0])
                        # Dimmer color for random rays
                        dim_color = (
                            led.color[0] * 0.5,
                            led.color[1] * 0.5,
                            led.color[2] * 0.5,
                        )
                        handle = server.scene.add_line_segments(
                            f"/rays/led_{led_idx}/random_{k}",
                            points=points.reshape(1, 2, 3),
                            colors=dim_color,
                            line_width=1.0,
                        )
                        ray_handles.append(handle)

        # Draw camera FOV rectangle on wall
        if show_camera_fov.value:
            # Use correct wall distance based on mode
            if room_mode_enable.value:
                wall_dist = room_front_dist.value
            else:
                wall_dist = wall_dist_slider.value
            
            cam_x = camera_pos_x.value
            cam_y = camera_pos_y.value

            # Footprint of the (possibly pitched) camera FOV on the wall.
            # With pitch != 0 this is a trapezoid, not a rectangle.
            z_bot_cm, z_top_cm, w_bot_cm, w_top_cm = _camera_fov_wall_trapezoid(
                wall_dist - cam_x, camera_pitch.value,
                camera_fov_h.value, camera_fov_v.value,
            )
            z_bot = z_bot_cm / 100.0  # metres
            z_top = z_top_cm / 100.0
            w_bot = w_bot_cm / 100.0
            w_top = w_top_cm / 100.0
            y_c = cam_y / 100.0  # camera Y offset shifts the footprint sideways
            wall_x = wall_dist / 100.0 - 0.008  # Slightly in front of wall

            y_bot_l, y_bot_r = y_c - w_bot, y_c + w_bot
            y_top_l, y_top_r = y_c - w_top, y_c + w_top

            # In room mode, clamp the FOV footprint to the front wall boundaries
            if room_mode_enable.value:
                max_half_w = room_side_dist.value / 100.0        # wall half-width in m
                max_half_h = room_top_bottom_dist.value / 100.0  # wall half-height in m
                y_bot_l = float(np.clip(y_bot_l, -max_half_w, max_half_w))
                y_bot_r = float(np.clip(y_bot_r, -max_half_w, max_half_w))
                y_top_l = float(np.clip(y_top_l, -max_half_w, max_half_w))
                y_top_r = float(np.clip(y_top_r, -max_half_w, max_half_w))
                z_bot = float(np.clip(z_bot, -max_half_h, max_half_h))
                z_top = float(np.clip(z_top, -max_half_h, max_half_h))
            
            # Four corner lines
            corners = [
                [[wall_x, y_bot_l, z_bot], [wall_x, y_bot_r, z_bot]],  # Bottom
                [[wall_x, y_bot_r, z_bot], [wall_x, y_top_r, z_top]],  # Right
                [[wall_x, y_top_r, z_top], [wall_x, y_top_l, z_top]],  # Top
                [[wall_x, y_top_l, z_top], [wall_x, y_bot_l, z_bot]],  # Left
            ]
            
            handle = server.scene.add_line_segments(
                "/camera/fov_border",
                points=np.array(corners),
                colors=(0.0, 1.0, 0.0),  # Green
                line_width=6.0,  # Thicker lines
            )
            camera_fov_handles.append(handle)

            # Main-camera marker + optical-axis vector (like the VIO cameras)
            cam_pos_cm = np.array([cam_x, cam_y, 0.0], dtype=float)
            cam_axis = vio_optical_axis(camera_pitch.value, 0.0)
            cam_pos_m = cam_pos_cm / 100.0
            cam_axis_end_m = (cam_pos_cm + cam_axis * 8.0) / 100.0
            cam_sph = server.scene.add_icosphere(
                "/camera/marker",
                radius=0.006,
                color=(0.0, 1.0, 0.0),
                position=tuple(cam_pos_m + cam_axis * 0.008),
            )
            camera_fov_handles.append(cam_sph)
            cam_axis_h = server.scene.add_line_segments(
                "/camera/axis",
                points=np.array([[cam_pos_m, cam_axis_end_m]]),
                colors=(0.0, 1.0, 0.0),
                line_width=4.0,
            )
            camera_fov_handles.append(cam_axis_h)

        # Draw VIO fisheye FOV footprints on the wall(s).
        # In room mode the footprint is projected on ALL room walls (front,
        # left, right, top, bottom, and back if shown), not just the front one.
        if show_vio_fov.value:
            # Wall patches: (axis, plane_coord_cm, u_min, u_max, v_min, v_max, inward_sign)
            # axis 0=x, 1=y, 2=z; (u, v) are the two remaining axes in
            # ascending order; inward_sign offsets the overlay into the room.
            if room_mode_enable.value:
                _fd = room_front_dist.value
                _sd = room_side_dist.value
                _td = room_top_bottom_dist.value
                # Same extent as draw_room_walls: 2.5x depth behind the front wall
                _x_back_edge = _fd - (_fd - circle_center_slider.value) * 2.5
                if show_back_wall.value:
                    _x_back_edge = max(_x_back_edge, -room_back_dist.value)
                vio_walls = [
                    (0, _fd, -_sd, _sd, -_td, _td, -1.0),   # front
                    (1, -_sd, _x_back_edge, _fd, -_td, _td, +1.0),  # left
                    (1, _sd, _x_back_edge, _fd, -_td, _td, -1.0),   # right
                    (2, _td, _x_back_edge, _fd, -_sd, _sd, -1.0),   # top
                    (2, -_td, _x_back_edge, _fd, -_sd, _sd, +1.0),  # bottom
                ]
                if show_back_wall.value:
                    vio_walls.append((0, -room_back_dist.value, -_sd, _sd, -_td, _td, +1.0))
            else:
                _half = wall_view_size.value / 2.0
                vio_walls = [(0, wall_dist_slider.value, -_half, _half, -_half, _half, -1.0)]

            vio_hfov, vio_vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
            cam_pos = np.array([vio_pos_x.value, vio_pos_y.value, vio_pos_z.value], dtype=float)
            axis_len_cm = 8.0
            marker_specs = [
                (1, vio_cam1_pitch.value, vio_cam1_yaw.value, (1.0, 0.0, 1.0), 0.008),  # magenta
                (2, vio_cam2_pitch.value, vio_cam2_yaw.value, (0.0, 1.0, 1.0), 0.012),  # cyan
            ]
            for cam_id, pitch, yaw, color, inset_m in marker_specs:
                all_verts, all_faces, all_segs = [], [], []
                n_verts = 0
                for w_axis, plane_cm, u_min, u_max, v_min, v_max, inward in vio_walls:
                    # Skip walls the camera is not on the interior side of
                    if (cam_pos[w_axis] - plane_cm) * inward <= 1e-6:
                        continue
                    mask, us, vs = rasterize_fisheye_fov_on_plane(
                        cam_pos, pitch, yaw, vio_hfov, vio_vfov,
                        w_axis, plane_cm, u_min, u_max, v_min, v_max, n_grid=90,
                    )
                    plane_m = plane_cm / 100.0 + inward * inset_m
                    verts, faces, segs = fov_plane_mask_to_quads_and_contour(
                        mask, us, vs, w_axis, plane_m
                    )
                    if len(faces) > 0:
                        all_verts.append(verts)
                        all_faces.append(faces + np.uint32(n_verts))
                        n_verts += len(verts)
                    if len(segs) > 0:
                        all_segs.append(segs)
                if vio_fill_fov.value and all_faces:
                    fill = server.scene.add_mesh_simple(
                        name=f"/vio_cam{cam_id}/fov_fill",
                        vertices=np.concatenate(all_verts, axis=0),
                        faces=np.concatenate(all_faces, axis=0),
                        color=tuple(int(c * 255) for c in color),
                        opacity=0.28,
                        flat_shading=True,
                        side="double",
                    )
                    vio_fov_handles.append(fill)
                if all_segs:
                    handle = server.scene.add_line_segments(
                        f"/vio_cam{cam_id}/fov_border",
                        points=np.concatenate(all_segs, axis=0),
                        colors=color,
                        line_width=4.0,
                    )
                    vio_fov_handles.append(handle)

                axis = vio_optical_axis(pitch, yaw)
                pos_m = cam_pos / 100.0
                axis_end_m = (cam_pos + axis * axis_len_cm) / 100.0
                marker_pos = pos_m + axis * 0.008
                sph = server.scene.add_icosphere(
                    f"/vio_cam{cam_id}/marker",
                    radius=0.006,
                    color=color,
                    position=tuple(marker_pos),
                )
                vio_fov_handles.append(sph)
                axis_h = server.scene.add_line_segments(
                    f"/vio_cam{cam_id}/axis",
                    points=np.array([[pos_m, axis_end_m]]),
                    colors=color,
                    line_width=4.0,
                )
                vio_fov_handles.append(axis_h)

    @server.scene.on_click()
    def _on_scene_background_click(_event):
        if designer_mode[0]:
            _just_clicked_mesh[0] = False
            return
        if _just_clicked_mesh[0]:
            _just_clicked_mesh[0] = False
            return
        if read_cell_at_ray(_event.ray_origin, _event.ray_direction):
            return
        if selected_owner[0] is not None:
            select_panel(None)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_dist_init = wall_dist_slider.value
    wall_size_init = wall_view_size.value / 100.0  # cm -> m
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, wall_size_init, wall_size_init),
        color=(0.5, 0.5, 0.5),
        position=(wall_dist_init / 100.0, 0.0, 0.0),
    )

    # Grid on XY plane (millimeter resolution)
    grid_points = []
    for i in range(-10, 11):
        grid_points.append([[-1.0, i * 0.01, 0], [1.0, i * 0.01, 0]])  # 1mm spacing
        grid_points.append([[i * 0.01, -1.0, 0], [i * 0.01, 1.0, 0]])  # 1mm spacing

    static_scene_handles.append(server.scene.add_line_segments(
        "/grid",
        points=np.array(grid_points),
        colors=(0.3, 0.3, 0.3),  # Single color for all segments
        line_width=1.0,
    ))

    # Origin axes
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/x",
        points=np.array([[[0, 0, 0], [0.5, 0, 0]]]),
        colors=(1.0, 0.0, 0.0),
        line_width=3.0,
    ))
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/y",
        points=np.array([[[0, 0, 0], [0, 0.5, 0]]]),
        colors=(0.0, 1.0, 0.0),
        line_width=3.0,
    ))
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/z",
        points=np.array([[[0, 0, 0], [0, 0, 0.5]]]),
        colors=(0.0, 0.0, 1.0),
        line_width=3.0,
    ))

    # Callback to update wall position and size
    def update_wall():
        nonlocal wall_handle
        if room_mode_enable.value:
            # In room mode, don't update main wall
            return
        wall_dist = wall_dist_slider.value
        wall_size_m = wall_view_size.value / 100.0  # cm -> m
        try:
            wall_handle.remove()
        except (AttributeError, KeyError):
            pass
        wall_handle = server.scene.add_box(
            "/wall",
            dimensions=(0.01, wall_size_m, wall_size_m),
            color=(0.5, 0.5, 0.5),
            position=(wall_dist / 100.0, 0.0, 0.0),
        )

    # Function to update cell area info
    def update_cell_area_info():
        grid_size = int(intensity_grid_size.value)
        wall_size_cm = int(wall_view_size.value)
        cell_size_cm = wall_size_cm / grid_size
        cell_area_cm2 = cell_size_cm * cell_size_cm
        cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
        
        cell_area_html.content = (
            f"<div style='font-family: sans-serif; font-size: 11px; color: #666; margin-top: -8px; margin-bottom: 8px;'>"
            f"Cell: {cell_size_cm:.2f} cm × {cell_size_cm:.2f} cm = {cell_area_cm2:.2f} cm² ({cell_area_m2:.6f} m²)"
            "</div>"
        )
    
    # Initial cell area update
    update_cell_area_info()

    # Register callbacks
    viewing_angle_slider.on_update(lambda _: update_scene())
    diffuser_enable_chk.on_update(lambda _: update_scene())
    diffuser_angle_slider.on_update(lambda _: update_scene())
    rot_front_pos.on_update(lambda _: update_scene())
    rot_front_neg.on_update(lambda _: update_scene())
    rot_side_pos.on_update(lambda _: update_scene())
    rot_side_neg.on_update(lambda _: update_scene())
    rot_y_front_pos.on_update(lambda _: update_scene())
    rot_y_front_neg.on_update(lambda _: update_scene())
    rot_y_side_pos.on_update(lambda _: update_scene())
    rot_y_side_neg.on_update(lambda _: update_scene())
    # Group position offset callbacks
    offset_front_pos_x.on_update(lambda _: update_scene())
    offset_front_pos_y.on_update(lambda _: update_scene())
    offset_front_pos_z.on_update(lambda _: update_scene())
    offset_front_neg_x.on_update(lambda _: update_scene())
    offset_front_neg_y.on_update(lambda _: update_scene())
    offset_front_neg_z.on_update(lambda _: update_scene())
    offset_side_pos_x.on_update(lambda _: update_scene())
    offset_side_pos_y.on_update(lambda _: update_scene())
    offset_side_pos_z.on_update(lambda _: update_scene())
    offset_side_neg_x.on_update(lambda _: update_scene())
    offset_side_neg_y.on_update(lambda _: update_scene())
    offset_side_neg_z.on_update(lambda _: update_scene())
    radius_slider.on_update(lambda _: update_scene())
    circle_center_slider.on_update(lambda _: update_scene())
    def _on_global_rotation_change(_):
        """Handle global rotation slider: update LEDs immediately, debounce mesh update."""
        update_scene()
        # Schedule mesh update with debounce to avoid recomputing on every tick
        import threading
        if hasattr(_on_global_rotation_change, '_timer') and _on_global_rotation_change._timer is not None:
            _on_global_rotation_change._timer.cancel()
        def _deferred_mesh():
            try:
                update_stl_mesh(skip_lighting=True)
            except Exception:
                pass
        _on_global_rotation_change._timer = threading.Timer(0.3, _deferred_mesh)
        _on_global_rotation_change._timer.start()
    _on_global_rotation_change._timer = None
    global_rotation_z_slider.on_update(_on_global_rotation_change)
    global_pos_x_slider.on_update(_on_global_rotation_change)
    global_pos_y_slider.on_update(_on_global_rotation_change)
    global_pos_z_slider.on_update(_on_global_rotation_change)
    ray_length_slider.on_update(lambda _: update_scene())
    led_lumens_slider.on_update(lambda _: None)  # No auto-update, use manual button
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: None if _mode_toggle_syncing[0] else update_intensity_map())
    row1_chk.on_update(lambda _: update_scene())
    row2_chk.on_update(lambda _: update_scene())
    row3_chk.on_update(lambda _: update_scene())
    row4_chk.on_update(lambda _: update_scene())
    absorbers_enable.on_update(lambda _: (update_scene(), update_ui_visibility()))
    show_camera_fov.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_h.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_v.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_y.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    show_vio_fov.on_update(lambda _: update_scene())
    vio_fill_fov.on_update(lambda _: update_scene())
    vio_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_pos_y.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_pos_z.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam1_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam1_yaw.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam2_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam2_yaw.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_long_fov.on_update(lambda _: (_refresh_vio_fov_label(), update_scene(), _refresh_uniformity()))
    vio_landscape.on_update(lambda _: (_refresh_vio_fov_label(), update_scene(), _refresh_uniformity()))
    abs0_off_x.on_update(lambda _: update_scene())
    abs0_off_y.on_update(lambda _: update_scene())
    abs0_off_z.on_update(lambda _: update_scene())
    abs1_off_x.on_update(lambda _: update_scene())
    abs1_off_y.on_update(lambda _: update_scene())
    abs1_off_z.on_update(lambda _: update_scene())
    abs2_off_x.on_update(lambda _: update_scene())
    abs2_off_y.on_update(lambda _: update_scene())
    abs2_off_z.on_update(lambda _: update_scene())
    abs2_rot_z.on_update(lambda _: update_scene())
    abs3_off_x.on_update(lambda _: update_scene())
    abs3_off_y.on_update(lambda _: update_scene())
    abs3_off_z.on_update(lambda _: update_scene())
    abs3_rot_z.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: None)  # No auto-update - manual button only
    ray_uniformity_slider.on_update(lambda _: None)  # No auto-update for expensive params
    intensity_threshold_slider.on_update(lambda _: _refresh_uniformity())
    uniformity_percentile_slider.on_update(lambda _: _refresh_uniformity())
    vio_occupancy_lux.on_update(lambda _: _refresh_uniformity())
    intensity_grid_size.on_update(lambda _: update_cell_area_info())  # Update cell area when resolution changes
    wall_view_size.on_update(lambda _: update_cell_area_info())  # Update cell area when wall size changes
    
    # Room mode callback - draw/clear room walls when toggled
    def on_room_mode_toggle(_):
        nonlocal wall_handle
        if room_mode_enable.value:
            # Hide main wall and show room walls
            try:
                wall_handle.remove()
            except (KeyError, AttributeError):
                pass
            for handle in intensity_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            intensity_handles.clear()
            _mode_toggle_syncing[0] = True
            try:
                show_intensity_map.value = False
            finally:
                _mode_toggle_syncing[0] = False
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Room Mode is active. Enable 'Show Room Intensity' and click 'Update Room Intensity'.</div>"
                "</div>"
            )
            draw_room_walls()
        else:
            # Clear room intensity handles
            for handle in room_intensity_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            room_intensity_handles.clear()
            # Clear room wall handles
            for handle in room_wall_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            room_wall_handles.clear()
            _mode_toggle_syncing[0] = True
            try:
                show_room_intensity.value = False
            finally:
                _mode_toggle_syncing[0] = False
            _last_room_cache['grids'] = None
            # Restore main wall
            wall_dist = wall_dist_slider.value
            wall_size_m = wall_view_size.value / 100.0
            wall_handle = server.scene.add_box(
                "/wall",
                dimensions=(0.01, wall_size_m, wall_size_m),
                color=(0.5, 0.5, 0.5),
                position=(wall_dist / 100.0, 0.0, 0.0),
            )
            if _last_intensity_cache['grid'] is not None:
                _refresh_uniformity()
            else:
                legend_html.content = (
                    "<div style='font-family: sans-serif;'>"
                    "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                    "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click 'Update Intensity Map' to see the legend</div>"
                    "</div>"
                )
    
    room_mode_enable.on_update(on_room_mode_toggle)
    show_room_walls.on_update(lambda _: draw_room_walls())
    show_room_intensity.on_update(
        lambda _: None if _mode_toggle_syncing[0] else (
            (update_room_intensity_map() if (room_mode_enable.value and show_room_intensity.value) else draw_room_walls())
            if room_mode_enable.value else None
        )
    )
    room_front_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_side_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_top_bottom_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    show_back_wall.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_back_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    wall_view_size.on_update(lambda _: (update_wall(), update_cell_area_info(), update_scene()))  # Update wall size, cell area, VIO FOV clip

    def on_wall_dist_change(_):
        """Clear stale intensity map when wall distance changes."""
        # Remove old intensity visualization (values are no longer valid)
        for h in intensity_handles:
            try:
                h.remove()
            except KeyError:
                pass
        intensity_handles.clear()
        # Update legend to inform user that recalculation is needed
        legend_html.content = (
            "<div style='font-family: sans-serif;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
            "<div style='color:#F0AD4E;font-size:12px;'>⚠ Wall distance changed.<br>Click 'Update Intensity Map' to recalculate.</div>"
            "</div>"
        )
        update_wall()
        update_scene()

    wall_dist_slider.on_update(on_wall_dist_change)
    
    # Register LED control button callbacks
    # Group buttons
    for group_idx, btn in group_buttons.items():
        def make_group_handler(g_idx):
            def handler(_):
                start_idx = g_idx * 12
                end_idx = start_idx + 12
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_group_handler(group_idx))
    
    # Row buttons
    for (group_idx, row_idx), btn in row_buttons.items():
        def make_row_handler(g_idx, r_idx):
            def handler(_):
                start_idx = g_idx * 12 + r_idx * 3
                end_idx = start_idx + 3
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} Row {r_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_row_handler(group_idx, row_idx))
    
    # Individual LED buttons
    for led_idx, btn in led_buttons.items():
        def make_led_handler(l_idx):
            def handler(_):
                led_states[l_idx] = not led_states[l_idx]
                print(f"LED {l_idx} toggled to {led_states[l_idx]}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_led_handler(led_idx))
    
    # Button for manual intensity map update
    update_intensity_button.on_click(lambda _: update_intensity_map())
    
    # Button for exporting lux matrix
    export_lux_matrix_button.on_click(lambda _: export_lux_matrix())
    
    # Button for running benchmark (multi-distance)
    run_benchmark_button.on_click(lambda _: run_benchmark())
    
    # Buttons for CSV pattern import
    csv_import_btn.on_click(lambda _: import_csv_pattern())
    csv_clear_btn.on_click(lambda _: clear_csv_pattern())

    # Button for capturing FOV intensity image
    capture_fov_btn.on_click(lambda _: capture_camera_fov_image())
    
    # Button for room intensity map update
    update_room_button.on_click(lambda _: update_room_intensity_map())



    return SimpleNamespace(update_scene=update_scene, update_wall=update_wall)
