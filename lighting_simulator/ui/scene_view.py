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
    rasterize_fisheye_fov_on_plane, rasterize_pinhole_fov_on_plane, vio_hfov_vfov_deg, vio_optical_axis,
)
from lighting_simulator.domain.geometry import as_vec3 as _as_vec3
from lighting_simulator.domain.guides import circle_line_segments_m as _circle_line_segments_m
from lighting_simulator.domain.led import ROLES, led_role
from lighting_simulator.scene.layout import LedSpec, Panel
from lighting_simulator.domain.optics import effective_lambertian_exponent as _get_effective_n
from lighting_simulator.raytracing.mesh import ray_mesh_intersection as _ray_mesh_intersection
from lighting_simulator.simulation.settings import RoomSettings as _RoomSettings
from lighting_simulator.ui.mesh_lighting import _build_stl_transform


def build(ctx):
    state = ctx.state
    populate_inspector = ctx.populate_inspector
    clear_inspector = ctx.clear_inspector
    save_panel_as_template = ctx.save_panel_as_template
    enter_designer_ref = ctx.enter_designer_ref
    _just_clicked_mesh = ctx._just_clicked_mesh
    _last_intensity_cache = ctx._last_intensity_cache
    _last_room_cache = ctx._last_room_cache
    _mode_toggle_syncing = ctx._mode_toggle_syncing
    _refresh_uniformity = ctx._refresh_uniformity
    _refresh_vio_fov_label = ctx._refresh_vio_fov_label
    apply_view_mode = ctx.apply_view_mode
    camera_fov_h = ctx.camera_fov_h
    camera_fov_handles = ctx.camera_fov_handles
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    capture_camera_fov_image = ctx.capture_camera_fov_image
    capture_fov_btn = ctx.capture_fov_btn
    cell_area_html = ctx.cell_area_html
    clear_csv_pattern = ctx.clear_csv_pattern
    csv_clear_btn = ctx.csv_clear_btn
    csv_import_btn = ctx.csv_import_btn
    current_leds = ctx.current_leds
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
    guide_handles = ctx.guide_handles
    import_csv_pattern = ctx.import_csv_pattern
    imported_csv_handles = ctx.imported_csv_handles
    inspector_tab = ctx.inspector_tab
    intensity_grid_size = ctx.intensity_grid_size
    intensity_handles = ctx.intensity_handles
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_threshold_slider = ctx.intensity_threshold_slider
    led_handles = ctx.led_handles
    led_lumens_slider = ctx.led_lumens_slider
    legend_html = ctx.legend_html
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
    run_benchmark = ctx.run_benchmark
    run_benchmark_button = ctx.run_benchmark_button
    server = ctx.server
    show_back_wall = ctx.show_back_wall
    show_camera_fov = ctx.show_camera_fov
    show_intensity_map = ctx.show_intensity_map
    show_led_markers = ctx.show_led_markers
    show_random_rays = ctx.show_random_rays
    show_rays_output = ctx.show_rays_output
    show_room_intensity = ctx.show_room_intensity
    show_room_walls = ctx.show_room_walls
    show_tilt_fovs = ctx.show_tilt_fovs
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
    tilt_fov_deg = ctx.tilt_fov_deg
    uniformity_percentile_slider = ctx.uniformity_percentile_slider
    update_intensity_button = ctx.update_intensity_button
    update_intensity_map = ctx.update_intensity_map
    update_room_button = ctx.update_room_button
    update_room_intensity_map = ctx.update_room_intensity_map
    view_mode_dropdown = ctx.view_mode_dropdown
    flash_lumens_input = ctx.flash_lumens_input
    led_voltage_input = ctx.led_voltage_input
    led_efficacy_input = ctx.led_efficacy_input
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

    # Inspector LED-button behaviour: 'toggle' on/off, or assign a role ('vio' | 'flash' | 'both')
    _inspector_click_action = ['toggle']
    _ROLE_COLORS = {'vio': "#1E90FF", 'flash': "#FF8C00", 'both': "#FFFFFF"}
    _ROLE_LABELS = {'vio': "VIO (continuous)", 'flash': "Flash (pulse only)", 'both': "Both"}
    # 3-D marker colours (0-1 RGB), matching _ROLE_COLORS: lit / idle in this operating mode / off / selected
    _MARKER_LIT = {'vio': (0.12, 0.56, 1.0), 'flash': (1.0, 0.55, 0.0), 'both': (1.0, 1.0, 1.0)}
    _MARKER_IDLE = {'vio': (0.05, 0.2, 0.4), 'flash': (0.45, 0.22, 0.0), 'both': (0.4, 0.4, 0.4)}
    _MARKER_OFF = (0.16, 0.16, 0.16)
    _MARKER_SELECTED, _MARKER_SELECTED_DIM = (1.0, 0.95, 0.1), (0.45, 0.42, 0.05)

    def _clear_inspector():
        clear_inspector()

    def _inspector_add(handle):
        designer_ui_handles.append(handle)
        return handle

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

    def _designer_leds_to_specs(leds):
        """Designer LED dicts (panel-local Euler) -> LedSpec list."""
        specs = []
        for led in leds:
            R = _designer_rotation_matrix(led['rx'], led['ry'], led['rz'])
            specs.append(LedSpec(position=(led['x'], led['y'], led['z']),
                                 direction=R @ np.array([1.0, 0.0, 0.0]), row_dir=R @ np.array([0.0, 1.0, 0.0]),
                                 beam_angle=led['view_angle'], tilt=led.get('tilt', 0.0), size=led['size'],
                                 on=led.get('on', True), role=led.get('role', 'both'),
                                 profile=led.get('profile')))
        return specs

    def _designer_panel():
        """The designed panel at identity pose (for templates / new panels)."""
        state_d = designer_state[0]
        return Panel(name=state_d['name'].strip() or 'Untitled Panel', leds=_designer_leds_to_specs(state_d['leds']),
                     rows=[list(range(len(state_d['leds'])))] if state_d['leds'] else None)

    def _save_designer_template(_event=None):
        panel = _designer_panel()
        save_panel_as_template(panel, panel.name)

    def _save_designer(_event=None):
        state_d = designer_state[0]
        idx = state_d['editing_index']
        if idx is None:
            state.add_panel(_designer_panel())
        else:
            panel = state.panel(idx)
            if panel is not None:
                panel.leds = _designer_leds_to_specs(state_d['leds'])
                new_name = state_d['name'].strip()
                if new_name and new_name != panel.name:
                    panel.name = state.unique_name(new_name)
                if not panel.rows or max((i for r in panel.rows for i in r), default=-1) >= len(panel.leds):
                    panel.rows = [list(range(len(panel.leds)))]
        _exit_designer()
        state.notify("structure")

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
        nonlocal led_handles, ray_handles, camera_fov_handles, vio_fov_handles, guide_handles
        for handle in led_handles + ray_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        led_handles.clear()
        ray_handles.clear()
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
        populate_inspector(state.selected)

    def _build_designer_ui():
        """Rebuild designer controls inside the floating Selected inspector."""
        clear_inspector()
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
                    'size': 0.5, 'view_angle': 120.0, 'tilt': 0.0, 'on': True, 'role': 'both',
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

                role_dd = _inspector_add(server.gui.add_dropdown(
                    "Role", options=[_ROLE_LABELS[r] for r in ROLES],
                    initial_value=_ROLE_LABELS.get(led.get('role', 'both'), _ROLE_LABELS['both']),
                    hint="VIO: on in flight only. Flash: photogrammetry pulse only. Both: always on.",
                ))
                role_dd.on_update(lambda _: led.__setitem__(
                    'role', next(r for r, lab in _ROLE_LABELS.items() if lab == role_dd.value)))

            _inspector_add(server.gui.add_html("<hr style='margin:8px 0;'>"))
            save_btn = _inspector_add(server.gui.add_button("Save", color="green"))
            template_btn = _inspector_add(server.gui.add_button("Save As Template"))
            cancel_btn = _inspector_add(server.gui.add_button("Cancel", color="red"))
            save_btn.on_click(_save_designer)
            template_btn.on_click(_save_designer_template)
            cancel_btn.on_click(_exit_designer)

    def _enter_panel_designer(index=None):
        """Open an empty designer, or edit panel ``index`` in its own (panel-local) frame."""
        state_d = {'name': 'Untitled Panel', 'editing_index': index, 'leds': [], 'selected_led': None}
        panel = state.panel(index)
        if index is not None and panel is None:
            return
        if panel is not None:
            state_d['name'] = panel.name
            for l in panel.leds:
                rx, ry, rz = _designer_euler_from_axes(l.direction, l.resolved_row_dir())
                state_d['leds'].append({'x': l.position[0], 'y': l.position[1], 'z': l.position[2],
                                        'rx': rx, 'ry': ry, 'rz': rz, 'size': l.size, 'view_angle': l.beam_angle,
                                        'tilt': l.tilt, 'on': l.on, 'role': l.role, 'profile': l.profile})
        if state_d['leds']:
            state_d['selected_led'] = 0
        designer_state[0] = state_d
        designer_mode[0] = True
        # Isolate the view without destroying static nodes (wall/grid/axes/STL).
        _clear_normal_dynamic_scene()
        _set_normal_scene_visible(False)
        _build_designer_ui()
        update_designer_scene(full_rebuild=True)
        _designer_set_camera('YZ')

    enter_designer_ref[0] = _enter_panel_designer

    def _select_panel_real(owner):
        if designer_mode[0]:
            return
        idx = state.owner_index(owner) if owner is not None else None
        if idx == state.selected:
            return
        state.select(idx)

    def update_scene():
        """Redraw the scene based on current slider values (without intensity map)."""
        nonlocal led_handles, ray_handles, camera_fov_handles, vio_fov_handles, guide_handles
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
        for handle in led_handles + ray_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles.clear()
        ray_handles.clear()
        camera_fov_handles.clear()
        vio_fov_handles.clear()
        guide_handles.clear()
        wall_dist = wall_dist_slider.value
        ray_length = ray_length_slider.value

        leds = state.leds(lumens=float(led_lumens_slider.value))
        apply_view_mode(leds)

        # Diffuser lens: widen every beam to at least the diffuser angle (flux loss is applied in the tracers)
        if diffuser_enable_chk.value:
            diff_angle = float(diffuser_angle_slider.value)
            for led in leds:
                led.viewing_angle = max(led.viewing_angle, diff_angle)

        # Save LEDs for reuse in room intensity calculation (in place: ui.room_mode holds a reference)
        current_leds[:] = leds

        # Construction circles + axis of the selected panel's duct guide
        guide = state.guides.get(state.selected) if state.selected is not None else None
        if guide is not None:
            origin = _as_vec3(guide.get('origin', (0, 0, 0)))
            axis = _as_vec3(guide.get('axis', (0, 0, 1)))
            an = np.linalg.norm(axis)
            axis = axis / an if an > 1e-12 else np.array([0.0, 0.0, 1.0])
            circles = guide.get('circles') or []
            for ci, circ in enumerate(circles):
                segs = _circle_line_segments_m(_as_vec3(circ.get('center', origin)), float(circ.get('radius', 0.0)),
                                               _as_vec3(circ.get('normal', axis)), n_seg=64)
                if segs.shape[0]:
                    guide_handles.append(server.scene.add_line_segments(
                        f"/guides/panel_{state.selected}/circle_{ci}", points=segs, colors=(0.55, 0.75, 0.95), line_width=1.5))
            ts = [float(np.dot(_as_vec3(c.get('center', origin)) - origin, axis)) for c in circles] or [0.0]
            t_lo, t_hi = min(ts) - 3.0, max(ts) + 3.0
            if abs(t_hi - t_lo) < 4.0:
                t_lo, t_hi = -8.0, 8.0
            guide_handles.append(server.scene.add_line_segments(
                f"/guides/panel_{state.selected}/axis",
                points=np.array([[(origin + axis * t_lo) / 100.0, (origin + axis * t_hi) / 100.0]]),
                colors=(1.0, 0.85, 0.2), line_width=3.0))

        absorbers = []  # box occluders are gone; the STL mesh is the only occluder

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
                
                # Role colour when lit; dim role colour when idle in this operating mode;
                # neutral dark grey when the user switched it off; yellow when its panel is selected
                _role = led_role(led)
                _idle = bool(getattr(led, 'mode_idle', False))
                if led_enabled:
                    square_color = _MARKER_LIT[_role]
                elif _idle:
                    square_color = _MARKER_IDLE[_role]
                else:
                    square_color = _MARKER_OFF
                _owner = getattr(led, 'owner', None)
                if state.selected is not None and state.owner_index(_owner) == state.selected:
                    square_color = _MARKER_SELECTED if led_enabled else _MARKER_SELECTED_DIM
                
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
                            _select_panel_real(own)
                        return _on_click
                    handle.on_click(_make_owner_click(_owner))
                
                # Draw small center source sphere (only if enabled)
                if led_enabled:
                    handle = server.scene.add_icosphere(
                        f"/leds/led_{led_idx}_source",
                        radius=0.0012,
                        color=_MARKER_LIT[_role],
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

        # Wall patches shared by the tilted-camera and VIO overlays:
        # (axis, plane_coord_cm, u_min, u_max, v_min, v_max, inward_sign); axis 0=x, 1=y, 2=z,
        # (u, v) are the two remaining axes in ascending order, inward_sign offsets the overlay into the room.
        if room_mode_enable.value:
            _fd = room_front_dist.value
            _sd = room_side_dist.value
            _td = room_top_bottom_dist.value
            # Same extent as draw_room_walls: 2.5x depth behind the front wall
            _x_back_edge = _fd - (_fd - _RoomSettings.led_x_center) * 2.5
            if show_back_wall.value:
                _x_back_edge = max(_x_back_edge, -room_back_dist.value)
            overlay_walls = [
                (0, _fd, -_sd, _sd, -_td, _td, -1.0),   # front
                (1, -_sd, _x_back_edge, _fd, -_td, _td, +1.0),  # left
                (1, _sd, _x_back_edge, _fd, -_td, _td, -1.0),   # right
                (2, _td, _x_back_edge, _fd, -_sd, _sd, -1.0),   # top
                (2, -_td, _x_back_edge, _fd, -_sd, _sd, +1.0),  # bottom
            ]
            if show_back_wall.value:
                overlay_walls.append((0, -room_back_dist.value, -_sd, _sd, -_td, _td, +1.0))
        else:
            _half = wall_view_size.value / 2.0
            overlay_walls = [(0, wall_dist_slider.value, -_half, _half, -_half, _half, -1.0)]

        # Up / down copies of the main camera (separate uniformity metrics in the legend)
        if show_camera_fov.value and show_tilt_fovs.value:
            t = float(tilt_fov_deg.value)
            cam_pos = np.array([camera_pos_x.value, camera_pos_y.value, 0.0], dtype=float)
            for tag, pitch, color in (('up', camera_pitch.value + t, (0.65, 1.0, 0.25)),
                                      ('down', camera_pitch.value - t, (0.0, 0.8, 0.55))):
                all_segs = []
                for w_axis, plane_cm, u_min, u_max, v_min, v_max, inward in overlay_walls:
                    if (cam_pos[w_axis] - plane_cm) * inward <= 1e-6:
                        continue
                    mask, us, vs = rasterize_pinhole_fov_on_plane(
                        cam_pos, pitch, camera_fov_h.value, camera_fov_v.value,
                        w_axis, plane_cm, u_min, u_max, v_min, v_max, n_grid=90,
                    )
                    _, _, segs = fov_plane_mask_to_quads_and_contour(mask, us, vs, w_axis, plane_cm / 100.0 + inward * 0.006)
                    if len(segs) > 0:
                        all_segs.append(segs)
                if all_segs:
                    camera_fov_handles.append(server.scene.add_line_segments(
                        f"/camera/tilt_{tag}/fov_border", points=np.concatenate(all_segs, axis=0),
                        colors=color, line_width=3.0,
                    ))
                axis = vio_optical_axis(pitch, 0.0)
                camera_fov_handles.append(server.scene.add_line_segments(
                    f"/camera/tilt_{tag}/axis",
                    points=np.array([[cam_pos / 100.0, (cam_pos + axis * 6.0) / 100.0]]),
                    colors=color, line_width=3.0,
                ))

        # Draw VIO fisheye FOV footprints on the wall(s).
        # In room mode the footprint is projected on ALL room walls (front,
        # left, right, top, bottom, and back if shown), not just the front one.
        if show_vio_fov.value:
            vio_walls = overlay_walls

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
        if state.selected is not None:
            state.select(None)

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
    diffuser_enable_chk.on_update(lambda _: update_scene())
    diffuser_angle_slider.on_update(lambda _: update_scene())
    ray_length_slider.on_update(lambda _: update_scene())
    led_lumens_slider.on_update(lambda _: None)  # No auto-update, use manual button
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: None if _mode_toggle_syncing[0] else update_intensity_map())
    show_camera_fov.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_h.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_v.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_y.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    show_tilt_fovs.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    tilt_fov_deg.on_update(lambda _: (update_scene(), _refresh_uniformity()))
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

    def on_view_mode_change(_):
        """Operating mode / flash electrical changed: redraw markers and flag the intensity map as stale."""
        update_scene()
        if room_mode_enable.value:
            stale = _last_room_cache['grids'] is not None
        else:
            stale = _last_intensity_cache['grid'] is not None
        if stale:
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                f"<div style='color:#F0AD4E;font-size:12px;'>⚠ Operating mode is now <b>{view_mode_dropdown.value}</b>."
                "<br>Click 'Update Intensity Map' / 'Update Room Intensity' to recalculate.</div></div>"
            )

    view_mode_dropdown.on_update(on_view_mode_change)
    flash_lumens_input.on_update(on_view_mode_change)
    led_voltage_input.on_update(on_view_mode_change)
    led_efficacy_input.on_update(on_view_mode_change)
    
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



    def hide_wall():
        """Remove the flat wall (sphere mode); ``update_wall`` brings it back."""
        try:
            wall_handle.remove()
        except (AttributeError, KeyError):
            pass

    return SimpleNamespace(update_scene=update_scene, update_wall=update_wall, hide_wall=hide_wall)
