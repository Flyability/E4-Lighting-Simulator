"""Wall intensity map: engine wrappers, colour mapping, legend/uniformity HTML, current-scene LED build.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import time as _time
import numpy as np
import trimesh
from lighting_simulator.analysis.uniformity import compute_uniformity_html as _compute_uniformity_html
from lighting_simulator.camera.fov import (
    camera_fov_wall_trapezoid as _camera_fov_wall_trapezoid, points_in_fisheye_fov, points_in_pinhole_fov,
    vio_hfov_vfov_deg,
)
from lighting_simulator.domain.guides import dynamic_group_world_geometry as _dynamic_group_world_geometry
from lighting_simulator.domain.led_factory import create_leds
from lighting_simulator.scene.absorbers import build_elios_absorbers, rotate_absorbers_z
from lighting_simulator.scene.builder import apply_diffuser, apply_global_transform
from lighting_simulator.scene.stl import global_z_rotation_4x4, stl_mesh_data as stl_mesh_data_payload
from lighting_simulator.simulation import (
    EmissionSettings, RoomSettings, WallSettings, room_wall_cell_centers, wall_grid_cell_centers_cm,
)
from lighting_simulator.simulation.room_geometry import wall_cell_areas_m2, wall_grid_indices
from lighting_simulator.simulation import wall as _wall_engine
from lighting_simulator.simulation import room as _room_engine
from lighting_simulator.ui.mesh_lighting import _build_stl_transform


def build(ctx):
    _absorber_config = ctx._absorber_config
    _expand_mirror_configs = ctx._expand_mirror_configs
    _panel_slot_data = ctx._panel_slot_data
    bw_scale_chk = ctx.bw_scale_chk
    calibration_factor_slider = ctx.calibration_factor_slider
    camera_fov_h = ctx.camera_fov_h
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    cell_readout_chk = ctx.cell_readout_chk
    cell_readout_html = ctx.cell_readout_html
    circle_center_slider = ctx.circle_center_slider
    custom_groups = ctx.custom_groups
    custom_reflectance_slider = ctx.custom_reflectance_slider
    diffuser_angle_slider = ctx.diffuser_angle_slider
    diffuser_enable_chk = ctx.diffuser_enable_chk
    diffuser_transmission_slider = ctx.diffuser_transmission_slider
    global_pos_x_slider = ctx.global_pos_x_slider
    global_pos_y_slider = ctx.global_pos_y_slider
    global_pos_z_slider = ctx.global_pos_z_slider
    global_rotation_z_slider = ctx.global_rotation_z_slider
    individual_leds = ctx.individual_leds
    intensity_grid_size = ctx.intensity_grid_size
    intensity_handles = ctx.intensity_handles
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_threshold_slider = ctx.intensity_threshold_slider
    led_lumens_slider = ctx.led_lumens_slider
    led_states = ctx.led_states
    legend_html = ctx.legend_html
    legend_max_input = ctx.legend_max_input
    max_bounces_slider_room = ctx.max_bounces_slider_room
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
    radius_slider = ctx.radius_slider
    ray_uniformity_slider = ctx.ray_uniformity_slider
    reflections_enable = ctx.reflections_enable
    room_mode_enable = ctx.room_mode_enable
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
    server = ctx.server
    show_intensity_map = ctx.show_intensity_map
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_mesh_data = ctx.stl_mesh_data
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale
    uniformity_percentile_slider = ctx.uniformity_percentile_slider
    vio_occupancy_lux = ctx.vio_occupancy_lux
    viewing_angle_slider = ctx.viewing_angle_slider
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

    def _emission_settings():
        """Engine emission parameters from the GUI.

        Per-LED flux already carries the diffuser transmission (applied in
        _build_current_leds_and_absorbers), so only the fallback flux is scaled here.
        """
        default_lumens = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        if diffuser_enable_chk.value:
            default_lumens *= float(diffuser_transmission_slider.value) / 100.0
        return EmissionSettings(
            default_lumens=default_lumens,
            ray_uniformity=float(ray_uniformity_slider.value),
        )

    def compute_wall_intensity(
        leds, wall_dist, num_rays_per_led, grid_size=50, wall_size=80, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays onto the wall. Returns (lux grid, wall_size)."""
        settings = WallSettings(
            wall_dist=float(wall_dist), grid_size=int(grid_size),
            wall_size=float(wall_size), rays_per_pixel=int(num_rays_per_led),
        )
        grid = _wall_engine.compute_wall_intensity(
            leds, settings, _emission_settings(),
            absorbers=absorbers, stl_mesh_data=stl_mesh_data,
        )
        return grid, wall_size

    def compute_room_intensity(
        leds, front_dist, side_dist, top_bottom_dist, num_rays_per_led, grid_size=20, back_dist=None, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays inside the room. Returns (per-wall lux grids, wall_specs)."""
        reflections_on = bool(reflections_enable.value)
        settings = RoomSettings(
            front_dist=float(front_dist), side_dist=float(side_dist),
            top_bottom_dist=float(top_bottom_dist), back_dist=back_dist,
            grid_size=int(grid_size), rays_per_pixel=int(num_rays_per_led),
            led_x_center=float(circle_center_slider.value),
            max_bounces=int(max_bounces_slider_room.value) if reflections_on else 0,
            wall_reflectance=float(custom_reflectance_slider.value) if reflections_on else 0.0,
        )
        return _room_engine.compute_room_intensity(
            leds, settings, _emission_settings(),
            absorbers=absorbers, stl_mesh_data=stl_mesh_data,
        )


    def intensity_to_color(value, max_val):
        """Convert intensity to colormap (inferno-like or black-to-white)."""
        # Handle invalid values
        if max_val == 0 or not np.isfinite(value) or not np.isfinite(max_val):
            return (0.0, 0.0, 0.0)
        
        # Below-threshold cells are rendered pitch black (threshold in lux, 0 = off)
        if value < float(intensity_threshold_slider.value):
            return (0.0, 0.0, 0.0)
        
        t = np.clip(value / max_val, 0.0, 1.0)
        
        # Black-to-white grayscale mode
        if bw_scale_chk.value:
            return (t, t, t)
        
        # Simple inferno-like gradient: black -> purple -> red -> orange -> yellow
        if t < 0.25:
            r, g, b = t * 4 * 0.5, 0, t * 4 * 0.5
        elif t < 0.5:
            r, g, b = 0.5 + (t - 0.25) * 4 * 0.5, 0, 0.5 - (t - 0.25) * 4 * 0.5
        elif t < 0.75:
            r, g, b = 1.0, (t - 0.5) * 4 * 0.5, 0
        else:
            r, g, b = 1.0, 0.5 + (t - 0.75) * 4 * 0.5, (t - 0.75) * 4
        return (r, g, b)

    # Cache for last computed intensity grid so uniformity can be recalculated
    # when FOV changes without re-running ray tracing
    _last_intensity_cache = {'grid': None, 'wall_size_cm': None, 'wall_dist': None,
                             'cell_area_m2': None, 'max_lux': None, 'color_scale_max': None}
    _last_room_cache = {
        'grids': None, 'wall_specs': None,
        'front_dist': None, 'side_dist': None, 'top_bottom_dist': None, 'back_dist': None,
        'max_lux': None, 'color_scale_max': None, 'avg_cell_area_m2': None,
    }
    # Guard programmatic checkbox writes in on_room_mode_toggle so we don't
    # retrigger update_intensity_map / update_room_intensity_map.
    _mode_toggle_syncing = [False]

    def _build_lux_legend_html(max_lux, color_scale_max, cell_area_m2, cell_caption="lm/cell"):
        """Intensity-scale HTML used by both wall and room legend panels."""
        _legend_cap = float(legend_max_input.value)
        if color_scale_max <= _legend_cap:
            _step = max(1, _legend_cap / 8)
            legend_vals_lux = np.arange(0, _legend_cap + 1, _step)
        else:
            legend_vals_lux = np.linspace(0, color_scale_max, 9)
        scale_label = f"(scale 0\u2013{int(color_scale_max)} lx" + (
            ", FIXED)" if color_scale_max <= _legend_cap else ", AUTO)")
        html_lines = [
            "<div style='font-family: sans-serif;'>",
            "<div style='font-weight:600;margin-bottom:2px;'>Intensity legend (lux)</div>",
            f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} \u2014 peak {max_lux:.0f} lx</div>",
        ]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, color_scale_max)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            lumen_val = lux_val * cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.0f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} {cell_caption})</div></div>"
            )
        html_lines.append("</div>")
        return "".join(html_lines)

    def _empty_fov_html():
        return (
            "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>Pattern Uniformity</div>"
            "<div style='color:#888;font-size:12px;'>No intensity data inside camera FOV</div></div>"
        )

    def _wall_grid_cell_centers_cm(grid, wall_size_cm, wall_dist):
        return wall_grid_cell_centers_cm(grid.shape, wall_size_cm, wall_dist)

    def _collect_room_fov_lux(cache):
        """Lux values of room-wall cells that fall inside the main-camera FOV."""
        grids = cache.get('grids')
        specs = cache.get('wall_specs')
        if not grids or not specs:
            return np.array([])
        cam_pos = np.array([camera_pos_x.value, camera_pos_y.value, 0.0], dtype=float)
        parts = []
        for name, grid in grids.items():
            spec = specs.get(name)
            if spec is None:
                continue
            pts = room_wall_cell_centers(
                name, spec,
                cache['front_dist'], cache['side_dist'],
                cache['top_bottom_dist'], cache['back_dist'],
            )
            mask = points_in_pinhole_fov(
                cam_pos, camera_pitch.value,
                camera_fov_h.value, camera_fov_v.value, pts,
            )
            if np.any(mask):
                parts.append(grid[mask])
        if not parts:
            return np.array([])
        return np.concatenate(parts)

    def _collect_active_cell_samples():
        """(points_cm Nx3, lux N) for the active intensity mode, or (None, None)."""
        if room_mode_enable.value:
            cache = _last_room_cache
            if cache['grids'] is None:
                return None, None
            pts_list, lux_list = [], []
            for name, grid in cache['grids'].items():
                spec = cache['wall_specs'].get(name)
                if spec is None:
                    continue
                pts = room_wall_cell_centers(
                    name, spec,
                    cache['front_dist'], cache['side_dist'],
                    cache['top_bottom_dist'], cache['back_dist'],
                )
                pts_list.append(pts.reshape(-1, 3))
                lux_list.append(np.asarray(grid).reshape(-1))
            if not pts_list:
                return None, None
            return np.concatenate(pts_list, axis=0), np.concatenate(lux_list)
        cache = _last_intensity_cache
        if cache['grid'] is None:
            return None, None
        pts = _wall_grid_cell_centers_cm(cache['grid'], cache['wall_size_cm'], cache['wall_dist'])
        return pts.reshape(-1, 3), np.asarray(cache['grid']).reshape(-1)

    def _compute_vio_occupancy_html():
        """% of VIO-FOV wall cells whose lux is at or above the VIO occupancy threshold."""
        pts, lux = _collect_active_cell_samples()
        if pts is None or pts.size == 0:
            return ""
        threshold = float(vio_occupancy_lux.value)
        cam_pos = np.array([vio_pos_x.value, vio_pos_y.value, vio_pos_z.value], dtype=float)
        hfov, vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
        mask1 = points_in_fisheye_fov(
            cam_pos, vio_cam1_pitch.value, vio_cam1_yaw.value, hfov, vfov, pts)
        mask2 = points_in_fisheye_fov(
            cam_pos, vio_cam2_pitch.value, vio_cam2_yaw.value, hfov, vfov, pts)
        good = lux >= threshold if threshold > 0 else lux > 0

        def _row(mask):
            n = int(np.count_nonzero(mask))
            if n == 0:
                return None, "\u2014"
            pct = 100.0 * float(np.count_nonzero(mask & good)) / n
            return pct, f"{pct:.1f}%"

        _, s1 = _row(mask1)
        _, s2 = _row(mask2)
        union_pct, s_u = _row(mask1 | mask2)
        union_color = "#4CAF50" if (union_pct is not None and union_pct >= 50.0) else "#FF9800"
        if union_pct is None:
            union_color = "#888"
        return (
            "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>VIO FOV Lighting Occupancy</div>"
            f"<div style='color:#888;font-size:10px;margin-bottom:6px;'>"
            f"Share of FOV wall cells {'at or above' if threshold > 0 else 'brighter than'} {threshold:.0f} lx"
            f"{' (any light)' if threshold <= 0 else ''}</div>"
            "<table style='font-size:11px;color:#ccc;border-collapse:collapse;width:100%;'>"
            f"<tr><td style='padding:1px 6px 1px 0;color:#ff00ff;'>Cam 1 (up)</td><td>{s1}</td></tr>"
            f"<tr><td style='padding:1px 6px 1px 0;color:#00ffff;'>Cam 2 (down)</td><td>{s2}</td></tr>"
            f"<tr><td style='padding:1px 6px 1px 0;'>Union (Cam 1 \u222a Cam 2)</td>"
            f"<td style='color:{union_color};font-weight:700;'>{s_u}</td></tr>"
            "</table>"
            "<div style='font-size:10px;color:#888;margin-top:6px;'>"
            "Denominator is FOV-covered wall cells, not empty space.</div></div>"
        )

    def _wall_metrics_html(grid, wall_size_cm, wall_dist):
        _trap = _camera_fov_wall_trapezoid(
            wall_dist - camera_pos_x.value, camera_pitch.value,
            camera_fov_h.value, camera_fov_v.value,
        )
        html = _compute_uniformity_html(
            grid,
            fov_trapezoid=(*_trap, camera_pos_y.value),
            wall_size_cm=wall_size_cm,
            min_percentile=float(uniformity_percentile_slider.value),
        )
        if not html:
            html = _empty_fov_html()
        return html + _compute_vio_occupancy_html()

    def _room_metrics_html():
        fov_lux = _collect_room_fov_lux(_last_room_cache)
        if fov_lux.size == 0:
            html = _empty_fov_html()
        else:
            html = (_compute_uniformity_html(fov_lux.reshape(1, -1),
                                             min_percentile=float(uniformity_percentile_slider.value))
                    or _empty_fov_html())
        return html + _compute_vio_occupancy_html()

    def _refresh_uniformity():
        """Recalculate FOV-only uniformity (and VIO occupancy) from cache (cheap)."""
        if room_mode_enable.value:
            cache = _last_room_cache
            if cache['grids'] is None:
                return
            legend = _build_lux_legend_html(
                cache['max_lux'], cache['color_scale_max'],
                cache['avg_cell_area_m2'], cell_caption="lm/cell avg",
            )
            legend_html.content = legend + _room_metrics_html()
            return
        cache = _last_intensity_cache
        if cache['grid'] is None:
            return
        legend = _build_lux_legend_html(
            cache['max_lux'], cache['color_scale_max'],
            cache['cell_area_m2'], cell_caption="lm/cell",
        )
        legend_html.content = legend + _wall_metrics_html(
            cache['grid'], cache['wall_size_cm'], cache['wall_dist'],
        )

    def _build_current_leds_and_absorbers():
        """Build LEDs and absorbers from current GUI state."""
        front_angle = 0.0
        side_angle = 90.0
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value

        rotations = [
            rot_front_pos.value, rot_front_neg.value,
            rot_side_pos.value, rot_side_neg.value,
        ]
        rotations_y = [
            rot_y_front_pos.value, rot_y_front_neg.value,
            rot_y_side_pos.value, rot_y_side_neg.value,
        ]
        offsets = [
            (offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value),
            (offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value),
            (offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value),
            (offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value),
        ]

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
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                translated_positions, rotated_directions, rotated_row_dirs = _dynamic_group_world_geometry(group)
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                config['led_beam_tilts'] = group.get('led_beam_tilts') or []
                if rotated_row_dirs:
                    config['led_row_directions'] = rotated_row_dirs
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
            if group.get('panel_slot') is not None:
                config['owner'] = ('slot', group['panel_slot'])
            else:
                config['owner'] = ('custom_group', group['id'])
            custom_groups_configs.append(config)

        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),
                'pos_x': led['pos_x'].value, 'pos_y': led['pos_y'].value, 'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value, 'rot_y': led['rot_y'].value, 'rot_z': led['rot_z'].value,
                'size': led['size'].value, 'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value, 'beam_tilt': led['beam_tilt'].value,
            }
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
            front_angle, side_angle, viewing_angle, radius, circle_center_x,
            default_lumens=float(led_lumens_slider.value),
            group_rotations=rotations, group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states, group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )

        _g_rot_z_deg = float(global_rotation_z_slider.value)
        apply_global_transform(
            leds, _g_rot_z_deg,
            (global_pos_x_slider.value, global_pos_y_slider.value, global_pos_z_slider.value),
        )

        if diffuser_enable_chk.value:
            apply_diffuser(leds, diffuser_angle_slider.value,
                           float(diffuser_transmission_slider.value) / 100.0)

        absorbers = build_elios_absorbers(
            _absorber_config(), radius, circle_center_x, front_angle,
        )
        rotate_absorbers_z(absorbers, _g_rot_z_deg)

        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            if abs(_g_rot_z_deg) > 0.01:
                transform = global_z_rotation_4x4(_g_rot_z_deg) @ transform
            stl_mesh_for_raytracing = stl_mesh_data_payload(stl_mesh_data[0], transform)

        return leds, absorbers, stl_mesh_for_raytracing

    def update_intensity_map():
        """Update only the intensity map on the wall (expensive operation)."""
        nonlocal intensity_handles, legend_html

        if room_mode_enable.value:
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Room Mode is active — use 'Update Room Intensity' in Advanced → Room Mode.</div>"
                "</div>"
            )
            return
        
        import time as _time
        t_total_start = _time.perf_counter()
        
        # Clear previous intensity handles
        for handle in intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        intensity_handles.clear()
        if not show_intensity_map.value:
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click 'Update Intensity Map' to see the legend</div>"
                "</div>"
            )
            return
        
        # Get current values
        wall_dist = wall_dist_slider.value
        grid_size = int(intensity_grid_size.value)
        wall_size = max(int(wall_view_size.value), 80)  # min 80cm to always cover +-40cm export range
        
        leds, absorbers, stl_mesh_for_raytracing = _build_current_leds_and_absorbers()
        
        # Compute intensity with rays_per_pixel from slider
        t_raytrace_start = _time.perf_counter()
        rays_per_pixel = int(intensity_rays_slider.value)
        intensity_grid, actual_wall_size = compute_wall_intensity(
            leds, wall_dist, rays_per_pixel, grid_size, wall_size, absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        t_raytrace_end = _time.perf_counter()
        print(f"  [TIMING] Ray tracing: {t_raytrace_end - t_raytrace_start:.2f}s")
        # Clean up any NaN or Inf values in the grid
        intensity_grid = np.nan_to_num(intensity_grid, nan=0.0, posinf=0.0, neginf=0.0)
        max_lux = intensity_grid.max()  # Grid now contains lux (lm/m²)
        # Use fixed scale; fall back to actual max if it exceeds the cap
        _legend_cap = float(legend_max_input.value)
        color_scale_max = _legend_cap if max_lux <= _legend_cap else max_lux
        
        # Calculate cell area for lux to lumen conversion
        cell_size_cm = actual_wall_size / grid_size
        cell_area_cm2 = cell_size_cm * cell_size_cm
        cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
        
        # === DIAGNOSTIC OUTPUT FOR FLUX CONSERVATION ===
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
        total_emitted_lumens = sum(float(getattr(led, 'lumens', None) or lumens_per_led) for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        # Convert lux to lumen: multiply each cell by its area and sum
        total_wall_lumens = np.sum(intensity_grid * cell_area_m2)
        conservation_ratio = (total_wall_lumens / total_emitted_lumens * 100) if total_emitted_lumens > 0 else 0
        
        # Calculate 7mm² sensor reading at center
        # Grid contains lux (lm/m²), convert to lumens for sensor area
        sensor_area_cm2 = 0.07  # 7mm² = 0.07cm²
        sensor_area_m2 = sensor_area_cm2 / 10000.0
        
        # Grid now stores lux (lm/m²), convert to lumens for sensor
        center_idx = grid_size // 2
        center_cell_lux = intensity_grid[center_idx, center_idx]
        
        # Convert sensor area to m²
        sensor_area_m2 = sensor_area_cm2 / 10000.0
        
        # Calculate lumens on sensor: Lumen = Lux × Area
        sensor_lumens_from_center_cell = center_cell_lux * sensor_area_m2
        
        print(f"\n=== FLUX CONSERVATION CHECK ===")
        print(f"Active LEDs: {num_active_leds}")
        print(f"Lumens per LED: {lumens_per_led:.1f} lm")
        print(f"Total emitted: {total_emitted_lumens:.1f} lm")
        print(f"Total on wall: {total_wall_lumens:.1f} lm")
        print(f"Conservation: {conservation_ratio:.1f}%")
        print(f"Wall distance: {wall_dist:.1f} cm")
        print(f"7mm² sensor at center: {sensor_lumens_from_center_cell:.4f} lm")
        print(f"================================\n")
        
        cell_size_cm = actual_wall_size / grid_size
        cell_size_m = cell_size_cm / 100.0
        half_size = actual_wall_size / 2
        
        t_viz_start = _time.perf_counter()
        
        # Build a single colored mesh for the entire intensity grid (much faster than per-cell boxes)
        # Each cell = 2 triangles (quad), with vertex colors for smooth rendering
        vertices_list = []
        faces_list = []
        colors_list = []
        
        x_pos = wall_dist / 100.0 - 0.008  # slightly in front of the wall
        vert_idx = 0
        gap = 0.025  # Small gap between cells (2.5% of cell)
        
        for gz in range(grid_size):
            for gy in range(grid_size):
                intensity = intensity_grid[gz, gy]
                if intensity > 0:
                    color = intensity_to_color(intensity, color_scale_max)
                    color_uint8 = [int(c * 255) for c in color] + [255]
                    
                    y_center = (-half_size + gy * cell_size_cm + cell_size_cm / 2) / 100.0
                    z_center = (-half_size + gz * cell_size_cm + cell_size_cm / 2) / 100.0
                    half_cell = cell_size_m * 0.5 * (1.0 - gap)
                    
                    # 4 corners of the quad
                    v0 = [x_pos, y_center - half_cell, z_center - half_cell]
                    v1 = [x_pos, y_center + half_cell, z_center - half_cell]
                    v2 = [x_pos, y_center + half_cell, z_center + half_cell]
                    v3 = [x_pos, y_center - half_cell, z_center + half_cell]
                    
                    vertices_list.extend([v0, v1, v2, v3])
                    # Double-sided: both winding orders so visible from any angle
                    faces_list.append([vert_idx, vert_idx + 1, vert_idx + 2])
                    faces_list.append([vert_idx, vert_idx + 2, vert_idx + 3])
                    faces_list.append([vert_idx, vert_idx + 2, vert_idx + 1])
                    faces_list.append([vert_idx, vert_idx + 3, vert_idx + 2])
                    colors_list.extend([color_uint8] * 4)
                    vert_idx += 4
        
        if len(vertices_list) > 0:
            vertices_np = np.array(vertices_list, dtype=np.float32)
            faces_np = np.array(faces_list, dtype=np.uint32)
            colors_np = np.array(colors_list, dtype=np.uint8)
            
            intensity_mesh = trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False
            )
            from trimesh.visual import ColorVisuals
            intensity_mesh.visual = ColorVisuals(mesh=intensity_mesh, vertex_colors=colors_np)
            
            handle = server.scene.add_mesh_trimesh(
                name="/intensity_map",
                mesh=intensity_mesh,
                visible=True,
            )
            intensity_handles.append(handle)
        
        t_viz_end = _time.perf_counter()
        print(f"  [TIMING] Visualization: {t_viz_end - t_viz_start:.2f}s ({vert_idx // 4} cells)")
        print(f"  [TIMING] Total update_intensity_map: {t_viz_end - t_total_start:.2f}s")
        
        # Update legend (grid now stores lux = lm/m²)
        legend = _build_lux_legend_html(max_lux, color_scale_max, cell_area_m2, cell_caption="lm/cell")

        # Cache grid so FOV / VIO / threshold changes can recalculate metrics cheaply
        _last_intensity_cache['grid'] = intensity_grid
        _last_intensity_cache['wall_size_cm'] = actual_wall_size
        _last_intensity_cache['wall_dist'] = wall_dist
        _last_intensity_cache['cell_area_m2'] = cell_area_m2
        _last_intensity_cache['max_lux'] = max_lux
        _last_intensity_cache['color_scale_max'] = color_scale_max

        legend_html.content = legend + _wall_metrics_html(
            intensity_grid, actual_wall_size, wall_dist,
        )

    _READOUT_HINT = ("<div style='color:#888;font-size:11px;margin:-4px 0 8px;'>"
                     "Click a cell in the 3-D view to read its lux.</div>")

    @cell_readout_chk.on_update
    def _(_):
        cell_readout_html.content = _READOUT_HINT if cell_readout_chk.value else ""

    def _readout_html(lux, area_m2, where):
        return (
            "<div style='font-family:sans-serif;font-size:12px;padding:6px 8px;margin:-4px 0 8px;"
            "border:1px solid #ccc;border-radius:4px;background:#f7f7f7;'>"
            f"<b>{lux:,.0f} lux</b> &nbsp;({lux * area_m2:.3f} lm/cell)<br>"
            f"<span style='color:#666;'>{where}</span></div>"
        )

    def _read_room_cell_at_ray(o, d):
        cache = _last_room_cache
        grids, specs = cache['grids'], cache['wall_specs']
        if not grids or not specs:
            return False
        front, side, tb, back = cache['front_dist'], cache['side_dist'], cache['top_bottom_dist'], cache['back_dist']
        # (plane axis, plane value cm, coord1 axis, coord2 axis) per wall; only walls facing the ray
        planes = {'front': (0, front, 1, 2), 'left': (1, -side, 0, 2), 'right': (1, side, 0, 2),
                  'top': (2, tb, 0, 1), 'bottom': (2, -tb, 0, 1)}
        if back is not None:
            planes['back'] = (0, -float(back), 1, 2)
        areas = wall_cell_areas_m2(specs)
        best = None
        for name, (ax, plane_cm, a1, a2) in planes.items():
            grid = grids.get(name)
            spec = specs.get(name)
            if grid is None or spec is None or abs(d[ax]) < 1e-9:
                continue
            t = (plane_cm / 100.0 - o[ax]) / d[ax]
            if t <= 0 or (best is not None and t >= best[0]):
                continue
            hit = (o + t * d) * 100.0
            row, col = wall_grid_indices(name, spec, hit[a1], hit[a2])
            row, col = int(np.floor(row)), int(np.floor(col))
            if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                best = (t, name, row, col, hit)
        if best is None:
            return False
        _, name, row, col, hit = best
        cell_readout_html.content = _readout_html(
            float(grids[name][row, col]), float(areas[name]),
            f"{name} wall &nbsp;·&nbsp; row {row}, col {col} &nbsp;·&nbsp; "
            f"x = {hit[0]:+.0f}, y = {hit[1]:+.0f}, z = {hit[2]:+.0f} cm",
        )
        return True

    def read_cell_at_ray(ray_origin, ray_direction):
        """Show the lux of the displayed wall/room cell hit by a click ray. Returns True if a cell was hit."""
        if not cell_readout_chk.value:
            return False
        o = np.asarray(ray_origin, dtype=float)
        d = np.asarray(ray_direction, dtype=float)
        if room_mode_enable.value:
            return _read_room_cell_at_ray(o, d)
        grid = _last_intensity_cache['grid']
        if grid is None or not intensity_handles:
            return False
        wall_cm = float(_last_intensity_cache['wall_size_cm'])
        if abs(d[0]) < 1e-9:
            return False
        t = (float(_last_intensity_cache['wall_dist']) / 100.0 - o[0]) / d[0]
        if t <= 0:
            return False
        y_cm, z_cm = (o[1] + t * d[1]) * 100.0, (o[2] + t * d[2]) * 100.0
        n = grid.shape[0]
        cell_cm = wall_cm / n
        gy = int((y_cm + wall_cm / 2) // cell_cm)
        gz = int((z_cm + wall_cm / 2) // cell_cm)
        if not (0 <= gy < n and 0 <= gz < n):
            return False
        cell_readout_html.content = _readout_html(
            float(grid[gz, gy]), float(_last_intensity_cache['cell_area_m2']),
            f"row {gz}, col {gy} &nbsp;·&nbsp; y = {y_cm:+.1f} cm, z = {z_cm:+.1f} cm",
        )
        return True

    # ── CSV Pattern Import logic ──────────────────────────────────────────


    return SimpleNamespace(_build_current_leds_and_absorbers=_build_current_leds_and_absorbers, _build_lux_legend_html=_build_lux_legend_html, _last_intensity_cache=_last_intensity_cache, _last_room_cache=_last_room_cache, _mode_toggle_syncing=_mode_toggle_syncing, _refresh_uniformity=_refresh_uniformity, _room_metrics_html=_room_metrics_html, compute_room_intensity=compute_room_intensity, compute_wall_intensity=compute_wall_intensity, intensity_to_color=intensity_to_color, read_cell_at_ray=read_cell_at_ray, update_intensity_map=update_intensity_map)
