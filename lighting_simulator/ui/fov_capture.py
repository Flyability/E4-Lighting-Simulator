"""Main-camera FOV capture: render the wall illuminance as seen by the camera and export image + CSV.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import os
import time
import numpy as np
from lighting_simulator.camera.fov import camera_fov_wall_trapezoid as _camera_fov_wall_trapezoid
from lighting_simulator.domain.guides import dynamic_group_world_geometry as _dynamic_group_world_geometry
from lighting_simulator.domain.led_factory import create_leds
from lighting_simulator.domain.optics import effective_lambertian_exponent as _get_effective_n
from lighting_simulator.raytracing.boxes import ray_box_intersection_batch as _ray_box_intersection_batch_np
from lighting_simulator.raytracing.mesh import (
    batch_ray_mesh_intersection as _batch_ray_mesh_intersection,
    prepare_mesh_ray_accelerator as _prepare_mesh_ray_accelerator,
)
from lighting_simulator.scene.stl import _rot4_x, _rot4_y, _rot4_z


def build(ctx):
    _expand_mirror_configs = ctx._expand_mirror_configs
    _panel_slot_data = ctx._panel_slot_data
    apply_view_mode = ctx.apply_view_mode
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
    absorbers_enable = ctx.absorbers_enable
    calibration_factor_slider = ctx.calibration_factor_slider
    camera_fov_h = ctx.camera_fov_h
    camera_fov_v = ctx.camera_fov_v
    camera_pitch = ctx.camera_pitch
    camera_pos_x = ctx.camera_pos_x
    camera_pos_y = ctx.camera_pos_y
    circle_center_slider = ctx.circle_center_slider
    custom_groups = ctx.custom_groups
    diffuser_angle_slider = ctx.diffuser_angle_slider
    diffuser_enable_chk = ctx.diffuser_enable_chk
    diffuser_transmission_slider = ctx.diffuser_transmission_slider
    individual_leds = ctx.individual_leds
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_to_color = ctx.intensity_to_color
    led_lumens_slider = ctx.led_lumens_slider
    led_states = ctx.led_states
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
    room_front_dist = ctx.room_front_dist
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
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_mesh_data = ctx.stl_mesh_data
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale
    viewing_angle_slider = ctx.viewing_angle_slider
    wall_dist_slider = ctx.wall_dist_slider

    def capture_camera_fov_image():
        """Capture intensity image within camera FOV at 1cm resolution."""
        from datetime import datetime
        from PIL import Image
        
        # Get current camera and wall settings
        # Use correct wall distance based on mode
        if room_mode_enable.value:
            wall_dist = room_front_dist.value
        else:
            wall_dist = wall_dist_slider.value
        
        cam_x = camera_pos_x.value
        cam_y = camera_pos_y.value
        fov_h_deg = camera_fov_h.value
        fov_v_deg = camera_fov_v.value
        pitch_deg = camera_pitch.value

        # FOV footprint on the wall (trapezoid when the camera is pitched)
        fov_z_bot, fov_z_top, fov_w_bot, fov_w_top = _camera_fov_wall_trapezoid(
            wall_dist - cam_x, pitch_deg, fov_h_deg, fov_v_deg
        )
        fov_half_w_max = max(fov_w_bot, fov_w_top)
        fov_width_cm = 2.0 * fov_half_w_max   # bounding-box width on wall
        fov_height_cm = fov_z_top - fov_z_bot  # bounding-box height on wall
        
        # Cell resolution: 1cm × 1cm (10mm²)
        cell_size_cm = 1.0
        grid_width = int(np.ceil(fov_width_cm / cell_size_cm))
        grid_height = int(np.ceil(fov_height_cm / cell_size_cm))
        
        # Create grid for FOV region
        fov_grid = np.zeros((grid_height, grid_width))
        
        # Get LEDs configuration (fixed angles: front=0°, side=90°)
        front_angle = 0.0  # Fixed front angle
        side_angle = 90.0  # Fixed side angle
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value
        
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
                'led_roles': group.get('led_roles') or [],
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
                'led_on': led.get('led_on', True),
                'role': led.get('role', 'both'),
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
        apply_view_mode(leds)
        
        # ── Apply diffuser lens effect (FOV camera) ──
        if diffuser_enable_chk.value:
            _diff_angle = float(diffuser_angle_slider.value)
            _diff_trans = float(diffuser_transmission_slider.value) / 100.0
            for led in leds:
                led.viewing_angle = max(led.viewing_angle, _diff_angle)
                if led.lumens is not None:
                    led.lumens = led.lumens * _diff_trans

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
        
        # Add abs2 and abs3 at origin with offsets (if absorbers enabled)
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
        
        # Ray tracing for FOV region
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
        rays_per_pixel = int(intensity_rays_slider.value)
        
        # Count active LEDs
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        if num_active_leds == 0:
            print("No active LEDs")
            return
        
        # Calculate rays per LED to achieve target rays per pixel
        total_pixels = grid_width * grid_height
        num_rays_per_led = max(1, int((total_pixels * rays_per_pixel) / num_active_leds))
        
        print(f"Capturing FOV image: {grid_width}x{grid_height} pixels ({total_pixels} total)...")
        print(f"Active LEDs: {num_active_leds}, Rays per LED: {num_rays_per_led}, Total rays: {num_active_leds * num_rays_per_led}")
        print(f"Target: {rays_per_pixel} rays/pixel, Actual: {(num_active_leds * num_rays_per_led) / total_pixels:.2f} rays/pixel")
        
        # Pre-build STL mesh accelerator ONCE (outside LED loop)
        fov_stl_accel = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_obj = stl_mesh_data[0]
            transform = np.eye(4)
            scale = float(stl_scale.value)
            if np.isfinite(scale) and scale > 0:
                transform[:3, :3] *= scale
            rot_x_v = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
            rot_y_v = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0
            rot_z_v = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
            if rot_x_v != 0:
                transform = _rot4_x(np.radians(rot_x_v)) @ transform
            if rot_y_v != 0:
                transform = _rot4_y(np.radians(rot_y_v)) @ transform
            if rot_z_v != 0:
                transform = _rot4_z(np.radians(rot_z_v)) @ transform
            pos_xv = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
            pos_yv = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
            pos_zv = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
            transform[:3, 3] = [pos_xv, pos_yv, pos_zv]
            fov_mesh_data = {
                'vertices': mesh_obj.vertices,
                'faces': mesh_obj.faces,
                'transform': transform,
            }
            fov_stl_accel = _prepare_mesh_ray_accelerator(fov_mesh_data)
            print(f"STL mesh absorber active ({len(mesh_obj.faces)} triangles)")
        
        led_total_lumens_emitted = 0.0
        
        for led_idx, led in enumerate(leds):
            if hasattr(led, 'enabled') and not led.enabled:
                continue
            
            idx = getattr(led, 'led_index', led_idx)
            np.random.seed((42 + idx) % (2**32))
            
            z_axis = led.direction
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            uniformity = float(ray_uniformity_slider.value)
            n = _get_effective_n(led, uniformity)
            max_theta = np.radians(led.viewing_angle / 2.0)
            cos_max = np.cos(max_theta)
            
            # --- Generate ALL rays for this LED at once ---
            u = np.random.uniform(0, 1, (num_rays_per_led, 2))
            cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            phi = 2 * np.pi * u[:, 1]
            
            sin_theta = np.sin(theta)
            local_dirs = np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta,
            ])
            
            world_dirs = (local_dirs[:, 0:1] * x_axis +
                          local_dirs[:, 1:2] * y_axis +
                          local_dirs[:, 2:3] * z_axis)
            norms_wd = np.linalg.norm(world_dirs, axis=1, keepdims=True)
            world_dirs = world_dirs / norms_wd
            
            # Calculate lumens per ray with cone normalization and lens efficiency
            cos_max_n1 = cos_max ** (n + 1.0)
            denom = 1.0 - cos_max_n1
            norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
            cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
            intensity_coefficients = np.power(cos_theta_clamped, n)
            fov_led_lumens = float(getattr(led, 'lumens', None) or lumens_per_led)
            lumens_per_ray_arr = (fov_led_lumens / max(1, num_rays_per_led)) * intensity_coefficients * norm_factor
            led_total_lumens_emitted += np.sum(lumens_per_ray_arr)
            
            # --- Check box absorber intersection (vectorized) ---
            hit_absorbed = np.zeros(num_rays_per_led, dtype=bool)
            if absorbers:
                ray_origins = np.broadcast_to(led.position, (num_rays_per_led, 3)).copy().astype(np.float32)
                hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
            
            # --- Batch STL mesh intersection ---
            if fov_stl_accel is not None:
                not_abs = np.where(~hit_absorbed)[0]
                if len(not_abs) > 0:
                    origins = np.tile(led.position, (len(not_abs), 1)).astype(np.float64)
                    mesh_hits = _batch_ray_mesh_intersection(origins, world_dirs[not_abs], fov_stl_accel)
                    hit_absorbed[not_abs[mesh_hits]] = True
            
            # --- Wall hits ---
            alive = ~hit_absorbed
            towards_wall = world_dirs[:, 0] > 0
            valid = alive & towards_wall
            vi = np.where(valid)[0]
            
            if len(vi) > 0:
                t = (wall_dist - led.position[0]) / world_dirs[vi, 0]
                pos_t = t > 0
                vi2 = vi[pos_t]
                t2 = t[pos_t]
                
                hit_y = led.position[1] + world_dirs[vi2, 1] * t2
                hit_z = led.position[2] + world_dirs[vi2, 2] * t2
                
                # Keep only hits inside the (possibly pitched) FOV trapezoid,
                # centred at the camera's Y position
                z_span = max(fov_z_top - fov_z_bot, 1e-9)
                frac_z = np.clip((hit_z - fov_z_bot) / z_span, 0.0, 1.0)
                half_w_at_z = fov_w_bot + (fov_w_top - fov_w_bot) * frac_z
                in_fov = (
                    (hit_z >= fov_z_bot) & (hit_z <= fov_z_top)
                    & (np.abs(hit_y - cam_y) <= half_w_at_z)
                )
                fi = np.where(in_fov)[0]
                
                grid_x = ((hit_y[fi] - cam_y + fov_half_w_max) / cell_size_cm).astype(int)
                grid_y = ((hit_z[fi] - fov_z_bot) / cell_size_cm).astype(int)
                
                in_bounds = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
                bi = np.where(in_bounds)[0]
                
                lux_values = lumens_per_ray_arr[vi2[fi[bi]]]
                np.add.at(fov_grid, (grid_y[bi], grid_x[bi]), lux_values)
        
        # Diagnostic: print first LED's flux conservation
        print(f"FOV Capture: First LED emitted {led_total_lumens_emitted:.2f} lm total (target: {lumens_per_led:.2f} lm)")
        
        # Convert to lux: Lux = Lumen / Area_m²
        cell_area_m2 = (cell_size_cm / 100.0) ** 2
        lux_grid = fov_grid / cell_area_m2
        
        # Clean up any NaN or Inf values
        fov_grid = np.nan_to_num(fov_grid, nan=0.0, posinf=0.0, neginf=0.0)
        lux_grid = np.nan_to_num(lux_grid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get max lux for color mapping
        max_lux = lux_grid.max()
        
        # Create image using same colormap as render (intensity_to_color)
        img_rgb = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                lux_val = lux_grid[i, j]
                color = intensity_to_color(lux_val, max_lux)
                img_rgb[i, j] = [int(c * 255) for c in color]
        
        # Add legend to the right (50 pixels wide)
        legend_width = 50
        legend_steps = 100
        full_width = grid_width + legend_width + 10  # 10px padding
        img_with_legend = np.ones((grid_height, full_width, 3), dtype=np.uint8) * 255  # White background
        
        # Copy main image
        img_with_legend[:, :grid_width, :] = img_rgb
        
        # Draw legend bar
        legend_x_start = grid_width + 5
        legend_x_end = legend_x_start + 30
        
        for i in range(legend_steps):
            # Map i to grid_height
            y_start = int(i * grid_height / legend_steps)
            y_end = int((i + 1) * grid_height / legend_steps)
            
            # Intensity from top (max) to bottom (min)
            intensity_fraction = 1.0 - (i / legend_steps)
            lux_val = intensity_fraction * max_lux
            color = intensity_to_color(lux_val, max_lux)
            rgb = [int(c * 255) for c in color]
            
            img_with_legend[y_start:y_end, legend_x_start:legend_x_end, :] = rgb
        
        # Use PIL to add text labels
        from PIL import ImageDraw, ImageFont
        img_pil = Image.fromarray(img_with_legend, 'RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use a default font, fallback to PIL default
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # Add text labels at key points
        num_labels = 6
        for i in range(num_labels):
            fraction = i / (num_labels - 1)
            y_pos = int((1.0 - fraction) * grid_height)
            lux_val = fraction * max_lux
            
            # Draw tick mark
            draw.line([(legend_x_end, y_pos), (legend_x_end + 3, y_pos)], fill=(0, 0, 0), width=1)
            
            # Draw text
            text = f"{lux_val:.0f}"
            draw.text((legend_x_end + 5, y_pos - 5), text, fill=(0, 0, 0), font=font)
        
        # Add "lux" label
        draw.text((legend_x_start, 5), "lux", fill=(0, 0, 0), font=font)
        
        # Save image (next to the other exports instead of the working directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("exports", exist_ok=True)
        filename = os.path.join("exports", f"fov_intensity_{timestamp}.png")
        img_pil.save(filename)
        
        print(f"FOV image saved to {filename}")
        print(f"Image size: {grid_width} x {grid_height} pixels (1 pixel = 1cm²)")
        print(f"FOV dimensions: {fov_width_cm:.2f} x {fov_height_cm:.2f} cm")
        print(f"Total lumens in FOV: {fov_grid.sum():.2f} lm")
        print(f"Max illuminance: {max_lux:.2f} lux")

    FIXED_LEGEND_MAX = 3500.0  # Default fixed absolute legend cap (overridden by GUI)



    return SimpleNamespace(capture_camera_fov_image=capture_camera_fov_image)
