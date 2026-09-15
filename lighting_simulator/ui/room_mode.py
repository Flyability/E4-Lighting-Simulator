"""Room mode: wireframe room walls and the six-wall intensity map.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import time as _time
import numpy as np
import trimesh
from lighting_simulator.scene.absorbers import build_elios_absorbers, rotate_absorbers_z
from lighting_simulator.scene.stl import global_z_rotation_4x4, stl_mesh_data as stl_mesh_data_payload


def build(ctx):
    _build_lux_legend_html = ctx._build_lux_legend_html
    _build_stl_transform = ctx._build_stl_transform
    _last_room_cache = ctx._last_room_cache
    _room_metrics_html = ctx._room_metrics_html
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
    circle_center_slider = ctx.circle_center_slider
    compute_room_intensity = ctx.compute_room_intensity
    current_leds = ctx.current_leds
    global_rotation_z_slider = ctx.global_rotation_z_slider
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_to_color = ctx.intensity_to_color
    legend_html = ctx.legend_html
    legend_max_input = ctx.legend_max_input
    radius_slider = ctx.radius_slider
    room_back_dist = ctx.room_back_dist
    room_front_dist = ctx.room_front_dist
    room_grid_size = ctx.room_grid_size
    room_intensity_handles = ctx.room_intensity_handles
    room_mode_enable = ctx.room_mode_enable
    room_side_dist = ctx.room_side_dist
    room_top_bottom_dist = ctx.room_top_bottom_dist
    room_wall_handles = ctx.room_wall_handles
    server = ctx.server
    show_back_wall = ctx.show_back_wall
    show_room_intensity = ctx.show_room_intensity
    show_room_walls = ctx.show_room_walls
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_mesh_data = ctx.stl_mesh_data
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale

    def draw_room_walls():
        """Draw room walls as wireframe/transparent boxes (no intensity calculation)."""
        
        # Clear previous room wall handles
        for handle in room_wall_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_wall_handles.clear()
        if not room_mode_enable.value or not show_room_walls.value:
            return
        
        # Get room dimensions
        front_dist = room_front_dist.value
        side_dist = room_side_dist.value
        top_bottom_dist = room_top_bottom_dist.value
        
        # Get LED position range (LEDs are at negative X)
        led_x_min = circle_center_slider.value  # Typically -35 cm
        
        # Wall color (solid gray like front wall)
        wall_color = (0.5, 0.5, 0.5)
        
        # Front wall (YZ plane at x=front_dist)
        front_width = 2 * side_dist / 100.0  # Y direction
        front_height = 2 * top_bottom_dist / 100.0  # Z direction
        handle = server.scene.add_box(
            "/room_walls/front",
            dimensions=(0.01, front_width, front_height),
            color=wall_color,
            position=(front_dist / 100.0, 0, 0),
        )
        room_wall_handles.append(handle)
        
        # Left wall (XZ plane at y=-side_dist) - starts from front wall, extends backward
        left_width = (front_dist - led_x_min) / 100.0 * 2.5  # X direction: 2.5x depth
        left_height = 2 * top_bottom_dist / 100.0  # Z direction
        # Center: front wall is at front_dist, extend backward by left_width
        left_center_x = (front_dist - left_width * 100.0 / 2) / 100.0
        handle = server.scene.add_box(
            "/room_walls/left",
            dimensions=(left_width, 0.01, left_height),
            color=wall_color,
            position=(left_center_x, -side_dist / 100.0, 0),
        )
        room_wall_handles.append(handle)
        
        # Right wall (XZ plane at y=+side_dist) - starts from front wall, extends backward
        handle = server.scene.add_box(
            "/room_walls/right",
            dimensions=(left_width, 0.01, left_height),
            color=wall_color,
            position=(left_center_x, side_dist / 100.0, 0),
        )
        room_wall_handles.append(handle)
        
        # Top wall (XY plane at z=+top_bottom_dist) - starts from front wall, extends backward
        top_width = (front_dist - led_x_min) / 100.0 * 2.5  # X direction: 2.5x depth
        top_depth = 2 * side_dist / 100.0  # Y direction
        # Center: front wall is at front_dist, extend backward by top_width
        top_center_x = (front_dist - top_width * 100.0 / 2) / 100.0
        handle = server.scene.add_box(
            "/room_walls/top",
            dimensions=(top_width, top_depth, 0.01),
            color=wall_color,
            position=(top_center_x, 0, top_bottom_dist / 100.0),
        )
        room_wall_handles.append(handle)
        
        # Bottom wall (XY plane at z=-top_bottom_dist) - starts from front wall, extends backward
        handle = server.scene.add_box(
            "/room_walls/bottom",
            dimensions=(top_width, top_depth, 0.01),
            color=wall_color,
            position=(top_center_x, 0, -top_bottom_dist / 100.0),
        )
        room_wall_handles.append(handle)
        
        # Back wall (YZ plane at x=-back_dist) - optional
        if show_back_wall.value:
            back_dist = room_back_dist.value
            back_width = 2 * side_dist / 100.0  # Y direction
            back_height = 2 * top_bottom_dist / 100.0  # Z direction
            # Back wall is at negative X (symmetric to front wall)
            back_x_pos = -back_dist / 100.0
            handle = server.scene.add_box(
                "/room_walls/back",
                dimensions=(0.01, back_width, back_height),
                color=wall_color,
                position=(back_x_pos, 0, 0),
            )
            room_wall_handles.append(handle)
    
    def update_room_intensity_map():
        """Update intensity map for all 5 room walls."""
        
        # Clear previous room intensity handles
        for handle in room_intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_intensity_handles.clear()
        if not room_mode_enable.value or not show_room_intensity.value:
            return
        
        # Hide room walls when showing intensity (they would cover the intensity cells)
        for handle in room_wall_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_wall_handles.clear()
        # Get room dimensions
        front_dist = room_front_dist.value
        side_dist = room_side_dist.value
        top_bottom_dist = room_top_bottom_dist.value
        grid_size = int(room_grid_size.value)
        
        # Use LEDs from current scene (already created in update_scene)
        leds = current_leds
        if not leds:
            print("Error: No LEDs available. Update scene first.")
            return
        
        # Build absorbers (still needed for room mode)
        absorbers = []
        
        # Get current geometry values for absorber calculation
        front_angle = 0.0
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value
        
        angles_deg = [front_angle, -front_angle, 90.0, -90.0]
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
        
        # Compute room intensity
        rays_per_pixel = int(intensity_rays_slider.value)
        
        # Prepare STL mesh data for ray tracing if enabled
        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_ref = stl_mesh_data[0]
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            # Apply global Z rotation to STL transform (room mode)
            _g_rot_room = global_rotation_z_slider.value
            if abs(_g_rot_room) > 0.01:
                T_global_room = np.eye(4)
                _gr = np.radians(_g_rot_room)
                T_global_room[:3, :3] = np.array([[np.cos(_gr), -np.sin(_gr), 0],
                                                   [np.sin(_gr),  np.cos(_gr), 0],
                                                   [0,            0,           1]])
                transform = T_global_room @ transform
            stl_mesh_for_raytracing = {
                'vertices': mesh_ref.vertices,
                'faces': mesh_ref.faces,
                'transform': transform
            }
            print(f"STL mesh enabled as light absorber ({len(mesh_ref.faces)} triangles)")
        
        # ── Apply global Z rotation to absorbers (room mode) ──
        _g_rot_room_deg = global_rotation_z_slider.value
        if abs(_g_rot_room_deg) > 0.01:
            _gr2 = np.radians(_g_rot_room_deg)
            _cg2, _sg2 = np.cos(_gr2), np.sin(_gr2)
            for a in absorbers:
                cx, cy, cz = a['center']
                a['center'] = (_cg2 * cx - _sg2 * cy, _sg2 * cx + _cg2 * cy, cz)
                if a.get('rotation') is not None:
                    qw0, qx0, qy0, qz0 = a['rotation']
                    _hf = _gr2 / 2.0
                    gqw, gqz = np.cos(_hf), np.sin(_hf)
                    a['rotation'] = (gqw*qw0 - gqz*qz0, gqw*qx0 - gqz*qy0, gqw*qy0 + gqz*qx0, gqw*qz0 + gqz*qw0)
                else:
                    _hf = _gr2 / 2.0
                    a['rotation'] = (np.cos(_hf), 0.0, 0.0, np.sin(_hf))

        grids, wall_specs = compute_room_intensity(
            leds, front_dist, side_dist, top_bottom_dist, rays_per_pixel, grid_size, 
            back_dist=room_back_dist.value if show_back_wall.value else None,
            absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        
        # Find max lux across all walls for color normalization
        max_lux = max(grid.max() for grid in grids.values()) if grids else 0.0
        # Use fixed lx scale; fall back to actual max if it exceeds the cap
        _legend_cap = float(legend_max_input.value)
        color_scale_max = _legend_cap if max_lux <= _legend_cap else max_lux
        
        print(f"\n=== ROOM INTENSITY VISUALIZATION ===")
        print(f"Max illuminance across all walls: {max_lux:.4f} lux")
        for wall_name, grid in grids.items():
            cells_with_intensity = np.count_nonzero(grid > 0)
            # Convert lux to lumens: multiply by cell area for that wall
            wall_spec = wall_specs[wall_name]
            if wall_name in ('front', 'back'):
                cell_area_cm2 = (wall_spec['size_y']/wall_spec['grid_y']) * (wall_spec['size_z']/wall_spec['grid_z'])
            elif wall_name in ('left', 'right'):
                cell_area_cm2 = (wall_spec['size_x']/wall_spec['grid_x']) * (wall_spec['size_z']/wall_spec['grid_z'])
            else:  # top/bottom
                cell_area_cm2 = (wall_spec['size_x']/wall_spec['grid_x']) * (wall_spec['size_y']/wall_spec['grid_y'])
            cell_area_m2 = cell_area_cm2 / 10000.0
            total_lumen = np.sum(grid) * cell_area_m2
            print(f"  {wall_name.capitalize()}: {cells_with_intensity} cells with intensity (total: {total_lumen:.1f} lm)")
        
        # Visualize each wall using batched meshes with flat shading (no 3D lighting artifacts)
        cells_created = {'front': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        if show_back_wall.value and 'back' in grids:
            cells_created['back'] = 0
        gap = 0.025  # 2.5% gap between cells
        inward_offset = 0.003  # 3mm inward offset to avoid z-fighting with room walls
        
        for wall_name, intensity_grid in grids.items():
            wall_spec = wall_specs[wall_name]
            grid_shape = intensity_grid.shape
            
            positions_list = []
            colors_list = []
            
            for gi in range(grid_shape[0]):
                for gj in range(grid_shape[1]):
                    intensity = intensity_grid[gi, gj]
                    color = intensity_to_color(intensity, color_scale_max)
                    color_uint8 = [int(c * 255) for c in color]
                    
                    if wall_name == 'front':
                        size_y = wall_spec['size_y']
                        size_z = wall_spec['size_z']
                        grid_y = wall_spec['grid_y']
                        grid_z = wall_spec['grid_z']
                        cell_size_y = size_y / grid_y
                        cell_size_z = size_z / grid_z
                        x_pos = front_dist / 100.0 - inward_offset
                        y_c = (-size_y/2 + gj * cell_size_y + cell_size_y / 2) / 100.0
                        z_c = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        positions_list.append([x_pos, y_c, z_c])
                    
                    elif wall_name == 'back':
                        size_y = wall_spec['size_y']
                        size_z = wall_spec['size_z']
                        grid_y = wall_spec['grid_y']
                        grid_z = wall_spec['grid_z']
                        cell_size_y = size_y / grid_y
                        cell_size_z = size_z / grid_z
                        x_pos = -room_back_dist.value / 100.0 + inward_offset
                        y_c = (-size_y/2 + gj * cell_size_y + cell_size_y / 2) / 100.0
                        z_c = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        positions_list.append([x_pos, y_c, z_c])
                    
                    elif wall_name == 'left':
                        size_x = wall_spec['size_x']
                        size_z = wall_spec['size_z']
                        grid_x = wall_spec['grid_x']
                        grid_z = wall_spec['grid_z']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_z = size_z / grid_z
                        x_c = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = -side_dist / 100.0 + inward_offset
                        z_c = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        positions_list.append([x_c, y_pos, z_c])
                    
                    elif wall_name == 'right':
                        size_x = wall_spec['size_x']
                        size_z = wall_spec['size_z']
                        grid_x = wall_spec['grid_x']
                        grid_z = wall_spec['grid_z']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_z = size_z / grid_z
                        x_c = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = side_dist / 100.0 - inward_offset
                        z_c = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        positions_list.append([x_c, y_pos, z_c])
                    
                    elif wall_name == 'top':
                        size_x = wall_spec['size_x']
                        size_y = wall_spec['size_y']
                        grid_x = wall_spec['grid_x']
                        grid_y = wall_spec['grid_y']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_y = size_y / grid_y
                        x_c = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_c = (-size_y/2 + gi * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = top_bottom_dist / 100.0 - inward_offset
                        positions_list.append([x_c, y_c, z_pos])
                    
                    elif wall_name == 'bottom':
                        size_x = wall_spec['size_x']
                        size_y = wall_spec['size_y']
                        grid_x = wall_spec['grid_x']
                        grid_y = wall_spec['grid_y']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_y = size_y / grid_y
                        x_c = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_c = (-size_y/2 + gi * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = -top_bottom_dist / 100.0 + inward_offset
                        positions_list.append([x_c, y_c, z_pos])
                    else:
                        continue
                    
                    colors_list.append(color_uint8)
                    cells_created[wall_name] += 1
            
            # Create batched mesh for this wall
            if len(positions_list) > 0:
                n_cells = len(positions_list)
                positions_np = np.array(positions_list, dtype=np.float32)
                colors_np = np.array(colors_list, dtype=np.uint8)
                
                # Build wall-specific scaled quad vertices
                if wall_name == 'front':
                    cell_h = (wall_spec['size_y'] / wall_spec['grid_y']) / 100.0 * (1.0 - gap)
                    cell_v = (wall_spec['size_z'] / wall_spec['grid_z']) / 100.0 * (1.0 - gap)
                    # Quad in YZ plane, normal -X (inward)
                    wall_verts = np.array([
                        [0.0, -cell_h/2, -cell_v/2],
                        [0.0, -cell_h/2,  cell_v/2],
                        [0.0,  cell_h/2,  cell_v/2],
                        [0.0,  cell_h/2, -cell_v/2],
                    ], dtype=np.float32)
                elif wall_name == 'back':
                    cell_h = (wall_spec['size_y'] / wall_spec['grid_y']) / 100.0 * (1.0 - gap)
                    cell_v = (wall_spec['size_z'] / wall_spec['grid_z']) / 100.0 * (1.0 - gap)
                    # Quad in YZ plane, normal +X (inward)
                    wall_verts = np.array([
                        [0.0, -cell_h/2, -cell_v/2],
                        [0.0,  cell_h/2, -cell_v/2],
                        [0.0,  cell_h/2,  cell_v/2],
                        [0.0, -cell_h/2,  cell_v/2],
                    ], dtype=np.float32)
                elif wall_name in ('left', 'right'):
                    cell_h = (wall_spec['size_x'] / wall_spec['grid_x']) / 100.0 * (1.0 - gap)
                    cell_v = (wall_spec['size_z'] / wall_spec['grid_z']) / 100.0 * (1.0 - gap)
                    # Quad in XZ plane (left: normal +Y inward, right: normal -Y inward)
                    if wall_name == 'left':
                        wall_verts = np.array([
                            [-cell_h/2, 0.0, -cell_v/2],
                            [-cell_h/2, 0.0,  cell_v/2],
                            [ cell_h/2, 0.0,  cell_v/2],
                            [ cell_h/2, 0.0, -cell_v/2],
                        ], dtype=np.float32)
                    else:  # right
                        wall_verts = np.array([
                            [-cell_h/2, 0.0, -cell_v/2],
                            [ cell_h/2, 0.0, -cell_v/2],
                            [ cell_h/2, 0.0,  cell_v/2],
                            [-cell_h/2, 0.0,  cell_v/2],
                        ], dtype=np.float32)
                elif wall_name == 'top':
                    cell_h = (wall_spec['size_x'] / wall_spec['grid_x']) / 100.0 * (1.0 - gap)
                    cell_v = (wall_spec['size_y'] / wall_spec['grid_y']) / 100.0 * (1.0 - gap)
                    # Quad in XY plane, normal -Z (inward, visible from below)
                    wall_verts = np.array([
                        [-cell_h/2, -cell_v/2, 0.0],
                        [-cell_h/2,  cell_v/2, 0.0],
                        [ cell_h/2,  cell_v/2, 0.0],
                        [ cell_h/2, -cell_v/2, 0.0],
                    ], dtype=np.float32)
                else:  # bottom
                    cell_h = (wall_spec['size_x'] / wall_spec['grid_x']) / 100.0 * (1.0 - gap)
                    cell_v = (wall_spec['size_y'] / wall_spec['grid_y']) / 100.0 * (1.0 - gap)
                    # Quad in XY plane, normal +Z (inward, visible from above)
                    wall_verts = np.array([
                        [-cell_h/2, -cell_v/2, 0.0],
                        [ cell_h/2, -cell_v/2, 0.0],
                        [ cell_h/2,  cell_v/2, 0.0],
                        [-cell_h/2,  cell_v/2, 0.0],
                    ], dtype=np.float32)
                
                wall_faces = np.array([[0,1,2],[0,2,3]], dtype=np.uint32)
                q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                wxyzs_np = np.tile(q, (n_cells, 1)).astype(np.float32)
                
                handle = server.scene.add_batched_meshes_simple(
                    name=f"/room_intensity/{wall_name}",
                    vertices=wall_verts,
                    faces=wall_faces,
                    batched_wxyzs=wxyzs_np,
                    batched_positions=positions_np,
                    batched_colors=colors_np,
                    flat_shading=True,
                    side='front',
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=True,
                )
                room_intensity_handles.append(handle)
        
        print(f"Cells visualized:")
        for wall_name, count in cells_created.items():
            print(f"  {wall_name.capitalize()}: {count} cells created")
        print(f"===================================\n")
        
        # Calculate average cell area across all walls for lux to lumen conversion
        total_cells = 0
        total_area_cm2 = 0
        for wall_name, spec in wall_specs.items():
            if wall_name in ('front', 'back'):
                cell_width_cm = spec['size_y'] / spec['grid_y']
                cell_height_cm = spec['size_z'] / spec['grid_z']
            elif wall_name in ['left', 'right']:
                cell_width_cm = spec['size_x'] / spec['grid_x']
                cell_height_cm = spec['size_z'] / spec['grid_z']
            else:  # top, bottom
                cell_width_cm = spec['size_x'] / spec['grid_x']
                cell_height_cm = spec['size_y'] / spec['grid_y']
            cell_area = cell_width_cm * cell_height_cm
            num_cells = spec.get('grid_y', spec.get('grid_x', 1)) * spec.get('grid_z', spec.get('grid_y', 1))
            total_cells += num_cells
            total_area_cm2 += cell_area * num_cells
        
        avg_cell_area_cm2 = total_area_cm2 / total_cells if total_cells > 0 else 1.0
        avg_cell_area_m2 = avg_cell_area_cm2 / 10000.0

        _last_room_cache['grids'] = grids
        _last_room_cache['wall_specs'] = wall_specs
        _last_room_cache['front_dist'] = front_dist
        _last_room_cache['side_dist'] = side_dist
        _last_room_cache['top_bottom_dist'] = top_bottom_dist
        _last_room_cache['back_dist'] = room_back_dist.value if show_back_wall.value else None
        _last_room_cache['max_lux'] = max_lux
        _last_room_cache['color_scale_max'] = color_scale_max
        _last_room_cache['avg_cell_area_m2'] = avg_cell_area_m2

        legend = _build_lux_legend_html(
            max_lux, color_scale_max, avg_cell_area_m2, cell_caption="lm/cell avg",
        )
        legend_html.content = legend + _room_metrics_html()


    return SimpleNamespace(draw_room_walls=draw_room_walls, update_room_intensity_map=update_room_intensity_map)
