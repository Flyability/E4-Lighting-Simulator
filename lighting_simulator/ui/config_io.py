"""Saved-configuration I/O: read the GUI into a config dict, apply a config dict to the GUI, new project, templates.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import json
import os
import time
import numpy as np
from lighting_simulator.domain.guides import (
    bake_and_disable_guide, enable_circular_guide, guide_is_enabled as _guide_is_enabled,
    restore_group_guide, serialize_guide,
)


def build(ctx):
    _ELIOS3_SLOTS = ctx._ELIOS3_SLOTS
    _clear_mirror_state = ctx._clear_mirror_state
    _enable_mirror_for = ctx._enable_mirror_for
    _mirror_primary = ctx._mirror_primary
    _panel_dropdowns = ctx._panel_dropdowns
    _panel_slot_data = ctx._panel_slot_data
    _refresh_vio_fov_label = ctx._refresh_vio_fov_label
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
    absorbers_folder = ctx.absorbers_folder
    base_groups_active = ctx.base_groups_active
    circle_center_slider = ctx.circle_center_slider
    clear_stl_model = ctx.clear_stl_model
    create_custom_group = ctx.create_custom_group
    create_individual_led = ctx.create_individual_led
    current_config_name = ctx.current_config_name
    custom_groups = ctx.custom_groups
    global_pos_x_slider = ctx.global_pos_x_slider
    global_pos_y_slider = ctx.global_pos_y_slider
    global_pos_z_slider = ctx.global_pos_z_slider
    global_rotation_z_slider = ctx.global_rotation_z_slider
    group_colors_hex = ctx.group_colors_hex
    individual_leds = ctx.individual_leds
    led_buttons = ctx.led_buttons
    led_config_folder = ctx.led_config_folder
    led_states = ctx.led_states
    load_stl_file = ctx.load_stl_file
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
    project_loaded = ctx.project_loaded
    radius_slider = ctx.radius_slider
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
    select_panel = ctx.select_panel
    server = ctx.server
    show_led_markers = ctx.show_led_markers
    show_vio_fov = ctx.show_vio_fov
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_file_path = ctx.stl_file_path
    stl_mesh_data = ctx.stl_mesh_data
    stl_opacity = ctx.stl_opacity
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale
    stl_visible = ctx.stl_visible
    stl_wireframe = ctx.stl_wireframe
    tab_panels = ctx.tab_panels
    template_folders = ctx.template_folders
    update_scene = ctx.update_scene
    viewing_angle_slider = ctx.viewing_angle_slider
    vio_cam1_pitch = ctx.vio_cam1_pitch
    vio_cam1_yaw = ctx.vio_cam1_yaw
    vio_cam2_pitch = ctx.vio_cam2_pitch
    vio_cam2_yaw = ctx.vio_cam2_yaw
    vio_fill_fov = ctx.vio_fill_fov
    vio_landscape = ctx.vio_landscape
    vio_long_fov = ctx.vio_long_fov
    vio_pos_x = ctx.vio_pos_x
    vio_pos_y = ctx.vio_pos_y
    vio_pos_z = ctx.vio_pos_z

    def _absorber_config():
        """Absorber offsets from the GUI, in the saved-config schema."""
        return {
            "enabled": absorbers_enable.value,
            "abs0": {"x": abs0_off_x.value, "y": abs0_off_y.value, "z": abs0_off_z.value},
            "abs1": {"x": abs1_off_x.value, "y": abs1_off_y.value, "z": abs1_off_z.value},
            "abs2": {"x": abs2_off_x.value, "y": abs2_off_y.value, "z": abs2_off_z.value, "rot_z": abs2_rot_z.value},
            "abs3": {"x": abs3_off_x.value, "y": abs3_off_y.value, "z": abs3_off_z.value, "rot_z": abs3_rot_z.value},
        }

    def get_current_config():
        """Retrieve current GUI values for saving."""
        # Save custom groups
        custom_groups_data = []
        for group in custom_groups:
            # For panel slot groups, save actual rotation slider values (not baked)
            is_panel_slot = group.get('panel_slot') is not None
            group_cfg = {
                'enabled': group['enable'].value,
                'position': [group['pos_x'].value, group['pos_y'].value, group['pos_z'].value],
                'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0.0,
                'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0.0,
                'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0.0,
                'led_states': group['led_states'][:],
                'template_name': group.get('template_name'),  # Save template association
                'initial_pos': group.get('initial_pos', [0.0, 0.0, 0.0]),  # Save initial position
                'initial_rot': group.get('initial_rot', [0, 0, 0]),  # Save initial rotation
                'panel_slot': group.get('panel_slot'),  # Save panel slot index
                'panel_slot_name': group.get('panel_slot_name'),  # Save panel slot name
            }
            # Save dynamic group properties if present
            if group.get('is_dynamic', False):
                group_cfg['is_dynamic'] = True
                group_cfg['num_leds'] = group.get('num_leds', 12)
                # CRITICAL: Save ORIGINAL positions (not the current rotated ones!)
                group_cfg['led_positions'] = group.get('original_led_positions', group.get('led_positions', []))
                group_cfg['led_rotations'] = group.get('original_led_rotations', group.get('led_rotations', []))
                group_cfg['led_row_directions'] = group.get('original_led_row_directions', group.get('led_row_directions', []))
                group_cfg['led_sizes'] = group.get('led_sizes', [])
                group_cfg['led_viewing_angles'] = group.get('led_viewing_angles', [])
                group_cfg['led_rows'] = group.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
                group_cfg['led_euler_angles'] = group.get('led_euler_angles', [])
                group_cfg['led_beam_tilts'] = group.get('led_beam_tilts', [])
                group_cfg['led_lumens'] = group.get('led_lumens', [])
            # Save lumens override settings for custom group
            group_cfg['lumens_override_enabled'] = group.get('lumens_override') and group['lumens_override'].value
            group_cfg['lumens_value'] = group['lumens_value'].value if group.get('lumens_value') else 100
            if _guide_is_enabled(group):
                group_cfg['guide'] = serialize_guide(group['guide'])
            custom_groups_data.append(group_cfg)
        
        # Process individual LEDs: separate template-sourced from standalone
        template_leds = {}  # {(template_name, group_index): [leds]}
        standalone_leds = []
        
        for led in individual_leds:
            template_source = led.get('template_source')
            if template_source:
                # This LED came from a template - group it for saving as custom group
                group_index = led.get('group_index')
                key = (template_source, group_index)
                if key not in template_leds:
                    template_leds[key] = []
                template_leds[key].append(led)
            else:
                # Standalone individual LED
                standalone_leds.append(led)
        
        # Convert template-sourced LEDs back into custom groups for saving
        for (template_name, group_index), leds_list in template_leds.items():
            if not leds_list:
                continue
            
            # Sort LEDs by position to maintain consistent ordering
            leds_list_sorted = sorted(leds_list, key=lambda l: (l['pos_z'].value, l['pos_y'].value, l['pos_x'].value))
            
            # Extract LED data
            num_leds = len(leds_list_sorted)
            led_positions = [(led['pos_x'].value, led['pos_y'].value, led['pos_z'].value) for led in leds_list_sorted]
            led_sizes = [led['size'].value for led in leds_list_sorted]
            led_viewing_angles = [led['viewing_angle'].value for led in leds_list_sorted]
            group_led_states = [led['led_on'] for led in leds_list_sorted]
            
            # Convert rotation angles to direction vectors
            led_rotations = []
            led_euler_angles = []  # Store original Euler angles for lossless roundtrip
            led_beam_tilts = []  # Store beam tilt angles for lossless roundtrip
            for led in leds_list_sorted:
                rot_x = np.radians(led['rot_x'].value)
                rot_y = np.radians(led['rot_y'].value)
                rot_z = np.radians(led['rot_z'].value)
                
                # Store original Euler angles
                led_euler_angles.append((led['rot_x'].value, led['rot_y'].value, led['rot_z'].value))
                led_beam_tilts.append(led['beam_tilt'].value)
                
                # Apply rotations sequentially: Rz first, then Ry, then Rx (matches rendering)
                direction = np.array([1.0, 0.0, 0.0])
                Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0], [np.sin(rot_z), np.cos(rot_z), 0], [0, 0, 1]])
                direction = Rz @ direction
                Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
                direction = Ry @ direction
                Rx = np.array([[1, 0, 0], [0, np.cos(rot_x), -np.sin(rot_x)], [0, np.sin(rot_x), np.cos(rot_x)]])
                direction = Rx @ direction
                
                direction = direction / np.linalg.norm(direction)
                led_rotations.append(tuple(direction))
            
            # Auto-detect row organization based on Z coordinate
            z_tolerance = 0.5
            led_rows = []
            current_row = []
            current_z = None
            
            for idx, led in enumerate(leds_list_sorted):
                z = led['pos_z'].value
                if current_z is None or abs(z - current_z) < z_tolerance:
                    current_row.append(idx)
                    current_z = z if current_z is None else current_z
                else:
                    if current_row:
                        led_rows.append(current_row)
                    current_row = [idx]
                    current_z = z
            
            if current_row:
                led_rows.append(current_row)
            
            # If no rows detected, create one row with all LEDs
            if not led_rows:
                led_rows = [list(range(num_leds))]
            
            # Check if LEDs have original group position saved
            original_pos = None
            original_rot = None
            for led in leds_list_sorted:
                if led.get('original_group_pos') is not None:
                    original_pos = led['original_group_pos']
                    original_rot = led['original_group_rot']
                    break
            
            # If we have original group position, use it; otherwise calculate average
            if original_pos is not None:
                group_pos = original_pos
            else:
                # Calculate average position (center of group)
                group_pos = [
                    sum(p[0] for p in led_positions) / num_leds,
                    sum(p[1] for p in led_positions) / num_leds,
                    sum(p[2] for p in led_positions) / num_leds
                ]
            
            # ALWAYS save in world-space with zero group rotation.
            # This avoids error-prone inverse rotation transforms.
            # On reload, R_group=identity ⇒ data is used as-is.
            group_rot = [0, 0, 0]
            led_positions_relative = [(p[0] - group_pos[0], p[1] - group_pos[1], p[2] - group_pos[2]) for p in led_positions]
            
            # Compute row directions from direction vectors, applying square_roll
            led_row_directions = []
            for i, direction in enumerate(led_rotations):
                dir_arr = np.array(direction)
                row_dir = np.cross(dir_arr, np.array([0, 0, 1]))
                norm = np.linalg.norm(row_dir)
                if norm > 1e-6:
                    row_dir = row_dir / norm
                else:
                    row_dir = np.array([0, -1, 0])
                # Apply square_roll (rotation around LED direction) using Rodrigues
                sq_roll_deg = leds_list_sorted[i].get('square_roll')
                if sq_roll_deg is not None:
                    sq_val = sq_roll_deg.value if hasattr(sq_roll_deg, 'value') else sq_roll_deg
                    if abs(sq_val) > 0.01:
                        sq_rad = np.radians(sq_val)
                        k = dir_arr / np.linalg.norm(dir_arr)
                        row_dir = row_dir * np.cos(sq_rad) + np.cross(k, row_dir) * np.sin(sq_rad) + k * np.dot(k, row_dir) * (1 - np.cos(sq_rad))
                led_row_directions.append(tuple(row_dir))
            
            # Check if all LEDs are enabled
            all_enabled = all(led['enable'].value for led in leds_list_sorted)
            
            # Create custom group config
            group_cfg = {
                'enabled': all_enabled,
                'position': group_pos,
                'rotation_x': group_rot[0],
                'rotation_y': group_rot[1],
                'rotation_z': group_rot[2],
                'led_states': group_led_states,
                'is_dynamic': True,
                'num_leds': num_leds,
                'led_positions': led_positions_relative,
                'led_rotations': led_rotations,
                'led_row_directions': led_row_directions,
                'led_sizes': led_sizes,
                'led_viewing_angles': led_viewing_angles,
                'led_rows': led_rows,
                'led_euler_angles': led_euler_angles,
                'led_beam_tilts': led_beam_tilts,
                'template_name': template_name if template_name != "unnamed" else None,
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_rot': [0, 0, 0]
            }
            # Save lumens override from individual LEDs in this template group
            # Check if any LED in the group had lumens override enabled
            any_lumens_override = False
            lumens_val = 100
            for led_item in leds_list_sorted:
                if led_item.get('lumens_override') and led_item['lumens_override'].value:
                    any_lumens_override = True
                    lumens_val = led_item['lumens_value'].value if led_item.get('lumens_value') else 100
                    break
            group_cfg['lumens_override_enabled'] = any_lumens_override
            group_cfg['lumens_value'] = lumens_val
            
            custom_groups_data.append(group_cfg)
        
        # Save standalone individual LEDs
        individual_leds_data = []
        for led in standalone_leds:
            individual_leds_data.append({
                'enabled': led['enable'].value,
                'led_on': led['led_on'],
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
                'lumens_override_enabled': led.get('lumens_override') and led['lumens_override'].value,
                'lumens_value': led['lumens_value'].value if led.get('lumens_value') else 100,
                'ext_lens_enabled': led.get('ext_lens_enable') and led['ext_lens_enable'].value,
                'ext_lens_angle': led['ext_lens_angle'].value if led.get('ext_lens_angle') else 30,
                'ext_lens_efficiency': led['ext_lens_efficiency'].value if led.get('ext_lens_efficiency') else 80,
            })
        
        return {
            "viewing_angle": viewing_angle_slider.value,
            "radius": radius_slider.value,
            "circle_center_x": circle_center_slider.value,
            "group_rotations": [
                rot_front_pos.value,
                rot_front_neg.value,
                rot_side_pos.value,
                rot_side_neg.value,
            ],
            "group_rotations_y": [
                rot_y_front_pos.value,
                rot_y_front_neg.value,
                rot_y_side_pos.value,
                rot_y_side_neg.value,
            ],
            "group_offsets": [
                [offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value],
                [offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value],
                [offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value],
                [offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value],
            ],
            "row_enabled": [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            "led_states": led_states[:],
            "global_rotation_z": global_rotation_z_slider.value,
            "global_pos_x": global_pos_x_slider.value,
            "global_pos_y": global_pos_y_slider.value,
            "global_pos_z": global_pos_z_slider.value,
            "custom_groups": custom_groups_data,
            "individual_leds": individual_leds_data,
            "absorbers": _absorber_config(),
            "stl_model": {
                "file_path": stl_file_path.value,
                "absorber_enable": stl_absorber_enable.value,
                "visible": stl_visible.value,
                "scale": stl_scale.value,
                "position": [stl_pos_x.value, stl_pos_y.value, stl_pos_z.value],
                "rotation": [stl_rot_x.value, stl_rot_y.value, stl_rot_z.value],
                "opacity": stl_opacity.value,
                "wireframe": stl_wireframe.value
            } if stl_mesh_data[0] is not None else None,
            "vio_cameras": {
                "show": show_vio_fov.value,
                "fill": vio_fill_fov.value,
                "position": [vio_pos_x.value, vio_pos_y.value, vio_pos_z.value],
                "cam1_pitch": vio_cam1_pitch.value,
                "cam1_yaw": vio_cam1_yaw.value,
                "cam2_pitch": vio_cam2_pitch.value,
                "cam2_yaw": vio_cam2_yaw.value,
                "long_fov": vio_long_fov.value,
                "landscape": vio_landscape.value,
            },
            "mirror_primary": _encode_mirror_primary(),
        }

    def _encode_mirror_primary():
        owner = _mirror_primary[0]
        if owner is None:
            return None
        kind, key = owner
        if kind == 'slot':
            return {"kind": "slot", "key": int(key)}
        if kind == 'custom_group':
            for idx, g in enumerate(custom_groups):
                if g.get('id') == key:
                    return {"kind": "custom_group", "key": idx}
        return None

    def clear_all_custom_groups():
        """Remove all custom groups."""
        nonlocal custom_groups
        num_groups = len(custom_groups)
        if num_groups > 0:
            print(f"Clearing {num_groups} custom group(s)...")
        # Remove all custom groups from GUI and list
        for group in custom_groups[:]:
            try:
                group['folder'].remove()
            except (KeyError, AttributeError):
                pass
        custom_groups.clear()
        # Update scene to remove visual elements
        update_scene()
    
    def clear_all_individual_leds():
        """Remove all individual LEDs."""
        nonlocal individual_leds
        num_leds = len(individual_leds)
        if num_leds > 0:
            print(f"Clearing {num_leds} individual LED(s)...")
        # Remove all individual LEDs from GUI and list
        for led in individual_leds[:]:
            try:
                led['folder'].remove()
            except (KeyError, AttributeError):
                pass
        individual_leds.clear()
        # Update scene to remove visual elements
        update_scene()
    
    def clear_all_template_folders():
        """Remove all loaded template folders."""
        nonlocal template_folders, custom_groups
        for template_data in template_folders[:]:
            try:
                template_data['folder'].remove()
                # Also remove groups associated with this template
                for group in template_data.get('groups', []):
                    if group in custom_groups:
                        custom_groups.remove(group)
            except (KeyError, AttributeError):
                pass
        template_folders.clear()
        # Reset panel configurator slots
        for si in range(len(_panel_slot_data)):
            _panel_slot_data[si] = None
        # Reset panel configurator dropdowns (if already created)
        for dd in _panel_dropdowns:
            try:
                dd.value = "-- Nessuno --"
            except Exception:
                pass
        _clear_mirror_state()
    
    def apply_config(cfg):
        """Update GUI elements with values from config."""
        nonlocal loading_in_progress
        loading_in_progress[0] = True  # Disable callbacks during loading
        select_panel(None)
        
        # Clear all existing custom groups first (this calls update_scene())
        clear_all_custom_groups()
        clear_all_template_folders()
        
        viewing_angle_slider.value = cfg.get("viewing_angle", 120)
        radius_slider.value = cfg.get("radius", 35)
        circle_center_slider.value = cfg.get("circle_center_x", -35)
        global_rotation_z_slider.value = cfg.get("global_rotation_z", 0)
        global_pos_x_slider.value = cfg.get("global_pos_x", 0)
        global_pos_y_slider.value = cfg.get("global_pos_y", 0)
        global_pos_z_slider.value = cfg.get("global_pos_z", 0)
        
        rots = cfg.get("group_rotations", [0.7, -0.7, 18, -18])
        rot_front_pos.value = rots[0]
        rot_front_neg.value = rots[1]
        rot_side_pos.value = rots[2]
        rot_side_neg.value = rots[3]
        
        rots_y = cfg.get("group_rotations_y", [0, 0, 0, 0])
        rot_y_front_pos.value = rots_y[0]
        rot_y_front_neg.value = rots_y[1]
        rot_y_side_pos.value = rots_y[2]
        rot_y_side_neg.value = rots_y[3]
        
        offs = cfg.get("group_offsets", [[0.0, 1.6, 0.0], [0.0, -1.6, 0.0], [-1.3, -33.1, 0.0], [-1.3, 33.1, 0.0]])
        offset_front_pos_x.value = offs[0][0]
        offset_front_pos_y.value = offs[0][1]
        offset_front_pos_z.value = offs[0][2]
        offset_front_neg_x.value = offs[1][0]
        offset_front_neg_y.value = offs[1][1]
        offset_front_neg_z.value = offs[1][2]
        offset_side_pos_x.value = offs[2][0]
        offset_side_pos_y.value = offs[2][1]
        offset_side_pos_z.value = offs[2][2]
        offset_side_neg_x.value = offs[3][0]
        offset_side_neg_y.value = offs[3][1]
        offset_side_neg_z.value = offs[3][2]
        
        rows = cfg.get("row_enabled", [False, True, True, False])
        row1_chk.value = rows[0]
        row2_chk.value = rows[1]
        row3_chk.value = rows[2]
        row4_chk.value = rows[3]
        
        # Update global led_states and button appearances
        # Default to all False (no base groups active) if not specified
        nonlocal led_states
        led_states[:] = cfg.get("led_states", [False] * 48)
        update_all_led_buttons()
        
        # Recreate custom groups from config (skip intermediate scene updates)
        custom_groups_data = cfg.get("custom_groups", [])
        if len(custom_groups_data) > 0:
            print(f"Recreating {len(custom_groups_data)} custom group(s)...")
        
        # Group by template_name to recreate master folders
        # NOTE: Every group is treated as standalone so each gets its own
        # independent position/rotation/lumens controls. No shared master folder.
        groups_by_template = {}
        standalone_groups = []
        for group_cfg in custom_groups_data:
            # All groups are standalone — each gets its own independent controls
            standalone_groups.append(group_cfg)
        
        # Create standalone groups (no template)
        for group_cfg in standalone_groups:
            # Check if this is a dynamic group
            if group_cfg.get('is_dynamic', False):
                # Create dynamic group with saved properties
                group_data = create_custom_group(
                    skip_update_scene=True,
                    num_leds=group_cfg.get('num_leds', 12),
                    led_rows=group_cfg.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
                )
                # Store dynamic group properties
                group_data['is_dynamic'] = True
                group_data['led_positions'] = group_cfg.get('led_positions', [])
                group_data['led_rotations'] = group_cfg.get('led_rotations', [])
                group_data['led_row_directions'] = group_cfg.get('led_row_directions', [])
                group_data['led_euler_angles'] = group_cfg.get('led_euler_angles', [])
                group_data['led_beam_tilts'] = group_cfg.get('led_beam_tilts', [])
                group_data['led_sizes'] = group_cfg.get('led_sizes', [])
                group_data['led_viewing_angles'] = group_cfg.get('led_viewing_angles', [])
                group_data['led_lumens'] = group_cfg.get('led_lumens', [])
                # IMPORTANT: Positions in saved config are already RELATIVE
                # They were saved with original_led_positions, use directly
                group_data['original_led_positions'] = [tuple(pos) for pos in group_data['led_positions']]
                group_data['original_led_rotations'] = [tuple(rot) for rot in group_data['led_rotations']]
                if group_data['led_row_directions']:
                    group_data['original_led_row_directions'] = [tuple(rd) for rd in group_data['led_row_directions']]
                # Calculate rotation center
                if group_data['led_positions']:
                    positions_array = np.array(group_data['led_positions'])
                    group_data['rotation_center'] = tuple(positions_array.mean(axis=0))
                else:
                    group_data['rotation_center'] = (0.0, 0.0, 0.0)
            else:
                # Standard 12-LED group
                group_data = create_custom_group(skip_update_scene=True)
            
            # Apply saved position and rotation values
            pos = group_cfg.get('position', [0, 0, 0])
            group_data['pos_x'].value = pos[0]
            group_data['pos_y'].value = pos[1]
            group_data['pos_z'].value = pos[2]
            # For panel slot groups, restore actual rotation values;
            # for others, rotations are baked into led_positions/led_rotations
            # unless a construction guide is active (theta is applied on top of Euler).
            _restore_euler = (
                group_cfg.get('panel_slot') is not None
                or (isinstance(group_cfg.get('guide'), dict) and group_cfg['guide'].get('enabled'))
            )
            if _restore_euler:
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = int(round(group_cfg.get('rotation_z', 0)))
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = int(round(group_cfg.get('rotation_y', 0)))
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = int(round(group_cfg.get('rotation_x', 0)))
                if group_cfg.get('panel_slot') is not None:
                    group_data['panel_slot'] = group_cfg.get('panel_slot')
                    group_data['panel_slot_name'] = group_cfg.get('panel_slot_name')
            else:
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = 0
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = 0
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = 0
            
            # Load LED states BEFORE enabling the group
            led_states_cfg = group_cfg.get('led_states', [])
            for i, state in enumerate(led_states_cfg):
                if i < len(group_data['led_states']):
                    group_data['led_states'][i] = state
            
            # Update button colors to match loaded LED states
            if 'update_button_colors' in group_data and group_data['update_button_colors']:
                group_data['update_button_colors']()
            
            # Restore lumens override settings
            if group_data.get('lumens_override') and group_cfg.get('lumens_override_enabled'):
                group_data['lumens_override'].value = True
                group_data['lumens_value'].value = group_cfg.get('lumens_value', 100)
            
            # Enable the group AFTER all parameters are loaded
            group_data['enable'].value = group_cfg.get('enabled', True)
            restore_group_guide(group_data, group_cfg)
        
        # Recreate template folders with master controls
        for template_name, template_groups_cfg in groups_by_template.items():
            if len(template_groups_cfg) == 0:
                continue
            
            print(f"Recreating template folder: {template_name} with {len(template_groups_cfg)} group(s)")
            
            # Create master folder
            with tab_panels:
                template_folder = server.gui.add_folder(f"Template: {template_name}")
            
            with template_folder:
                master_enable = server.gui.add_checkbox("Enable All", initial_value=True)
                master_pos_x = server.gui.add_slider("Master Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_pos_y = server.gui.add_slider("Master Position Y (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_pos_z = server.gui.add_slider("Master Position Z (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_rot_x = server.gui.add_slider("Master Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
                master_rot_y = server.gui.add_slider("Master Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
                master_rot_z = server.gui.add_slider("Master Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
                server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>")
                master_lumens_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
                master_lumens_slider = server.gui.add_slider("Lumens per LED (lm)", min=1, max=900000, step=1, initial_value=100)
                server.gui.add_html("<hr style='margin:8px 0;'>")
                remove_template_btn = server.gui.add_button(f"Remove All ({len(template_groups_cfg)} groups)", color="red")
            
            # Create all groups from this template
            created_groups = []
            for group_cfg in template_groups_cfg:
                # Extract position early so it's available for both dynamic and standard groups
                pos = group_cfg.get('position', [0, 0, 0])
                # Check if this is a dynamic group
                if group_cfg.get('is_dynamic', False):
                    # Create dynamic group with saved properties
                    group_data = create_custom_group(
                        skip_update_scene=True,
                        num_leds=group_cfg.get('num_leds', 12),
                        led_rows=group_cfg.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
                    )
                    # Store dynamic group properties
                    group_data['is_dynamic'] = True
                    group_data['led_positions'] = group_cfg.get('led_positions', [])
                    group_data['led_rotations'] = group_cfg.get('led_rotations', [])
                    group_data['led_row_directions'] = group_cfg.get('led_row_directions', [])
                    group_data['led_euler_angles'] = group_cfg.get('led_euler_angles', [])
                    group_data['led_beam_tilts'] = group_cfg.get('led_beam_tilts', [])
                    group_data['led_sizes'] = group_cfg.get('led_sizes', [])
                    group_data['led_viewing_angles'] = group_cfg.get('led_viewing_angles', [])
                    # IMPORTANT: Positions in saved config are already RELATIVE
                    # They were saved from original_led_positions, so use them directly
                    group_data['original_led_positions'] = [tuple(pos) for pos in group_data['led_positions']]
                    group_data['original_led_rotations'] = [tuple(rot) for rot in group_data['led_rotations']]
                    if group_data['led_row_directions']:
                        group_data['original_led_row_directions'] = [tuple(rd) for rd in group_data['led_row_directions']]
                    # Calculate rotation center
                    if group_data['led_positions']:
                        positions_array = np.array(group_data['led_positions'])
                        group_data['rotation_center'] = tuple(positions_array.mean(axis=0))
                    else:
                        group_data['rotation_center'] = (0.0, 0.0, 0.0)
                else:
                    # Standard 12-LED group
                    group_data = create_custom_group(skip_update_scene=True)
                
                # Apply saved position and rotation values
                group_data['pos_x'].value = pos[0]
                group_data['pos_y'].value = pos[1]
                group_data['pos_z'].value = pos[2]
                # Restore saved rotation values
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = int(round(group_cfg.get('rotation_z', 0)))
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = int(round(group_cfg.get('rotation_y', 0)))
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = int(round(group_cfg.get('rotation_x', 0)))
                
                # Load LED states
                led_states_cfg = group_cfg.get('led_states', [])
                for i, state in enumerate(led_states_cfg):
                    if i < len(group_data['led_states']):
                        group_data['led_states'][i] = state
                
                # Update button colors
                if 'update_button_colors' in group_data and group_data['update_button_colors']:
                    group_data['update_button_colors']()
                
                # Restore lumens override settings
                if group_data.get('lumens_override') and group_cfg.get('lumens_override_enabled'):
                    group_data['lumens_override'].value = True
                    group_data['lumens_value'].value = group_cfg.get('lumens_value', 100)
                
                # Enable the group
                group_data['enable'].value = group_cfg.get('enabled', True)
                restore_group_guide(group_data, group_cfg)
                
                # Store template association and initial offsets
                group_data['template_name'] = template_name
                group_data['initial_pos'] = group_cfg.get('initial_pos', [0.0, 0.0, 0.0])
                group_data['initial_rot'] = group_cfg.get('initial_rot', [0, 0, 0])
                
                # Hide individual group folder - LED controls will be in master folder
                group_data['folder'].visible = False
                
                created_groups.append(group_data)
            
            # Restore master lumens from first group that has it enabled
            for group_cfg_check in template_groups_cfg:
                if group_cfg_check.get('lumens_override_enabled'):
                    master_lumens_chk.value = True
                    master_lumens_slider.value = group_cfg_check.get('lumens_value', 100)
                    break
            
            # Add LED controls in master folder for each group
            with template_folder:
                server.gui.add_html("<hr style='margin:8px 0;'><b>LED Controls:</b>")
                
                # Store button references for dynamic color updates
                master_led_buttons = []  # List of dicts with button references per group
                per_group_lumens_controls = []  # Per-group lumens UI controls
                per_group_pos_rot_controls = []  # Per-group position/rotation UI controls
                
                for group_idx, group in enumerate(created_groups):
                    with server.gui.add_folder(f"Group {group_idx + 1}"):
                        # Store button references for this group
                        group_buttons = {
                            'all_btn': None,
                            'row_btns': {},
                            'led_btns': {}
                        }
                        
                        # Per-group enable checkbox
                        _grp_enable_init = group['enable'].value
                        grp_enable_chk = server.gui.add_checkbox("Enable", initial_value=_grp_enable_init)
                        
                        # Per-group position sliders
                        server.gui.add_html("<b>Position:</b>")
                        grp_pos_x = server.gui.add_slider("Pos X (cm)", min=-100, max=100, step=0.1, initial_value=group['pos_x'].value)
                        grp_pos_y = server.gui.add_slider("Pos Y (cm)", min=-100, max=100, step=0.1, initial_value=group['pos_y'].value)
                        grp_pos_z = server.gui.add_slider("Pos Z (cm)", min=-100, max=100, step=0.1, initial_value=group['pos_z'].value)
                        
                        # Per-group rotation sliders
                        server.gui.add_html("<b>Rotation:</b>")
                        grp_rot_lr = server.gui.add_slider("Left/Right (\u00b0)", min=-180, max=180, step=1, initial_value=group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0)
                        grp_rot_ud = server.gui.add_slider("Up/Down (\u00b0)", min=-180, max=180, step=1, initial_value=group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0)
                        grp_rot_roll = server.gui.add_slider("Roll (\u00b0)", min=-180, max=180, step=1, initial_value=group['rot_roll'].value if 'rot_roll' in group else 0)
                        
                        per_group_pos_rot_controls.append({
                            'group': group,
                            'enable': grp_enable_chk,
                            'pos_x': grp_pos_x, 'pos_y': grp_pos_y, 'pos_z': grp_pos_z,
                            'rot_lr': grp_rot_lr, 'rot_ud': grp_rot_ud, 'rot_roll': grp_rot_roll
                        })
                        
                        # Per-group lumens controls
                        server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens:</b>")
                        _grp_lum_init = group['lumens_override'].value if group.get('lumens_override') else False
                        _grp_lum_val = group['lumens_value'].value if group.get('lumens_value') else 100
                        grp_lumens_chk = server.gui.add_checkbox("Custom lumens", initial_value=_grp_lum_init)
                        grp_lumens_slider = server.gui.add_slider("Lumens (lm)", min=1, max=900000, step=1, initial_value=_grp_lum_val)
                        per_group_lumens_controls.append({'chk': grp_lumens_chk, 'slider': grp_lumens_slider, 'group': group})
                        
                        server.gui.add_html("<hr style='margin:4px 0;'>")
                        
                        # ALL button for this group
                        group_all_btn = server.gui.add_button("ALL LEDs", color="#666666")
                        group_buttons['all_btn'] = group_all_btn
                        
                        server.gui.add_html("<hr style='margin:4px 0;'>")
                        
                        # Row buttons
                        led_rows = group.get('led_rows', [[0,1,2], [3,4,5], [6,7,8], [9,10,11]])
                        for row_idx, led_indices in enumerate(led_rows):
                            row_btn = server.gui.add_button(f"Row {row_idx + 1}", color="#666666")
                            group_buttons['row_btns'][row_idx] = row_btn
                            
                            # Create row handler
                            def make_row_click_handler(grp, r_idx, leds_in_row, update_colors_fn):
                                def handler(_):
                                    # Toggle all LEDs in this row
                                    all_on = all(grp['led_states'][i] for i in leds_in_row if i < len(grp['led_states']))
                                    for led_i in leds_in_row:
                                        if led_i < len(grp['led_states']):
                                            grp['led_states'][led_i] = not all_on
                                    # Update both standard and master button colors
                                    if grp.get('update_button_colors'):
                                        grp['update_button_colors']()
                                    update_colors_fn()
                                    update_scene()
                                return handler
                            
                            # Will set handler after update function is defined
                        
                        server.gui.add_html("<hr style='margin:4px 0;'>")
                        
                        # Individual LED buttons
                        num_leds = group.get('num_leds', 12)
                        for led_idx in range(num_leds):
                            initial_color = "#FF00FF" if group['led_states'][led_idx] else "#444444"
                            led_btn = server.gui.add_button(f"LED {led_idx + 1}", color=initial_color)
                            group_buttons['led_btns'][led_idx] = led_btn
                            
                            # Create LED handler
                            def make_led_click_handler(grp, l_idx, update_colors_fn):
                                def handler(_):
                                    if l_idx < len(grp['led_states']):
                                        grp['led_states'][l_idx] = not grp['led_states'][l_idx]
                                    # Update both standard and master button colors
                                    if grp.get('update_button_colors'):
                                        grp['update_button_colors']()
                                    update_colors_fn()
                                    update_scene()
                                return handler
                            
                            # Will set handler after update function is defined
                        
                        # ALL button handler
                        def make_all_click_handler(grp, update_colors_fn):
                            def handler(_):
                                # Toggle all LEDs in this group
                                all_on = all(grp['led_states'])
                                for i in range(len(grp['led_states'])):
                                    grp['led_states'][i] = not all_on
                                # Update both standard and master button colors
                                if grp.get('update_button_colors'):
                                    grp['update_button_colors']()
                                update_colors_fn()
                                update_scene()
                            return handler
                        
                        # Will set handler after update function is defined
                        
                        master_led_buttons.append({
                            'group': group,
                            'buttons': group_buttons,
                            'led_rows': led_rows
                        })
                
                # Create function to update all master button colors
                def update_master_led_button_colors():
                    """Update colors of all LED control buttons in master folder."""
                    for group_data in master_led_buttons:
                        grp = group_data['group']
                        btns = group_data['buttons']
                        led_rows = group_data['led_rows']
                        
                        # Update individual LED buttons
                        for led_idx, led_btn in btns['led_btns'].items():
                            if led_idx < len(grp['led_states']):
                                color = "#FF00FF" if grp['led_states'][led_idx] else "#444444"
                                led_btn.color = color
                        
                        # Update row buttons
                        for row_idx, led_indices in enumerate(led_rows):
                            if row_idx in btns['row_btns']:
                                any_on = any(grp['led_states'][i] for i in led_indices if i < len(grp['led_states']))
                                btns['row_btns'][row_idx].color = "#FF00FF" if any_on else "#666666"
                        
                        # Update ALL button
                        if btns['all_btn']:
                            any_on = any(grp['led_states'])
                            btns['all_btn'].color = "#FF00FF" if any_on else "#666666"
                
                # Now set all the click handlers with the update function
                for group_data in master_led_buttons:
                    grp = group_data['group']
                    btns = group_data['buttons']
                    led_rows = group_data['led_rows']
                    
                    # Set ALL button handler
                    btns['all_btn'].on_click(make_all_click_handler(grp, update_master_led_button_colors))
                    
                    # Set row button handlers
                    for row_idx, led_indices in enumerate(led_rows):
                        if row_idx in btns['row_btns']:
                            btns['row_btns'][row_idx].on_click(
                                make_row_click_handler(grp, row_idx, led_indices, update_master_led_button_colors)
                            )
                    
                    # Set LED button handlers
                    for led_idx, led_btn in btns['led_btns'].items():
                        led_btn.on_click(make_led_click_handler(grp, led_idx, update_master_led_button_colors))
                
                # Initial color update
                update_master_led_button_colors()
                
                # Wire per-group lumens handlers
                def make_group_lumens_handler(grp, chk, slider):
                    def handler(_):
                        if loading_in_progress[0]:
                            return
                        if grp.get('lumens_override'):
                            grp['lumens_override'].value = chk.value
                        if grp.get('lumens_value'):
                            grp['lumens_value'].value = slider.value
                        update_scene()
                    return handler
                
                for pg_ctrl in per_group_lumens_controls:
                    _h = make_group_lumens_handler(pg_ctrl['group'], pg_ctrl['chk'], pg_ctrl['slider'])
                    pg_ctrl['chk'].on_update(_h)
                    pg_ctrl['slider'].on_update(_h)
                
                # Wire per-group position/rotation handlers
                def make_group_pos_rot_handler(grp, ctrl):
                    def handler(_):
                        if loading_in_progress[0]:
                            return
                        grp['enable'].value = ctrl['enable'].value
                        grp['pos_x'].value = ctrl['pos_x'].value
                        grp['pos_y'].value = ctrl['pos_y'].value
                        grp['pos_z'].value = ctrl['pos_z'].value
                        if 'rot_tilt_lr' in grp:
                            grp['rot_tilt_lr'].value = int(ctrl['rot_lr'].value)
                        if 'rot_tilt_ud' in grp:
                            grp['rot_tilt_ud'].value = int(ctrl['rot_ud'].value)
                        if 'rot_roll' in grp:
                            grp['rot_roll'].value = int(ctrl['rot_roll'].value)
                        # Apply rotation if group supports it
                        if grp.get('apply_rotation'):
                            grp['apply_rotation']()
                        update_scene()
                    return handler
                
                for pg_ctrl in per_group_pos_rot_controls:
                    _h = make_group_pos_rot_handler(pg_ctrl['group'], pg_ctrl)
                    pg_ctrl['enable'].on_update(_h)
                    pg_ctrl['pos_x'].on_update(_h)
                    pg_ctrl['pos_y'].on_update(_h)
                    pg_ctrl['pos_z'].on_update(_h)
                    pg_ctrl['rot_lr'].on_update(_h)
                    pg_ctrl['rot_ud'].on_update(_h)
                    pg_ctrl['rot_roll'].on_update(_h)
            
            # Setup master control callbacks
            def make_update_handler(groups_list, m_enable, m_pos_x, m_pos_y, m_pos_z, m_rot_x, m_rot_y, m_rot_z, pg_pr_controls):
                def update_all_from_master(_):
                    if loading_in_progress[0]:
                        return
                    loading_in_progress[0] = True
                    
                    master_pos_offset = np.array([m_pos_x.value, m_pos_y.value, m_pos_z.value])
                    
                    # Build master rotation matrix (extrinsic X-Y-Z)
                    roll_rad = np.radians(m_rot_x.value)
                    pitch_rad = np.radians(m_rot_y.value)
                    yaw_rad = np.radians(m_rot_z.value)
                    Rx = np.array([[1,0,0],[0,np.cos(roll_rad),-np.sin(roll_rad)],[0,np.sin(roll_rad),np.cos(roll_rad)]])
                    Ry = np.array([[np.cos(pitch_rad),0,np.sin(pitch_rad)],[0,1,0],[-np.sin(pitch_rad),0,np.cos(pitch_rad)]])
                    Rz = np.array([[np.cos(yaw_rad),-np.sin(yaw_rad),0],[np.sin(yaw_rad),np.cos(yaw_rad),0],[0,0,1]])
                    R_master = Rz @ Ry @ Rx
                    
                    # Compute template center (centroid of all initial positions)
                    all_init_pos = [np.array(g.get('initial_pos', [0.0, 0.0, 0.0])) for g in groups_list]
                    template_center = np.mean(all_init_pos, axis=0) if all_init_pos else np.zeros(3)
                    
                    for gi, group in enumerate(groups_list):
                        group['enable'].value = m_enable.value
                        if m_enable.value:
                            init_pos = np.array(group.get('initial_pos', [0.0, 0.0, 0.0]))
                            init_rot = np.array(group.get('initial_rot', [0, 0, 0]))
                            
                            # RIGID BODY: rotate group position around template center
                            rotated_pos = R_master @ (init_pos - template_center) + template_center + master_pos_offset
                            group['pos_x'].value = float(rotated_pos[0])
                            group['pos_y'].value = float(rotated_pos[1])
                            group['pos_z'].value = float(rotated_pos[2])
                            
                            # Keep per-group rotation sliders at initial values
                            group['rot_roll'].value = int(init_rot[0])
                            group['rot_tilt_ud'].value = int(init_rot[1])
                            group['rot_tilt_lr'].value = int(init_rot[2])
                            
                            # Sync per-group UI sliders
                            if gi < len(pg_pr_controls):
                                pg_pr_controls[gi]['enable'].value = m_enable.value
                                pg_pr_controls[gi]['pos_x'].value = float(rotated_pos[0])
                                pg_pr_controls[gi]['pos_y'].value = float(rotated_pos[1])
                                pg_pr_controls[gi]['pos_z'].value = float(rotated_pos[2])
                                pg_pr_controls[gi]['rot_roll'].value = int(init_rot[0])
                                pg_pr_controls[gi]['rot_ud'].value = int(init_rot[1])
                                pg_pr_controls[gi]['rot_lr'].value = int(init_rot[2])
                            
                            # RIGID BODY: rotate internal LED geometry with R_master
                            # Same approach as apply_rotation_transform in elios3_pannel
                            if group.get('is_dynamic', False):
                                # Get the base originals (stored at template load time)
                                base_positions = group.get('_master_base_led_positions', group.get('original_led_positions'))
                                base_rotations = group.get('_master_base_led_rotations', group.get('original_led_rotations'))
                                base_row_dirs = group.get('_master_base_led_row_directions', group.get('original_led_row_directions'))
                                
                                # Store base originals once (before any master rotation)
                                if '_master_base_led_positions' not in group and base_positions is not None:
                                    group['_master_base_led_positions'] = [tuple(p) for p in base_positions]
                                if '_master_base_led_rotations' not in group and base_rotations is not None:
                                    group['_master_base_led_rotations'] = [tuple(r) for r in base_rotations]
                                if '_master_base_led_row_directions' not in group and base_row_dirs is not None:
                                    group['_master_base_led_row_directions'] = [tuple(rd) for rd in base_row_dirs]
                                
                                base_positions = group.get('_master_base_led_positions')
                                base_rotations = group.get('_master_base_led_rotations')
                                base_row_dirs = group.get('_master_base_led_row_directions')
                                
                                # Rotate positions around LED centroid (local axes, not global origin)
                                if base_positions is not None:
                                    led_center = np.mean([np.array(p) for p in base_positions], axis=0)
                                    rotated = [tuple(R_master @ (np.array(p) - led_center) + led_center) for p in base_positions]
                                    group['original_led_positions'] = rotated
                                    group['led_positions'] = rotated
                                
                                # Rotate directions with R_master
                                if base_rotations is not None:
                                    rotated = [tuple(R_master @ np.array(r)) for r in base_rotations]
                                    group['original_led_rotations'] = rotated
                                    group['led_rotations'] = rotated
                                
                                # Rotate row directions with R_master
                                if base_row_dirs is not None:
                                    rotated = [tuple(R_master @ np.array(rd)) for rd in base_row_dirs]
                                    group['original_led_row_directions'] = rotated
                                    group['led_row_directions'] = rotated
                    
                    loading_in_progress[0] = False
                    update_scene()
                return update_all_from_master
            
            def make_remove_handler(groups_list, folder):
                def remove_all(_):
                    for group in groups_list:
                        custom_groups.remove(group)
                        group['folder'].remove()
                    folder.remove()
                    for template_data in template_folders[:]:
                        if template_data['folder'] == folder:
                            template_folders.remove(template_data)
                    update_scene()
                return remove_all
            
            remove_handler = make_remove_handler(created_groups, template_folder)
            
            # Lumens override callback: propagate master lumens to all sub-groups
            def make_lumens_handler(groups_list, m_lumens_chk, m_lumens_slider, pg_controls):
                def update_lumens(_):
                    if loading_in_progress[0]:
                        return
                    loading_in_progress[0] = True
                    for i, group in enumerate(groups_list):
                        if group.get('lumens_override'):
                            group['lumens_override'].value = m_lumens_chk.value
                        if group.get('lumens_value'):
                            group['lumens_value'].value = m_lumens_slider.value
                        # Sync per-group UI controls
                        if i < len(pg_controls):
                            pg_controls[i]['chk'].value = m_lumens_chk.value
                            pg_controls[i]['slider'].value = m_lumens_slider.value
                    loading_in_progress[0] = False
                    update_scene()
                return update_lumens
            
            update_handler = make_update_handler(created_groups, master_enable, master_pos_x, master_pos_y, master_pos_z, master_rot_x, master_rot_y, master_rot_z, per_group_pos_rot_controls)
            lumens_handler = make_lumens_handler(created_groups, master_lumens_chk, master_lumens_slider, per_group_lumens_controls)
            
            master_enable.on_update(update_handler)
            master_pos_x.on_update(update_handler)
            master_pos_y.on_update(update_handler)
            master_pos_z.on_update(update_handler)
            master_rot_x.on_update(update_handler)
            master_rot_y.on_update(update_handler)
            master_rot_z.on_update(update_handler)
            master_lumens_chk.on_update(lumens_handler)
            master_lumens_slider.on_update(lumens_handler)
            remove_template_btn.on_click(remove_handler)
            
            # Store template folder data
            template_folders.append({
                'folder': template_folder,
                'groups': created_groups
            })
        
        # Load absorbers configuration if present
        absorbers_cfg = cfg.get("absorbers", {})
        if absorbers_cfg:
            absorbers_enable.value = absorbers_cfg.get('enabled', False)
            
            abs0_data = absorbers_cfg.get('abs0', {})
            abs0_off_x.value = abs0_data.get('x', -1)
            abs0_off_y.value = abs0_data.get('y', 2.5)
            abs0_off_z.value = abs0_data.get('z', 0.0)
            
            abs1_data = absorbers_cfg.get('abs1', {})
            abs1_off_x.value = abs1_data.get('x', -1)
            abs1_off_y.value = abs1_data.get('y', -2.5)
            abs1_off_z.value = abs1_data.get('z', 0.0)
            
            abs2_data = absorbers_cfg.get('abs2', {})
            abs2_off_x.value = abs2_data.get('x', -1.8)
            abs2_off_y.value = abs2_data.get('y', -10.5)
            abs2_off_z.value = abs2_data.get('z', 0.0)
            abs2_rot_z.value = abs2_data.get('rot_z', -14)
            
            abs3_data = absorbers_cfg.get('abs3', {})
            abs3_off_x.value = abs3_data.get('x', -1.8)
            abs3_off_y.value = abs3_data.get('y', 10.5)
            abs3_off_z.value = abs3_data.get('z', 0.0)
            abs3_rot_z.value = abs3_data.get('rot_z', 14)
        
        # Load STL model configuration if present
        stl_cfg = cfg.get("stl_model")
        if stl_cfg:
            # Clear existing model first
            clear_stl_model()
            
            # Load file path and try to load the model
            file_path = stl_cfg.get('file_path', '')
            if file_path and os.path.exists(file_path):
                stl_file_path.value = file_path
                load_stl_file()  # Load the mesh
                
                # Apply saved settings
                stl_absorber_enable.value = stl_cfg.get('absorber_enable', True)
                stl_visible.value = stl_cfg.get('visible', True)
                stl_scale.value = stl_cfg.get('scale', 1.0)
                
                position = stl_cfg.get('position', [0, 0, 0])
                stl_pos_x.value = position[0]
                stl_pos_y.value = position[1]
                stl_pos_z.value = position[2]
                
                rotation = stl_cfg.get('rotation', [0, 0, 0])
                stl_rot_x.value = rotation[0]
                stl_rot_y.value = rotation[1]
                stl_rot_z.value = rotation[2]
                
                stl_opacity.value = stl_cfg.get('opacity', 0.8)
                stl_wireframe.value = stl_cfg.get('wireframe', False)
                
                print(f"✓ STL model loaded from config: {os.path.basename(file_path)}")
            else:
                if file_path:
                    print(f"⚠️ STL file not found: {file_path}")
        else:
            # No STL model in config, clear any existing model
            clear_stl_model()
        
        # Recreate individual LEDs from config (skip intermediate scene updates)
        clear_all_individual_leds()
        individual_leds_data = cfg.get("individual_leds", [])
        if len(individual_leds_data) > 0:
            print(f"Recreating {len(individual_leds_data)} individual LED(s)...")
        for led_cfg in individual_leds_data:
            led_data = create_individual_led(skip_update_scene=True)
            # Apply saved values
            led_data['enable'].value = led_cfg.get('enabled', True)
            led_data['led_on'] = led_cfg.get('led_on', True)
            # Update button color to match state
            led_data['led_on_btn'].color = "#00FFFF" if led_data['led_on'] else "#444444"
            led_data['pos_x'].value = led_cfg.get('pos_x', 0.0)
            led_data['pos_y'].value = led_cfg.get('pos_y', 0.0)
            led_data['pos_z'].value = led_cfg.get('pos_z', 0.0)
            led_data['rot_x'].value = led_cfg.get('rot_x', 0.0)
            led_data['rot_y'].value = led_cfg.get('rot_y', 0.0)
            led_data['rot_z'].value = led_cfg.get('rot_z', 0.0)
            led_data['size'].value = led_cfg.get('size', 0.5)
            led_data['viewing_angle'].value = led_cfg.get('viewing_angle', 120)
            led_data['square_roll'].value = led_cfg.get('square_roll', 0)
            led_data['beam_tilt'].value = led_cfg.get('beam_tilt', 0)
            # Restore lumens override settings for individual LED
            if led_data.get('lumens_override') and led_cfg.get('lumens_override_enabled'):
                led_data['lumens_override'].value = True
                led_data['lumens_value'].value = led_cfg.get('lumens_value', 100)
            # Restore external lens settings
            if led_data.get('ext_lens_enable') and led_cfg.get('ext_lens_enabled'):
                led_data['ext_lens_enable'].value = True
                led_data['ext_lens_angle'].value = led_cfg.get('ext_lens_angle', 30)
                led_data['ext_lens_efficiency'].value = led_cfg.get('ext_lens_efficiency', 80)
        
        vio_cfg = cfg.get("vio_cameras")
        if vio_cfg:
            show_vio_fov.value = vio_cfg.get("show", True)
            vio_fill_fov.value = vio_cfg.get("fill", False)
            pos = vio_cfg.get("position", [3.0, 0.0, 0.0])
            vio_pos_x.value = pos[0]
            vio_pos_y.value = pos[1] if len(pos) > 1 else 0.0
            vio_pos_z.value = pos[2] if len(pos) > 2 else 0.0
            vio_cam1_pitch.value = vio_cfg.get("cam1_pitch", 45)
            vio_cam1_yaw.value = vio_cfg.get("cam1_yaw", 0)
            vio_cam2_pitch.value = vio_cfg.get("cam2_pitch", -45)
            vio_cam2_yaw.value = vio_cfg.get("cam2_yaw", 0)
            vio_long_fov.value = vio_cfg.get("long_fov", 170)
            vio_landscape.value = vio_cfg.get("landscape", True)
            _refresh_vio_fov_label()
        
        # Restore mirror-primary panel (old configs omit this key → no-op)
        _clear_mirror_state()
        mp = cfg.get("mirror_primary")
        if isinstance(mp, dict):
            owner = None
            if mp.get("kind") == "slot":
                try:
                    si = int(mp.get("key"))
                    if 0 <= si < len(_ELIOS3_SLOTS):
                        owner = ('slot', si)
                except (TypeError, ValueError):
                    pass
            elif mp.get("kind") == "custom_group":
                try:
                    gi = int(mp.get("key"))
                    if 0 <= gi < len(custom_groups):
                        owner = ('custom_group', custom_groups[gi]['id'])
                except (TypeError, ValueError):
                    pass
            if owner is not None:
                _enable_mirror_for(owner)

        # Re-enable callbacks and do final scene update
        loading_in_progress[0] = False
        
        # Final scene update after all groups are recreated
        update_scene()
        
        # Force refresh LED markers to ensure all handles are created correctly
        if show_led_markers.value:
            show_led_markers.value = False
            time.sleep(0.05)
            update_scene()
            time.sleep(0.05)
            show_led_markers.value = True
            update_scene()
        
        # Update UI visibility indicators
        update_ui_visibility()

    def update_all_led_buttons():
        """Sync button colors with current led_states."""
        for i in range(48):
            color = group_colors_hex[i // 12] if led_states[i] else "#444444"
            led_buttons[i].color = color
    
    def update_ui_visibility():
        """Update UI folder visibility based on current project configuration."""
        nonlocal led_config_folder, absorbers_folder, current_config_name
        
        # Check if any base LED groups are active
        any_base_leds = any(led_states[:48])
        base_groups_active[0] = any_base_leds
        
        # Show/hide LED Configuration folder based on base LED state
        led_config_folder.visible = any_base_leds
        
        # Show/hide Absorbers folder only when elios3 is loaded
        absorbers_folder.visible = current_config_name[0] == "elios3"

    def new_project():
        """Initialize a new empty project with default geometry settings."""
        nonlocal led_states, project_loaded, current_config_name, led_config_folder, absorbers_folder, loading_in_progress
        loading_in_progress[0] = True  # Disable callbacks during reset
        
        print("Creating new empty project...")
        
        select_panel(None)
        
        # Clear current config name
        current_config_name[0] = ""
        
        # Disable all LEDs first (before clearing custom groups)
        led_states[:] = [False] * 48
        update_all_led_buttons()
        
        # Clear custom groups and individual LEDs (this calls update_scene() with led_states already disabled)
        clear_all_custom_groups()
        clear_all_individual_leds()
        clear_all_template_folders()
        
        # Reset to default geometry values
        viewing_angle_slider.value = 120
        radius_slider.value = 35
        circle_center_slider.value = -35
        global_rotation_z_slider.value = 0
        global_pos_x_slider.value = 0
        global_pos_y_slider.value = 0
        global_pos_z_slider.value = 0
        
        # Reset all group rotations to 0
        rot_front_pos.value = 0.0
        rot_front_neg.value = 0.0
        rot_side_pos.value = 0.0
        rot_side_neg.value = 0.0
        
        rot_y_front_pos.value = 0.0
        rot_y_front_neg.value = 0.0
        rot_y_side_pos.value = 0.0
        rot_y_side_neg.value = 0.0
        
        # Reset all group offsets to 0
        offset_front_pos_x.value = 0.0
        offset_front_pos_y.value = 0.0
        offset_front_pos_z.value = 0.0
        offset_front_neg_x.value = 0.0
        offset_front_neg_y.value = 0.0
        offset_front_neg_z.value = 0.0
        offset_side_pos_x.value = 0.0
        offset_side_pos_y.value = 0.0
        offset_side_pos_z.value = 0.0
        offset_side_neg_x.value = 0.0
        offset_side_neg_y.value = 0.0
        offset_side_neg_z.value = 0.0
        
        # Disable all rows
        row1_chk.value = False
        row2_chk.value = False
        row3_chk.value = False
        row4_chk.value = False
        
        # Reset absorbers
        absorbers_enable.value = False
        abs0_off_x.value = -1
        abs0_off_y.value = 2.5
        abs0_off_z.value = 0.0
        abs1_off_x.value = -1
        abs1_off_y.value = -2.5
        abs1_off_z.value = 0.0
        abs2_off_x.value = -1.8
        abs2_off_y.value = -10.5
        abs2_off_z.value = 0.0
        abs2_rot_z.value = -14
        abs3_off_x.value = -1.8
        abs3_off_y.value = 10.5
        abs3_off_z.value = 0.0
        abs3_rot_z.value = 14
        
        # Clear STL model
        clear_stl_model()
        stl_file_path.value = ""
        stl_absorber_enable.value = True
        stl_visible.value = True
        stl_scale.value = 1.0
        stl_pos_x.value = 0.0
        stl_pos_y.value = 0.0
        stl_pos_z.value = 0.0
        stl_rot_x.value = 0.0
        stl_rot_y.value = 0.0
        stl_rot_z.value = 0.0
        stl_opacity.value = 0.8
        stl_wireframe.value = False
        
        # Re-enable callbacks
        loading_in_progress[0] = False
        
        # Update scene to confirm empty project (led_states already disabled above)
        update_scene()
        
        # Refresh LED markers to ensure clean state
        show_led_markers.value = False
        time.sleep(0.1)
        show_led_markers.value = True
        
        # Explicitly hide both UI folders before updating visibility
        led_config_folder.visible = False
        absorbers_folder.visible = False
        
        # Update UI visibility indicators (should keep them hidden)
        update_ui_visibility()
        
        project_loaded[0] = True
        print("✓ New empty project created - Add custom LED groups to get started!")
    


    return SimpleNamespace(_absorber_config=_absorber_config, apply_config=apply_config, get_current_config=get_current_config, new_project=new_project, update_all_led_buttons=update_all_led_buttons, update_ui_visibility=update_ui_visibility)
