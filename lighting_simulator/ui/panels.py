"""Panel system: custom groups, individual LEDs, template loading, panel slots and XZ mirroring helpers.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import json
import os
import time
import numpy as np
from lighting_simulator.domain.geometry import as_vec3 as _as_vec3
from lighting_simulator.domain.guides import (
    bake_and_disable_guide, enable_circular_guide, guide_is_enabled as _guide_is_enabled,
    restore_group_guide, serialize_guide,
)
from lighting_simulator.domain.mirroring import expand_mirror_configs


def build(ctx):
    _ELIOS3_SLOTS = ctx._ELIOS3_SLOTS
    _Rz_matrix = ctx._Rz_matrix
    _clear_mirror_state = ctx._clear_mirror_state
    _mirror_primary = ctx._mirror_primary
    _panel_dropdowns = ctx._panel_dropdowns
    _panel_slot_data = ctx._panel_slot_data
    custom_groups = ctx.custom_groups
    custom_groups_folder = ctx.custom_groups_folder
    custom_groups_templates_dir = ctx.custom_groups_templates_dir
    individual_leds = ctx.individual_leds
    individual_leds_folder = ctx.individual_leds_folder
    loading_in_progress = ctx.loading_in_progress
    next_custom_group_id = ctx.next_custom_group_id
    next_individual_led_id = ctx.next_individual_led_id
    select_panel = ctx.select_panel
    selected_owner = ctx.selected_owner
    server = ctx.server
    show_led_markers = ctx.show_led_markers
    tab_panels = ctx.tab_panels
    template_folders = ctx.template_folders
    update_scene = ctx.update_scene

    def load_custom_group_from_template(template_name):
        """Load all custom groups and individual LEDs from template and add them to the scene."""
        nonlocal loading_in_progress
        loading_in_progress[0] = True  # Disable callbacks during loading
        
        path = os.path.join(custom_groups_templates_dir, f"{template_name}.json")
        if not os.path.exists(path):
            print(f"Error: Template '{template_name}' not found")
            return None
        
        with open(path, "r") as f:
            template = json.load(f)
        
        # Check if this is a new multi-group template or old single-group template
        groups_data = template.get('groups', [])
        if not groups_data and 'enabled' in template:
            # Old format - single group template
            groups_data = [{
                'enabled': template.get('enabled', True),
                'position': template.get('position', [0, 0, 0]),
                'rotation_y': template.get('rotation_y', 0),
                'rotation_z': template.get('rotation_z', 0),
                'led_states': template.get('led_states', [True] * 12)
            }]
        
        # Create a master folder with shared controls for this template
        with tab_panels:
            template_folder = server.gui.add_folder(f"Template: {template.get('name', template_name)}")
        
        with template_folder:
            master_enable = server.gui.add_checkbox("Enable All", initial_value=True)
            master_pos_x = server.gui.add_slider("Master Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
            master_pos_y = server.gui.add_slider("Master Position Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            master_pos_z = server.gui.add_slider("Master Position Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            master_rot_x = server.gui.add_slider("Master Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
            master_rot_y = server.gui.add_slider("Master Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
            master_rot_z = server.gui.add_slider("Master Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
            server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>")
            master_lumens_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
            master_lumens_slider = server.gui.add_slider("Lumens per LED (lm)", min=1, max=900000, step=1, initial_value=100)
            server.gui.add_html("<hr style='margin:8px 0;'>")
            remove_template_btn = server.gui.add_button("Remove Template", color="red")
        
        # Create all groups from template
        created_groups = []
        initial_positions = []  # Store initial offset for each group
        initial_rotations = []
        
        for group_cfg in groups_data:
            # Check if this is a dynamic group (from individual LEDs)
            if 'num_leds' in group_cfg and 'led_rows' in group_cfg:
                # Dynamic group with custom LED organization
                group_data = create_custom_group(
                    skip_update_scene=True,
                    num_leds=group_cfg['num_leds'],
                    led_rows=group_cfg['led_rows'],
                    group_name=group_cfg.get('name', None)
                )
            else:
                # Standard 12-LED group
                group_data = create_custom_group(skip_update_scene=True)
            
            # Load position and rotation
            pos = group_cfg.get('position', [0, 0, 0])
            group_data['pos_x'].value = pos[0]
            group_data['pos_y'].value = pos[1]
            group_data['pos_z'].value = pos[2]
            # Rotations: restore from template config (may be non-zero now)
            if 'rot_tilt_lr' in group_data:
                group_data['rot_tilt_lr'].value = int(round(group_cfg.get('rotation_z', 0)))
            if 'rot_tilt_ud' in group_data:
                group_data['rot_tilt_ud'].value = int(round(group_cfg.get('rotation_y', 0)))
            if 'rot_roll' in group_data:
                group_data['rot_roll'].value = int(round(group_cfg.get('rotation_x', 0)))
            
            # Load dynamic group properties if present
            if group_cfg.get('is_dynamic', False):
                group_data['is_dynamic'] = True
                group_data['led_positions'] = group_cfg.get('led_positions', [])
                group_data['led_rotations'] = group_cfg.get('led_rotations', [])
                group_data['led_row_directions'] = group_cfg.get('led_row_directions', [])
                group_data['led_euler_angles'] = group_cfg.get('led_euler_angles', [])
                group_data['led_beam_tilts'] = group_cfg.get('led_beam_tilts', [])
                group_data['led_sizes'] = group_cfg.get('led_sizes', [])
                group_data['led_viewing_angles'] = group_cfg.get('led_viewing_angles', [])
                group_data['led_lumens'] = group_cfg.get('led_lumens', [])
                # IMPORTANT: Positions in template are already RELATIVE (not absolute!)
                # They were saved with original_led_positions, so use them directly
                group_data['original_led_positions'] = [tuple(pos) for pos in group_data['led_positions']]
                group_data['original_led_rotations'] = [tuple(rot) for rot in group_data['led_rotations']]
                if group_data['led_row_directions']:
                    group_data['original_led_row_directions'] = [tuple(rd) for rd in group_data['led_row_directions']]
            
            # Load LED states BEFORE enabling the group to avoid IndexError
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
                # Also set master lumens controls to match
                master_lumens_chk.value = True
                master_lumens_slider.value = group_cfg.get('lumens_value', 100)
            
            # Enable the group AFTER all parameters are loaded
            group_data['enable'].value = group_cfg.get('enabled', True)
            restore_group_guide(group_data, group_cfg)
            
            # Store initial offset for this group
            init_pos = [pos[0], pos[1], pos[2]]
            init_rot = [
                group_cfg.get('rotation_x', 0),
                group_cfg.get('rotation_y', 0),
                group_cfg.get('rotation_z', 0)
            ]
            initial_positions.append(init_pos)
            initial_rotations.append(init_rot)
            
            # Mark as belonging to this template
            group_data['template_name'] = template.get('name', template_name)
            group_data['initial_pos'] = init_pos
            group_data['initial_rot'] = init_rot
            
            # Hide individual group folder - LED controls will be in master folder
            group_data['folder'].visible = False
            
            created_groups.append(group_data)
        
        # Check if there are individual LEDs to convert into a custom group
        individual_leds_data = template.get('individual_leds', [])
        if individual_leds_data:
            # Convert individual LEDs to a custom group
            # 1. Sort by Z coordinate (descending - highest first)
            sorted_leds = sorted(individual_leds_data, key=lambda led: led.get('pos_z', 0.0), reverse=True)
            
            # 2. Group LEDs by Z coordinate (with tolerance of 0.5 cm)
            z_tolerance = 0.5
            led_rows_indices = []
            current_row = []
            current_z = None
            
            for idx, led in enumerate(sorted_leds):
                led_z = led.get('pos_z', 0.0)
                if current_z is None or abs(led_z - current_z) <= z_tolerance:
                    # Same row
                    current_row.append(idx)
                    if current_z is None:
                        current_z = led_z
                else:
                    # New row
                    if current_row:
                        led_rows_indices.append(current_row)
                    current_row = [idx]
                    current_z = led_z
            
            # Don't forget last row
            if current_row:
                led_rows_indices.append(current_row)
            
            num_leds = len(sorted_leds)
            
            # 3. Create LED states array (all on by default)
            initial_led_states = [led.get('led_on', True) for led in sorted_leds]
            
            # 4. Extract LED positions and sizes, convert rotations to direction vectors
            led_positions = [(led.get('pos_x', 0.0), led.get('pos_y', 0.0), led.get('pos_z', 0.0)) for led in sorted_leds]
            led_sizes = [led.get('size', 0.5) for led in sorted_leds]
            led_viewing_angles = [led.get('viewing_angle', 120) for led in sorted_leds]
            
            # Convert rotation angles to direction vectors (saves correctly for group rotation)
            led_rotations = []
            for led in sorted_leds:
                rot_x_deg = led.get('rot_x', 0.0)
                rot_y_deg = led.get('rot_y', 0.0)
                rot_z_deg = led.get('rot_z', 0.0)
                
                # Start with direction pointing along +X axis
                direction = np.array([1.0, 0.0, 0.0])
                
                # Apply LED's rotations: Z, then Y, then X
                rot_z_rad = np.radians(rot_z_deg)
                Rz = np.array([
                    [np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
                    [np.sin(rot_z_rad), np.cos(rot_z_rad), 0],
                    [0, 0, 1]
                ])
                direction = Rz @ direction
                
                rot_y_rad = np.radians(rot_y_deg)
                Ry = np.array([
                    [np.cos(rot_y_rad), 0, np.sin(rot_y_rad)],
                    [0, 1, 0],
                    [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]
                ])
                direction = Ry @ direction
                
                rot_x_rad = np.radians(rot_x_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
                    [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]
                ])
                direction = Rx @ direction
                
                # Normalize and store as direction vector
                direction = direction / np.linalg.norm(direction)
                led_rotations.append(tuple(direction))
            
            # Compute row_direction for each LED, applying square_roll if present
            led_row_directions = []
            for i, direction_tuple in enumerate(led_rotations):
                d = np.array(direction_tuple)
                # Default row_dir perpendicular to LED direction
                row_dir = np.cross(d, [0, 0, 1])
                if np.linalg.norm(row_dir) < 0.01:
                    row_dir = np.cross(d, [0, 1, 0])
                row_dir = row_dir / np.linalg.norm(row_dir)
                # Apply square_roll (rotation around LED direction) using Rodrigues
                sq_roll_deg = sorted_leds[i].get('square_roll', 0.0)
                if abs(sq_roll_deg) > 0.01:
                    sq_rad = np.radians(sq_roll_deg)
                    k = d / np.linalg.norm(d)
                    row_dir = row_dir * np.cos(sq_rad) + np.cross(k, row_dir) * np.sin(sq_rad) + k * np.dot(k, row_dir) * (1 - np.cos(sq_rad))
                led_row_directions.append(tuple(row_dir))
            
            # 5. Create custom group with dynamic structure
            group_data = create_custom_group(
                skip_update_scene=True,
                num_leds=num_leds,
                led_rows=led_rows_indices,
                group_name=f"{template.get('name', template_name)}"
            )
            
            # Store additional data for dynamic group
            group_data['is_dynamic'] = True
            group_data['led_positions'] = led_positions
            group_data['led_rotations'] = led_rotations
            group_data['led_sizes'] = led_sizes
            group_data['led_viewing_angles'] = led_viewing_angles
            # IMPORTANT: Save original positions/rotations immediately (never modify these)
            group_data['original_led_positions'] = [tuple(pos) for pos in led_positions]
            group_data['original_led_rotations'] = [tuple(rot) for rot in led_rotations]
            group_data['led_row_directions'] = led_row_directions
            group_data['original_led_row_directions'] = [tuple(rd) for rd in led_row_directions]
            # Store original Euler angles for lossless roundtrip
            led_euler_angles = [(led.get('rot_x', 0.0), led.get('rot_y', 0.0), led.get('rot_z', 0.0)) for led in sorted_leds]
            group_data['led_euler_angles'] = led_euler_angles
            # Store beam tilt angles for lossless roundtrip
            led_beam_tilts = [led.get('beam_tilt', 0.0) for led in sorted_leds]
            group_data['led_beam_tilts'] = led_beam_tilts
            # Calculate rotation center
            if led_positions:
                positions_array = np.array(led_positions)
                group_data['rotation_center'] = tuple(positions_array.mean(axis=0))
            else:
                group_data['rotation_center'] = (0.0, 0.0, 0.0)
            
            # Set LED states
            for i, state in enumerate(initial_led_states):
                if i < len(group_data['led_states']):
                    group_data['led_states'][i] = state
            
            # Update button colors
            if 'update_button_colors' in group_data and group_data['update_button_colors']:
                group_data['update_button_colors']()
            
            # Store initial offset (individual LEDs converted to group at 0,0,0)
            initial_positions.append([0.0, 0.0, 0.0])
            initial_rotations.append([0, 0, 0])
            
            # Mark as belonging to this template
            group_data['template_name'] = template.get('name', template_name)
            group_data['initial_pos'] = [0.0, 0.0, 0.0]
            group_data['initial_rot'] = [0, 0, 0]
            
            # Hide individual group folder - LED controls will be in master folder
            group_data['folder'].visible = False
            
            created_groups.append(group_data)
            
            print(f"✓ Converted {num_leds} individual LED(s) into 1 custom group with {len(led_rows_indices)} row(s)")

        
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
                    grp_rot_lr = server.gui.add_slider("Left/Right (°)", min=-180, max=180, step=1, initial_value=group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0)
                    grp_rot_ud = server.gui.add_slider("Up/Down (°)", min=-180, max=180, step=1, initial_value=group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0)
                    grp_rot_roll = server.gui.add_slider("Roll (°)", min=-180, max=180, step=1, initial_value=group['rot_roll'].value if 'rot_roll' in group else 0)
                    
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
        
        # Setup master control callbacks to sync all groups
        def update_all_from_master(_):
            if loading_in_progress[0]:
                return
            loading_in_progress[0] = True
            
            master_pos_offset = np.array([master_pos_x.value, master_pos_y.value, master_pos_z.value])
            
            # Build master rotation matrix (extrinsic X-Y-Z)
            roll_rad = np.radians(master_rot_x.value)
            pitch_rad = np.radians(master_rot_y.value)
            yaw_rad = np.radians(master_rot_z.value)
            Rx = np.array([[1,0,0],[0,np.cos(roll_rad),-np.sin(roll_rad)],[0,np.sin(roll_rad),np.cos(roll_rad)]])
            Ry = np.array([[np.cos(pitch_rad),0,np.sin(pitch_rad)],[0,1,0],[-np.sin(pitch_rad),0,np.cos(pitch_rad)]])
            Rz = np.array([[np.cos(yaw_rad),-np.sin(yaw_rad),0],[np.sin(yaw_rad),np.cos(yaw_rad),0],[0,0,1]])
            R_master = Rz @ Ry @ Rx
            
            # Compute template center (centroid of all initial positions)
            template_center = np.mean([np.array(p) for p in initial_positions], axis=0) if initial_positions else np.zeros(3)
            
            for idx, group in enumerate(created_groups):
                group['enable'].value = master_enable.value
                
                # RIGID BODY: rotate group position around template center
                initial_pos = np.array(initial_positions[idx])
                rotated_pos = R_master @ (initial_pos - template_center) + template_center + master_pos_offset
                
                group['pos_x'].value = float(rotated_pos[0])
                group['pos_y'].value = float(rotated_pos[1])
                group['pos_z'].value = float(rotated_pos[2])
                
                # Keep per-group rotation sliders at initial values
                group['rot_roll'].value = initial_rotations[idx][0]
                group['rot_tilt_ud'].value = initial_rotations[idx][1]
                group['rot_tilt_lr'].value = initial_rotations[idx][2]
                
                # Sync per-group UI sliders
                if idx < len(per_group_pos_rot_controls):
                    per_group_pos_rot_controls[idx]['enable'].value = master_enable.value
                    per_group_pos_rot_controls[idx]['pos_x'].value = float(rotated_pos[0])
                    per_group_pos_rot_controls[idx]['pos_y'].value = float(rotated_pos[1])
                    per_group_pos_rot_controls[idx]['pos_z'].value = float(rotated_pos[2])
                    per_group_pos_rot_controls[idx]['rot_roll'].value = initial_rotations[idx][0]
                    per_group_pos_rot_controls[idx]['rot_ud'].value = initial_rotations[idx][1]
                    per_group_pos_rot_controls[idx]['rot_lr'].value = initial_rotations[idx][2]
                
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
        
        def remove_all_groups(_):
            """Remove all groups from this template."""
            for group in created_groups:
                if group in custom_groups:
                    custom_groups.remove(group)
                group['folder'].remove()
            template_folder.remove()
            # Remove from template_folders list
            for template_data in template_folders[:]:
                if template_data['folder'] == template_folder:
                    template_folders.remove(template_data)
                    break
            update_scene()
        
        # Register callbacks
        master_enable.on_update(update_all_from_master)
        master_pos_x.on_update(update_all_from_master)
        master_pos_y.on_update(update_all_from_master)
        master_pos_z.on_update(update_all_from_master)
        master_rot_x.on_update(update_all_from_master)
        master_rot_y.on_update(update_all_from_master)
        master_rot_z.on_update(update_all_from_master)
        remove_template_btn.on_click(remove_all_groups)
        
        # Lumens override callback: propagate master lumens to all sub-groups
        def update_lumens_from_master(_):
            if loading_in_progress[0]:
                return
            loading_in_progress[0] = True
            for i, group in enumerate(created_groups):
                if group.get('lumens_override'):
                    group['lumens_override'].value = master_lumens_chk.value
                if group.get('lumens_value'):
                    group['lumens_value'].value = master_lumens_slider.value
                # Sync per-group UI controls
                if i < len(per_group_lumens_controls):
                    per_group_lumens_controls[i]['chk'].value = master_lumens_chk.value
                    per_group_lumens_controls[i]['slider'].value = master_lumens_slider.value
            loading_in_progress[0] = False
            update_scene()
        
        master_lumens_chk.on_update(update_lumens_from_master)
        master_lumens_slider.on_update(update_lumens_from_master)
        
        # Store template folder data for cleanup on new project/load
        template_folders.append({
            'folder': template_folder,
            'groups': created_groups
        })
        
        # Re-enable callbacks and update scene
        loading_in_progress[0] = False
        update_scene()
        
        # Force refresh LED markers to ensure all handles are created correctly
        if show_led_markers.value:
            show_led_markers.value = False
            time.sleep(0.05)
            update_scene()
            time.sleep(0.05)
            show_led_markers.value = True
            update_scene()
        
        print(f"✓ Loaded {len(created_groups)} custom group(s) from template: {template.get('name', template_name)}")
        return created_groups


    def create_custom_group(skip_update_scene=False, num_leds=12, led_rows=None, group_name=None):
        """Create a new custom LED group with all controls.
        
        Args:
            skip_update_scene: If True, skip the final update_scene() call.
                             Useful when loading multiple groups from config.
            num_leds: Number of LEDs in this group (default 12).
            led_rows: List of lists defining LED organization per row.
                     E.g., [[0,1,2], [3,4,5], [6,7,8], [9,10,11]] for 4 rows of 3 LEDs.
                     If None, defaults to 4 rows of 3 LEDs each.
            group_name: Optional custom name for the group folder.
        """
        group_id = next_custom_group_id[0]
        next_custom_group_id[0] += 1
        
        # Default row organization if not specified
        if led_rows is None:
            led_rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        # Create LED state array for this group (Row 2 and 3 on by default)
        led_states = [False] * num_leds
        # Turn on middle rows by default (if we have 4 rows, turn on rows 2 and 3)
        if len(led_rows) >= 4:
            for led_idx in led_rows[1] + led_rows[2]:
                if led_idx < num_leds:
                    led_states[led_idx] = True
        elif len(led_rows) >= 2:
            # Turn on first row if we have at least 2 rows
            for led_idx in led_rows[0]:
                if led_idx < num_leds:
                    led_states[led_idx] = True
        
        # Create group folder with controls
        with custom_groups_folder:
            folder_name = group_name if group_name else f"Custom Group {group_id}"
            group_folder = server.gui.add_folder(folder_name)
            group_folder.visible = False  # Edited via 3D click → inspector panel
        
        with group_folder:
            enable_chk = server.gui.add_checkbox("Enable", initial_value=True)
            pos_x = server.gui.add_slider("Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
            pos_y = server.gui.add_slider("Position Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            pos_z = server.gui.add_slider("Position Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            rot_tilt_lr = server.gui.add_slider("Tilt Left/Right (°)", min=-180, max=180, step=1, initial_value=0)
            rot_tilt_ud = server.gui.add_slider("Tilt Up/Down (°)", min=-180, max=180, step=1, initial_value=0)
            rot_roll = server.gui.add_slider("Rotate on axis (°)", min=-180, max=180, step=1, initial_value=0)
            
            server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>")
            group_lumens_override_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
            group_lumens_slider = server.gui.add_slider("Lumens per LED (lm)", min=1, max=900000, step=1, initial_value=100)
            
            remove_btn = server.gui.add_button("Remove Group", color="red")
            
            server.gui.add_html("<hr style='margin:4px 0;'><b>LED Controls:</b>")
            
            # Group ALL button
            all_btn = server.gui.add_button("ALL LEDs", color="#FF00FF")
            
            server.gui.add_html("<hr style='margin:4px 0;'>")
            
            # Row and LED buttons
            row_buttons = {}
            led_buttons = {}
            
            for row_idx, led_indices in enumerate(led_rows):
                html_content = f"""
                <div style='margin:6px 0 2px 0;'>
                    <span style='font-weight:600;font-size:11px;'>Row {row_idx + 1}:</span>
                </div>
                """
                server.gui.add_html(html_content)
                
                row_btn = server.gui.add_button(f"Row {row_idx + 1}", color="#666666")
                row_buttons[row_idx] = row_btn
                
                for led_in_row_idx, led_idx in enumerate(led_indices):
                    # Set initial color based on LED state
                    initial_color = "#FF00FF" if led_states[led_idx] else "#444444"
                    led_btn = server.gui.add_button(f"L{led_in_row_idx+1}", color=initial_color)
                    led_buttons[led_idx] = led_btn
        
        # Store group data
        group_data = {
            'id': group_id,
            'folder': group_folder,
            'enable': enable_chk,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'rot_tilt_lr': rot_tilt_lr,
            'rot_tilt_ud': rot_tilt_ud,
            'rot_roll': rot_roll,
            'lumens_override': group_lumens_override_chk,
            'lumens_value': group_lumens_slider,
            'remove_btn': remove_btn,
            'led_states': led_states,
            'led_rows': led_rows,  # Store row organization
            'num_leds': num_leds,  # Store total LED count
            'all_btn': all_btn,
            'row_buttons': row_buttons,
            'led_buttons': led_buttons,
            'update_button_colors': None,  # Will be set after function definition
            'template_name': None,  # Will be set if created from template
            'initial_pos': [0.0, 0.0, 0.0],  # Initial position before master transforms
            'initial_rot': [0, 0, 0],  # Initial rotation before master transforms
            'rotation_center': None,  # Center point for rotations (calculated once)
            'original_led_positions': None,  # Original LED positions (never modified)
            'original_led_rotations': None,  # Original LED directions (never modified)
            'led_row_directions': None,  # Row direction for each LED (for square orientation)
            'original_led_row_directions': None,  # Original row directions (never modified)
            'led_euler_angles': None,  # Original Euler angles for lossless roundtrip
            'led_beam_tilts': None,  # Beam tilt angles for lossless roundtrip
            'apply_rotation': None,  # Will be set to apply_rotation_transform()
            'guide': None,  # Circular construction guide (cylinder) or None
            'guide_error': None,
        }
        
        # Function to update button colors based on LED states
        def update_button_colors():
            """Update all button colors to match current LED states."""
            for led_idx in range(num_leds):
                if led_idx in led_buttons:
                    color = "#FF00FF" if led_states[led_idx] else "#444444"
                    led_buttons[led_idx].color = color
            # Update row button colors
            for row_idx, led_indices in enumerate(led_rows):
                any_on = any(led_states[i] for i in led_indices if i < num_leds)
                row_buttons[row_idx].color = "#FF00FF" if any_on else "#666666"
            # Update ALL button color
            any_on = any(led_states)
            all_btn.color = "#FF00FF" if any_on else "#666666"
        
        # Store reference to update function in group_data
        group_data['update_button_colors'] = update_button_colors
        # apply_rotation will be set after apply_rotation_transform is defined
        
        # Setup callbacks for this group
        def on_all_click(_):
            all_on = all(led_states)
            new_state = not all_on
            for i in range(num_leds):
                led_states[i] = new_state
            update_button_colors()
            update_scene()
        
        def make_row_handler(r_idx, led_indices):
            def handler(_):
                all_on = all(led_states[i] for i in led_indices if i < num_leds)
                new_state = not all_on
                for i in led_indices:
                    if i < num_leds:
                        led_states[i] = new_state
                update_button_colors()
                update_scene()
            return handler
        
        def make_led_handler(l_idx):
            def handler(_):
                led_states[l_idx] = not led_states[l_idx]
                update_button_colors()
                update_scene()
            return handler
        
        def on_remove(_):
            """Remove this group."""
            if group_data in custom_groups:
                custom_groups.remove(group_data)
            group_folder.remove()
            if selected_owner[0] == ('custom_group', group_id):
                select_panel(None)
            else:
                update_scene()
        
        all_btn.on_click(on_all_click)
        for row_idx, led_indices in enumerate(led_rows):
            row_buttons[row_idx].on_click(make_row_handler(row_idx, led_indices))
        for led_idx in led_buttons.keys():
            led_buttons[led_idx].on_click(make_led_handler(led_idx))
        remove_btn.on_click(on_remove)
        
        # Rotation callbacks - apply rotation to LED directions only (keep positions fixed)
        def apply_rotation_transform():
            """Apply current rotation sliders to group's LED directions (NOT positions)."""
            if not group_data.get('is_dynamic', False):
                return  # Only applies to dynamic groups with led_positions/led_rotations
            
            # Get rotation angles from sliders (Euler angles)
            roll_deg = rot_roll.value        # Rotation around X axis
            pitch_deg = rot_tilt_ud.value    # Rotation around Y axis
            yaw_deg = rot_tilt_lr.value      # Rotation around Z axis
            
            # Check if original rotations are available
            if group_data.get('original_led_rotations') is None:
                return  # No original rotations saved yet
            
            # Build rotation matrices for Euler angles (extrinsic rotations)
            # Applied to the FIXED initial reference frame
            roll_rad = np.radians(roll_deg)
            pitch_rad = np.radians(pitch_deg)
            yaw_rad = np.radians(yaw_deg)
            
            # Rotation matrix around X axis (roll)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)]
            ])
            
            # Rotation matrix around Y axis (pitch)
            Ry = np.array([
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
            ])
            
            # Rotation matrix around Z axis (yaw)
            Rz = np.array([
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1]
            ])
            
            # Compose rotations: extrinsic X-Y-Z means R = Rz @ Ry @ Rx
            # (applied right to left: first X, then Y, then Z in fixed frame)
            R_total = Rz @ Ry @ Rx
            
            # RIGID BODY rotation: rotate BOTH positions AND directions
            # R is orthogonal, so all distances are preserved
            original_positions = group_data.get('original_led_positions')
            original_rotations = group_data['original_led_rotations']
            
            # Rotate positions (relative to group center)
            if original_positions is not None:
                rotated_positions = []
                for pos in original_positions:
                    pos_rotated = R_total @ np.array(pos)
                    rotated_positions.append(tuple(pos_rotated))
                group_data['led_positions'] = rotated_positions
            
            # Rotate direction vectors
            rotated_directions = []
            for direction in original_rotations:
                dir_rotated = R_total @ np.array(direction)
                rotated_directions.append(tuple(dir_rotated))
            
            group_data['led_rotations'] = rotated_directions
            
            # Rotate row direction vectors
            original_row_dirs = group_data.get('original_led_row_directions')
            if original_row_dirs is not None:
                rotated_row_dirs = []
                for rd in original_row_dirs:
                    rd_rotated = R_total @ np.array(rd)
                    rotated_row_dirs.append(tuple(rd_rotated))
                group_data['led_row_directions'] = rotated_row_dirs
        
        # Store reference so external code can trigger the same rotation logic
        group_data['apply_rotation'] = apply_rotation_transform

        def on_rot_tilt_lr_update(_):
            if not loading_in_progress[0]:
                apply_rotation_transform()
                update_scene()
        
        def on_rot_tilt_ud_update(_):
            if not loading_in_progress[0]:
                apply_rotation_transform()
                update_scene()
        
        def on_rot_roll_update(_):
            if not loading_in_progress[0]:
                apply_rotation_transform()
                update_scene()
        
        # Register slider callbacks (check loading flag to prevent updates during config load)
        enable_chk.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_x.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_y.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_z.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        rot_tilt_lr.on_update(on_rot_tilt_lr_update)
        rot_tilt_ud.on_update(on_rot_tilt_ud_update)
        rot_roll.on_update(on_rot_roll_update)
        group_lumens_override_chk.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        group_lumens_slider.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        
        custom_groups.append(group_data)
        
        # Set initial button colors to match LED states
        update_button_colors()
        
        # Only update scene if not skipping (e.g., when manually adding a group)
        if not skip_update_scene:
            update_scene()
            select_panel(('custom_group', group_id))
        
        return group_data
    

    def _convert_individual_leds_to_dynamic_group(individual_leds_data):
        """Convert an individual_leds list into a single dynamic-format group config
        with positions made RELATIVE (centroid subtracted)."""
        if not individual_leds_data:
            return None

        sorted_leds = sorted(individual_leds_data,
                             key=lambda l: l.get('pos_z', 0.0), reverse=True)

        # Group into rows by Z coordinate
        z_tol = 0.5
        led_rows_indices, current_row, current_z = [], [], None
        for idx, led in enumerate(sorted_leds):
            lz = led.get('pos_z', 0.0)
            if current_z is None or abs(lz - current_z) <= z_tol:
                current_row.append(idx)
                if current_z is None:
                    current_z = lz
            else:
                if current_row:
                    led_rows_indices.append(current_row)
                current_row = [idx]
                current_z = lz
        if current_row:
            led_rows_indices.append(current_row)

        num_leds = len(sorted_leds)
        led_states = [l.get('led_on', True) for l in sorted_leds]
        led_positions = [(l.get('pos_x', 0.0), l.get('pos_y', 0.0), l.get('pos_z', 0.0))
                         for l in sorted_leds]
        led_sizes = [l.get('size', 0.5) for l in sorted_leds]
        led_viewing_angles = [l.get('viewing_angle', 120) for l in sorted_leds]

        # Euler → direction vectors (Rz · Ry · Rx · [1,0,0])
        led_rotations = []
        for led in sorted_leds:
            d = np.array([1.0, 0.0, 0.0])
            for axis, ang in [('z', led.get('rot_z', 0.0)),
                              ('y', led.get('rot_y', 0.0)),
                              ('x', led.get('rot_x', 0.0))]:
                rad = np.radians(ang)
                cs, sn = np.cos(rad), np.sin(rad)
                if axis == 'z':
                    M = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
                elif axis == 'y':
                    M = np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])
                else:
                    M = np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]])
                d = M @ d
            d = d / max(np.linalg.norm(d), 1e-10)
            led_rotations.append(tuple(d))

        # Row directions with Rodrigues square_roll
        led_row_directions = []
        for i, dt in enumerate(led_rotations):
            dv = np.array(dt)
            rd = np.cross(dv, [0, 0, 1])
            if np.linalg.norm(rd) < 0.01:
                rd = np.cross(dv, [0, 1, 0])
            rd = rd / np.linalg.norm(rd)
            sq = sorted_leds[i].get('square_roll', 0.0)
            if abs(sq) > 0.01:
                sr = np.radians(sq)
                k = dv / np.linalg.norm(dv)
                rd = (rd * np.cos(sr)
                      + np.cross(k, rd) * np.sin(sr)
                      + k * np.dot(k, rd) * (1 - np.cos(sr)))
            led_row_directions.append(tuple(rd))

        # Subtract centroid → relative positions
        pa = np.array(led_positions)
        ctr = pa.mean(axis=0)
        relative_positions = [tuple(p - ctr) for p in pa]

        led_euler_angles = [(l.get('rot_x', 0.0), l.get('rot_y', 0.0), l.get('rot_z', 0.0))
                            for l in sorted_leds]
        led_beam_tilts = [l.get('beam_tilt', 0.0) for l in sorted_leds]

        return {
            'enabled': True,
            'position': [0.0, 0.0, 0.0],
            'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0,
            'led_states': led_states,
            'is_dynamic': True,
            'num_leds': num_leds,
            'led_positions': relative_positions,
            'led_rotations': list(led_rotations),
            'led_row_directions': led_row_directions,
            'led_sizes': led_sizes,
            'led_viewing_angles': led_viewing_angles,
            'led_rows': led_rows_indices,
            'led_euler_angles': led_euler_angles,
            'led_beam_tilts': led_beam_tilts,
            'lumens_override_enabled': False,
            'lumens_value': 100,
        }

    def _generate_standard_12led_group(led_states=None):
        """Generate a standard 12-LED Elios-style panel group at origin."""
        led_spacing_cm = 0.8
        inclinations = [90, 30, -30, -90]
        row_offsets = [-0.85, -0.55, 0.55, 0.85]

        # Local frame (forward = +X, row direction = -Y, up = +Z)
        x_axis = np.array([0.0, -1.0, 0.0])
        rolled_z = np.array([0.0, 0.0, 1.0])
        radial = np.array([1.0, 0.0, 0.0])

        positions, directions, row_dirs = [], [], []
        for row_idx, alpha_deg in enumerate(inclinations):
            alpha = np.radians(alpha_deg)
            row_center = rolled_z * row_offsets[row_idx]
            rotated_dir = np.cos(alpha) * radial + (-np.sin(alpha)) * rolled_z
            rotated_dir = rotated_dir / np.linalg.norm(rotated_dir)
            if row_idx in (0, 3):
                row_center = row_center - radial * 0.5
            for off in [-led_spacing_cm, 0.0, led_spacing_cm]:
                positions.append(tuple(row_center + x_axis * off))
                directions.append(tuple(rotated_dir))
                row_dirs.append(tuple(x_axis))

        if led_states is None:
            led_states = [True] * 12
        return {
            'enabled': True,
            'position': [0.0, 0.0, 0.0],
            'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0,
            'led_states': led_states[:],
            'is_dynamic': True,
            'num_leds': 12,
            'led_positions': positions,
            'led_rotations': directions,
            'led_row_directions': row_dirs,
            'led_sizes': [0.5] * 12,
            'led_viewing_angles': [120] * 12,
            'led_rows': [[0,1,2], [3,4,5], [6,7,8], [9,10,11]],
            'led_euler_angles': [],
            'led_beam_tilts': [0] * 12,
            'lumens_override_enabled': False,
            'lumens_value': 100,
        }

    def _load_template_into_slot(slot_idx, template_name, as_individual):
        """Load a template into a Panel Configurator slot, rotated to face outward."""
        nonlocal loading_in_progress
        slot = _ELIOS3_SLOTS[slot_idx]
        centroid = np.array(slot["centroid"])
        R = _Rz_matrix(slot["angle_deg"])

        # Full rotation for individual LEDs: Rz(config_rot+config_lr) @ Rx(config_roll) @ Rz(angle_deg)
        _cfg_lr = slot.get("config_rot", 0) + slot.get("config_lr", 0)
        _cfg_roll = slot.get("config_roll", 0)
        _lr_rad = np.radians(_cfg_lr)
        _roll_rad = np.radians(_cfg_roll)
        _Rz_cfg = np.array([[np.cos(_lr_rad), -np.sin(_lr_rad), 0],
                            [np.sin(_lr_rad),  np.cos(_lr_rad), 0],
                            [0, 0, 1]])
        _Rx_cfg = np.array([[1, 0, 0],
                            [0, np.cos(_roll_rad), -np.sin(_roll_rad)],
                            [0, np.sin(_roll_rad),  np.cos(_roll_rad)]])
        R_full = _Rz_cfg @ _Rx_cfg @ R  # coarse(angle_deg) + config offsets

        # Remove previous content in this slot if any
        _clear_panel_slot(slot_idx)

        path = os.path.join(custom_groups_templates_dir, f"{template_name}.json")
        if not os.path.exists(path):
            print(f"Panel Configurator: template '{template_name}' not found")
            return

        with open(path, "r") as f:
            template = json.load(f)

        groups_data = template.get('groups', [])
        if not groups_data and 'enabled' in template:
            groups_data = [{
                'enabled': template.get('enabled', True),
                'position': template.get('position', [0, 0, 0]),
                'led_states': template.get('led_states', [True] * 12),
            }]

        # ---- Normalize ALL formats into dynamic group configs ----
        # 1) Convert individual_leds → dynamic group (centered at origin)
        individual_leds_data = template.get('individual_leds', [])
        if individual_leds_data:
            converted = _convert_individual_leds_to_dynamic_group(individual_leds_data)
            if converted:
                groups_data.append(converted)

        # 2) Convert standard groups (no is_dynamic, no led_positions) → dynamic
        for gi in range(len(groups_data)):
            grp = groups_data[gi]
            if not grp.get('is_dynamic', False) and 'led_positions' not in grp:
                std = _generate_standard_12led_group(grp.get('led_states', [True] * 12))
                std['lumens_override_enabled'] = grp.get('lumens_override_enabled', False)
                std['lumens_value'] = grp.get('lumens_value', 100)
                std['enabled'] = grp.get('enabled', True)
                groups_data[gi] = std

        if not groups_data:
            print(f"Panel Configurator: template '{template_name}' has no LED data")
            loading_in_progress[0] = False
            return

        # ---- Align every group to +X-forward convention ----
        # Templates from individual_leds or pre-rotated configs may have
        # world-frame directions.  Detect the mean XY direction and un-rotate
        # so that the panel faces +X.  update_scene() will apply
        # rot_tilt_lr = slot_angle to orient it to the correct direction.
        for gi in range(len(groups_data)):
            grp = groups_data[gi]
            dirs = grp.get('led_rotations', [])
            if not dirs:
                continue
            mean_d = np.mean([np.array(d) for d in dirs], axis=0)
            azimuth = np.arctan2(mean_d[1], mean_d[0])          # current heading
            if abs(azimuth) < np.radians(2):                    # already ~+X
                continue
            # Build Rz(-azimuth) to bring mean direction back to +X
            Runrot = _Rz_matrix(np.degrees(-azimuth))
            grp['led_positions'] = [tuple(Runrot @ np.array(p))
                                    for p in grp.get('led_positions', [])]
            grp['led_rotations'] = [tuple(Runrot @ np.array(d))
                                    for d in dirs]
            rds = grp.get('led_row_directions', [])
            if rds:
                grp['led_row_directions'] = [tuple(Runrot @ np.array(rd))
                                             for rd in rds]

        loading_in_progress[0] = True

        slot_label = slot["name"]
        template_display = template.get('name', template_name)
        mode_label = "Individual LEDs" if as_individual else "Solid Group"

        # Create a master folder for this slot
        with tab_panels:
            slot_folder = server.gui.add_folder(f"Slot {slot_label}: {template_display} ({mode_label})")
        slot_folder.visible = False  # Edited via 3D click → inspector panel

        with slot_folder:
            slot_enable = server.gui.add_checkbox("Enable Slot", initial_value=True)
            slot_pos_x = server.gui.add_slider("Offset X (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_pos_y = server.gui.add_slider("Offset Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_pos_z = server.gui.add_slider("Offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_rot_x = server.gui.add_slider("Rotate on axis (°)", min=-180, max=180, step=1, initial_value=0)
            slot_rot_y = server.gui.add_slider("Tilt Up/Down (°)", min=-180, max=180, step=1, initial_value=0)
            slot_rot_z = server.gui.add_slider("Tilt Left/Right (°)", min=-180, max=180, step=1, initial_value=0)
            server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>")
            slot_lumens_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
            slot_lumens_slider = server.gui.add_slider("Lumens per LED (lm)", min=1, max=900000, step=1, initial_value=100)
            slot_remove_btn = server.gui.add_button("Remove Slot", color="red")

        created_groups = []
        created_individual_leds = []
        _slot_led_buttons = []
        _update_slot_btn_colors = None

        if as_individual:
            # --- Load as individual LEDs ---
            for grp_cfg in groups_data:
                is_dynamic = grp_cfg.get('is_dynamic', False)
                num_leds = grp_cfg.get('num_leds', 12)
                led_positions_raw = grp_cfg.get('led_positions', [(0,0,0)] * num_leds) if is_dynamic else []
                led_rotations_raw = grp_cfg.get('led_rotations', [(1,0,0)] * num_leds) if is_dynamic else []
                led_sizes_raw = grp_cfg.get('led_sizes', [0.5] * num_leds) if is_dynamic else []
                led_va_raw = grp_cfg.get('led_viewing_angles', [120] * num_leds) if is_dynamic else []
                led_states_raw = grp_cfg.get('led_states', [True] * num_leds)
                led_row_dirs_raw = grp_cfg.get('led_row_directions', []) if is_dynamic else []
                led_beam_tilts_raw = grp_cfg.get('led_beam_tilts', []) if is_dynamic else []

                if not is_dynamic:
                    # Static group — generate default positions
                    led_rows = grp_cfg.get('led_rows', [[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
                    led_positions_raw = []
                    led_rotations_raw = []
                    led_sizes_raw = []
                    led_va_raw = []
                    for row_idx, led_indices in enumerate(led_rows):
                        for led_idx_in_row, _ in enumerate(led_indices):
                            led_positions_raw.append((led_idx_in_row * 1.5, 0.0, -row_idx * 2.0))
                            led_rotations_raw.append((1, 0, 0))
                            led_sizes_raw.append(0.5)
                            led_va_raw.append(120)

                for led_idx in range(num_leds):
                    # Rotate position with full rotation (coarse + config offsets), then add centroid
                    raw_pos = np.array(led_positions_raw[led_idx]) if led_idx < len(led_positions_raw) else np.zeros(3)
                    world_pos = R_full @ raw_pos + centroid

                    # Rotate direction vector
                    raw_dir = np.array(led_rotations_raw[led_idx]) if led_idx < len(led_rotations_raw) else np.array([1,0,0])
                    world_dir = R_full @ raw_dir
                    world_dir = world_dir / max(np.linalg.norm(world_dir), 1e-10)

                    # Derive Euler angles from direction
                    dy_clamped = np.clip(world_dir[1], -1.0, 1.0)
                    rot_z_rad = np.arcsin(dy_clamped)
                    cos_rz = np.cos(rot_z_rad)
                    rot_y_rad = np.arctan2(-world_dir[2], world_dir[0]) if abs(cos_rz) > 1e-6 else 0.0
                    rot_angles = [0.0, float(np.degrees(rot_y_rad)), float(np.degrees(rot_z_rad))]

                    led_data = create_individual_led(skip_update_scene=True)
                    led_data['enable'].value = True
                    led_data['led_on'] = led_states_raw[led_idx] if led_idx < len(led_states_raw) else True
                    led_data['led_on_btn'].color = "#00FF00" if led_data['led_on'] else "#FF0000"
                    led_data['pos_x'].value = float(world_pos[0])
                    led_data['pos_y'].value = float(world_pos[1])
                    led_data['pos_z'].value = float(world_pos[2])
                    led_data['rot_x'].value = rot_angles[0]
                    led_data['rot_y'].value = rot_angles[1]
                    led_data['rot_z'].value = rot_angles[2]
                    led_data['size'].value = float(led_sizes_raw[led_idx]) if led_idx < len(led_sizes_raw) else 0.5
                    led_data['viewing_angle'].value = float(led_va_raw[led_idx]) if led_idx < len(led_va_raw) else 120
                    led_data['panel_slot'] = slot_idx

                    if led_beam_tilts_raw and led_idx < len(led_beam_tilts_raw):
                        led_data['beam_tilt'].value = int(round(led_beam_tilts_raw[led_idx]))

                    created_individual_leds.append(led_data)

            # Store initial positions/rotations for slot offset/rotation callbacks
            _ind_init_positions = []
            _ind_init_rotations = []
            for led in created_individual_leds:
                _ind_init_positions.append([led['pos_x'].value, led['pos_y'].value, led['pos_z'].value])
                _ind_init_rotations.append([led['rot_x'].value, led['rot_y'].value, led['rot_z'].value])

            # Slot callbacks for individual LEDs (offset + rotation)
            def _make_ind_slot_update(s_enable, s_px, s_py, s_pz, s_rx, s_ry, s_rz,
                                      c_leds, init_pos, init_rot, ctr,
                                      base_roll=0, base_lr=0):
                def handler(_):
                    if loading_in_progress[0]:
                        return
                    offset = np.array([s_px.value, s_py.value, s_pz.value])
                    roll_r = np.radians(base_roll + s_rx.value)
                    pitch_r = np.radians(s_ry.value)
                    yaw_r = np.radians(base_lr + s_rz.value)
                    Rx_ = np.array([[1,0,0],[0,np.cos(roll_r),-np.sin(roll_r)],[0,np.sin(roll_r),np.cos(roll_r)]])
                    Ry_ = np.array([[np.cos(pitch_r),0,np.sin(pitch_r)],[0,1,0],[-np.sin(pitch_r),0,np.cos(pitch_r)]])
                    Rz_ = np.array([[np.cos(yaw_r),-np.sin(yaw_r),0],[np.sin(yaw_r),np.cos(yaw_r),0],[0,0,1]])
                    R_s = Rz_ @ Ry_ @ Rx_
                    for li, led in enumerate(c_leds):
                        led['enable'].value = s_enable.value
                        ip = np.array(init_pos[li])
                        rp = R_s @ (ip - ctr) + ctr + offset
                        led['pos_x'].value = float(rp[0])
                        led['pos_y'].value = float(rp[1])
                        led['pos_z'].value = float(rp[2])
                        # Recompute direction from initial Euler + slot rotation
                        ir = init_rot[li]
                        d = np.array([1.0, 0.0, 0.0])
                        for axis, ang in [('z', ir[2]), ('y', ir[1]), ('x', ir[0])]:
                            rad = np.radians(ang)
                            cs, sn = np.cos(rad), np.sin(rad)
                            if axis == 'z':
                                M = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
                            elif axis == 'y':
                                M = np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])
                            else:
                                M = np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]])
                            d = M @ d
                        d = R_s @ d
                        d = d / max(np.linalg.norm(d), 1e-10)
                        dy_c = np.clip(d[1], -1.0, 1.0)
                        rz_r = np.arcsin(dy_c)
                        crz = np.cos(rz_r)
                        ry_r = np.arctan2(-d[2], d[0]) if abs(crz) > 1e-6 else 0.0
                        led['rot_x'].value = 0.0
                        led['rot_y'].value = float(np.degrees(ry_r))
                        led['rot_z'].value = float(np.degrees(rz_r))
                    update_scene()
                return handler

            ind_center = np.mean(_ind_init_positions, axis=0) if _ind_init_positions else np.zeros(3)
            _ind_cb = _make_ind_slot_update(slot_enable, slot_pos_x, slot_pos_y, slot_pos_z,
                                            slot_rot_x, slot_rot_y, slot_rot_z,
                                            created_individual_leds, _ind_init_positions,
                                            _ind_init_rotations, ind_center,
                                            base_roll=slot.get("config_roll", 0),
                                            base_lr=slot.get("config_rot", 0) + slot.get("config_lr", 0))
            slot_enable.on_update(_ind_cb)
            slot_pos_x.on_update(_ind_cb)
            slot_pos_y.on_update(_ind_cb)
            slot_pos_z.on_update(_ind_cb)
            slot_rot_x.on_update(_ind_cb)
            slot_rot_y.on_update(_ind_cb)
            slot_rot_z.on_update(_ind_cb)

            # Lumens callback for individual LEDs
            def _make_ind_lumens_cb(c_leds):
                def handler(_):
                    if loading_in_progress[0]:
                        return
                    for led in c_leds:
                        if led.get('lumens_override'):
                            led['lumens_override'].value = slot_lumens_chk.value
                        if led.get('lumens_value'):
                            led['lumens_value'].value = slot_lumens_slider.value
                    update_scene()
                return handler

            _ind_lum_cb = _make_ind_lumens_cb(created_individual_leds)
            slot_lumens_chk.on_update(_ind_lum_cb)
            slot_lumens_slider.on_update(_ind_lum_cb)

        else:
            # --- Load as solid group(s) ---
            initial_positions = []
            initial_rotations = []

            for grp_cfg in groups_data:
                if 'num_leds' in grp_cfg and 'led_rows' in grp_cfg:
                    group_data = create_custom_group(
                        skip_update_scene=True,
                        num_leds=grp_cfg['num_leds'],
                        led_rows=grp_cfg['led_rows'],
                        group_name=grp_cfg.get('name', None)
                    )
                else:
                    group_data = create_custom_group(skip_update_scene=True)

                # Set position to slot centroid
                group_data['pos_x'].value = float(centroid[0])
                group_data['pos_y'].value = float(centroid[1])
                group_data['pos_z'].value = float(centroid[2])
                # Set initial rotation offsets matching Elios3 config
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = int(round(slot["config_rot"] + slot.get("config_lr", 0)))
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = 0
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = int(round(slot.get("config_roll", 0)))

                # Pre-bake Rz(angle_deg) into original positions for coarse
                # slot orientation; config_rot in rot_tilt_lr provides the
                # fine-tuning (same split as standard Elios3 groups).
                if grp_cfg.get('is_dynamic', False):
                    group_data['is_dynamic'] = True
                    raw_positions = grp_cfg.get('led_positions', [])
                    raw_rotations = grp_cfg.get('led_rotations', [])
                    raw_row_dirs = grp_cfg.get('led_row_directions', [])

                    baked_pos = [tuple(R @ np.array(p)) for p in raw_positions]
                    baked_rot = [tuple(R @ np.array(d)) for d in raw_rotations]
                    baked_row = [tuple(R @ np.array(rd)) for rd in raw_row_dirs]

                    group_data['led_positions'] = baked_pos
                    group_data['led_rotations'] = baked_rot
                    group_data['led_row_directions'] = baked_row
                    group_data['led_euler_angles'] = grp_cfg.get('led_euler_angles', [])
                    group_data['led_beam_tilts'] = grp_cfg.get('led_beam_tilts', [])
                    group_data['led_sizes'] = grp_cfg.get('led_sizes', [])
                    group_data['led_viewing_angles'] = grp_cfg.get('led_viewing_angles', [])
                    group_data['led_lumens'] = grp_cfg.get('led_lumens', [])

                    group_data['original_led_positions'] = list(baked_pos)
                    group_data['original_led_rotations'] = list(baked_rot)
                    if baked_row:
                        group_data['original_led_row_directions'] = list(baked_row)

                # Load LED states
                led_states_cfg = grp_cfg.get('led_states', [])
                for i, state in enumerate(led_states_cfg):
                    if i < len(group_data['led_states']):
                        group_data['led_states'][i] = state
                if 'update_button_colors' in group_data and group_data['update_button_colors']:
                    group_data['update_button_colors']()

                # Restore lumens override if present
                if group_data.get('lumens_override') and grp_cfg.get('lumens_override_enabled'):
                    group_data['lumens_override'].value = True
                    group_data['lumens_value'].value = grp_cfg.get('lumens_value', 100)

                group_data['enable'].value = grp_cfg.get('enabled', True)

                group_data['template_name'] = template_display
                group_data['panel_slot'] = slot_idx
                group_data['panel_slot_name'] = slot["name"]
                group_data['initial_pos'] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
                group_data['initial_rot'] = [0, 0, 0]
                group_data['folder'].visible = False

                initial_positions.append(group_data['initial_pos'])
                initial_rotations.append(group_data['initial_rot'])
                created_groups.append(group_data)

            # -- LED Controls (tracked buttons) --
            _slot_led_buttons = []

            with slot_folder:
                server.gui.add_html("<hr style='margin:8px 0;'><b>LED Controls:</b>")
                for g_idx, group in enumerate(created_groups):
                    with server.gui.add_folder(f"Group {g_idx + 1} LEDs"):
                        _gb = {'all_btn': None, 'row_btns': {}, 'led_btns': {}}
                        led_rows = group.get('led_rows', [[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

                        _ab = server.gui.add_button("ALL LEDs", color="#666666")
                        _gb['all_btn'] = _ab
                        server.gui.add_html("<hr style='margin:4px 0;'>")

                        for r_idx, led_indices in enumerate(led_rows):
                            _rb = server.gui.add_button(f"Row {r_idx + 1}", color="#666666")
                            _gb['row_btns'][r_idx] = _rb

                        server.gui.add_html("<hr style='margin:4px 0;'>")

                        for l_idx in range(group.get('num_leds', 12)):
                            _c = "#FF00FF" if group['led_states'][l_idx] else "#444444"
                            _lb = server.gui.add_button(f"LED {l_idx + 1}", color=_c)
                            _gb['led_btns'][l_idx] = _lb

                        _slot_led_buttons.append({
                            'group': group, 'buttons': _gb, 'led_rows': led_rows
                        })

            # Function to sync all slot LED button colours
            def _update_slot_btn_colors():
                for _gd in _slot_led_buttons:
                    _grp = _gd['group']
                    _btns = _gd['buttons']
                    _lrs = _gd['led_rows']
                    for _li, _lb in _btns['led_btns'].items():
                        if _li < len(_grp['led_states']):
                            _lb.color = "#FF00FF" if _grp['led_states'][_li] else "#444444"
                    for _ri, _lis in enumerate(_lrs):
                        if _ri in _btns['row_btns']:
                            _any = any(_grp['led_states'][i] for i in _lis if i < len(_grp['led_states']))
                            _btns['row_btns'][_ri].color = "#FF00FF" if _any else "#666666"
                    if _btns['all_btn']:
                        _btns['all_btn'].color = "#FF00FF" if any(_grp['led_states']) else "#666666"

            # Wire LED button handlers
            def _mk_all_h(grp, uf):
                def h(_):
                    _ao = all(grp['led_states'])
                    for i in range(len(grp['led_states'])):
                        grp['led_states'][i] = not _ao
                    if grp.get('update_button_colors'):
                        grp['update_button_colors']()
                    uf()
                    update_scene()
                return h

            def _mk_row_h(grp, leds_in_row, uf):
                def h(_):
                    _ao = all(grp['led_states'][i] for i in leds_in_row if i < len(grp['led_states']))
                    for li in leds_in_row:
                        if li < len(grp['led_states']):
                            grp['led_states'][li] = not _ao
                    if grp.get('update_button_colors'):
                        grp['update_button_colors']()
                    uf()
                    update_scene()
                return h

            def _mk_led_h(grp, idx, uf):
                def h(_):
                    if idx < len(grp['led_states']):
                        grp['led_states'][idx] = not grp['led_states'][idx]
                    if grp.get('update_button_colors'):
                        grp['update_button_colors']()
                    uf()
                    update_scene()
                return h

            for _gd in _slot_led_buttons:
                _grp = _gd['group']
                _btns = _gd['buttons']
                _lrs = _gd['led_rows']
                _btns['all_btn'].on_click(_mk_all_h(_grp, _update_slot_btn_colors))
                for _ri, _lis in enumerate(_lrs):
                    if _ri in _btns['row_btns']:
                        _btns['row_btns'][_ri].on_click(_mk_row_h(_grp, _lis, _update_slot_btn_colors))
                for _li, _lb in _btns['led_btns'].items():
                    _lb.on_click(_mk_led_h(_grp, _li, _update_slot_btn_colors))

            _update_slot_btn_colors()

            # -- Slot master callback (enable / offset / rotation) --
            # Uses the SAME rotation mechanism as standard Elios3 groups:
            # slot_rot_x → rot_roll (Rotate on axis, around panel forward axis)
            # slot_rot_y → rot_tilt_ud (Tilt Up/Down)
            # slot_rot_z → rot_tilt_lr = slot_angle + user_offset
            # R_total = Rz(slot_angle + user_z) @ Ry(user_y) @ Rx(user_roll)
            # Since Rx is applied FIRST to the +X-forward template, roll works
            # around the panel's own forward direction — identical to Elios3.
            def _make_slot_update(s_enable, s_px, s_py, s_pz, s_rx, s_ry, s_rz,
                                  c_groups, init_pos, base_lr, base_roll):
                def handler(_):
                    if loading_in_progress[0]:
                        return
                    loading_in_progress[0] = True
                    offset = np.array([s_px.value, s_py.value, s_pz.value])

                    for gi, g in enumerate(c_groups):
                        g['enable'].value = s_enable.value
                        # Position = initial + offset
                        base = np.array(init_pos[gi])
                        g['pos_x'].value = float(base[0] + offset[0])
                        g['pos_y'].value = float(base[1] + offset[1])
                        g['pos_z'].value = float(base[2] + offset[2])
                        # Map slot rotation sliders → per-group rotation sliders
                        # base offsets replicate the Elios3 config defaults
                        if 'rot_roll' in g:
                            g['rot_roll'].value = int(round(base_roll + s_rx.value))
                        if 'rot_tilt_ud' in g:
                            g['rot_tilt_ud'].value = int(round(s_ry.value))
                        if 'rot_tilt_lr' in g:
                            g['rot_tilt_lr'].value = int(round(base_lr + s_rz.value))
                        # Trigger the group's own rotation transform
                        if callable(g.get('apply_rotation')):
                            g['apply_rotation']()

                    loading_in_progress[0] = False
                    update_scene()
                return handler

            _slot_cb = _make_slot_update(slot_enable, slot_pos_x, slot_pos_y, slot_pos_z,
                                        slot_rot_x, slot_rot_y, slot_rot_z,
                                        created_groups, initial_positions,
                                        slot["config_rot"] + slot.get("config_lr", 0),
                                        slot.get("config_roll", 0))
            slot_enable.on_update(_slot_cb)
            slot_pos_x.on_update(_slot_cb)
            slot_pos_y.on_update(_slot_cb)
            slot_pos_z.on_update(_slot_cb)
            slot_rot_x.on_update(_slot_cb)
            slot_rot_y.on_update(_slot_cb)
            slot_rot_z.on_update(_slot_cb)

            # Lumens override callback
            def _make_lumens_cb(c_groups):
                def handler(_):
                    if loading_in_progress[0]:
                        return
                    for g in c_groups:
                        if g.get('lumens_override'):
                            g['lumens_override'].value = slot_lumens_chk.value
                        if g.get('lumens_value'):
                            g['lumens_value'].value = slot_lumens_slider.value
                    update_scene()
                return handler

            _lum_cb = _make_lumens_cb(created_groups)
            slot_lumens_chk.on_update(_lum_cb)
            slot_lumens_slider.on_update(_lum_cb)

        # Remove slot callback (shared between both modes)
        def _make_remove_cb(si):
            def handler(_):
                _clear_panel_slot(si)
                if si < len(_panel_dropdowns):
                    _panel_dropdowns[si].value = "-- Nessuno --"
                update_scene()
            return handler
        slot_remove_btn.on_click(_make_remove_cb(slot_idx))

        # Store slot data
        _panel_slot_data[slot_idx] = {
            'template_name': template_name,
            'as_individual': as_individual,
            'folder': slot_folder,
            'groups': created_groups,
            'individual_leds': created_individual_leds,
            'slot_label': slot_label,
            'template_display': template_display,
            'controls': {
                'enable': slot_enable,
                'pos_x': slot_pos_x,
                'pos_y': slot_pos_y,
                'pos_z': slot_pos_z,
                'rot_x': slot_rot_x,
                'rot_y': slot_rot_y,
                'rot_z': slot_rot_z,
                'lumens_chk': slot_lumens_chk,
                'lumens_slider': slot_lumens_slider,
            },
            'led_button_groups': _slot_led_buttons,
            'update_btn_colors': _update_slot_btn_colors,
        }

        # Also register in template_folders for cleanup on new project
        template_folders.append({
            'folder': slot_folder,
            'groups': created_groups,
        })

        loading_in_progress[0] = False
        update_scene()
        select_panel(('slot', slot_idx))
        print(f"✓ Panel Configurator: Loaded '{template_display}' into slot {slot_label} ({mode_label})")

    def _clear_panel_slot(slot_idx):
        """Remove all content from a panel slot."""
        if _mirror_primary[0] == ('slot', slot_idx):
            _clear_mirror_state()
        data = _panel_slot_data[slot_idx]
        if data is None:
            return
        # Remove groups
        for g in data.get('groups', []):
            if g in custom_groups:
                custom_groups.remove(g)
            try:
                g['folder'].remove()
            except Exception:
                pass
        # Remove individual LEDs
        for led in data.get('individual_leds', []):
            if led in individual_leds:
                individual_leds.remove(led)
            try:
                led['folder'].remove()
            except Exception:
                pass
        # Remove slot folder
        try:
            data['folder'].remove()
        except Exception:
            pass
        # Remove from template_folders
        for tf in template_folders[:]:
            if tf.get('folder') == data.get('folder'):
                template_folders.remove(tf)
                break
        _panel_slot_data[slot_idx] = None
        if selected_owner[0] == ('slot', slot_idx):
            select_panel(None)


    def load_template_as_individual_leds(template_name):
        """Load a template and create individual editable LEDs instead of a group."""
        nonlocal loading_in_progress
        loading_in_progress[0] = True
        
        path = os.path.join(custom_groups_templates_dir, f"{template_name}.json")
        if not os.path.exists(path):
            print(f"Template not found: {template_name}")
            loading_in_progress[0] = False
            return
        
        with open(path, "r") as f:
            template = json.load(f)
        
        # Get groups data from template
        groups_data = template.get('groups', [])
        if not groups_data and 'enabled' in template:
            # Old single-group format
            groups_data = [template]
        
        total_leds_created = 0
        
        # Process each group in the template
        for group_cfg in groups_data:
            num_leds = group_cfg.get('num_leds', 12)
            led_rows = group_cfg.get('led_rows', [[0,1,2], [3,4,5], [6,7,8], [9,10,11]])
            led_states = group_cfg.get('led_states', [True] * num_leds)
            
            # Get LED positions and rotations
            is_dynamic = group_cfg.get('is_dynamic', False)
            if is_dynamic:
                # Dynamic group with custom LED positions
                led_positions = group_cfg.get('led_positions', [(0, 0, 0)] * num_leds)
                led_rotations = group_cfg.get('led_rotations', [(1, 0, 0)] * num_leds)
                led_sizes = group_cfg.get('led_sizes', [0.5] * num_leds)
                if not led_sizes:
                    led_sizes = [0.5] * num_leds
                led_viewing_angles = group_cfg.get('led_viewing_angles', [120] * num_leds)
                if not led_viewing_angles:
                    led_viewing_angles = [120] * num_leds
                led_row_directions = group_cfg.get('led_row_directions', [])
            else:
                # Static group - generate positions based on rows
                led_positions = []
                led_rotations = []
                led_sizes = []
                led_viewing_angles = []
                
                for row_idx, led_indices in enumerate(led_rows):
                    y_pos = 0.0
                    z_pos = -row_idx * 2.0  # Space rows by 2cm
                    for led_idx_in_row, led_idx in enumerate(led_indices):
                        x_pos = led_idx_in_row * 1.5  # Space LEDs by 1.5cm
                        led_positions.append((x_pos, y_pos, z_pos))
                        led_rotations.append((1, 0, 0))  # Forward direction
                        led_sizes.append(0.5)
                        led_viewing_angles.append(120)
            
            # Get group position and rotation offset
            # Support both old format (position list) and new format (pos_x/y/z)
            if 'position' in group_cfg:
                pos = group_cfg['position']
                group_pos = [pos[0] if len(pos) > 0 else 0.0, 
                            pos[1] if len(pos) > 1 else 0.0, 
                            pos[2] if len(pos) > 2 else 0.0]
            else:
                group_pos = [
                    group_cfg.get('pos_x', 0.0),
                    group_cfg.get('pos_y', 0.0),
                    group_cfg.get('pos_z', 0.0)
                ]
            
            # Group rotation is now always [0,0,0] (data saved in world-space)
            # Keep group_rot for metadata only
            group_rot = [
                group_cfg.get('rotation_x', group_cfg.get('rot_x', 0)),
                group_cfg.get('rotation_y', group_cfg.get('rot_y', 0)),
                group_cfg.get('rotation_z', group_cfg.get('rot_z', 0))
            ]
            
            # Create individual LED for each LED in the group
            for led_idx in range(num_leds):
                # Compute world position: relative position + group offset
                # (positions in template are relative to group_pos)
                led_pos_local = np.array(led_positions[led_idx])
                led_pos_final = led_pos_local + np.array(group_pos)
                
                # ALWAYS derive rotation angles from led_rotations (direction vectors).
                # led_euler_angles may be stale (not updated after group master rotation).
                # led_rotations are the ground truth in world-space.
                # Convention: direction = Rx(rx) @ Ry(ry) @ Rz(rz) @ [1,0,0]
                # With rx=0: dx = cos(ry)*cos(rz), dy = sin(rz), dz = -sin(ry)*cos(rz)
                led_dir = np.array(led_rotations[led_idx])
                led_dir = led_dir / np.linalg.norm(led_dir)  # ensure unit vector
                
                forward = np.array([1, 0, 0])
                if np.allclose(led_dir, forward, atol=1e-6):
                    rot_angles = [0.0, 0.0, 0.0]
                elif np.allclose(led_dir, -forward, atol=1e-6):
                    rot_angles = [0.0, 180.0, 0.0]
                else:
                    # rz from dy = sin(rz)
                    dy_clamped = np.clip(led_dir[1], -1.0, 1.0)
                    rot_z_rad = np.arcsin(dy_clamped)
                    cos_rz = np.cos(rot_z_rad)
                    if abs(cos_rz) > 1e-6:
                        rot_y_rad = np.arctan2(-led_dir[2], led_dir[0])
                    else:
                        rot_y_rad = 0.0
                    rot_angles = [0.0, np.degrees(rot_y_rad), np.degrees(rot_z_rad)]
                
                # Create individual LED
                led_data = create_individual_led(skip_update_scene=True)
                
                # Set LED properties
                led_data['enable'].value = group_cfg.get('enabled', True)
                led_data['led_on'] = led_states[led_idx]
                led_data['led_on_btn'].color = "#00FF00" if led_states[led_idx] else "#FF0000"
                led_data['pos_x'].value = float(led_pos_final[0])
                led_data['pos_y'].value = float(led_pos_final[1])
                led_data['pos_z'].value = float(led_pos_final[2])
                led_data['rot_x'].value = float(rot_angles[0])
                led_data['rot_y'].value = float(rot_angles[1])
                led_data['rot_z'].value = float(rot_angles[2])
                led_data['size'].value = float(led_sizes[led_idx])
                led_data['viewing_angle'].value = float(led_viewing_angles[led_idx])
                
                # Recover beam_tilt from stored led_beam_tilts
                led_beam_tilts_data = group_cfg.get('led_beam_tilts', [])
                if led_beam_tilts_data and led_idx < len(led_beam_tilts_data):
                    led_data['beam_tilt'].value = int(round(led_beam_tilts_data[led_idx]))
                
                # Recover square_roll from led_row_directions if available
                if led_row_directions and led_idx < len(led_row_directions):
                    saved_row_dir = np.array(led_row_directions[led_idx])
                    # Compute LED direction from rotation angles
                    led_dir = np.array([1.0, 0.0, 0.0])
                    rz_r = np.radians(rot_angles[2]); ry_r = np.radians(rot_angles[1]); rx_r = np.radians(rot_angles[0])
                    Rz_m = np.array([[np.cos(rz_r), -np.sin(rz_r), 0], [np.sin(rz_r), np.cos(rz_r), 0], [0, 0, 1]])
                    Ry_m = np.array([[np.cos(ry_r), 0, np.sin(ry_r)], [0, 1, 0], [-np.sin(ry_r), 0, np.cos(ry_r)]])
                    Rx_m = np.array([[1, 0, 0], [0, np.cos(rx_r), -np.sin(rx_r)], [0, np.sin(rx_r), np.cos(rx_r)]])
                    led_dir = Rx_m @ Ry_m @ Rz_m @ led_dir
                    led_dir = led_dir / np.linalg.norm(led_dir)
                    # Compute default row_dir for this LED's direction (same logic as create_leds)
                    default_row_dir = np.cross(led_dir, np.array([0, 0, 1]))
                    if np.linalg.norm(default_row_dir) < 1e-6:
                        default_row_dir = np.cross(led_dir, np.array([0, 1, 0]))
                    default_row_dir = default_row_dir / np.linalg.norm(default_row_dir)
                    # Compute angle between default and saved row_dir around LED direction
                    dot_val = np.clip(np.dot(default_row_dir, saved_row_dir), -1.0, 1.0)
                    cross_val = np.cross(default_row_dir, saved_row_dir)
                    sign = np.sign(np.dot(cross_val, led_dir))
                    angle_rad = np.arccos(dot_val) * (sign if sign != 0 else 1)
                    led_data['square_roll'].value = int(round(np.degrees(angle_rad)))
                
                # Mark as part of this template for potential regrouping
                led_data['template_source'] = template_name
                led_data['group_index'] = len(groups_data) if len(groups_data) > 1 else None
                led_data['original_group_pos'] = group_pos  # Save original group position
                led_data['original_group_rot'] = group_rot  # Save original group rotation
                
                total_leds_created += 1
        
        # Process individual LEDs from template (if any)
        individual_leds_data = template.get('individual_leds', [])
        for led_cfg in individual_leds_data:
            led_data = create_individual_led(skip_update_scene=True)
            
            # Set LED properties
            led_data['enable'].value = led_cfg.get('enabled', True)
            led_data['led_on'] = led_cfg.get('led_on', True)
            led_data['led_on_btn'].color = "#00FF00" if led_cfg.get('led_on', True) else "#FF0000"
            led_data['pos_x'].value = led_cfg.get('pos_x', 0.0)
            led_data['pos_y'].value = led_cfg.get('pos_y', 0.0)
            led_data['pos_z'].value = led_cfg.get('pos_z', 0.0)
            led_data['rot_x'].value = led_cfg.get('rot_x', 0)
            led_data['rot_y'].value = led_cfg.get('rot_y', 0)
            led_data['rot_z'].value = led_cfg.get('rot_z', 0)
            led_data['size'].value = led_cfg.get('size', 0.5)
            led_data['viewing_angle'].value = led_cfg.get('viewing_angle', 120)
            led_data['square_roll'].value = led_cfg.get('square_roll', 0)
            led_data['beam_tilt'].value = led_cfg.get('beam_tilt', 0)
            
            # Restore lumens override settings for individual LED
            if led_data.get('lumens_override') and led_cfg.get('lumens_override_enabled'):
                led_data['lumens_override'].value = True
                led_data['lumens_value'].value = led_cfg.get('lumens_value', 100)
            
            # Mark as part of this template
            led_data['template_source'] = template_name
            led_data['group_index'] = None
            led_data['original_group_pos'] = None  # No group position for standalone LEDs
            led_data['original_group_rot'] = None
            
            total_leds_created += 1
        
        loading_in_progress[0] = False
        update_scene()
        
        print(f"✓ Loaded {total_leds_created} individual LED(s) from template: {template_name}")
        print("  Edit each LED individually - they will be saved as a group when you save the project")

    # Individual LEDs folder (single LED management)

    def create_individual_led(skip_update_scene=False):
        """Create a new individual LED with position and rotation controls."""
        led_id = next_individual_led_id[0]
        next_individual_led_id[0] += 1
        
        # Create LED folder with controls
        with individual_leds_folder:
            led_folder = server.gui.add_folder(f"LED {led_id}")
        
        with led_folder:
            enable_chk = server.gui.add_checkbox("Enable", initial_value=True)
            
            server.gui.add_html("<b>LED Control:</b>")
            led_on_btn = server.gui.add_button("💡 LED", color="#00FFFF")  # Cyan color when on
            
            server.gui.add_html("<b>Position (cm):</b>")
            pos_x = server.gui.add_slider("X", min=-100, max=100, step=0.1, initial_value=0.0)
            pos_y = server.gui.add_slider("Y", min=-50, max=50, step=0.1, initial_value=0.0)
            pos_z = server.gui.add_slider("Z", min=-50, max=50, step=0.1, initial_value=0.0)
            
            server.gui.add_html("<b>Rotation (degrees):</b>")
            rot_x = server.gui.add_slider("Rotation X (red axis)", min=-180, max=180, step=1, initial_value=0)
            rot_y = server.gui.add_slider("Rotation Y (green axis)", min=-180, max=180, step=1, initial_value=0)
            rot_z = server.gui.add_slider("Rotation Z (blue axis)", min=-180, max=180, step=1, initial_value=0)
            
            server.gui.add_html("<b>Size:</b>")
            size_slider = server.gui.add_slider("Square side (cm)", min=0.1, max=5.0, step=0.1, initial_value=0.5)
            
            server.gui.add_html("<b>Viewing Angle:</b>")
            viewing_angle_slider = server.gui.add_slider("Viewing angle (°)", min=10, max=130, step=5, initial_value=120)
            
            server.gui.add_html("<b>Rotazione quadrato:</b>")
            square_roll_slider = server.gui.add_slider("Rotate on axis (°)", min=-180, max=180, step=1, initial_value=0)
            
            server.gui.add_html("<b>Beam Tilt:</b>")
            beam_tilt_slider = server.gui.add_slider("Tilt beam sopra/sotto (°)", min=-180, max=180, step=1, initial_value=0)
            
            server.gui.add_html("<b>Lumens Override:</b>")
            lumens_override_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
            lumens_slider_ind = server.gui.add_slider("Lumens (lm)", min=1, max=900000, step=1, initial_value=100)
            
            server.gui.add_html("<b>Lente esterna (collimatrice):</b>")
            ext_lens_chk = server.gui.add_checkbox("Enable external lens", initial_value=False)
            ext_lens_angle = server.gui.add_slider("Lens beam angle (°)", min=5, max=120, step=5, initial_value=30)
            ext_lens_efficiency = server.gui.add_slider("Lens efficiency (%)", min=10, max=100, step=1, initial_value=80)
            
            remove_btn = server.gui.add_button("Remove LED", color="red")
        
        # Store LED data
        led_data = {
            'id': led_id,
            'folder': led_folder,
            'enable': enable_chk,
            'led_on': True,  # LED state (on/off)
            'led_on_btn': led_on_btn,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'rot_x': rot_x,
            'rot_y': rot_y,
            'rot_z': rot_z,
            'size': size_slider,
            'viewing_angle': viewing_angle_slider,
            'square_roll': square_roll_slider,
            'beam_tilt': beam_tilt_slider,
            'lumens_override': lumens_override_chk,
            'lumens_value': lumens_slider_ind,
            'ext_lens_enable': ext_lens_chk,
            'ext_lens_angle': ext_lens_angle,
            'ext_lens_efficiency': ext_lens_efficiency,
            'remove_btn': remove_btn,
        }
        
        # Setup callbacks
        def on_led_toggle(_):
            """Toggle LED on/off state."""
            led_data['led_on'] = not led_data['led_on']
            # Update button color
            led_on_btn.color = "#00FFFF" if led_data['led_on'] else "#444444"
            update_scene()
        
        def on_remove(_):
            """Remove this LED."""
            individual_leds.remove(led_data)
            led_folder.remove()
            update_scene()
        
        led_on_btn.on_click(on_led_toggle)
        enable_chk.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_x.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_y.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        pos_z.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        rot_x.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        rot_y.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        rot_z.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        size_slider.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        viewing_angle_slider.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        square_roll_slider.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        beam_tilt_slider.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        lumens_override_chk.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        lumens_slider_ind.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        ext_lens_chk.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        ext_lens_angle.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        ext_lens_efficiency.on_update(lambda _: update_scene() if not loading_in_progress[0] else None)
        remove_btn.on_click(on_remove)
        
        individual_leds.append(led_data)
        
        if not skip_update_scene:
            update_scene()
        
        return led_data
    


    return SimpleNamespace(_clear_panel_slot=_clear_panel_slot, _load_template_into_slot=_load_template_into_slot, create_custom_group=create_custom_group, create_individual_led=create_individual_led, load_custom_group_from_template=load_custom_group_from_template, load_template_as_individual_leds=load_template_as_individual_leds)
