#!/usr/bin/env python3
"""
Fix rendering functions to apply rigid body rotation to BOTH positions and directions.
"""

filepath = r'c:\Users\gianmatteo.marietti_\Downloads\E4_LightingSim\interactive_lighting.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Old pattern: only translates positions, only rotates directions
old_pattern = """                # Compose: extrinsic X-Y-Z means R = Rz @ Ry @ Rx
                R_total = Rz @ Ry @ Rx
                
                # Get ORIGINAL positions (never rotate, only translate)
                # Save originals if missing (for backward compatibility)
                position_offset = np.array([group['pos_x'].value, group['pos_y'].value, group['pos_z'].value])
                if 'original_led_positions' not in group and 'led_positions' in group:
                    # Calculate relative positions by subtracting current offset
                    relative_positions = []
                    for led_pos in group.get('led_positions', []):
                        relative_pos = np.array(led_pos) - position_offset
                        relative_positions.append(tuple(relative_pos))
                    group['original_led_positions'] = relative_positions
                if 'original_led_rotations' not in group and 'led_rotations' in group:
                    group['original_led_rotations'] = [tuple(rot) for rot in group.get('led_rotations', [])]

                # Use original_led_positions if available, otherwise led_positions  
                original_positions = group.get('original_led_positions', group.get('led_positions', []))
                
                # Keep positions fixed, only apply translation
                translated_positions = []
                for orig_pos in original_positions:
                    final_pos = np.array(orig_pos) + position_offset
                    translated_positions.append(tuple(final_pos))
                
                # Apply rotations ONLY to LED directions using rotation matrix
                # Use original rotations if available
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                rotated_directions = []
                for orig_dir in original_rotations:
                    # Apply rotation matrix
                    rotated_dir = R_total @ np.array(orig_dir)
                    rotated_directions.append(tuple(rotated_dir))"""

# New pattern: RIGID BODY rotation - rotate BOTH positions and directions
new_pattern = """                # Compose: extrinsic X-Y-Z means R = Rz @ Ry @ Rx
                R_total = Rz @ Ry @ Rx
                
                # RIGID BODY rotation: rotate BOTH positions AND directions
                # R is orthogonal => all distances are preserved
                position_offset = np.array([group['pos_x'].value, group['pos_y'].value, group['pos_z'].value])
                
                # Save originals if missing (for backward compatibility)
                if 'original_led_positions' not in group and 'led_positions' in group:
                    relative_positions = []
                    for led_pos in group.get('led_positions', []):
                        relative_pos = np.array(led_pos) - position_offset
                        relative_positions.append(tuple(relative_pos))
                    group['original_led_positions'] = relative_positions
                if 'original_led_rotations' not in group and 'led_rotations' in group:
                    group['original_led_rotations'] = [tuple(rot) for rot in group.get('led_rotations', [])]

                # Get originals
                original_positions = group.get('original_led_positions', group.get('led_positions', []))
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                
                # Rotate relative positions then translate (rigid body)
                translated_positions = []
                for orig_pos in original_positions:
                    rotated_pos = R_total @ np.array(orig_pos)
                    final_pos = rotated_pos + position_offset
                    translated_positions.append(tuple(final_pos))
                
                # Rotate direction vectors
                rotated_directions = []
                for orig_dir in original_rotations:
                    rotated_dir = R_total @ np.array(orig_dir)
                    rotated_directions.append(tuple(rotated_dir))"""

count = content.count(old_pattern)
print(f"Found {count} occurrences of old rendering pattern")
content = content.replace(old_pattern, new_pattern)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Updated {count} rendering functions to use rigid body rotation")
