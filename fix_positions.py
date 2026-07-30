#!/usr/bin/env python3
"""
Fix the backward compatibility checks to save relative positions instead of absolute posit ions.
"""

# Read the file
filepath = r'c:\Users\gianmatteo.marietti_\Downloads\E4_LightingSim\interactive_lighting.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")

# Fix location 1: capture_camera_fov_image (around line 4277)
# Check lines 4275-4284 (0-indexed: 4274-4283)
print("\n=== Checking location 1 (around line 4277) ===")
for i in range(4274, 4285):
    print(f"Line {i+1}: {lines[i][:60]}...")

if 'original_led_positions' in lines[4276] and "'led_positions'" in lines[4276]:
    print("\n✓ Found first location - Applying fix...")
    # The old section spans lines 4275-4284 (9 lines)
    # Replace with new code (15 lines)
    new_lines_1 = [
        '                # Get ORIGINAL positions (never rotate, only translate)\n',
        '                # Save originals if missing (for backward compatibility)\n',
        '                position_offset = np.array([group[\'pos_x\'].value, group[\'pos_y\'].value, group[\'pos_z\'].value])\n',
        '                if \'original_led_positions\' not in group and \'led_positions\' in group:\n',
        '                    # Calculate relative positions by subtracting current offset\n',
        '                    relative_positions = []\n',
        '                    for led_pos in group.get(\'led_positions\', []):\n',
        '                        relative_pos = np.array(led_pos) - position_offset\n',
        '                        relative_positions.append(tuple(relative_pos))\n',
        '                    group[\'original_led_positions\'] = relative_positions\n',
        '                if \'original_led_rotations\' not in group and \'led_rotations\' in group:\n',
        '                    group[\'original_led_rotations\'] = [tuple(rot) for rot in group.get(\'led_rotations\', [])]\n',
        '\n',
        '                # Use original_led_positions if available, otherwise led_positions  \n',
        '                original_positions = group.get(\'original_led_positions\', group.get(\'led_positions\', []))\n',
    ]
    lines[4274:4284] = new_lines_1
    print(f"  Replaced {9} lines with {len(new_lines_1)} lines")
else:
    print("\n✗ Location 1 not found where expected!")

# After first replacement, line numbers shift
# Location 2 was at 5711, but with +6 lines difference, it's now at 5717
shift = len(new_lines_1) - 10  # difference from original 10 lines to new lines
print(f"\nLine shift after first fix: {shift}")

# Fix location 2: update_scene (around line 5711 + shift)
loc2_start = 5711 + shift
print(f"\n=== Checking location 2 (around line {loc2_start}) ===")
for i in range(loc2_start-3, loc2_start+7):
    if i < len(lines):
        print(f"Line {i+1}: {lines[i][:60]}...")

# Find the exact location by searching for the pattern
found_loc2 = False
for i in range(loc2_start-10, min(loc2_start+10, len(lines))):
    if 'original_led_positions' in lines[i] and "'led_positions'" in lines[i] and i != 4276:
        print(f"\n✓ Found second location at line {i+1} - Applying fix...")
        # Replace the 9-line section with 15 new lines
        new_lines_2 = [
            '                # Get ORIGINAL positions (never rotate, only translate)\n',
            '                # Save originals if missing (for backward compatibility)\n',
            '                position_offset = np.array([group[\'pos_x\'].value, group[\'pos_y\'].value, group[\'pos_z\'].value])\n',
            '                if \'original_led_positions\' not in group and \'led_positions\' in group:\n',
            '                    # Calculate relative positions by subtracting current offset\n',
            '                    relative_positions = []\n',
            '                    for led_pos in group.get(\'led_positions\', []):\n',
            '                        relative_pos = np.array(led_pos) - position_offset\n',
            '                        relative_positions.append(tuple(relative_pos))\n',
            '                    group[\'original_led_positions\'] = relative_positions\n',
            '                if \'original_led_rotations\' not in group and \'led_rotations\' in group:\n',
            '                    group[\'original_led_rotations\'] = [tuple(rot) for rot in group.get(\'led_rotations\', [])]\n',
            '\n',
            '                # Use original_led_positions if available, otherwise led_positions  \n',
            '                original_positions = group.get(\'original_led_positions\', group.get(\'led_positions\', []))\n',
        ]
        # Find end of section (skip to line with position_offset = ...)
        end_idx = i
        for j in range(i, min(i+15, len(lines))):
            if 'position_offset = np.array' in lines[j]:
                end_idx = j + 1
                break
        
        lines[i-2:end_idx] = new_lines_2
        print(f"  Replaced lines {i-1} to {end_idx} with {len(new_lines_2)} new lines")
        found_loc2 = True
        break

if not found_loc2:
    print("\n✗ Location 2 not found!")

# Write back
print("\n=== Writing changes back to file ===")
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"✓ Done! File updated with {len(lines)} total lines")
