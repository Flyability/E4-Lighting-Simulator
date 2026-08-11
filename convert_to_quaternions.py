#!/usr/bin/env python3
"""
Convert all 3 rendering functions from Euler matrices to quaternions.
"""

# Read the file
filepath = r'c:\Users\gianmatteo.marietti_\Downloads\E4_LightingSim\interactive_lighting.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find and replace
old_pattern = """                # Convert to radians
                roll_rad = np.radians(roll_deg)
                pitch_rad = np.radians(pitch_deg)
                yaw_rad = np.radians(yaw_deg)
                
                # Rotation matrices (Yaw->Pitch->Roll order)
                # Roll: rotation around X axis
                R_roll = np.array([
                    [1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]
                ])
                
                # Pitch: rotation around Y axis
                R_pitch = np.array([
                    [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
                ])
                
                # Yaw: rotation around Z axis
                R_yaw = np.array([
                    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]
                ])
                
                # Combined rotation: apply Yaw first, then Pitch, then Roll
                R_combined = R_roll @ R_pitch @ R_yaw"""

new_pattern = """                # Use quaternions via scipy for robust rotation
                # Apply intrinsic rotations: Z (yaw) -> Y (pitch) -> X (roll)
                rotation = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)"""

# Replace all occurrences
count = content.count(old_pattern)
print(f"Found {count} occurrences of Euler matrix pattern")
content = content.replace(old_pattern, new_pattern)

# Pattern 2: Replace the direction rotation code
old_dir_pattern = """                # Apply rotations ONLY to LED directions
                # Use original rotations if available
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                rotated_directions = []
                for orig_dir in original_rotations:
                    # Direct rotation of direction vector
                    dir_array = np.array(orig_dir)
                    rotated_dir = R_combined @ dir_array
                    rotated_directions.append(tuple(rotated_dir))"""

new_dir_pattern = """                # Apply rotations ONLY to LED directions using quaternions
                # Use original rotations if available
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                rotated_directions = []
                for orig_dir in original_rotations:
                    # Apply rotation using quaternions
                    rotated_dir = rotation.apply(np.array(orig_dir))
                    rotated_directions.append(tuple(rotated_dir))"""

count2 = content.count(old_dir_pattern)
print(f"Found {count2} occurrences of direction rotation pattern")
content = content.replace(old_dir_pattern, new_dir_pattern)

# Write back
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Successfully converted {count} matrix sections and {count2} direction rotations to quaternions")
