#!/usr/bin/env python3
"""
Update all 3 rendering functions to use explicit Euler rotation matrices
with extrinsic X-Y-Z rotation order (applied to fixed reference frame).
"""

# Read the file
filepath = r'c:\Users\gianmatteo.marietti_\Downloads\E4_LightingSim\interactive_lighting.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern 1: Old rotation angle extraction
old_angles = """                # Get rotation angles (Yaw, Pitch, Roll)
                yaw_deg = group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0  # Tilt Left/Right: Yaw (rotation around Z)
                pitch_deg = group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0  # Tilt Up/Down: Pitch (rotation around Y)
                roll_deg = group['rot_roll'].value if 'rot_roll' in group else 0  # Rotate on itself: Roll (rotation around X)
                
                # Use quaternions via scipy for robust rotation
                # Apply intrinsic rotations: Z (yaw) -> Y (pitch) -> X (roll)
                rotation = Rotation.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)"""

new_angles = """                # Get rotation angles (Euler angles in fixed frame)
                roll_deg = group['rot_roll'].value if 'rot_roll' in group else 0  # Rotation around X
                pitch_deg = group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0  # Rotation around Y
                yaw_deg = group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0  # Rotation around Z
                
                # Build Euler rotation matrices (extrinsic X-Y-Z)
                roll_rad = np.radians(roll_deg)
                pitch_rad = np.radians(pitch_deg)
                yaw_rad = np.radians(yaw_deg)
                
                # Rotation matrix around X axis
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]
                ])
                
                # Rotation matrix around Y axis
                Ry = np.array([
                    [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
                ])
                
                # Rotation matrix around Z axis
                Rz = np.array([
                    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]
                ])
                
                # Compose: extrinsic X-Y-Z means R = Rz @ Ry @ Rx
                R_total = Rz @ Ry @ Rx"""

# Replace all occurrences
count1 = content.count(old_angles)
print(f"Found {count1} occurrences of old rotation pattern")
content = content.replace(old_angles, new_angles)

# Pattern 2: Old direction rotation using quaternions
old_rotation = """                # Apply rotations ONLY to LED directions using quaternions
                # Use original rotations if available
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                rotated_directions = []
                for orig_dir in original_rotations:
                    # Apply rotation using quaternions
                    rotated_dir = rotation.apply(np.array(orig_dir))
                    rotated_directions.append(tuple(rotated_dir))"""

new_rotation = """                # Apply rotations ONLY to LED directions using rotation matrix
                # Use original rotations if available
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                rotated_directions = []
                for orig_dir in original_rotations:
                    # Apply rotation matrix
                    rotated_dir = R_total @ np.array(orig_dir)
                    rotated_directions.append(tuple(rotated_dir))"""

count2 = content.count(old_rotation)
print(f"Found {count2} occurrences of old direction rotation pattern")
content = content.replace(old_rotation, new_rotation)

# Write back
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Updated {count1} angle extraction sections and {count2} direction rotation sections")
print("✓ All functions now use explicit Euler rotation matrices with extrinsic X-Y-Z order")
