"""
Interactive lighting design tool using Viser.
Allows real-time adjustment of LED parameters with sliders.
"""

import numpy as np
import viser
import viser.transforms as tf
import time
import multiprocessing
import json
import os
from functools import partial
import trimesh


class LED:
    """Square planar LED with hemispherical (Lambertian) emission pattern."""

    def __init__(
        self,
        width=1.0,
        viewing_angle=60,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        color=(1.0, 0.0, 0.0),
    ):
        self.width = width
        self.viewing_angle = viewing_angle  # total cone angle in degrees (e.g., 120°)
        self.position = np.array(position)
        # Normalize direction, use default (1,0,0) if direction is zero vector
        dir_array = np.array(direction)
        dir_norm = np.linalg.norm(dir_array)
        if dir_norm < 1e-10:  # Avoid division by zero
            self.direction = np.array([1.0, 0.0, 0.0])
        else:
            self.direction = dir_array / dir_norm
        self.color = color

    def get_visualization_rays(self, ray_length=60.0):
        """Get chief ray + marginal rays for visualization."""
        rays = []

        # Build local coordinate system
        z_axis = self.direction
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Chief ray
        rays.append((self.position.copy(), self.direction.copy()))

        # Marginal rays at viewing angle edges (half-angle from center)
        theta = np.radians(self.viewing_angle / 2.0)  # Convert total angle to half-angle
        s, c = np.sin(theta), np.cos(theta)

        for local_dir in [(s, 0, c), (-s, 0, c), (0, s, c), (0, -s, c)]:
            world_dir = (
                local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis
            )
            world_dir = world_dir / np.linalg.norm(world_dir)
            rays.append((self.position.copy(), world_dir))

        return rays


def create_leds(
    front_angle_deg,
    side_angle_deg,
    viewing_angle,
    radius,
    circle_center_x,
    group_rotations=None,
    group_rotations_y=None,
    row_enabled=None,
    led_states=None,
    group_offsets=None,
    custom_groups_configs=None,
    individual_leds_configs=None,
    create_base_groups=True,
):
    """Create 4 LED groups; each group contains 3 LEDs spaced by 3 mm.
    Optionally add multiple custom groups at specified positions.
    
    Args:
        create_base_groups: If False, skip creating the 4 base LED groups (front+/-, side+/-)
    """
    angles_deg = [front_angle_deg, -front_angle_deg, side_angle_deg, -side_angle_deg]
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]

    if group_rotations is None:
        group_rotations = [0.0, 0.0, 0.0, 0.0]
    
    if group_rotations_y is None:
        group_rotations_y = [0.0, 0.0, 0.0, 0.0]
    
    if group_offsets is None:
        group_offsets = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    
    if custom_groups_configs is None:
        custom_groups_configs = []
    
    if individual_leds_configs is None:
        individual_leds_configs = []

    leds = []
    led_index = 0  # Track global LED index
    front_x_positions = {}  # Store X positions of front groups for side groups alignment
    
    # Only create base groups if requested
    if create_base_groups:
        for i, angle_deg in enumerate(angles_deg):
            angle_rad = np.radians(angle_deg)
            x = circle_center_x + radius * np.cos(angle_rad)
            y = radius * np.sin(angle_rad)
            z = 0.0

            # Apply Y offset (green axis) based on group type to position only
            # Front groups (i=0, i=1): ±6.5 cm (13 cm apart)
            # Side groups (i=2, i=3): positioned so first LED is 3cm from last LED of front group
            # Front last LED: 6.5 + 0.8 = 7.3 cm, Side first LED: 7.3 + 3.0 = 10.3 cm, Side center: 10.3 + 0.8 = 11.1 cm
            if i in (0, 1):  # Front
                y_offset = 6.5 if i == 0 else -6.5
                front_x_positions[i] = x  # Store X for corresponding side group
            else:  # Side (i in (2, 3))
                y_offset = 11.1 if i == 2 else -11.1
                # Use same X as corresponding front group (side positive with front positive, side negative with front negative)
                corresponding_front_idx = 0 if i == 2 else 1
                x = front_x_positions[corresponding_front_idx]
            y = y + y_offset
            
            # Apply group offset (translation)
            offset_x, offset_y, offset_z = group_offsets[i]
            x += offset_x
            y += offset_y
            z += offset_z

            # Direction: radially outward from circle center (after offset)
            dir_x = x - circle_center_x
            dir_y = y
            dir_z = 0

            # Build local in-plane axes for arranging rows and LEDs
            z_axis = np.array((dir_x, dir_y, dir_z), dtype=float)
            if np.linalg.norm(z_axis) == 0:
                z_axis = np.array((1.0, 0.0, 0.0))
            else:
                z_axis = z_axis / np.linalg.norm(z_axis)

            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            # Parameters: LED squares are 5mm (0.5cm) with 3mm spacing between them
            # Distance between centers = square_size + spacing = 0.5cm + 0.3cm = 0.8cm
            led_spacing_cm = 0.8  # Distance between LED centers (5mm square + 3mm gap)
            # Between rows: 6mm spacing (edge to edge) + 5mm square = 1.1cm center-to-center
            row_spacing_cm = 1.1  # 11 mm between row centers (5mm square + 6mm gap)

            # Four rows with specified in-plane inclinations (degrees)
            inclinations = [90,30, -30,-90]
            # Row centers offsets along Z-axis (blue axis)
            # Rows 2 and 3 (central): 1.1 cm apart, rows 1 and 4 (outer): 0.3 cm from central rows
            row_offsets = [-0.85, -0.55, 0.55, 0.85]  # cm along Z axis

            for row_idx, alpha_deg in enumerate(inclinations):
                # Always create LEDs even if row is disabled (for visualization)
                alpha = np.radians(alpha_deg)
                
                # Center point for this row: distributed along Z axis (blue axis, vertical)
                center_off = row_offsets[row_idx]
                row_center = np.array((x, y, z)) + np.array([0, 0, 1]) * center_off

                # Compute rotated LED direction for this row
                # Base azimuth: radial direction in XY plane (points toward group's angle)
                radial = np.array((dir_x, dir_y, 0.0), dtype=float)
                if np.linalg.norm(radial) == 0:
                    radial_unit = np.array((1.0, 0.0, 0.0))
                else:
                    radial_unit = radial / np.linalg.norm(radial)
                
                # Calculate local tangent axis (Y local): perpendicular to radial in XY plane
                # For radial = (rx, ry, 0), tangent = (-ry, rx, 0) normalized
                tangent_axis = np.array([-radial_unit[1], radial_unit[0], 0.0])
                
                # First apply group rotation around LOCAL Y axis (tangent axis - tilts forward/backward)
                rot_y_deg = float(group_rotations_y[i])
                rot_y_rad = np.radians(rot_y_deg)
                c = np.cos(rot_y_rad)
                s = np.sin(rot_y_rad)
                t = 1.0 - c
                # Rodrigues rotation matrix around tangent_axis
                ux, uy, uz = tangent_axis[0], tangent_axis[1], tangent_axis[2]
                rot_y_matrix = np.array([
                    [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy],
                    [t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux],
                    [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c   ]
                ])
                
                # Apply rotation to directions and positions (entire block rotates together)
                rotated_radial = rot_y_matrix @ radial_unit
                rotated_row_dir = rot_y_matrix @ x_axis
                # Rotate row_center position around group center
                row_center_rel = row_center - np.array([x, y, z])
                row_center_rotated = rot_y_matrix @ row_center_rel
                row_center = np.array([x, y, z]) + row_center_rotated
                
                # Then apply group azimuth rotation Z (rotate around Z axis - blue axis)
                rot_deg = float(group_rotations[i])
                rot_rad = np.radians(rot_deg)
                ca_r, sa_r = np.cos(rot_rad), np.sin(rot_rad)
                rotated_radial = np.array([
                    ca_r * rotated_radial[0] - sa_r * rotated_radial[1],
                    sa_r * rotated_radial[0] + ca_r * rotated_radial[1],
                    rotated_radial[2],
                ])
                # Apply rotation Z to row direction
                rotated_row_dir = np.array([
                    ca_r * rotated_row_dir[0] - sa_r * rotated_row_dir[1],
                    sa_r * rotated_row_dir[0] + ca_r * rotated_row_dir[1],
                    rotated_row_dir[2],
                ])
                # Apply rotation Z to row_center as well (RIGID BODY)
                row_center_rel = row_center - np.array([x, y, z])
                row_center = np.array([x, y, z]) + np.array([
                    ca_r * row_center_rel[0] - sa_r * row_center_rel[1],
                    sa_r * row_center_rel[0] + ca_r * row_center_rel[1],
                    row_center_rel[2],
                ])
                
                # Tilt around axis perpendicular to radial: apply to rotated Z axis so rows follow group rotation
                z_unit = np.array((0.0, 0.0, 1.0))
                # Apply Y local rotation to Z axis as well
                rotated_z = rot_y_matrix @ z_unit
                # Apply Z rotation to rotated Z axis
                rotated_z = np.array([
                    ca_r * rotated_z[0] - sa_r * rotated_z[1],
                    sa_r * rotated_z[0] + ca_r * rotated_z[1],
                    rotated_z[2],
                ])
                rotated_dir = np.cos(alpha) * rotated_radial + (-np.sin(alpha)) * rotated_z
                rotated_dir = rotated_dir / np.linalg.norm(rotated_dir)

                # If row 1 or 4 (indices 0 or 3), move row center 0.5 cm back along rotated radial direction
                if row_idx in (0, 3):
                    row_center = row_center - rotated_radial * 0.5

                # Three LEDs along the row (spaced along green/Y axis direction: rotated_row_dir)
                for led_in_row, off in enumerate([-led_spacing_cm, 0.0, led_spacing_cm]):
                    pos = tuple(row_center + rotated_row_dir * off)
                    
                    # Check if this LED should be enabled based on row_enabled and led_states
                    is_row_enabled = row_enabled is None or row_enabled[row_idx]
                    is_led_enabled = led_states is None or led_states[led_index]
                    
                    # Create LED object (always create it to maintain fixed indices)
                    led = LED(
                        width=0.5,  # 5mm square = 0.5cm
                        viewing_angle=viewing_angle,
                        position=pos,
                        direction=tuple(rotated_dir),
                        color=colors[i],
                    )
                    # Add enabled flag and metadata to LED object
                    led.enabled = is_row_enabled and is_led_enabled
                    led.led_index = led_index  # Store the global index
                    led.row_direction = rotated_row_dir.copy()  # Store rotated row direction for consistent square orientation
                    leds.append(led)
                    led_index += 1

    # Add custom groups if any are enabled
    for custom_group_config in custom_groups_configs:
        if not custom_group_config.get('enabled', False):
            continue
        
        # Check if this is a dynamic group (from individual LEDs)
        is_dynamic = 'num_leds' in custom_group_config and 'led_positions' in custom_group_config
        
        if is_dynamic:
            # Dynamic group: create LEDs with specific positions and rotations
            custom_led_states_array = custom_group_config.get('led_states', [])
            led_positions = custom_group_config.get('led_positions', [])
            led_rotations = custom_group_config.get('led_rotations', [])
            led_sizes = custom_group_config.get('led_sizes', [])
            led_viewing_angles = custom_group_config.get('led_viewing_angles', [])
            led_row_dirs = custom_group_config.get('led_row_directions', [])
            num_leds = custom_group_config.get('num_leds', 0)
            
            custom_color = (1.0, 0.0, 1.0)  # Magenta for custom group
            
            for i in range(num_leds):
                if i >= len(led_positions):
                    break
                
                pos_x, pos_y, pos_z = led_positions[i]
                # led_rotations now contains direction vectors, not angles
                if i < len(led_rotations):
                    direction = np.array(led_rotations[i])
                    # Ensure direction is not zero vector
                    if np.linalg.norm(direction) < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                size = led_sizes[i] if i < len(led_sizes) else 0.5
                led_view_angle = led_viewing_angles[i] if i < len(led_viewing_angles) else viewing_angle
                is_led_on = custom_led_states_array[i] if i < len(custom_led_states_array) else True
                
                # Get row direction: from config if available, else compute from direction
                if i < len(led_row_dirs):
                    row_dir = np.array(led_row_dirs[i])
                else:
                    # Compute default row direction from LED direction
                    row_dir = np.cross(direction, [0, 0, 1])
                    if np.linalg.norm(row_dir) < 0.01:
                        row_dir = np.cross(direction, [0, 1, 0])
                    row_dir = row_dir / np.linalg.norm(row_dir)
                
                led = LED(
                    width=size,
                    viewing_angle=led_view_angle,
                    position=(pos_x, pos_y, pos_z),
                    direction=tuple(direction),
                    color=custom_color,
                )
                led.enabled = is_led_on
                led.led_index = led_index
                led.row_direction = row_dir
                led.is_custom = True  # Mark as custom group LED
                led.is_dynamic_group = True  # Mark as dynamic group LED
                leds.append(led)
                led_index += 1
        else:
            # Standard custom group: 12 LEDs with fixed geometry
            custom_pos = custom_group_config.get('position', (0.0, 0.0, 0.0))
            custom_rot_x_deg = custom_group_config.get('rotation_x', 0.0)
            custom_rot_y_deg = custom_group_config.get('rotation_y', 0.0)
            custom_rot_z_deg = custom_group_config.get('rotation_z', 0.0)
            custom_led_states_array = custom_group_config.get('led_states', [True] * 12)
            custom_row_enabled = custom_group_config.get('row_enabled', [True] * 4)
            
            # Custom group positioned at specified location with Front+ characteristics (angle 0°)
            x, y, z = custom_pos
            
            # Direction: radially outward from (0, 0, 0) for angle 0° (Front+ style)
            dir_x = 1.0  # Points along +X axis (like Front+)
            dir_y = 0.0
            dir_z = 0.0

            # Build local in-plane axes
            z_axis = np.array((dir_x, dir_y, dir_z), dtype=float)
            z_axis = z_axis / np.linalg.norm(z_axis)

            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            # Apply roll rotation (rotation_x) around radial/forward axis BEFORE the row loop
            # For custom groups at 0°, radial = [1,0,0], so roll = standard Rx matrix
            roll_rad = np.radians(custom_rot_x_deg)
            cr, sr = np.cos(roll_rad), np.sin(roll_rad)
            roll_matrix = np.array([
                [1, 0, 0],
                [0, cr, -sr],
                [0, sr, cr]
            ])
            # Roll rotates the local frame axes (but not the radial/forward axis itself)
            x_axis = roll_matrix @ x_axis
            y_axis = roll_matrix @ y_axis
            # The vertical direction used for row offsets also gets rolled
            rolled_z = roll_matrix @ np.array([0.0, 0.0, 1.0])

            # LED spacing parameters
            led_spacing_cm = 0.8
            row_spacing_cm = 1.1
            inclinations = [90, 30, -30, -90]
            row_offsets = [-0.85, -0.55, 0.55, 0.85]

            custom_color = (1.0, 0.0, 1.0)  # Magenta for custom group

            for row_idx, alpha_deg in enumerate(inclinations):
                alpha = np.radians(alpha_deg)
                center_off = row_offsets[row_idx]
                # Use rolled Z direction for row offsets (rigid body)
                row_center = np.array((x, y, z)) + rolled_z * center_off

                # Compute rotated LED direction for this row
                radial_unit = np.array([1.0, 0.0, 0.0])  # Points along +X
                
                # Calculate local tangent axis (Y local): perpendicular to radial in XY plane
                # For radial = (1, 0, 0), tangent = (0, 1, 0)
                tangent_axis = np.array([-radial_unit[1], radial_unit[0], 0.0])
                
                # Apply custom group rotation around LOCAL Y axis (tangent axis - tilts forward/backward)
                rot_y_rad = np.radians(custom_rot_y_deg)
                c = np.cos(rot_y_rad)
                s = np.sin(rot_y_rad)
                t = 1.0 - c
                # Rodrigues rotation matrix around tangent_axis
                ux, uy, uz = tangent_axis[0], tangent_axis[1], tangent_axis[2]
                rot_y_matrix = np.array([
                    [t*ux*ux + c,    t*ux*uy - s*uz, t*ux*uz + s*uy],
                    [t*ux*uy + s*uz, t*uy*uy + c,    t*uy*uz - s*ux],
                    [t*ux*uz - s*uy, t*uy*uz + s*ux, t*uz*uz + c   ]
                ])
                
                # Apply rotation to directions and positions (entire block rotates together)
                rotated_radial = rot_y_matrix @ radial_unit
                rotated_x_axis = rot_y_matrix @ x_axis
                # Rotate row_center position around group center
                row_center_rel = row_center - np.array([x, y, z])
                row_center_rotated = rot_y_matrix @ row_center_rel
                row_center = np.array([x, y, z]) + row_center_rotated
                
                # Then apply custom group rotation Z (rotate around Z axis - blue axis)
                rot_z_rad = np.radians(custom_rot_z_deg)
                cz, sz = np.cos(rot_z_rad), np.sin(rot_z_rad)
                # Rotation matrix Z applied to already Y-rotated radial
                rotated_radial = np.array([
                    cz * rotated_radial[0] - sz * rotated_radial[1],
                    sz * rotated_radial[0] + cz * rotated_radial[1],
                    rotated_radial[2],
                ])
                # Apply rotation Z to row direction
                rotated_row_dir = np.array([
                    cz * rotated_x_axis[0] - sz * rotated_x_axis[1],
                    sz * rotated_x_axis[0] + cz * rotated_x_axis[1],
                    rotated_x_axis[2],
                ])
                # Apply rotation Z to row_center as well (RIGID BODY: all positions must rotate together)
                row_center_rel = row_center - np.array([x, y, z])
                row_center = np.array([x, y, z]) + np.array([
                    cz * row_center_rel[0] - sz * row_center_rel[1],
                    sz * row_center_rel[0] + cz * row_center_rel[1],
                    row_center_rel[2],
                ])
                
                # Tilt around axis perpendicular to radial: apply to rotated Z axis so rows follow group rotation
                z_unit = rolled_z.copy()  # Use rolled Z (accounts for rotation_x/roll)
                # Apply Y local rotation to Z axis as well
                rotated_z = rot_y_matrix @ z_unit
                # Apply Z rotation to rotated Z axis
                rotated_z = np.array([
                    cz * rotated_z[0] - sz * rotated_z[1],
                    sz * rotated_z[0] + cz * rotated_z[1],
                    rotated_z[2],
                ])
                rotated_dir = np.cos(alpha) * rotated_radial + (-np.sin(alpha)) * rotated_z
                rotated_dir = rotated_dir / np.linalg.norm(rotated_dir)

                # If row 1 or 4, move row center 0.5 cm back
                if row_idx in (0, 3):
                    row_center = row_center - rotated_radial * 0.5

                # Three LEDs along the row
                for led_in_row, off in enumerate([-led_spacing_cm, 0.0, led_spacing_cm]):
                    pos = tuple(row_center + rotated_row_dir * off)
                    
                    custom_led_idx = row_idx * 3 + led_in_row
                    is_row_enabled = custom_row_enabled[row_idx] if row_idx < len(custom_row_enabled) else True
                    is_led_enabled = custom_led_states_array[custom_led_idx] if custom_led_idx < len(custom_led_states_array) else True
                    
                    led = LED(
                        width=0.5,  # 5mm square = 0.5cm
                        viewing_angle=viewing_angle,
                        position=pos,
                        direction=tuple(rotated_dir),
                        color=custom_color,
                    )
                    led.enabled = is_row_enabled and is_led_enabled
                    led.led_index = led_index
                    led.row_direction = rotated_row_dir.copy()
                    led.is_custom = True  # Mark as custom group LED
                    leds.append(led)
                    led_index += 1

    # Add individual LEDs
    for individual_led_config in individual_leds_configs:
        if not individual_led_config.get('enabled', True):
            print(f"DEBUG: Skipping individual LED - enabled=False")
            continue
        
        # Get LED on/off state (but still create the LED object)
        led_is_on = individual_led_config.get('led_on', True)
        print(f"DEBUG: Creating individual LED - enabled=True, led_on={led_is_on}")
        
        # Get LED parameters
        pos_x = individual_led_config.get('pos_x', 0.0)
        pos_y = individual_led_config.get('pos_y', 0.0)
        pos_z = individual_led_config.get('pos_z', 0.0)
        rot_x_deg = individual_led_config.get('rot_x', 0.0)
        rot_y_deg = individual_led_config.get('rot_y', 0.0)
        rot_z_deg = individual_led_config.get('rot_z', 0.0)
        size = individual_led_config.get('size', 0.5)
        led_viewing_angle = individual_led_config.get('viewing_angle', viewing_angle)
        
        # Calculate direction vector from rotations
        # Start with direction pointing along +X axis (1, 0, 0)
        # Apply rotations: first Z, then Y, then X
        direction = np.array([1.0, 0.0, 0.0])
        
        # Rotation around Z axis (blue)
        rot_z_rad = np.radians(rot_z_deg)
        cos_z, sin_z = np.cos(rot_z_rad), np.sin(rot_z_rad)
        rot_z_matrix = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        direction = rot_z_matrix @ direction
        
        # Rotation around Y axis (green)
        rot_y_rad = np.radians(rot_y_deg)
        cos_y, sin_y = np.cos(rot_y_rad), np.sin(rot_y_rad)
        rot_y_matrix = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        direction = rot_y_matrix @ direction
        
        # Rotation around X axis (red)
        rot_x_rad = np.radians(rot_x_deg)
        cos_x, sin_x = np.cos(rot_x_rad), np.sin(rot_x_rad)
        rot_x_matrix = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ])
        direction = rot_x_matrix @ direction
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Compute row_direction based on square_roll angle
        # First, compute default row_direction perpendicular to direction
        default_row_dir = np.cross(direction, np.array([0, 0, 1]))
        if np.linalg.norm(default_row_dir) < 1e-6:
            default_row_dir = np.cross(direction, np.array([0, 1, 0]))
        default_row_dir = default_row_dir / np.linalg.norm(default_row_dir)
        
        # Apply square_roll: rotate row_direction around the LED's direction vector
        square_roll_deg = individual_led_config.get('square_roll', 0.0)
        if abs(square_roll_deg) > 0.01:
            sq_roll_rad = np.radians(square_roll_deg)
            # Rodrigues' rotation formula: rotate default_row_dir around direction by sq_roll_rad
            k = direction  # rotation axis (already normalized)
            v = default_row_dir
            row_dir = v * np.cos(sq_roll_rad) + np.cross(k, v) * np.sin(sq_roll_rad) + k * np.dot(k, v) * (1 - np.cos(sq_roll_rad))
        else:
            row_dir = default_row_dir
        
        # Create LED
        individual_color = (0.0, 1.0, 1.0)  # Cyan for individual LEDs
        led = LED(
            width=size,
            viewing_angle=led_viewing_angle,
            position=(pos_x, pos_y, pos_z),
            direction=tuple(direction),
            color=individual_color,
        )
        led.enabled = led_is_on  # LED is active only if turned on
        led.led_index = led_index
        led.row_direction = row_dir
        led.is_individual = True  # Mark as individual LED
        leds.append(led)
        led_index += 1

    return leds


# Helper functions for multiprocessing (must be at module level for pickle serialization)
def _ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2):
    """
    Möller–Trumbore ray-triangle intersection algorithm.
    Returns distance t if intersection exists, None otherwise.
    """
    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < epsilon:
        return None  # Ray is parallel to triangle
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return None
    
    t = f * np.dot(edge2, q)
    
    if t > epsilon:
        return t
    
    return None

def _ray_mesh_intersection(ray_origin, ray_direction, mesh_data):
    """
    Check if ray intersects with STL mesh using trimesh's optimized BVH ray tracing.
    Returns minimum distance if intersection exists, None otherwise.
    mesh_data: dict with 'vertices', 'faces', 'transform' (4x4 matrix)
    """
    if mesh_data is None:
        return None
    
    # Check if we have a cached trimesh object, otherwise create one
    if 'trimesh_obj' not in mesh_data:
        # Create trimesh object with BVH acceleration structure
        import trimesh
        mesh_data['trimesh_obj'] = trimesh.Trimesh(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces'],
            process=False  # Don't validate/repair mesh (faster)
        )
    
    mesh = mesh_data['trimesh_obj']
    transform = mesh_data.get('transform', np.eye(4))
    
    # Apply transformation to ray origin and direction (inverse transform)
    # Instead of transforming mesh vertices, transform ray to mesh's local space
    try:
        inv_transform = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        # Singular matrix, use identity
        inv_transform = np.eye(4)
    
    # Transform ray to mesh local coordinates
    ray_origin_local = (inv_transform @ np.append(ray_origin, 1))[:3]
    ray_direction_local = (inv_transform[:3, :3] @ ray_direction)
    ray_direction_local = ray_direction_local / np.linalg.norm(ray_direction_local)
    
    # Use trimesh's optimized ray intersection (uses BVH internally)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin_local],
        ray_directions=[ray_direction_local],
        multiple_hits=False  # Only need closest hit
    )
    
    if len(locations) == 0:
        return None
    
    # Calculate distance in world space
    hit_point_local = locations[0]
    hit_point_world = (transform @ np.append(hit_point_local, 1))[:3]
    distance = np.linalg.norm(hit_point_world - ray_origin)
    
    return distance

def _ray_box_intersection(pos, direction, box):
    """Check ray-box intersection for absorbers (supports rotation via quaternion)."""
    center = np.array(box['center'], dtype=float)
    half = np.array(box['half_sizes'], dtype=float)
    rotation = box.get('rotation', None)
    
    # If box has rotation, transform ray to box's local space
    if rotation is not None:
        qw, qx, qy, qz = rotation
        # Convert quaternion to rotation matrix
        # Rotation matrix from quaternion (w, x, y, z)
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        # Transform ray to local space (inverse rotation)
        R_inv = R.T  # Orthogonal matrix, inverse = transpose
        local_pos = R_inv @ (pos - center)
        local_dir = R_inv @ direction
        # Now test in local axis-aligned box space
        pos = local_pos
        direction = local_dir
        center = np.array([0.0, 0.0, 0.0])  # Center is at origin in local space
    
    # Standard axis-aligned box intersection test
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

def _calculate_lambertian_exponent(viewing_angle, ray_uniformity):
    """Calculate Lambertian exponent for LED."""
    theta_half = np.radians(viewing_angle / 2.0)
    cos_half = np.cos(theta_half)
    if cos_half > 0.01:
        n_base = np.log(0.5) / np.log(cos_half)
        n_base = np.clip(n_base, 0.1, 10.0)
    else:
        n_base = 1.0
    n = n_base * (1.0 + ray_uniformity * 2.0)
    n = np.clip(n, 0.1, 30.0)
    return n

def _process_led_wall_worker(args):
    """Worker function for single wall ray tracing (multiprocessing)."""
    led, params = args
    
    # Unpack parameters
    wall_dist = params['wall_dist']
    rays_per_led = params['rays_per_led']
    grid_size = params['grid_size']
    wall_size = params['wall_size']
    lumens_per_led = params['lumens_per_led']
    absorbers = params['absorbers']
    ray_uniformity = params['ray_uniformity']
    led_idx = params['led_idx']
    
    # Initialize local grid (stores lux = lm/m²)
    local_grid = np.zeros((grid_size, grid_size))
    cell_size = wall_size / grid_size  # cm
    cell_area_cm2 = cell_size * cell_size
    cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
    half_size = wall_size / 2
    
    # Set seed for reproducibility
    np.random.seed((42 + led_idx) % (2**32))
    
    # Build local coordinate system
    z_axis = led.direction
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, [0, 0, 1])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Calculate emission parameters
    max_theta = np.radians(led.viewing_angle / 2.0)
    n = _calculate_lambertian_exponent(led.viewing_angle, ray_uniformity)
    norm_factor = n + 1.0
    
    # Trace rays
    for _ in range(rays_per_led):
        # Sample direction
        u1, u2 = np.random.uniform(0, 1, 2)
        cos_max = np.cos(max_theta)
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = 2 * np.pi * u2
        
        local_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        world_dir = (
            local_dir[0] * x_axis +
            local_dir[1] * y_axis +
            local_dir[2] * z_axis
        )
        world_dir = world_dir / np.linalg.norm(world_dir)
        
        # Calculate lumens for this ray
        cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
        intensity_coefficient = np.power(cos_theta_clamped, n)
        lumens_per_ray = (lumens_per_led / rays_per_led) * intensity_coefficient * norm_factor
        
        # Check absorber intersection
        hit_absorbed = False
        if absorbers:
            for a in absorbers:
                t_hit = _ray_box_intersection(led.position, world_dir, a)
                if t_hit is not None and t_hit > 0:
                    hit_absorbed = True
                    break
        
        # Check STL mesh intersection
        if not hit_absorbed and params.get('stl_mesh_data') is not None:
            t_hit = _ray_mesh_intersection(led.position, world_dir, params['stl_mesh_data'])
            if t_hit is not None and t_hit > 0:
                hit_absorbed = True
        
        if hit_absorbed:
            continue
        
        # Check wall intersection
        if world_dir[0] > 0:
            t = (wall_dist - led.position[0]) / world_dir[0]
            if t > 0:
                hit_y = led.position[1] + world_dir[1] * t
                hit_z = led.position[2] + world_dir[2] * t
                
                grid_y = int((hit_y + half_size) / cell_size)
                grid_z = int((hit_z + half_size) / cell_size)
                
                if 0 <= grid_y < grid_size and 0 <= grid_z < grid_size:
                    # Convert lumens to lux (lm/m²)
                    lux_contribution = lumens_per_ray / cell_area_m2
                    local_grid[grid_z, grid_y] += lux_contribution
    
    return local_grid

def _process_led_worker(args):
    """Worker function to process rays for a single LED (for multiprocessing)."""
    led, params = args
    
    # Unpack parameters
    front_dist = params['front_dist']
    side_dist = params['side_dist']
    top_bottom_dist = params['top_bottom_dist']
    back_dist = params.get('back_dist')
    led_x_center = params.get('led_x_center', -35)
    num_rays_per_led = params['num_rays_per_led']
    grid_size = params['grid_size']
    lumens_per_led = params['lumens_per_led']
    absorbers = params['absorbers']
    stl_mesh_data = params.get('stl_mesh_data')
    ray_uniformity = params['ray_uniformity']
    grid_shapes = params['grid_shapes']
    wall_specs = params['wall_specs']
    
    # Initialize local grids for this LED (stores lux = lm/m²)
    local_grids = {
        'front': np.zeros(grid_shapes['front']),
        'left': np.zeros(grid_shapes['left']),
        'right': np.zeros(grid_shapes['right']),
        'top': np.zeros(grid_shapes['top']),
        'bottom': np.zeros(grid_shapes['bottom'])
    }
    local_ray_hits = {'front': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    
    # Add back wall if enabled
    if back_dist is not None:
        local_grids['back'] = np.zeros(grid_shapes['back'])
        local_ray_hits['back'] = 0
    
    local_total_rays = 0
    
    # Calculate cell areas for each wall (in m²)
    cell_areas_m2 = {}
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
        cell_area_cm2 = cell_width_cm * cell_height_cm
        cell_areas_m2[wall_name] = cell_area_cm2 / 10000.0  # Convert cm² to m²
    
    # Build local coordinate system from LED direction
    z_axis = led.direction
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, [0, 0, 1])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Calculate rays per LED
    rays_traced = num_rays_per_led * grid_size * grid_size
    
    for _ in range(rays_traced):
        local_total_rays += 1
        
        # Sample rays within viewing angle cone in LED frame
        u1, u2 = np.random.uniform(0, 1, 2)
        max_theta = np.radians(led.viewing_angle / 2.0)  # Use full viewing angle
        cos_max = np.cos(max_theta)
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = 2 * np.pi * u2
        
        # Local direction in LED frame
        local_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Transform to world coordinates
        world_dir = (
            local_dir[0] * x_axis
            + local_dir[1] * y_axis
            + local_dir[2] * z_axis
        )
        world_dir = world_dir / np.linalg.norm(world_dir)
        
        # Calculate lumens for this ray
        n = _calculate_lambertian_exponent(led.viewing_angle, ray_uniformity)
        norm_factor = n + 1.0
        cos_n_theta = cos_theta ** n
        lumens_per_ray = (lumens_per_led / rays_traced) * cos_n_theta * norm_factor
        
        # Check absorber intersection
        hit_absorbed = False
        if absorbers:
            for a in absorbers:
                t_hit = _ray_box_intersection(led.position, world_dir, a)
                if t_hit is not None and t_hit > 0:
                    hit_absorbed = True
                    break
        
        # Check STL mesh intersection
        if not hit_absorbed and stl_mesh_data is not None:
            t_hit = _ray_mesh_intersection(led.position, world_dir, stl_mesh_data)
            if t_hit is not None and t_hit > 0:
                hit_absorbed = True
        
        if hit_absorbed:
            continue
        
        # Calculate intersection with each wall
        intersections = []
        
        # Front wall
        if world_dir[0] > 0:
            t = (front_dist - led.position[0]) / world_dir[0]
            if t > 0:
                y = led.position[1] + world_dir[1] * t
                z = led.position[2] + world_dir[2] * t
                intersections.append(('front', t, y, z))
        
        # Left wall
        if world_dir[1] < 0:
            t = (-side_dist - led.position[1]) / world_dir[1]
            if t > 0:
                x = led.position[0] + world_dir[0] * t
                z = led.position[2] + world_dir[2] * t
                intersections.append(('left', t, x, z))
        
        # Right wall
        if world_dir[1] > 0:
            t = (side_dist - led.position[1]) / world_dir[1]
            if t > 0:
                x = led.position[0] + world_dir[0] * t
                z = led.position[2] + world_dir[2] * t
                intersections.append(('right', t, x, z))
        
        # Top wall
        if world_dir[2] > 0:
            t = (top_bottom_dist - led.position[2]) / world_dir[2]
            if t > 0:
                x = led.position[0] + world_dir[0] * t
                y = led.position[1] + world_dir[1] * t
                intersections.append(('top', t, x, y))
        
        # Bottom wall
        if world_dir[2] < 0:
            t = (-top_bottom_dist - led.position[2]) / world_dir[2]
            if t > 0:
                x = led.position[0] + world_dir[0] * t
                y = led.position[1] + world_dir[1] * t
                intersections.append(('bottom', t, x, y))
        
        # Back wall (at negative X, symmetric to front wall)
        if back_dist is not None and world_dir[0] < 0:
            back_x_pos = -back_dist
            t = (back_x_pos - led.position[0]) / world_dir[0]
            if t > 0:
                y = led.position[1] + world_dir[1] * t
                z = led.position[2] + world_dir[2] * t
                intersections.append(('back', t, y, z))
        
        if not intersections:
            continue
        
        wall_name, t_min, coord1, coord2 = min(intersections, key=lambda x: x[1])
        wall_spec = wall_specs[wall_name]
        
        # Map coordinates to grid indices
        if wall_name == 'front' or wall_name == 'back':
            size_y = wall_spec['size_y']
            size_z = wall_spec['size_z']
            grid_size_y = wall_spec['grid_y']
            grid_size_z = wall_spec['grid_z']
            y_idx = int((coord1 + size_y/2) / (size_y/grid_size_y))
            z_idx = int((coord2 + size_z/2) / (size_z/grid_size_z))
            grid_i = z_idx
            grid_j = y_idx
        elif wall_name in ['left', 'right']:
            size_x = wall_spec['size_x']
            size_z = wall_spec['size_z']
            grid_size_x = wall_spec['grid_x']
            grid_size_z = wall_spec['grid_z']
            x_min = wall_spec['x_min']
            x_idx = int((coord1 - x_min) / (size_x/grid_size_x))
            z_idx = int((coord2 + size_z/2) / (size_z/grid_size_z))
            grid_i = z_idx
            grid_j = x_idx
        else:  # top, bottom
            size_x = wall_spec['size_x']
            size_y = wall_spec['size_y']
            grid_size_x = wall_spec['grid_x']
            grid_size_y = wall_spec['grid_y']
            x_min = wall_spec['x_min']
            x_idx = int((coord1 - x_min) / (size_x/grid_size_x))
            y_idx = int((coord2 + size_y/2) / (size_y/grid_size_y))
            grid_i = y_idx
            grid_j = x_idx
        
        # Clamp indices to grid bounds
        grid_shape = local_grids[wall_name].shape
        grid_i = max(0, min(grid_shape[0] - 1, grid_i))
        grid_j = max(0, min(grid_shape[1] - 1, grid_j))
        
        # Convert lumens to lux (lm/m²)
        lux_contribution = lumens_per_ray / cell_areas_m2[wall_name]
        local_grids[wall_name][grid_i, grid_j] += lux_contribution
        local_ray_hits[wall_name] += 1
    
    return local_grids, local_ray_hits, local_total_rays


def main():
    # Create Viser server
    server = viser.ViserServer()
    print(f"Viser server running at: http://localhost:8080")
    print("\n" + "="*60)
    print("  LED Lighting Simulation - Interactive Tool")
    print("="*60)
    print("\n📋 To get started:")
    print("  1. Create a 🆕 New Project (empty, add custom groups)")
    print("  2. Or 📂 Load an existing configuration (e.g., Elios 3)")
    print("\n💡 All LEDs are disabled until you load or create a project.")
    print("="*60 + "\n")

    # --- Configuration Management ---
    config_dir = "configs"
    custom_groups_templates_dir = "custom_groups_templates"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(custom_groups_templates_dir):
        os.makedirs(custom_groups_templates_dir)
    
    # Save the Elios 3 configuration if it doesn't exist
    elios3_config = {
        "name": "Elios 3",
        "description": "Configurazione standard: front +/-, side +/-",
        "viewing_angle": 120,
        "radius": 35,
        "circle_center_x": -35,
        "group_rotations": [0.7, -0.7, 18, -18],
        "group_rotations_y": [0, 0, 0, 0],
        "group_offsets": [
            [0.0, 1.6, 0.0],
            [0.0, -1.6, 0.0],
            [-1.3, -33.1, 0.0],
            [-1.3, 33.1, 0.0]
        ],
        "row_enabled": [False, True, True, False],
        "led_states": [True] * 48,
        "custom_groups": []
    }
    elios3_path = os.path.join(config_dir, "elios3.json")
    if not os.path.exists(elios3_path):
        with open(elios3_path, "w") as f:
            json.dump(elios3_config, f, indent=4)

    # LED state array: 4 groups × 4 rows × 3 LEDs = 48 LEDs total
    # Start with all LEDs disabled (user must load a project or create new)
    led_states = [False] * 48
    
    # Flag to track if a project is loaded
    project_loaded = [False]  # Use list for mutability in nested functions
    current_config_name = [""]  # Track which configuration is loaded
    loading_in_progress = [False]  # Flag to prevent callbacks during config loading
    
    # Custom groups - list of dictionaries, each containing group configuration
    custom_groups = []  # Each group: {id, enable, pos_x, pos_y, pos_z, rot, led_states, buttons, folder}
    next_custom_group_id = [0]  # Counter for unique IDs (use list to allow modification in nested functions)
    
    # Individual LEDs - list of single LED configurations
    individual_leds = []  # Each LED: {id, enable, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, size, folder}
    next_individual_led_id = [0]  # Counter for unique IDs
    
    # Template folders - track loaded templates with master controls
    template_folders = []  # List of template folder handles to remove on new project/load
    
    # Store button handles for LED control
    led_buttons = {}
    row_buttons = {}
    group_buttons = {}
    
    # Group colors (defined early for use in config functions)
    group_colors_hex = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33"]

    def get_current_config():
        """Retrieve current GUI values for saving."""
        # Save custom groups
        custom_groups_data = []
        for group in custom_groups:
            group_cfg = {
                'enabled': group['enable'].value,
                'position': [group['pos_x'].value, group['pos_y'].value, group['pos_z'].value],
                'rotation_x': 0.0,  # Rotations are baked into led_positions/led_rotations
                'rotation_y': 0.0,
                'rotation_z': 0.0,
                'led_states': group['led_states'][:],
                'template_name': group.get('template_name'),  # Save template association
                'initial_pos': group.get('initial_pos', [0.0, 0.0, 0.0]),  # Save initial position
                'initial_rot': group.get('initial_rot', [0, 0, 0]),  # Save initial rotation
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
            for led in leds_list_sorted:
                rot_x = np.radians(led['rot_x'].value)
                rot_y = np.radians(led['rot_y'].value)
                rot_z = np.radians(led['rot_z'].value)
                
                # Build rotation matrix
                Rx = np.array([[1, 0, 0], [0, np.cos(rot_x), -np.sin(rot_x)], [0, np.sin(rot_x), np.cos(rot_x)]])
                Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
                Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0], [np.sin(rot_z), np.cos(rot_z), 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx
                
                # Apply rotation to forward direction (1, 0, 0)
                direction = R @ np.array([1, 0, 0])
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
                group_rot = original_rot if original_rot is not None else [0, 0, 0]
            else:
                # Calculate average position (center of group)
                group_pos = [
                    sum(p[0] for p in led_positions) / num_leds,
                    sum(p[1] for p in led_positions) / num_leds,
                    sum(p[2] for p in led_positions) / num_leds
                ]
                group_rot = [0, 0, 0]
            
            # Convert LED positions to relative (from group center)
            # If we have original rotation, need to reverse it
            if original_rot is not None and any(original_rot):
                # Build inverse rotation matrix
                rot_x_rad = np.radians(group_rot[0])
                rot_y_rad = np.radians(group_rot[1])
                rot_z_rad = np.radians(group_rot[2])
                
                Rx = np.array([[1, 0, 0], [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)], [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])
                Ry = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)], [0, 1, 0], [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])
                Rz = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0], [np.sin(rot_z_rad), np.cos(rot_z_rad), 0], [0, 0, 1]])
                R_group = Rz @ Ry @ Rx
                R_group_inv = R_group.T  # Inverse is transpose for rotation matrices
                
                # Convert positions to relative by removing group offset and rotation
                led_positions_relative = []
                for pos in led_positions:
                    pos_offset = np.array([pos[0] - group_pos[0], pos[1] - group_pos[1], pos[2] - group_pos[2]])
                    pos_local = R_group_inv @ pos_offset
                    led_positions_relative.append(tuple(pos_local))
                
                # Also reverse-transform the direction vectors
                led_rotations_original = []
                for direction in led_rotations:
                    dir_local = R_group_inv @ np.array(direction)
                    led_rotations_original.append(tuple(dir_local))
                led_rotations = led_rotations_original
            else:
                led_positions_relative = [(p[0] - group_pos[0], p[1] - group_pos[1], p[2] - group_pos[2]) for p in led_positions]
            
            # Compute row directions from direction vectors
            led_row_directions = []
            for direction in led_rotations:
                dir_arr = np.array(direction)
                row_dir = np.cross(dir_arr, np.array([0, 0, 1]))
                norm = np.linalg.norm(row_dir)
                if norm > 1e-6:
                    row_dir = row_dir / norm
                else:
                    row_dir = np.array([0, -1, 0])
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
                'template_name': template_name if template_name != "unnamed" else None,
                'initial_pos': [0.0, 0.0, 0.0],
                'initial_rot': [0, 0, 0]
            }
            
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
                'square_roll': led['square_roll'].value
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
            "custom_groups": custom_groups_data,
            "individual_leds": individual_leds_data,
            "absorbers": {
                "enabled": absorbers_enable.value,
                "abs0": {"x": abs0_off_x.value, "y": abs0_off_y.value, "z": abs0_off_z.value},
                "abs1": {"x": abs1_off_x.value, "y": abs1_off_y.value, "z": abs1_off_z.value},
                "abs2": {"x": abs2_off_x.value, "y": abs2_off_y.value, "z": abs2_off_z.value, "rot_z": abs2_rot_z.value},
                "abs3": {"x": abs3_off_x.value, "y": abs3_off_y.value, "z": abs3_off_z.value, "rot_z": abs3_rot_z.value}
            },
            "stl_model": {
                "file_path": stl_file_path.value,
                "absorber_enable": stl_absorber_enable.value,
                "visible": stl_visible.value,
                "scale": stl_scale.value,
                "position": [stl_pos_x.value, stl_pos_y.value, stl_pos_z.value],
                "rotation": [stl_rot_x.value, stl_rot_y.value, stl_rot_z.value],
                "opacity": stl_opacity.value,
                "wireframe": stl_wireframe.value
            } if stl_mesh_data[0] is not None else None
        }

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
    
    def apply_config(cfg):
        """Update GUI elements with values from config."""
        nonlocal loading_in_progress
        loading_in_progress[0] = True  # Disable callbacks during loading
        
        # Clear all existing custom groups first (this calls update_scene())
        clear_all_custom_groups()
        clear_all_template_folders()
        
        viewing_angle_slider.value = cfg.get("viewing_angle", 120)
        radius_slider.value = cfg.get("radius", 35)
        circle_center_slider.value = cfg.get("circle_center_x", -35)
        
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
        groups_by_template = {}
        standalone_groups = []
        for group_cfg in custom_groups_data:
            template_name = group_cfg.get('template_name')
            if template_name:
                if template_name not in groups_by_template:
                    groups_by_template[template_name] = []
                groups_by_template[template_name].append(group_cfg)
            else:
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
                group_data['led_sizes'] = group_cfg.get('led_sizes', [])
                group_data['led_viewing_angles'] = group_cfg.get('led_viewing_angles', [])
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
            # Rotations are always 0 (transformations are baked into led_positions/led_rotations)
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
            
            # Enable the group AFTER all parameters are loaded
            group_data['enable'].value = group_cfg.get('enabled', True)
        
        # Recreate template folders with master controls
        for template_name, template_groups_cfg in groups_by_template.items():
            if len(template_groups_cfg) == 0:
                continue
            
            print(f"Recreating template folder: {template_name} with {len(template_groups_cfg)} group(s)")
            
            # Create master folder
            template_folder = server.gui.add_folder(f"Template: {template_name}")
            
            with template_folder:
                master_enable = server.gui.add_checkbox("Enable All", initial_value=True)
                master_pos_x = server.gui.add_slider("Master Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_pos_y = server.gui.add_slider("Master Position Y (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_pos_z = server.gui.add_slider("Master Position Z (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
                master_rot_x = server.gui.add_slider("Master Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
                master_rot_y = server.gui.add_slider("Master Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
                master_rot_z = server.gui.add_slider("Master Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
                server.gui.add_html("<hr style='margin:8px 0;'>")
                remove_template_btn = server.gui.add_button(f"Remove All ({len(template_groups_cfg)} groups)", color="red")
            
            # Create all groups from this template
            created_groups = []
            for group_cfg in template_groups_cfg:
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
                    group_data['led_sizes'] = group_cfg.get('led_sizes', [])
                    group_data['led_viewing_angles'] = group_cfg.get('led_viewing_angles', [])
                    # IMPORTANT: Save RELATIVE positions as originals (subtract group offset)
                    # The config contains absolute positions, we need relative ones
                    group_offset = np.array([pos[0], pos[1], pos[2]])
                    relative_positions = []
                    for led_pos in group_data['led_positions']:
                        relative_pos = np.array(led_pos) - group_offset
                        relative_positions.append(tuple(relative_pos))
                    group_data['original_led_positions'] = relative_positions
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
                # Rotations are always 0 (transformations are baked into led_positions/led_rotations)
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = 0
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = 0
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = 0
                
                # Load LED states
                led_states_cfg = group_cfg.get('led_states', [])
                for i, state in enumerate(led_states_cfg):
                    if i < len(group_data['led_states']):
                        group_data['led_states'][i] = state
                
                # Update button colors
                if 'update_button_colors' in group_data and group_data['update_button_colors']:
                    group_data['update_button_colors']()
                
                # Enable the group
                group_data['enable'].value = group_cfg.get('enabled', True)
                
                # Store template association and initial offsets
                group_data['template_name'] = template_name
                group_data['initial_pos'] = group_cfg.get('initial_pos', [0.0, 0.0, 0.0])
                group_data['initial_rot'] = group_cfg.get('initial_rot', [0, 0, 0])
                
                # Hide individual group folder - LED controls will be in master folder
                group_data['folder'].visible = False
                
                created_groups.append(group_data)
            
            # Add LED controls in master folder for each group
            with template_folder:
                server.gui.add_html("<hr style='margin:8px 0;'><b>LED Controls:</b>")
                
                # Store button references for dynamic color updates
                master_led_buttons = []  # List of dicts with button references per group
                
                for group_idx, group in enumerate(created_groups):
                    with server.gui.add_folder(f"Group {group_idx + 1} LEDs"):
                        # Store button references for this group
                        group_buttons = {
                            'all_btn': None,
                            'row_btns': {},
                            'led_btns': {}
                        }
                        
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
            
            # Setup master control callbacks
            def make_update_handler(groups_list):
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
                    
                    for group in groups_list:
                        group['enable'].value = master_enable.value
                        if master_enable.value:
                            init_pos = np.array(group.get('initial_pos', [0.0, 0.0, 0.0]))
                            init_rot = np.array(group.get('initial_rot', [0, 0, 0]))
                            
                            # RIGID BODY: rotate group positions around template center
                            rotated_pos = R_master @ init_pos + master_pos_offset
                            group['pos_x'].value = float(rotated_pos[0])
                            group['pos_y'].value = float(rotated_pos[1])
                            group['pos_z'].value = float(rotated_pos[2])
                            
                            # Add master rotations to initial rotations
                            new_rot = init_rot + np.array([master_rot_x.value, master_rot_y.value, master_rot_z.value])
                            group['rot_roll'].value = int(new_rot[0])
                            group['rot_tilt_ud'].value = int(new_rot[1])
                            group['rot_tilt_lr'].value = int(new_rot[2])
                    
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
            
            update_handler = make_update_handler(created_groups)
            remove_handler = make_remove_handler(created_groups, template_folder)
            
            master_enable.on_update(update_handler)
            master_pos_x.on_update(update_handler)
            master_pos_y.on_update(update_handler)
            master_pos_z.on_update(update_handler)
            master_rot_x.on_update(update_handler)
            master_rot_y.on_update(update_handler)
            master_rot_z.on_update(update_handler)
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
    
    def save_custom_group_template(name, groups_list, individual_leds_list):
        """Save all custom groups and individual LEDs as a reusable template."""
        path = os.path.join(custom_groups_templates_dir, f"{name.lower().replace(' ', '_')}.json")
        template = {
            "name": name,
            "groups": groups_list,
            "individual_leds": individual_leds_list
        }
        with open(path, "w") as f:
            json.dump(template, f, indent=4)
        print(f"✓ Template saved with {len(groups_list)} custom group(s) and {len(individual_leds_list)} individual LED(s): {name}")
    
    def get_available_templates():
        """Get list of available custom group templates."""
        if not os.path.exists(custom_groups_templates_dir):
            return []
        files = [f for f in os.listdir(custom_groups_templates_dir) if f.endswith(".json")]
        return [f.replace(".json", "") for f in files]
    
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
        template_folder = server.gui.add_folder(f"Template: {template.get('name', template_name)}")
        
        with template_folder:
            master_enable = server.gui.add_checkbox("Enable All", initial_value=True)
            master_pos_x = server.gui.add_slider("Master Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
            master_pos_y = server.gui.add_slider("Master Position Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            master_pos_z = server.gui.add_slider("Master Position Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            master_rot_x = server.gui.add_slider("Master Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
            master_rot_y = server.gui.add_slider("Master Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
            master_rot_z = server.gui.add_slider("Master Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
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
            # Rotations are always 0 (transformations are baked into led_positions/led_rotations)
            if 'rot_tilt_lr' in group_data:
                group_data['rot_tilt_lr'].value = 0
            if 'rot_tilt_ud' in group_data:
                group_data['rot_tilt_ud'].value = 0
            if 'rot_roll' in group_data:
                group_data['rot_roll'].value = 0
            
            # Load dynamic group properties if present
            if group_cfg.get('is_dynamic', False):
                group_data['is_dynamic'] = True
                group_data['led_positions'] = group_cfg.get('led_positions', [])
                group_data['led_rotations'] = group_cfg.get('led_rotations', [])
                group_data['led_row_directions'] = group_cfg.get('led_row_directions', [])
                group_data['led_sizes'] = group_cfg.get('led_sizes', [])
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
            
            # Enable the group AFTER all parameters are loaded
            group_data['enable'].value = group_cfg.get('enabled', True)
            
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
            
            # Compute row_direction for each LED
            # Row direction is perpendicular to LED direction, lying in a plane
            # For the Elios3 panel geometry, row_dir = cross(direction, [1,0,0]) normalized
            # If direction is along X, use cross(direction, [0,0,1]) as fallback
            led_row_directions = []
            for direction_tuple in led_rotations:
                d = np.array(direction_tuple)
                # Use the same logic as standard panel: cross(z_axis, [0,0,1])
                row_dir = np.cross(d, [0, 0, 1])
                if np.linalg.norm(row_dir) < 0.01:
                    row_dir = np.cross(d, [0, 1, 0])
                row_dir = row_dir / np.linalg.norm(row_dir)
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
            
            for group_idx, group in enumerate(created_groups):
                with server.gui.add_folder(f"Group {group_idx + 1} LEDs"):
                    # Store button references for this group
                    group_buttons = {
                        'all_btn': None,
                        'row_btns': {},
                        'led_btns': {}
                    }
                    
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
            
            for idx, group in enumerate(created_groups):
                group['enable'].value = master_enable.value
                
                # RIGID BODY: rotate group positions around template center
                initial_pos = np.array(initial_positions[idx])
                rotated_pos = R_master @ initial_pos + master_pos_offset
                
                group['pos_x'].value = float(rotated_pos[0])
                group['pos_y'].value = float(rotated_pos[1])
                group['pos_z'].value = float(rotated_pos[2])
                
                # Add master rotations to initial rotations
                group['rot_roll'].value = initial_rotations[idx][0] + master_rot_x.value
                group['rot_tilt_ud'].value = initial_rotations[idx][1] + master_rot_y.value
                group['rot_tilt_lr'].value = initial_rotations[idx][2] + master_rot_z.value
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

    # --- GUI Controls ---
    with server.gui.add_folder("Project Management"):
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Start a new project or load existing</div>")
        
        new_project_btn = server.gui.add_button("🆕 New Project (Empty)", color="#4CAF50")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        
        def get_available_configs():
            files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
            return [f.replace(".json", "") for f in files]

        config_dropdown = server.gui.add_dropdown(
            "Select Configuration",
            options=get_available_configs(),
            initial_value=None
        )

        load_config_btn = server.gui.add_button("📂 Load Configuration")
        
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;margin-bottom:6px;'>Save Current Project</div>")
        
        save_name_input = server.gui.add_text("Project Name", initial_value="")
        save_type_dropdown = server.gui.add_dropdown(
            "Save As",
            options=["Full Configuration", "Custom Group Template"],
            initial_value="Full Configuration"
        )
        save_project_btn = server.gui.add_button("💾 Save Project")

        @new_project_btn.on_click
        def _(_):
            new_project()

        @load_config_btn.on_click
        def _(_):
            name = config_dropdown.value
            if not name:
                print("Error: Please select a configuration to load.")
                return
            path = os.path.join(config_dir, f"{name}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    cfg = json.load(f)
                    num_custom = len(cfg.get("custom_groups", []))
                    print(f"Loading configuration: {name} ({num_custom} custom groups)")
                    current_config_name[0] = name
                    project_loaded[0] = True
                    apply_config(cfg)
                # Set the loaded config name in the save field for easy re-saving
                save_name_input.value = name
                print(f"✓ Configuration loaded: {name}")
                print(f"  💡 Modify and click 'Save Project' to update the configuration")

        @save_project_btn.on_click
        def _(_):
            nonlocal template_dropdown
            name = save_name_input.value.strip()
            if not name:
                print("Error: Please enter a project name.")
                return
            
            save_type = save_type_dropdown.value
            
            if save_type == "Full Configuration":
                # Save complete configuration
                cfg = get_current_config()
                cfg["name"] = name
                
                path = os.path.join(config_dir, f"{name.lower().replace(' ', '_')}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4)
                
                print(f"✓ Configuration saved: {name}")
                # Refresh dropdown options
                config_dropdown.options = get_available_configs()
            else:
                # Save as custom group template - save groups separately
                if len(custom_groups) > 0 or len(individual_leds) > 0:
                    # Save custom groups
                    custom_groups_data = []
                    for group in custom_groups:
                        group_cfg = {
                            'enabled': group['enable'].value,
                            'position': [group['pos_x'].value, group['pos_y'].value, group['pos_z'].value],
                            'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                            'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                            'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                            'led_states': group['led_states'][:]
                        }
                        # Save dynamic group properties if present
                        if group.get('is_dynamic', False):
                            group_cfg['is_dynamic'] = True
                            group_cfg['num_leds'] = group.get('num_leds', 12)
                            # CRITICAL: Save ORIGINAL positions (not the current rotated ones!)
                            # This ensures template always contains undeformed geometry
                            group_cfg['led_positions'] = group.get('original_led_positions', group.get('led_positions', []))
                            group_cfg['led_rotations'] = group.get('original_led_rotations', group.get('led_rotations', []))
                            group_cfg['led_row_directions'] = group.get('original_led_row_directions', group.get('led_row_directions', []))
                            group_cfg['led_sizes'] = group.get('led_sizes', [])
                            group_cfg['led_viewing_angles'] = group.get('led_viewing_angles', [])
                            group_cfg['led_rows'] = group.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
                        custom_groups_data.append(group_cfg)
                    
                    # Save individual LEDs
                    individual_leds_data = []
                    for led in individual_leds:
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
                            'viewing_angle': led['viewing_angle'].value
                        })
                    
                    # Save as template with separate groups
                    save_custom_group_template(name, custom_groups_data, individual_leds_data)
                    # Refresh template dropdown list
                    template_dropdown.options = ["Empty"] + get_available_templates()
                    
                    print(f"✓ Template saved: {len(custom_groups_data)} group(s) + {len(individual_leds_data)} individual LED(s)")
                else:
                    print("Error: No custom groups or individual LEDs to save as template")

    # Store reference to LED Configuration folder and visibility state
    base_groups_active = [False]  # Track if base LED groups are active in current project
    
    led_config_folder = server.gui.add_folder("LED Configuration (Base Groups)")
    led_config_folder.visible = False  # Hidden by default (empty project)
    
    with led_config_folder:
        viewing_angle_slider = server.gui.add_slider(
            "Viewing angle (°) [GWP9LR35: 120°]", min=10, max=130, step=5, initial_value=120
        )
        # Per-group rotation sliders Z axis (rotate beam and visual together)
        rot_front_pos = server.gui.add_slider("Rotate front+ Z (°)", min=-180, max=180, step=1, initial_value=0.7)
        rot_front_neg = server.gui.add_slider("Rotate front- Z (°)", min=-180, max=180, step=1, initial_value=-0.7)
        rot_side_pos = server.gui.add_slider("Rotate side+ Z (°)", min=-180, max=180, step=1, initial_value=18)
        rot_side_neg = server.gui.add_slider("Rotate side- Z (°)", min=-180, max=180, step=1, initial_value=-18)
        
        # Per-group rotation sliders Y axis (local tangent axis - tilts forward/backward)
        rot_y_front_pos = server.gui.add_slider("Rotate front+ local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_front_neg = server.gui.add_slider("Rotate front- local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_side_pos = server.gui.add_slider("Rotate side+ local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_side_neg = server.gui.add_slider("Rotate side- local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        
        # Per-group translation sliders (move entire group along X, Y, Z axes)
        with server.gui.add_folder("Group Positions"):
            # Front+ (Red group)
            with server.gui.add_folder("Front+ (Red)"):
                offset_front_pos_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
                offset_front_pos_y = server.gui.add_slider("Offset Y (cm)", min=-30, max=30, step=0.1, initial_value=1.6)
                offset_front_pos_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Front- (Green group)
            with server.gui.add_folder("Front- (Green)"):
                offset_front_neg_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
                offset_front_neg_y = server.gui.add_slider("Offset Y (cm)", min=-30, max=30, step=0.1, initial_value=-1.6)
                offset_front_neg_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Side+ (Blue group)
            with server.gui.add_folder("Side+ (Blue)"):
                offset_side_pos_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=-1.3)
                offset_side_pos_y = server.gui.add_slider("Offset Y (cm)", min=-40, max=40, step=0.1, initial_value=-33.1)
                offset_side_pos_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Side- (Yellow group)
            with server.gui.add_folder("Side- (Yellow)"):
                offset_side_neg_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=-1.3)
                offset_side_neg_y = server.gui.add_slider("Offset Y (cm)", min=-40, max=50, step=0.1, initial_value=33.1)
                offset_side_neg_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)

    with server.gui.add_folder("Geometry"):
        radius_slider = server.gui.add_slider(
            "Circle radius (cm)", min=10, max=60, step=1, initial_value=35
        )
        wall_dist_slider = server.gui.add_slider(
            "Wall distance (cm)", min=20, max=100, step=5, initial_value=50
        )
        circle_center_slider = server.gui.add_slider(
            "Circle center X (cm)", min=-60, max=0, step=5, initial_value=-35
        )

    with server.gui.add_folder("Display"):
        ray_length_slider = server.gui.add_slider(
            "Ray length (cm)", min=20, max=100, step=5, initial_value=60
        )
        show_random_rays = server.gui.add_checkbox(
            "Show random rays (scales with intensity)", initial_value=False
        )
        show_rays_output = server.gui.add_checkbox(
            "Show rays in output", initial_value=False
        )
        show_led_markers = server.gui.add_checkbox(
            "Show LED markers", initial_value=True
        ) 
        show_intensity_map = server.gui.add_checkbox(
            "Show intensity on wall", initial_value=False
        )
        intensity_rays_slider = server.gui.add_slider(
            "Rays per pixel (↑quality, ↓speed)", min=10, max=50000, step=10, initial_value=50
        )
        ray_uniformity_slider = server.gui.add_slider(
            "Focus factor (0=Standard, 1=3x focused)", min=0.0, max=1.0, step=0.05, initial_value=0.0
        )
        led_lumens_slider = server.gui.add_slider(
            "LED lumens (lm/LED)", min=10, max=1000, step=10, initial_value=100
        )
        calibration_factor_slider = server.gui.add_slider(
            "Calibration factor", min=0.5, max=1.5, step=0.001, initial_value=1.0
        )
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=5, max=100, step=5, initial_value=30
        )
        # Cell area info (updated dynamically)
        cell_area_html = server.gui.add_html(
            "<div style='font-family: sans-serif; font-size: 11px; color: #666; margin-top: -8px; margin-bottom: 8px;'>"
            "Cell area: calculating..."
            "</div>"
        )
        # Intensity legend shown under the sliders as HTML with color swatches
        legend_html = server.gui.add_html(
            "<div style='font-family: sans-serif;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
            "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click<br>'Update Intensity Map' to populate legend</div>"
            "</div>"
        )
        wall_view_size = server.gui.add_slider(
            "Wall view size (cm)", min=100, max=200, step=10, initial_value=100
        )
        # Manual update button for intensity map (computationally expensive)
        update_intensity_button = server.gui.add_button("Update Intensity Map")
        # Reset button: some Viser button handles don't support on_update;
        # we'll detect clicks by polling `reset_button.value` in the main loop.
        reset_button = server.gui.add_button("Reset to original positions")
        # Per-row enable toggles
        row1_chk = server.gui.add_checkbox("Row 1 on", initial_value=False)
        row2_chk = server.gui.add_checkbox("Row 2 on", initial_value=True)
        row3_chk = server.gui.add_checkbox("Row 3 on", initial_value=True)
        row4_chk = server.gui.add_checkbox("Row 4 on", initial_value=False)
        # Absorber controls moved to dedicated folder for clarity

    # 3D Model Import (STL files)
    stl_mesh_handle = [None]  # Store mesh handle for removal/update
    stl_mesh_data = [None]  # Store loaded trimesh object
    
    with server.gui.add_folder("3D Models (STL)"):
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Import 3D CAD models</div>")
        stl_file_path = server.gui.add_text("STL File Path", initial_value=r"C:\Users\gianmatteo.marietti_\Downloads\109045 E3 CAGE ASSEMBLY_Coarse.STL")
        stl_load_button = server.gui.add_button("📂 Load STL", color="#4CAF50")
        stl_clear_button = server.gui.add_button("🗑️ Clear Model", color="#FF5555")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        stl_absorber_enable = server.gui.add_checkbox("Enable as Light Absorber", initial_value=True)
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>When enabled, the 3D model blocks light rays</div>")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        stl_visible = server.gui.add_checkbox("Show Model", initial_value=True)
        stl_scale = server.gui.add_slider("Scale", min=0.01, max=10.0, step=0.01, initial_value=0.1)
        stl_pos_x = server.gui.add_slider("Position X (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_pos_y = server.gui.add_slider("Position Y (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_pos_z = server.gui.add_slider("Position Z (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_rot_x = server.gui.add_slider("Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
        stl_rot_y = server.gui.add_slider("Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
        stl_rot_z = server.gui.add_slider("Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
        stl_opacity = server.gui.add_slider("Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.8)
        stl_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        update_mesh_lighting_btn = server.gui.add_button("🔆 Update Mesh Lighting", color="#FFA500")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Recalculate lighting based on current LED configuration</div>")
        
# Mesh lighting legend (gradient from dark blue to white)
        server.gui.add_html(
            "<div style='font-family: sans-serif; margin-top: 12px;'>" 
            "<div style='font-weight:600; margin-bottom:6px; font-size:12px;'>Mesh Lighting Intensity:</div>"
            "<div style='display:flex; align-items:center; gap:8px;'>"
            "<span style='font-size:10px; color:#888;'>Dark</span>"
            "<div style='flex:1; height:20px; background:linear-gradient(to right, "
            "#000033 0%, #000055 10%, #0000AA 20%, #1133CC 30%, #2255DD 40%, "
            "#3366EE 50%, #5588FF 60%, #77AAFF 65%, #99CCFF 75%, #BBDDFF 85%, #DDEEFF 92%, #FFFFFF 100%); "
            "border:1px solid #444; border-radius:3px;'></div>"
            "<span style='font-size:10px; color:#888;'>Bright</span>"
            "</div>"
            "<div style='font-size:10px; color:#666; margin-top:4px;'>"
            "• Color map: Dark Blue → White<br>"
            "• Physics: Lux = (Lumens × cos θ) / d² | Responds to LED lumens"
            "</div>"
            "</div>"
        )
        
        stl_info_html = server.gui.add_html(
            "<div style='font-family: sans-serif; font-size: 11px; color: #666;'>"
            "No model loaded"
            "</div>"
        )
        
        def load_stl_file():
            """Load STL file and display in scene."""
            file_path = stl_file_path.value.strip()
            if not file_path:
                print("⚠️ Please enter a file path")
                return
            
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return
            
            try:
                print(f"Loading STL file: {file_path}")
                mesh = trimesh.load(file_path)
                
                # Handle multiple meshes (Scene object)
                if isinstance(mesh, trimesh.Scene):
                    # Combine all meshes in the scene
                    mesh = trimesh.util.concatenate(
                        [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                    )
                
                # Center mesh at origin (move centroid to 0,0,0)
                mesh.vertices -= mesh.centroid
                
                stl_mesh_data[0] = mesh
                
                # Calculate mesh dimensions
                num_vertices = len(mesh.vertices)
                num_faces = len(mesh.faces)
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                
                # Auto-calculate ideal scale for this project
                # Project uses cm scale, typical diameter ~70cm (2 * radius 35cm)
                # Assume largest dimension should be around 70cm in project space
                target_size_cm = 70.0  # Target size in cm
                max_dimension = np.max(size)
                
                if max_dimension > 0:
                    # Calculate scale: target_size / current_size
                    # Note: mesh is in STL units (usually mm), we want it in cm for display
                    ideal_scale = target_size_cm / max_dimension
                    stl_scale.value = ideal_scale
                    print(f"Auto-scale applied: {ideal_scale:.4f} (model size: {max_dimension:.1f} → {target_size_cm}cm)")
                
                # Update info
                info_text = (
                    f"<div style='font-family: sans-serif; font-size: 11px; color: #4CAF50;'>"
                    f"✓ Model loaded<br>"
                    f"Vertices: {num_vertices:,}<br>"
                    f"Faces: {num_faces:,}<br>"
                    f"Original size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}<br>"
                    f"Scaled size: {size[0]*ideal_scale:.1f} × {size[1]*ideal_scale:.1f} × {size[2]*ideal_scale:.1f} cm"
                    f"</div>"
                )
                stl_info_html.content = info_text
                
                print(f"✓ STL loaded: {num_vertices:,} vertices, {num_faces:,} faces")
                update_stl_mesh()
                
            except Exception as e:
                print(f"❌ Error loading STL: {e}")
                stl_info_html.content = f"<div style='color:#FF5555;'>Error: {str(e)}</div>"
        
        def update_stl_mesh():
            """Update STL mesh visualization in scene."""
            nonlocal stl_mesh_handle
            
            try:
                # Remove existing mesh
                if stl_mesh_handle[0] is not None:
                    try:
                        stl_mesh_handle[0].remove()
                    except:
                        pass
                    stl_mesh_handle[0] = None
                
                # Add mesh if loaded and visible
                if stl_mesh_data[0] is not None and stl_visible.value:
                    mesh = stl_mesh_data[0].copy()
                    
                    # Validate values (protect against NaN)
                    scale = float(stl_scale.value)
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    
                    pos_x = float(stl_pos_x.value)
                    pos_y = float(stl_pos_y.value)
                    pos_z = float(stl_pos_z.value)
                    if not np.isfinite(pos_x):
                        pos_x = 0.0
                    if not np.isfinite(pos_y):
                        pos_y = 0.0
                    if not np.isfinite(pos_z):
                        pos_z = 0.0
                    
                    rot_x = float(stl_rot_x.value)
                    rot_y = float(stl_rot_y.value)
                    rot_z = float(stl_rot_z.value)
                    if not np.isfinite(rot_x):
                        rot_x = 0.0
                    if not np.isfinite(rot_y):
                        rot_y = 0.0
                    if not np.isfinite(rot_z):
                        rot_z = 0.0
                    
                    opacity = float(stl_opacity.value)
                    if not np.isfinite(opacity):
                        opacity = 0.8
                    
                    # Calculate original and scaled sizes for info display
                    orig_mesh = stl_mesh_data[0]
                    bounds = orig_mesh.bounds
                    orig_size = bounds[1] - bounds[0]
                    scaled_size = orig_size * scale
                    
                    # Update info display with current scale
                    num_vertices = len(orig_mesh.vertices)
                    num_faces = len(orig_mesh.faces)
                    info_text = (
                        f"<div style='font-family: sans-serif; font-size: 11px; color: #4CAF50;'>"
                        f"✓ Model loaded<br>"
                        f"Vertices: {num_vertices:,}<br>"
                        f"Faces: {num_faces:,}<br>"
                        f"Original size: {orig_size[0]:.1f} × {orig_size[1]:.1f} × {orig_size[2]:.1f}<br>"
                        f"Scaled size: {scaled_size[0]:.1f} × {scaled_size[1]:.1f} × {scaled_size[2]:.1f} cm"
                        f"</div>"
                    )
                    stl_info_html.content = info_text
                    
                    # Apply transformations by creating a transformed copy
                    # This prevents the original mesh from being modified
                    
                    # Get original vertices and faces
                    orig_vertices = mesh.vertices.copy()
                    orig_faces = mesh.faces.copy()
                    
                    # Build transformation matrix: Scale * Rotation * Translation
                    # 1. Scale matrix
                    T_scale = np.eye(4)
                    T_scale[0, 0] = scale
                    T_scale[1, 1] = scale
                    T_scale[2, 2] = scale
                    
                    # 2. Rotation matrices
                    T_rot = np.eye(4)
                    if rot_x != 0:
                        T_rot = T_rot @ trimesh.transformations.rotation_matrix(np.radians(rot_x), [1, 0, 0])
                    if rot_y != 0:
                        T_rot = T_rot @ trimesh.transformations.rotation_matrix(np.radians(rot_y), [0, 1, 0])
                    if rot_z != 0:
                        T_rot = T_rot @ trimesh.transformations.rotation_matrix(np.radians(rot_z), [0, 0, 1])
                    
                    # 3. Translation matrix (in cm, will be converted to meters later)
                    T_trans = trimesh.transformations.translation_matrix([pos_x, pos_y, pos_z])
                    
                    # 4. Final cm to meters conversion
                    T_to_meters = np.eye(4) * 0.01  # Scale by 0.01
                    T_to_meters[3, 3] = 1.0  # Keep homogeneous coordinate
                    
                    # Combine: first scale, then rotate, then translate, then convert to meters
                    T_combined = T_to_meters @ T_trans @ T_rot @ T_scale
                    
                    # Apply combined transform to vertices
                    vertices_homogeneous = np.hstack([orig_vertices, np.ones((len(orig_vertices), 1))])
                    vertices_transformed = (T_combined @ vertices_homogeneous.T).T[:, :3]
                    
                    # Transform normals (rotation only, no translation or scale for normals)
                    # Use the upper-left 3x3 of the rotation matrix
                    R_only = T_rot[:3, :3]
                    normals_transformed = (R_only @ mesh.vertex_normals.T).T
                    # Normalize
                    normals_transformed = normals_transformed / (np.linalg.norm(normals_transformed, axis=1, keepdims=True) + 1e-10)
                    
                    # Create new mesh with transformed vertices
                    from trimesh import Trimesh
                    mesh_transformed = Trimesh(
                        vertices=vertices_transformed.astype(np.float32),
                        faces=orig_faces,
                        vertex_normals=normals_transformed.astype(np.float32),
                        process=False  # Don't recompute normals
                    )
                    
                    vertices = mesh_transformed.vertices
                    faces = mesh_transformed.faces
                    vertex_normals = mesh_transformed.vertex_normals
                    
                    # For lighting calculation, vertices are already in world space (meters)
                    vertices_world = vertices
                    
                    # Calculate dynamic lighting based on current LEDs
                    base_color = (0.7, 0.7, 0.9)  # Light blue base color
                    vertex_colors = calculate_mesh_lighting(
                        vertices_world, 
                        vertex_normals, 
                        current_leds,
                        base_color=base_color,
                        led_lumens=led_lumens_slider.value
                    )
                    
                    # Set vertex colors on the mesh before adding to scene
                    # Convert to uint8 RGBA format (Viser/Three.js expects 0-255 range)
                    vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)
                    # Add alpha channel with user-controlled opacity
                    alpha_value = int(opacity * 255)
                    vertex_colors_rgba = np.concatenate([
                        vertex_colors_uint8,
                        np.full((len(vertex_colors_uint8), 1), alpha_value, dtype=np.uint8)
                    ], axis=1)
                    
                    # Assign to mesh visual
                    from trimesh.visual import ColorVisuals
                    mesh_transformed.visual = ColorVisuals(mesh=mesh_transformed, vertex_colors=vertex_colors_rgba)
                    
                    # Add to scene
                    # Note: add_mesh_trimesh doesn't support opacity and wireframe parameters
                    # Opacity is controlled via alpha channel in vertex colors
                    # Wireframe mode is not supported with per-vertex colors
                    # Position, rotation, and scale are already baked into the mesh
                    stl_mesh_handle[0] = server.scene.add_mesh_trimesh(
                        name="/stl_model",
                        mesh=mesh_transformed,
                        visible=True,
                    )
            except Exception as e:
                print(f"Error updating STL mesh: {e}")
        
        def clear_stl_model():
            """Clear loaded STL model."""
            nonlocal stl_mesh_handle
            if stl_mesh_handle[0] is not None:
                try:
                    stl_mesh_handle[0].remove()
                except:
                    pass
                stl_mesh_handle[0] = None
            stl_mesh_data[0] = None
            stl_info_html.content = "<div style='color:#666;'>No model loaded</div>"
            print("STL model cleared")
        
        # Button callbacks
        @stl_load_button.on_click
        def _(_):
            load_stl_file()
        
        @stl_clear_button.on_click
        def _(_):
            clear_stl_model()
        
        @update_mesh_lighting_btn.on_click
        def _(_):
            try:
                if stl_mesh_data[0] is not None:
                    update_stl_mesh()
                    print("✓ Mesh lighting updated")
                else:
                    print("⚠️ No mesh loaded")
            except Exception as e:
                print(f"Error updating mesh lighting: {e}")
        
        # Update mesh when parameters change (with error protection)
        def safe_update_stl(_):
            try:
                update_stl_mesh()
                # Update ray visualization with new STL mesh
                update_scene()
                # Note: Intensity calculations remain manual (use update buttons)
            except Exception as e:
                print(f"Error in STL update callback: {e}")
        
        def safe_update_stl_absorber(_):
            """Update when STL absorber enable/disable changes - affects ray blocking."""
            try:
                # Update ray visualization with new STL absorber state
                update_scene()
                # Note: Intensity calculations remain manual (use update buttons)
            except Exception as e:
                print(f"Error in STL absorber update callback: {e}")
        
        stl_visible.on_update(safe_update_stl)
        stl_scale.on_update(safe_update_stl)
        stl_pos_x.on_update(safe_update_stl)
        stl_pos_y.on_update(safe_update_stl)
        stl_pos_z.on_update(safe_update_stl)
        stl_rot_x.on_update(safe_update_stl)
        stl_rot_y.on_update(safe_update_stl)
        stl_rot_z.on_update(safe_update_stl)
        stl_opacity.on_update(safe_update_stl)
        stl_wireframe.on_update(safe_update_stl)
        stl_absorber_enable.on_update(safe_update_stl_absorber)

    # Room Mode - Cubic room with 5 or 6 walls (back wall optional)
    with server.gui.add_folder("Room Mode"):
        room_mode_enable = server.gui.add_checkbox("Enable Room Mode", initial_value=False)
        show_room_walls = server.gui.add_checkbox("Show Room Walls", initial_value=True)
        show_room_intensity = server.gui.add_checkbox("Show Room Intensity", initial_value=False)
        room_front_dist = server.gui.add_slider(
            "Front wall distance (cm)", min=20, max=200, step=10, initial_value=50
        )
        room_side_dist = server.gui.add_slider(
            "Side walls distance (cm)", min=20, max=300, step=10, initial_value=50
        )
        room_top_bottom_dist = server.gui.add_slider(
            "Top/Bottom walls distance (cm)", min=20, max=300, step=10, initial_value=50
        )
        show_back_wall = server.gui.add_checkbox("Show Back Wall", initial_value=False)
        room_back_dist = server.gui.add_slider(
            "Back wall distance (cm)", min=10, max=100, step=5, initial_value=50
        )
        room_grid_size = server.gui.add_slider(
            "Room walls grid resolution", min=10, max=50, step=5, initial_value=20
        )
        update_room_button = server.gui.add_button("Update Room Intensity")

    # Camera FOV visualization
    with server.gui.add_folder("Camera FOV"):
        show_camera_fov = server.gui.add_checkbox("Show Camera FOV", initial_value=True)
        camera_fov_h = server.gui.add_slider(
            "Horizontal FOV (°)", min=10, max=120, step=1, initial_value=75
        )
        camera_fov_v = server.gui.add_slider(
            "Vertical FOV (°)", min=10, max=90, step=1, initial_value=60
        )
        camera_pos_x = server.gui.add_slider(
            "Camera X pos (cm)", min=-100, max=100, step=1, initial_value=0
        )
        capture_fov_btn = server.gui.add_button("Capture FOV Image", color="green")

    # Store handles for dynamic objects
    camera_fov_handles = []
    led_handles = []
    ray_handles = []
    intensity_handles = []
    room_intensity_handles = []
    room_wall_handles = []
    absorber_handles = []
    
    # Store current LED objects (for reuse in room intensity calculation)
    current_leds = []

    def calculate_mesh_lighting(mesh_vertices, mesh_normals, leds, base_color=(0.7, 0.7, 0.9), led_lumens=100):
        """
        Calculate per-vertex lighting for STL mesh using physical lux calculation.
        
        Args:
            mesh_vertices: Nx3 array of vertex positions in meters (Viser units)
            mesh_normals: Nx3 array of vertex normals
            leds: List of LED objects with position, direction, enabled
            base_color: RGB tuple for base mesh color (0-1 range)
            led_lumens: Luminous flux per LED in lumens (default 100)
        
        Returns:
            Nx3 array of RGB colors for each vertex
        """
        num_vertices = len(mesh_vertices)
        vertex_colors = np.zeros((num_vertices, 3), dtype=np.float32)
        
        # Base ambient lighting (minimum visibility)
        ambient = 0.15  # 15% ambient light
        vertex_colors[:] = np.array(base_color) * ambient
        
        # Get only enabled LEDs
        active_leds = [led for led in leds if getattr(led, 'enabled', True)]
        
        if len(active_leds) == 0:
            return vertex_colors
        
        # Calculate luminous intensity (candelas) from lumens and viewing angle
        # For a cone with half-angle θ: solid_angle = 2π(1 - cos(θ))
        # Luminous intensity I = Φ / Ω (candelas = lumens / steradians)
        
        # For each active LED, calculate contribution to each vertex
        for led in active_leds:
            # LED position in cm, convert to meters
            led_pos_m = np.array(led.position) / 100.0
            led_dir = np.array(led.direction)
            led_dir_norm = led_dir / (np.linalg.norm(led_dir) + 1e-10)
            
            # Calculate solid angle for this LED's cone
            viewing_half_angle_rad = np.radians(led.viewing_angle / 2.0)
            solid_angle = 2 * np.pi * (1 - np.cos(viewing_half_angle_rad))  # steradians
            
            # Protect against division by zero or very small solid angles
            if solid_angle < 0.001:
                solid_angle = 0.001
            
            luminous_intensity = led_lumens / solid_angle  # candelas
            
            # Validate luminous intensity
            if not np.isfinite(luminous_intensity) or luminous_intensity <= 0:
                continue
            
            # Vector from LED to each vertex
            to_vertex = mesh_vertices - led_pos_m  # Shape: (N, 3)
            distances = np.linalg.norm(to_vertex, axis=1, keepdims=True) + 1e-6  # meters
            to_vertex_norm = to_vertex / distances  # Normalized direction to vertex
            
            # Check if vertex is within LED's viewing cone
            cos_angle_to_vertex = np.sum(led_dir_norm * to_vertex_norm, axis=1, keepdims=True)
            cos_half_angle = np.cos(viewing_half_angle_rad)
            
            # Only illuminate vertices within viewing cone
            in_cone = (cos_angle_to_vertex > cos_half_angle).flatten()
            
            if np.sum(in_cone) == 0:
                continue
            
            # Lambert's cosine law: intensity depends on angle of incidence
            # N = surface normal, L = direction from surface to light source
            cos_incident = np.sum(mesh_normals * (-to_vertex_norm), axis=1, keepdims=True)
            cos_incident = np.maximum(0, cos_incident)  # Only positive (facing the light)
            
            # Physical illuminance calculation (lux)
            # E = I × cos(θ_incident) / d²
            # where: E = illuminance (lux), I = luminous intensity (cd), d = distance (m)
            illuminance = np.zeros((num_vertices, 1), dtype=np.float32)
            illuminance[in_cone] = (luminous_intensity * cos_incident[in_cone] / 
                                   (distances[in_cone] ** 2))
            
            # Remove any NaN or infinite values
            illuminance = np.nan_to_num(illuminance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply cone edge falloff for smooth transition
            angle_factor = np.ones((num_vertices, 1), dtype=np.float32)
            angle_factor[in_cone] = np.maximum(0, 
                (cos_angle_to_vertex[in_cone] - cos_half_angle) / (1.0 - cos_half_angle)) ** 2
            illuminance *= angle_factor
            
            # Convert lux to color intensity (normalize to reasonable range)
            # Typical indoor lighting: 100-500 lux
            # Scale factor to map lux to [0, 1] color range
            lux_to_color_scale = 0.002  # Adjust to control brightness response
            color_intensity = np.clip(illuminance * lux_to_color_scale, 0.0, 1.0)
            
            # Ensure no NaN values in final color intensity
            color_intensity = np.nan_to_num(color_intensity, nan=0.0)
            
            # Add this LED's contribution to vertex colors (accumulate intensity)
            # Keep as (N, 1) for proper broadcasting with (N, 3)
            vertex_colors += color_intensity
        
        # Final safety check: remove any remaining NaN or infinite values
        vertex_colors = np.nan_to_num(vertex_colors, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Calculate total intensity per vertex (average across RGB channels)
        intensity_per_vertex = np.mean(vertex_colors, axis=1, keepdims=True)
        
        # Map accumulated intensity to blue-to-white color gradient
        # Dark blue (0,0,0.2) at low intensity → White (1,1,1) at high intensity
        blue_base = np.array([0.0, 0.0, 0.2])  # Dark blue
        white = np.array([1.0, 1.0, 1.0])  # White
        
        # Interpolate between blue and white based on intensity
        # Use a smooth curve for better visual appearance
        intensity_normalized = np.clip(intensity_per_vertex, 0.0, 1.0)
        final_colors = blue_base * (1.0 - intensity_normalized) + white * intensity_normalized
        final_colors = np.clip(final_colors, 0.0, 1.0)
        
        return final_colors

    # Absorbers folder (separate group for easier access)
    absorbers_folder = server.gui.add_folder("Absorbers")
    absorbers_folder.visible = False  # Hidden by default
    
    with absorbers_folder:
        absorbers_enable = server.gui.add_checkbox("Enable absorbers", initial_value=False)
        abs0_off_x = server.gui.add_slider("Abs0 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1)
        abs0_off_y = server.gui.add_slider("Abs0 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=2.5)
        abs0_off_z = server.gui.add_slider("Abs0 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs1_off_x = server.gui.add_slider("Abs1 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1)
        abs1_off_y = server.gui.add_slider("Abs1 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=-2.5)
        abs1_off_z = server.gui.add_slider("Abs1 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs2_off_x = server.gui.add_slider("Abs2 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1.8)
        abs2_off_y = server.gui.add_slider("Abs2 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=-10.5)
        abs2_off_z = server.gui.add_slider("Abs2 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs2_rot_z = server.gui.add_slider("Abs2 rotation Z (deg)", min=-180, max=180, step=1, initial_value=-14)
        abs3_off_x = server.gui.add_slider("Abs3 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1.8)
        abs3_off_y = server.gui.add_slider("Abs3 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=10.5)
        abs3_off_z = server.gui.add_slider("Abs3 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs3_rot_z = server.gui.add_slider("Abs3 rotation Z (deg)", min=-180, max=180, step=1, initial_value=14)

    # Custom LED Groups folder (dynamic groups)
    custom_groups_folder = server.gui.add_folder("Custom LED Groups")
    template_dropdown = None  # Will be initialized later
    
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
        
        with group_folder:
            enable_chk = server.gui.add_checkbox("Enable", initial_value=True)
            pos_x = server.gui.add_slider("Position X (cm)", min=-100, max=100, step=0.1, initial_value=0.0)
            pos_y = server.gui.add_slider("Position Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            pos_z = server.gui.add_slider("Position Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            rot_tilt_lr = server.gui.add_slider("Inclina Sinistra/Destra (°)", min=-180, max=180, step=1, initial_value=0)
            rot_tilt_ud = server.gui.add_slider("Inclina Alto/Basso (°)", min=-180, max=180, step=1, initial_value=0)
            rot_roll = server.gui.add_slider("Ruota su se stesso (°)", min=-180, max=180, step=1, initial_value=0)
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
            custom_groups.remove(group_data)
            group_folder.remove()
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
        
        custom_groups.append(group_data)
        
        # Set initial button colors to match LED states
        update_button_colors()
        
        # Only update scene if not skipping (e.g., when manually adding a group)
        if not skip_update_scene:
            update_scene()
        
        return group_data
    
    with custom_groups_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Add New Custom Group</div>")
        
        template_dropdown = server.gui.add_dropdown(
            "From Template",
            options=["Empty"] + get_available_templates(),
            initial_value="Empty"
        )
        
        load_mode_dropdown = server.gui.add_dropdown(
            "Load Mode",
            options=["As Group (Solid)", "As Individual LEDs (Editable)"],
            initial_value="As Group (Solid)"
        )
        
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>"
                           "• Group: Fast, moves as one unit<br>"
                           "• Individual LEDs: Edit each LED position/rotation/size separately</div>")
        
        add_custom_group_btn = server.gui.add_button("➕ Add Custom Group", color="green")
        
        @add_custom_group_btn.on_click
        def _(_):
            selected_template = template_dropdown.value
            load_mode = load_mode_dropdown.value
            
            if selected_template == "Empty":
                # Create empty custom group
                create_custom_group()
                print("✓ Empty custom group added")
            else:
                # Load from template
                if load_mode == "As Individual LEDs (Editable)":
                    # Load template as individual LEDs
                    load_template_as_individual_leds(selected_template)
                else:
                    # Load as solid group (default behavior)
                    load_custom_group_from_template(selected_template)
                # Refresh template list in case new templates were added
                template_dropdown.options = ["Empty"] + get_available_templates()
    
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
                led_viewing_angles = group_cfg.get('led_viewing_angles', [120] * num_leds)
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
            
            # Support both rotation_x/y/z and rot_x/y/z
            group_rot = [
                group_cfg.get('rotation_x', group_cfg.get('rot_x', 0)),
                group_cfg.get('rotation_y', group_cfg.get('rot_y', 0)),
                group_cfg.get('rotation_z', group_cfg.get('rot_z', 0))
            ]
            
            # Create rotation matrix for group rotation
            rot_x_rad = np.radians(group_rot[0])
            rot_y_rad = np.radians(group_rot[1])
            rot_z_rad = np.radians(group_rot[2])
            
            Rx = np.array([[1, 0, 0], [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)], [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])
            Ry = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)], [0, 1, 0], [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])
            Rz = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0], [np.sin(rot_z_rad), np.cos(rot_z_rad), 0], [0, 0, 1]])
            R_group = Rz @ Ry @ Rx
            
            # Create individual LED for each LED in the group
            for led_idx in range(num_leds):
                # Apply group rotation to LED position
                led_pos_local = np.array(led_positions[led_idx])
                led_pos_rotated = R_group @ led_pos_local
                led_pos_final = led_pos_rotated + np.array(group_pos)
                
                # Apply group rotation to LED direction
                led_dir_local = np.array(led_rotations[led_idx])
                led_dir_rotated = R_group @ led_dir_local
                
                # Convert direction to rotation angles
                # Forward direction is (1, 0, 0)
                # Calculate rotation needed to transform (1,0,0) to led_dir_rotated
                forward = np.array([1, 0, 0])
                
                # Calculate rotation axis and angle
                if np.allclose(led_dir_rotated, forward):
                    rot_angles = [0, 0, 0]
                elif np.allclose(led_dir_rotated, -forward):
                    rot_angles = [0, 180, 0]
                else:
                    # Use standard transformation
                    # Y rotation (tilt up/down)
                    rot_y = np.degrees(np.arctan2(led_dir_rotated[2], led_dir_rotated[0]))
                    # Z rotation (tilt left/right)
                    rot_z = np.degrees(np.arctan2(led_dir_rotated[1], np.sqrt(led_dir_rotated[0]**2 + led_dir_rotated[2]**2)))
                    rot_angles = [0, rot_y, rot_z]
                
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
    individual_leds_folder = server.gui.add_folder("Individual LEDs")
    
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
            square_roll_slider = server.gui.add_slider("Ruota su se stesso (°)", min=-180, max=180, step=1, initial_value=0)
            
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
        remove_btn.on_click(on_remove)
        
        individual_leds.append(led_data)
        
        if not skip_update_scene:
            update_scene()
        
        return led_data
    
    def export_individual_leds_simple():
        """Export all individual LEDs to a simple JSON format (preserves exact coordinates)."""
        if len(individual_leds) == 0:
            print("⚠️ No individual LEDs to export")
            return
        
        # Export directory
        export_dir = "exports"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Collect current LED data (exact coordinates, no transformations)
        leds_export = []
        for led_data in individual_leds:
            led_export = {
                "id": led_data['id'],
                "enabled": led_data['enable'].value,
                "led_on": led_data['led_on'],
                "position": {
                    "x": float(led_data['pos_x'].value),
                    "y": float(led_data['pos_y'].value),
                    "z": float(led_data['pos_z'].value)
                },
                "rotation": {
                    "x": float(led_data['rot_x'].value),
                    "y": float(led_data['rot_y'].value),
                    "z": float(led_data['rot_z'].value)
                },
                "size": float(led_data['size'].value),
                "viewing_angle": float(led_data['viewing_angle'].value),
                "square_roll": float(led_data['square_roll'].value)
            }
            
            # Add metadata if present (template source info)
            if 'template_source' in led_data:
                led_export['template_source'] = led_data['template_source']
            if 'group_index' in led_data:
                led_export['group_index'] = led_data['group_index']
            
            leds_export.append(led_export)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"individual_leds_{timestamp}.json"
        filepath = os.path.join(export_dir, filename)
        
        # Save to file
        export_data = {
            "format_version": "1.0",
            "description": "Individual LEDs export - exact coordinates (no transformations)",
            "export_date": timestamp,
            "num_leds": len(leds_export),
            "leds": leds_export
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported {len(leds_export)} individual LED(s) to: {filename}")
        return filepath
    
    with individual_leds_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Add New Individual LED</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Add single LEDs with custom position, rotation, and size</div>")
        
        add_individual_led_btn = server.gui.add_button("➕ Add LED", color="cyan")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export Individual LEDs</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Save exact coordinates without transformations</div>")
        
        export_individual_leds_btn = server.gui.add_button("💾 Export to JSON", color="#4CAF50")
        
        @add_individual_led_btn.on_click
        def _(_):
            create_individual_led()
            print("✓ Individual LED added")
        
        @export_individual_leds_btn.on_click
        def _(_):
            export_individual_leds_simple()


    # LED Control Matrix (individual LED and row control for base groups)
    # These controls are placed inside led_config_folder so they are hidden when no base groups are active
    group_names = ["Front+", "Front-", "Side+", "Side-"]
    
    # Create control folders for each group with HTML buttons inside led_config_folder
    with led_config_folder:
        for group_idx, (group_name, color_hex) in enumerate(zip(group_names, group_colors_hex)):
            with server.gui.add_folder(f"{group_name}"):
                # Group control button (toggle entire group)
                group_btn = server.gui.add_button(f"ALL", color=color_hex)
                group_buttons[group_idx] = group_btn
                
                server.gui.add_html("<hr style='margin:4px 0;'>")
                
                # Create controls for each row in the group
                for row_idx in range(4):
                    # Row header with row toggle button
                    html_content = f"""
                    <div style='margin:6px 0 2px 0;'>
                        <span style='font-weight:600;font-size:11px;'>Row {row_idx + 1}:</span>
                    </div>
                    """
                    server.gui.add_html(html_content)
                    
                    # Row toggle button
                    row_btn = server.gui.add_button(f"Row {row_idx + 1}", color="#666666")
                    row_buttons[(group_idx, row_idx)] = row_btn
                    
                    # LED buttons for this row (3 LEDs) - small square buttons
                    for led_in_row_idx in range(3):
                        led_global_idx = group_idx * 12 + row_idx * 3 + led_in_row_idx
                        led_btn = server.gui.add_button(f"L{led_in_row_idx+1}", color=color_hex)
                        led_buttons[led_global_idx] = led_btn


    def compute_wall_intensity(
        leds, wall_dist, num_rays_per_led, grid_size=50, wall_size=80, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays and compute intensity grid on wall."""
        # Grid covers -wall_size/2 to +wall_size/2 cm in Y and Z
        grid = np.zeros((grid_size, grid_size))
        cell_size = wall_size / grid_size  # cm per cell
        half_size = wall_size / 2

        # Assume uniform luminous flux per LED provided by GUI
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value) if 'led_lumens_slider' in globals() or True else 100.0
        
        # Count active LEDs to calculate rays per LED for target rays per pixel
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        if num_active_leds == 0:
            return grid, wall_size
        
        # Calculate rays per LED to achieve target rays per pixel (2 rays per pixel)
        total_pixels = grid_size * grid_size
        rays_per_led_calculated = max(1, int((total_pixels * num_rays_per_led) / num_active_leds))
        
        # Diagnostic: Print LED positions and approximate distances to wall
        num_cores = multiprocessing.cpu_count()
        print(f"\n=== SYSTEM INFO ===")
        print(f"CPU cores available: {num_cores}")
        print(f"=== LED GEOMETRY CHECK ===")
        print(f"Wall at x = {wall_dist:.1f} cm")
        print(f"\n=== RAY EMISSION MODEL ===")
        print(f"Each LED emits {lumens_per_led:.1f} lm total")
        print(f"Distribution: Lambertian I(θ) = I₀ × cos^n(θ)")
        print(f"Viewing angle: {leds[0].viewing_angle if leds else 120}° (50% intensity)")
        print(f"Emission cone: 0-90° (full hemisphere)")
        print(f"Rays per LED: {rays_per_led_calculated}")
        print(f"Each ray carries: (lm/LED / N_rays) × cos^n(θ) × (n+1)")
        print(f"This conserves flux: Σ[all rays] = {lumens_per_led:.1f} lm per LED")
        print(f"==========================")
        for i, led in enumerate(leds):
            if hasattr(led, 'enabled') and not led.enabled:
                continue
            # Approximate distance from LED to wall along LED direction
            if led.direction[0] > 0.01:  # LED pointing towards wall
                t_center = (wall_dist - led.position[0]) / led.direction[0]
                dist_along_axis = wall_dist - led.position[0]
                print(f"  LED {i}: pos=({led.position[0]:.1f}, {led.position[1]:.1f}, {led.position[2]:.1f}), "
                      f"dir=({led.direction[0]:.3f}, {led.direction[1]:.3f}, {led.direction[2]:.3f}), "
                      f"dist_to_wall={dist_along_axis:.1f}cm, t_center={t_center:.1f}cm")
            else:
                print(f"  LED {i}: pos=({led.position[0]:.1f}, {led.position[1]:.1f}, {led.position[2]:.1f}), "
                      f"dir=({led.direction[0]:.3f}, {led.direction[1]:.3f}, {led.direction[2]:.3f}), "
                      f"NOT pointing towards wall")
        print(f"==========================\n")
        print(f"=== STARTING RAY TRACING ===\n")
        
        # Prepare active LEDs with their indices
        active_leds = []
        for led_idx, led in enumerate(leds):
            if hasattr(led, 'enabled') and not led.enabled:
                continue
            idx = getattr(led, 'led_index', led_idx)
            active_leds.append((led, idx))
        
        if not active_leds:
            return grid, wall_size
        
        # Prepare parameters for worker processes
        ray_uniformity = float(ray_uniformity_slider.value) if 'ray_uniformity_slider' in globals() or True else 0.0
        
        worker_args = []
        for led, led_idx in active_leds:
            params = {
                'wall_dist': wall_dist,
                'rays_per_led': rays_per_led_calculated,
                'grid_size': grid_size,
                'wall_size': wall_size,
                'lumens_per_led': lumens_per_led,
                'absorbers': absorbers if absorbers else [],
                'stl_mesh_data': stl_mesh_data,
                'ray_uniformity': ray_uniformity,
                'led_idx': led_idx,
            }
            worker_args.append((led, params))
        
        # Use multiprocessing to parallelize LED processing
        num_processes = min(multiprocessing.cpu_count(), len(active_leds))
        print(f"Using {num_processes} CPU cores to process {len(active_leds)} LEDs in parallel...")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_process_led_wall_worker, worker_args)
        
        # Aggregate results from all workers
        for local_grid in results:
            grid += local_grid
        
        print(f"Ray tracing complete!\n")

        return grid, wall_size

    def compute_room_intensity(
        leds, front_dist, side_dist, top_bottom_dist, num_rays_per_led, grid_size=20, back_dist=None, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays and compute intensity grids on all room walls (5 or 6 with optional back wall) using multiprocessing."""
        
        # Calculate physical dimensions of each wall
        wall_width_x = front_dist + abs(circle_center_slider.value)  # Base depth
        wall_width_y = 2 * side_dist  # Front wall width, top/bottom wall width
        wall_height_z = 2 * top_bottom_dist  # Front wall height, left/right wall height
        
        # Increase depth (X axis) for side/top/bottom walls by at least 2x
        side_topbottom_depth_x = wall_width_x * 2.5  # 2.5x depth for lateral, top, bottom walls
        
        # Each wall gets grid_size x grid_size cells to ensure complete coverage
        # Cells adapt to exact wall dimensions
        front_grid_y = grid_size
        front_grid_z = grid_size
        
        side_grid_x = grid_size
        side_grid_z = grid_size
        
        topbottom_grid_x = grid_size
        topbottom_grid_y = grid_size
        
        # Initialize grids for each wall with appropriate dimensions
        # Grid indexed as [gi, gj] where gi is first axis (vertical/rows), gj is second axis (horizontal/cols)
        grids = {
            'front': np.zeros((front_grid_z, front_grid_y)),  # YZ plane: [Z, Y]
            'left': np.zeros((side_grid_z, side_grid_x)),   # XZ plane: [Z, X]
            'right': np.zeros((side_grid_z, side_grid_x)),  # XZ plane: [Z, X]
            'top': np.zeros((topbottom_grid_y, topbottom_grid_x)),    # XY plane: [Y, X]
            'bottom': np.zeros((topbottom_grid_y, topbottom_grid_x))  # XY plane: [Y, X]
        }
        
        # Add back wall if enabled
        if back_dist is not None:
            grids['back'] = np.zeros((front_grid_z, front_grid_y))  # YZ plane: [Z, Y] (same as front)
        
        # Wall dimensions for grid mapping (each wall needs proper size and range)
        # For extended side/top/bottom walls: x_min is the back edge (front_dist - depth)
        extended_x_min = front_dist - side_topbottom_depth_x  # Back edge of extended walls
        
        wall_specs = {
            'front': {'size_y': wall_width_y, 'size_z': wall_height_z, 'dims': ('y', 'z'), 
                     'grid_y': front_grid_y, 'grid_z': front_grid_z},
            'left': {'size_x': side_topbottom_depth_x, 'size_z': wall_height_z, 'dims': ('x', 'z'), 'x_min': extended_x_min,
                    'grid_x': side_grid_x, 'grid_z': side_grid_z},
            'right': {'size_x': side_topbottom_depth_x, 'size_z': wall_height_z, 'dims': ('x', 'z'), 'x_min': extended_x_min,
                     'grid_x': side_grid_x, 'grid_z': side_grid_z},
            'top': {'size_x': side_topbottom_depth_x, 'size_y': wall_width_y, 'dims': ('x', 'y'), 'x_min': extended_x_min,
                   'grid_x': topbottom_grid_x, 'grid_y': topbottom_grid_y},
            'bottom': {'size_x': side_topbottom_depth_x, 'size_y': wall_width_y, 'dims': ('x', 'y'), 'x_min': extended_x_min,
                      'grid_x': topbottom_grid_x, 'grid_y': topbottom_grid_y}
        }
        
        # Add back wall specs if enabled
        if back_dist is not None:
            wall_specs['back'] = {'size_y': wall_width_y, 'size_z': wall_height_z, 'dims': ('y', 'z'),
                                 'grid_y': front_grid_y, 'grid_z': front_grid_z}
        
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        if num_active_leds == 0:
            return grids, wall_specs
        
        print(f"\n=== ROOM MODE ===")
        print(f"Front wall: x={front_dist}cm, Left: y={-side_dist}cm, Right: y={+side_dist}cm")
        print(f"Top: z={+top_bottom_dist}cm, Bottom: z={-top_bottom_dist}cm")
        if back_dist is not None:
            print(f"Back wall: x={-back_dist}cm")
        print(f"Grid: {grid_size}×{grid_size} cells per wall (cells adapt to each wall's dimensions)")
        print(f"Cell sizes: Front {wall_width_y/front_grid_y:.2f}×{wall_height_z/front_grid_z:.2f}cm, "
              f"Side {side_topbottom_depth_x/side_grid_x:.2f}×{wall_height_z/side_grid_z:.2f}cm, "
              f"Top/Bottom {side_topbottom_depth_x/topbottom_grid_x:.2f}×{wall_width_y/topbottom_grid_y:.2f}cm")
        
        # Track ray hits per wall for debugging
        ray_hits = {'front': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        if back_dist is not None:
            ray_hits['back'] = 0
        total_rays = 0
        
        # MULTIPROCESSING: Prepare parameters for worker processes
        active_leds = [led for led in leds if not (hasattr(led, 'enabled') and not led.enabled)]
        
        if len(active_leds) == 0:
            return grids, wall_specs
        
        # Get ray uniformity from slider
        ray_uniformity = float(ray_uniformity_slider.value) if 'ray_uniformity_slider' in globals() else 0.0
        
        # Prepare grid shapes for workers
        grid_shapes = {
            'front': grids['front'].shape,
            'left': grids['left'].shape,
            'right': grids['right'].shape,
            'top': grids['top'].shape,
            'bottom': grids['bottom'].shape
        }
        if back_dist is not None:
            grid_shapes['back'] = grids['back'].shape
        
        # Package parameters for workers
        worker_params = {
            'front_dist': front_dist,
            'side_dist': side_dist,
            'top_bottom_dist': top_bottom_dist,
            'back_dist': back_dist,
            'led_x_center': circle_center_slider.value,
            'num_rays_per_led': num_rays_per_led,
            'grid_size': grid_size,
            'lumens_per_led': lumens_per_led,
            'absorbers': absorbers if absorbers else [],
            'stl_mesh_data': stl_mesh_data,
            'ray_uniformity': ray_uniformity,
            'grid_shapes': grid_shapes,
            'wall_specs': wall_specs
        }
        
        # Prepare arguments for each LED
        led_args = [(led, worker_params) for led in active_leds]
        
        # Use multiprocessing to parallelize LED processing
        num_processes = min(multiprocessing.cpu_count(), len(active_leds))
        print(f"Using {num_processes} CPU cores for parallel ray tracing...")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(_process_led_worker, led_args)
        
        # Combine results from all workers
        for led_grids, led_ray_hits, led_total_rays in results:
            for wall_name in grids.keys():
                grids[wall_name] += led_grids[wall_name]
                ray_hits[wall_name] += led_ray_hits[wall_name]
            total_rays += led_total_rays
        
        # Print summary
        total_emitted = num_active_leds * lumens_per_led
        total_on_walls = sum(grid.sum() for grid in grids.values())
        conservation_pct = (total_on_walls / total_emitted * 100) if total_emitted > 0 else 0
        
        print(f"\nActive LEDs: {num_active_leds}")
        print(f"Total rays traced: {total_rays}")
        print(f"Total emitted: {total_emitted:.1f} lm")
        print(f"Total on walls: {total_on_walls:.1f} lm ({conservation_pct:.1f}%)")
        for wall_name, grid in grids.items():
            hits = ray_hits[wall_name]
            pct = (hits / total_rays * 100) if total_rays > 0 else 0
            print(f"  {wall_name.capitalize()}: {grid.sum():.1f} lm ({hits} rays, {pct:.1f}%)")
        
        return grids, wall_specs

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
        fov_h_deg = camera_fov_h.value
        fov_v_deg = camera_fov_v.value
        
        # Calculate FOV dimensions on wall
        fov_h_rad = np.radians(fov_h_deg)
        fov_v_rad = np.radians(fov_v_deg)
        fov_width_cm = 2 * (wall_dist - cam_x) * np.tan(fov_h_rad / 2)
        fov_height_cm = 2 * (wall_dist - cam_x) * np.tan(fov_v_rad / 2)
        
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
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                
                # Get rotation angles (Euler angles in fixed frame)
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
                if 'original_led_row_directions' not in group and 'led_row_directions' in group:
                    group['original_led_row_directions'] = [tuple(rd) for rd in group.get('led_row_directions', [])]

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
                    rotated_directions.append(tuple(rotated_dir))
                
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_sizes'] = group.get('led_sizes', [])
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                
                # Rotate row direction vectors
                original_row_dirs = group.get('original_led_row_directions', group.get('led_row_directions', []))
                if original_row_dirs:
                    rotated_row_dirs = [tuple(R_total @ np.array(rd)) for rd in original_row_dirs]
                    config['led_row_directions'] = rotated_row_dirs
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
            }
            individual_leds_configs.append(config)
        
        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
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
            
            # IMPORTANT: viewing_angle defines the full emission cone angle
            # LEDs emit rays within the full viewing_angle cone
            # Calculate exponent n so that intensity drops to 50% at viewing_angle/2
            # I(θ) = I₀ × cos^n(θ), at θ_half: 0.5 = cos^n(θ_half)
            # n = ln(0.5) / ln(cos(θ_half))
            
            # Maximum emission angle is viewing_angle/2 (half-angle from center)
            max_theta = np.radians(led.viewing_angle / 2.0)  # Use full viewing angle
            
            # Calculate n from viewing angle
            theta_half = np.radians(led.viewing_angle / 2.0)
            cos_half = np.cos(theta_half)
            
            # Calculate base exponent for this viewing angle
            if cos_half > 0.01:
                n_base = np.log(0.5) / np.log(cos_half)
                # Clamp n_base to reasonable range to avoid numerical issues
                n_base = np.clip(n_base, 0.1, 10.0)
            else:
                n_base = 1.0
            
            # Apply uniformity factor to make beam more focused if desired
            uniformity = float(ray_uniformity_slider.value)
            n = n_base * (1.0 + uniformity * 2.0)
            n = np.clip(n, 0.1, 30.0)  # Final safety clamp
            
            # Calculate normalization factor for uniform solid angle sampling with cos^n(θ) weighting
            # For hemisphere (0 to 90°): norm_factor = (n+1)
            # PHYSICS: Each ray = I₀ × cos^n(θ) × dΩ where dΩ = 2π/N
            # With I₀ = Φ×(n+1)/(2π), we get: lumens_per_ray = Φ×(n+1)×cos^n(θ)/N
            norm_factor = n + 1.0
            
            led_total_lumens_emitted = 0.0  # Track for verification
            
            for _ in range(num_rays_per_led):
                # Uniform sampling in solid angle (physically correct)
                u1, u2 = np.random.uniform(0, 1, 2)
                
                # Sample uniformly within hemisphere (0 to 90°)
                cos_max = np.cos(max_theta)
                cos_theta = 1.0 - u1 * (1.0 - cos_max)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                phi = 2 * np.pi * u2
                
                local_dir = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                world_dir = (
                    local_dir[0] * x_axis
                    + local_dir[1] * y_axis
                    + local_dir[2] * z_axis
                )
                world_dir = world_dir / np.linalg.norm(world_dir)
                
                # Calculate lumens carried by this specific ray (emission in this direction)
                cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
                intensity_coefficient = np.power(cos_theta_clamped, n)
                lumens_per_ray = (lumens_per_led / max(1, num_rays_per_led)) * intensity_coefficient * norm_factor
                led_total_lumens_emitted += lumens_per_ray
                
                # Check absorber intersection
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
                
                hit_absorbed = False
                for a in absorbers:
                    t_hit = ray_box_intersection(led.position, world_dir, a)
                    if t_hit is not None and t_hit > 0:
                        hit_absorbed = True
                        break
                
                # Check STL mesh intersection
                if not hit_absorbed and stl_absorber_enable.value and stl_mesh_data[0] is not None:
                    mesh = stl_mesh_data[0].copy()
                    
                    # Build transformation matrix  (same as before)
                    transform = np.eye(4)
                    scale = float(stl_scale.value)
                    if np.isfinite(scale) and scale > 0:
                        transform[:3, :3] *= scale
                    
                    rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
                    rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0  
                    rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
                    
                    if rot_x != 0:
                        rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_x), [1, 0, 0])
                        transform = rot_mat @ transform
                    if rot_y != 0:
                        rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_y), [0, 1, 0])
                        transform = rot_mat @ transform
                    if rot_z != 0:
                        rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_z), [0, 0, 1])
                        transform = rot_mat @ transform
                    
                    pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
                    pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
                    pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
                    transform[:3, 3] = [pos_x, pos_y, pos_z]
                    
                    mesh_data_ray = {
                        'vertices': mesh.vertices,
                        'faces': mesh.faces,
                        'transform': transform
                    }
                    
                    t_hit = _ray_mesh_intersection(led.position, world_dir, mesh_data_ray)
                    if t_hit is not None and t_hit > 0:
                        hit_absorbed = True
                
                if hit_absorbed:
                    continue
                
                if world_dir[0] > 0:
                    t = (wall_dist - led.position[0]) / world_dir[0]
                    if t > 0:
                        hit_y = led.position[1] + world_dir[1] * t
                        hit_z = led.position[2] + world_dir[2] * t
                        
                        # Check if hit is within FOV bounds (centered at 0,0)
                        half_w = fov_width_cm / 2
                        half_h = fov_height_cm / 2
                        
                        if -half_w <= hit_y <= half_w and -half_h <= hit_z <= half_h:
                            # Convert to FOV grid indices
                            grid_x = int((hit_y + half_w) / cell_size_cm)
                            grid_y = int((hit_z + half_h) / cell_size_cm)
                            
                            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                                # Ray hits FOV region - add its lumens (already calculated above)
                                fov_grid[grid_y, grid_x] += lumens_per_ray
        
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
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fov_intensity_{timestamp}.png"
        img_pil.save(filename)
        
        print(f"FOV image saved to {filename}")
        print(f"Image size: {grid_width} x {grid_height} pixels (1 pixel = 1cm²)")
        print(f"FOV dimensions: {fov_width_cm:.2f} x {fov_height_cm:.2f} cm")
        print(f"Total lumens in FOV: {fov_grid.sum():.2f} lm")
        print(f"Max illuminance: {max_lux:.2f} lux")

    def intensity_to_color(value, max_val):
        """Convert intensity to inferno-like colormap."""
        # Handle invalid values
        if max_val == 0 or not np.isfinite(value) or not np.isfinite(max_val):
            return (0.0, 0.0, 0.0)
        
        t = np.clip(value / max_val, 0.0, 1.0)
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

    def update_intensity_map():
        """Update only the intensity map on the wall (expensive operation)."""
        nonlocal intensity_handles, legend_html
        
        # Clear previous intensity handles
        for handle in intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        intensity_handles = []
        
        if not show_intensity_map.value:
            # Update legend with message when intensity map is disabled
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
        wall_size = int(wall_view_size.value)
        
        # Get current LEDs configuration (fixed angles: front=0°, side=90°)
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
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                
                # Get rotation angles (Euler angles in fixed frame)
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
                if 'original_led_row_directions' not in group and 'led_row_directions' in group:
                    group['original_led_row_directions'] = [tuple(rd) for rd in group.get('led_row_directions', [])]

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
                    rotated_directions.append(tuple(rotated_dir))
                
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                
                # Rotate row direction vectors
                original_row_dirs = group.get('original_led_row_directions', group.get('led_row_directions', []))
                if original_row_dirs:
                    rotated_row_dirs = [tuple(R_total @ np.array(rd)) for rd in original_row_dirs]
                    config['led_row_directions'] = rotated_row_dirs
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
            }
            individual_leds_configs.append(config)
        
        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
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
        
        # Prepare STL mesh data for ray tracing if enabled
        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            # Prepare transformation matrix for STL mesh
            mesh = stl_mesh_data[0].copy()
            
            # Build transformation matrix
            transform = np.eye(4)
            
            # Apply scale
            scale = float(stl_scale.value)
            if np.isfinite(scale) and scale > 0:
                transform[:3, :3] *= scale
            
            # Apply rotations
            rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
            rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0  
            rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
            
            if rot_x != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_x), [1, 0, 0])
                transform = rot_mat @ transform
            if rot_y != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_y), [0, 1, 0])
                transform = rot_mat @ transform
            if rot_z != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_z), [0, 0, 1])
                transform = rot_mat @ transform
            
            # Apply translation (convert cm to same units as LED positions)
            pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
            pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
            pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
            transform[:3, 3] = [pos_x, pos_y, pos_z]
            
            stl_mesh_for_raytracing = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
                'transform': transform
            }
            print(f"STL mesh enabled as light absorber ({len(mesh.faces)} triangles)")
        
        # Compute intensity with rays_per_pixel from slider
        rays_per_pixel = int(intensity_rays_slider.value)
        intensity_grid, actual_wall_size = compute_wall_intensity(
            leds, wall_dist, rays_per_pixel, grid_size, wall_size, absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        # Clean up any NaN or Inf values in the grid
        intensity_grid = np.nan_to_num(intensity_grid, nan=0.0, posinf=0.0, neginf=0.0)
        max_lux = intensity_grid.max()  # Grid now contains lux (lm/m²)
        
        # Calculate cell area for lux to lumen conversion
        cell_size_cm = actual_wall_size / grid_size
        cell_area_cm2 = cell_size_cm * cell_size_cm
        cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
        
        # === DIAGNOSTIC OUTPUT FOR FLUX CONSERVATION ===
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        total_emitted_lumens = num_active_leds * lumens_per_led
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
        
        for gz in range(grid_size):
            for gy in range(grid_size):
                intensity = intensity_grid[gz, gy]
                if intensity > 0:
                    color = intensity_to_color(intensity, max_lux)
                    y_pos = (
                        -half_size + gy * cell_size_cm + cell_size_cm / 2
                    ) / 100.0
                    z_pos = (
                        -half_size + gz * cell_size_cm + cell_size_cm / 2
                    ) / 100.0
                    x_pos = wall_dist / 100.0 - 0.005
                    
                    handle = server.scene.add_box(
                        f"/intensity/cell_{gz}_{gy}",
                        dimensions=(0.001, cell_size_m * 0.95, cell_size_m * 0.95),
                        color=color,
                        position=(x_pos, y_pos, z_pos),
                    )
                    intensity_handles.append(handle)
        
        # Update legend (grid now stores lux = lm/m²)
        legend_steps = 6
        legend_vals_lux = np.linspace(0, max_lux, legend_steps)
        html_lines = ["<div style='font-family: sans-serif;'>",
                      "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend (lux)</div>"]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, max_lux)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            # Convert lux to lumens for this cell: Lumen = Lux × Area
            lumen_val = lux_val * cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.1f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} lm/cell)</div></div>"
            )
        html_lines.append("</div>")
        legend_html.content = "".join(html_lines)
    
    def draw_room_walls():
        """Draw room walls as wireframe/transparent boxes (no intensity calculation)."""
        nonlocal room_wall_handles
        
        # Clear previous room wall handles
        for handle in room_wall_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_wall_handles = []
        
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
        nonlocal room_intensity_handles, room_wall_handles
        
        # Clear previous room intensity handles
        for handle in room_intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_intensity_handles = []
        
        if not room_mode_enable.value or not show_room_intensity.value:
            return
        
        # Hide room walls when showing intensity (they would cover the intensity cells)
        for handle in room_wall_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        room_wall_handles = []
        
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
            # Prepare transformation matrix for STL mesh
            mesh = stl_mesh_data[0].copy()
            
            # Build transformation matrix
            transform = np.eye(4)
            
            # Apply scale
            scale = float(stl_scale.value)
            if np.isfinite(scale) and scale > 0:
                transform[:3, :3] *= scale
            
            # Apply rotations
            rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
            rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0  
            rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
            
            if rot_x != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_x), [1, 0, 0])
                transform = rot_mat @ transform
            if rot_y != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_y), [0, 1, 0])
                transform = rot_mat @ transform
            if rot_z != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_z), [0, 0, 1])
                transform = rot_mat @ transform
            
            # Apply translation (convert cm to same units as LED positions)
            pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
            pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
            pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
            transform[:3, 3] = [pos_x, pos_y, pos_z]
            
            stl_mesh_for_raytracing = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
                'transform': transform
            }
            print(f"STL mesh enabled as light absorber ({len(mesh.faces)} triangles)")
        
        grids, wall_specs = compute_room_intensity(
            leds, front_dist, side_dist, top_bottom_dist, rays_per_pixel, grid_size, 
            back_dist=room_back_dist.value if show_back_wall.value else None,
            absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        
        # Find max lux across all walls for color normalization
        max_lux = max(grid.max() for grid in grids.values()) if grids else 0.0
        
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
        
        # Visualize each wall
        cells_created = {'front': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        if show_back_wall.value and 'back' in grids:
            cells_created['back'] = 0
        for wall_name, intensity_grid in grids.items():
            wall_spec = wall_specs[wall_name]
            grid_shape = intensity_grid.shape  # Get actual grid dimensions for this wall
            
            for gi in range(grid_shape[0]):
                for gj in range(grid_shape[1]):
                    intensity = intensity_grid[gi, gj]
                    # Create ALL cells (even with zero intensity) to show full wall dimensions
                    color = intensity_to_color(intensity, max_lux)
                    
                    # Calculate cell position based on wall orientation
                    # Position cells exactly on wall surfaces (same as gray walls)
                    if wall_name == 'front':
                        # YZ plane at x=front_dist
                        size_y = wall_spec['size_y']
                        size_z = wall_spec['size_z']
                        grid_y = wall_spec['grid_y']
                        grid_z = wall_spec['grid_z']
                        cell_size_y = size_y / grid_y
                        cell_size_z = size_z / grid_z
                        x_pos = front_dist / 100.0
                        y_pos = (-size_y/2 + gj * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        dims = (0.01, cell_size_y / 100.0 * 0.98, cell_size_z / 100.0 * 0.98)
                    
                    elif wall_name == 'left':
                        # XZ plane at y=-side_dist
                        size_x = wall_spec['size_x']
                        size_z = wall_spec['size_z']
                        grid_x = wall_spec['grid_x']
                        grid_z = wall_spec['grid_z']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_z = size_z / grid_z
                        x_pos = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = -side_dist / 100.0
                        z_pos = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        dims = (cell_size_x / 100.0 * 0.98, 0.01, cell_size_z / 100.0 * 0.98)
                    
                    elif wall_name == 'right':
                        # XZ plane at y=+side_dist
                        size_x = wall_spec['size_x']
                        size_z = wall_spec['size_z']
                        grid_x = wall_spec['grid_x']
                        grid_z = wall_spec['grid_z']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_z = size_z / grid_z
                        x_pos = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = side_dist / 100.0
                        z_pos = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        dims = (cell_size_x / 100.0 * 0.98, 0.01, cell_size_z / 100.0 * 0.98)
                    
                    elif wall_name == 'top':
                        # XY plane at z=+top_bottom_dist
                        size_x = wall_spec['size_x']
                        size_y = wall_spec['size_y']
                        grid_x = wall_spec['grid_x']
                        grid_y = wall_spec['grid_y']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_y = size_y / grid_y
                        x_pos = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = (-size_y/2 + gi * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = top_bottom_dist / 100.0
                        dims = (cell_size_x / 100.0 * 0.98, cell_size_y / 100.0 * 0.98, 0.01)
                    
                    elif wall_name == 'bottom':
                        # XY plane at z=-top_bottom_dist
                        size_x = wall_spec['size_x']
                        size_y = wall_spec['size_y']
                        grid_x = wall_spec['grid_x']
                        grid_y = wall_spec['grid_y']
                        x_min = wall_spec['x_min']
                        cell_size_x = size_x / grid_x
                        cell_size_y = size_y / grid_y
                        x_pos = (x_min + gj * cell_size_x + cell_size_x / 2) / 100.0
                        y_pos = (-size_y/2 + gi * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = -top_bottom_dist / 100.0
                        dims = (cell_size_x / 100.0 * 0.98, cell_size_y / 100.0 * 0.98, 0.01)
                    
                    elif wall_name == 'back':
                        # YZ plane at x=-back_dist (symmetric to front wall)
                        size_y = wall_spec['size_y']
                        size_z = wall_spec['size_z']
                        grid_y = wall_spec['grid_y']
                        grid_z = wall_spec['grid_z']
                        cell_size_y = size_y / grid_y
                        cell_size_z = size_z / grid_z
                        back_x_pos = -room_back_dist.value
                        x_pos = back_x_pos / 100.0
                        y_pos = (-size_y/2 + gj * cell_size_y + cell_size_y / 2) / 100.0
                        z_pos = (-size_z/2 + gi * cell_size_z + cell_size_z / 2) / 100.0
                        dims = (0.01, cell_size_y / 100.0 * 0.98, cell_size_z / 100.0 * 0.98)
                    else:
                        continue
                    
                    handle = server.scene.add_box(
                        f"/room_intensity/{wall_name}/cell_{gi}_{gj}",
                        dimensions=dims,
                        color=color,
                        position=(x_pos, y_pos, z_pos),
                    )
                    room_intensity_handles.append(handle)
                    cells_created[wall_name] += 1
        
        print(f"Cells visualized:")
        for wall_name, count in cells_created.items():
            print(f"  {wall_name.capitalize()}: {count} cells created")
        print(f"===================================\n")
        
        # Update legend (grid stores lux)
        legend_steps = 6
        legend_vals_lux = np.linspace(0, max_lux, legend_steps)
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
        
        html_lines = ["<div style='font-family: sans-serif;'>",
                      "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend (lux)</div>"]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, max_lux)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            # Convert lux to lumens using average cell area: Lumen = Lux × Area
            lumen_val = lux_val * avg_cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.1f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} lm/cell avg)</div></div>"
            )
        html_lines.append("</div>")
        legend_html.content = "".join(html_lines)
    
    def update_scene():
        """Redraw the scene based on current slider values (without intensity map)."""
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles, current_leds

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
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles = []
        ray_handles = []
        absorber_handles = []
        camera_fov_handles = []

        # Get current values (fixed angles: front=0°, side=90°)
        front_angle = 0.0  # Fixed front angle
        side_angle = 90.0  # Fixed side angle
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        wall_dist = wall_dist_slider.value
        circle_center_x = circle_center_slider.value
        ray_length = ray_length_slider.value

        # Create LEDs
        # Read per-group rotation slider values
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
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                
                # Get rotation angles (Euler angles in fixed frame)
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
                if 'original_led_row_directions' not in group and 'led_row_directions' in group:
                    group['original_led_row_directions'] = [tuple(rd) for rd in group.get('led_row_directions', [])]

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
                    rotated_directions.append(tuple(rotated_dir))
                
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_sizes'] = group.get('led_sizes', [])
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                
                # Rotate row direction vectors
                original_row_dirs = group.get('original_led_row_directions', group.get('led_row_directions', []))
                if original_row_dirs:
                    rotated_row_dirs = [tuple(R_total @ np.array(rd)) for rd in original_row_dirs]
                    config['led_row_directions'] = rotated_row_dirs
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),  # Default to True if not set
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
            }
            individual_leds_configs.append(config)
        
        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
        # Save LEDs for reuse in room intensity calculation
        current_leds = leds
        
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

        # Draw absorber boxes (red) in the scene
        for idx, a in enumerate(absorbers):
            cx, cy, cz = a['center']
            hx, hy, hz = a['half_sizes']
            rot = a.get('rotation', None)
            # Viser add_box dimensions are in meters (x,y,z)
            dims = ((hx * 2) / 100.0, (hy * 2) / 100.0, (hz * 2) / 100.0)
            pos_m = (cx / 100.0, cy / 100.0, cz / 100.0)
            if rot is not None:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                    wxyz=rot,
                )
            else:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                )
            absorber_handles.append(handle)

        # Draw LEDs as squares with center source (if enabled)
        if show_led_markers.value:
            for i, led in enumerate(leds):
                led_idx = getattr(led, 'led_index', i)
                led_enabled = not (hasattr(led, 'enabled') and not led.enabled)
                
                # Build local coordinate system for LED
                # z_axis = LED direction (normal to square)
                z_axis = led.direction / np.linalg.norm(led.direction)
                
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
                
                # White color: bright if enabled, dim if disabled
                square_color = (1.0, 1.0, 1.0) if led_enabled else (0.3, 0.3, 0.3)
                
                # Draw square base with rotation
                handle = server.scene.add_box(
                    f"/leds/led_{led_idx}_base",
                    dimensions=dims,
                    color=square_color,
                    position=tuple(led.position / 100.0),  # Convert cm to m for viser
                    wxyz=tuple(quat_wxyz),
                )
                led_handles.append(handle)
                
                # Draw small center source sphere (only if enabled)
                if led_enabled:
                    handle = server.scene.add_icosphere(
                        f"/leds/led_{led_idx}_source",
                        radius=0.001,  # Very small 1mm source
                        color=led.color,
                        position=tuple(led.position / 100.0),
                    )
                    led_handles.append(handle)

        # Prepare STL mesh data for ray tracing visualization if enabled
        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh = stl_mesh_data[0].copy()
            
            # Build transformation matrix
            transform = np.eye(4)
            
            # Apply scale
            scale = float(stl_scale.value)
            if np.isfinite(scale) and scale > 0:
                transform[:3, :3] *= scale
            
            # Apply rotations
            rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
            rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0  
            rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
            
            if rot_x != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_x), [1, 0, 0])
                transform = rot_mat @ transform
            if rot_y != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_y), [0, 1, 0])
                transform = rot_mat @ transform
            if rot_z != 0:
                rot_mat = trimesh.transformations.rotation_matrix(np.radians(rot_z), [0, 0, 1])
                transform = rot_mat @ transform
            
            # Apply translation
            pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
            pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
            pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
            transform[:3, 3] = [pos_x, pos_y, pos_z]
            
            stl_mesh_for_raytracing = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
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
                        max_theta = np.radians(led.viewing_angle / 2.0)  # Half-angle for proper cone - use LED's specific viewing angle
                        
                        uniformity = float(ray_uniformity_slider.value)
                        n = 1.0 + uniformity * 3.0  # Exponent from 1 to 4
                        
                        # Cosine power distribution sampling with clamping
                        cos_max = np.cos(max_theta)
                        base = 1 - u1 * (1 - np.power(cos_max, n + 1))
                        base = np.clip(base, 0.0, 1.0)
                        cos_theta_sampled = np.power(base, 1.0 / (n + 1))
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
            fov_h_deg = camera_fov_h.value
            fov_v_deg = camera_fov_v.value
            
            # Calculate FOV dimensions on wall based on viewing angles
            # Using simple trigonometry: width = 2 * distance * tan(angle/2)
            fov_h_rad = np.radians(fov_h_deg)
            fov_v_rad = np.radians(fov_v_deg)
            
            # FOV dimensions at wall distance (in cm)
            fov_width_cm = 2 * (wall_dist - cam_x) * np.tan(fov_h_rad / 2)
            fov_height_cm = 2 * (wall_dist - cam_x) * np.tan(fov_v_rad / 2)
            
            # Draw FOV border lines only (no fill)
            half_w = fov_width_cm / 200.0  # Half width in meters
            half_h = fov_height_cm / 200.0  # Half height in meters
            wall_x = wall_dist / 100.0 - 0.008
            
            # Four corner lines
            corners = [
                [[wall_x, -half_w, -half_h], [wall_x, half_w, -half_h]],  # Bottom
                [[wall_x, half_w, -half_h], [wall_x, half_w, half_h]],    # Right
                [[wall_x, half_w, half_h], [wall_x, -half_w, half_h]],    # Top
                [[wall_x, -half_w, half_h], [wall_x, -half_w, -half_h]],  # Left
            ]
            
            handle = server.scene.add_line_segments(
                "/camera/fov_border",
                points=np.array(corners),
                colors=(0.0, 1.0, 0.0),  # Green
                line_width=6.0,  # Linee più spesse
            )
            camera_fov_handles.append(handle)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_dist_init = wall_dist_slider.value
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, 2.0, 2.0),  # 200cm x 200cm wall, thin
        color=(0.5, 0.5, 0.5),
        position=(wall_dist_init / 100.0, 0.0, 0.0),
    )

    # Grid on XY plane (millimeter resolution)
    grid_points = []
    for i in range(-10, 11):
        grid_points.append([[-1.0, i * 0.01, 0], [1.0, i * 0.01, 0]])  # 1mm spacing
        grid_points.append([[i * 0.01, -1.0, 0], [i * 0.01, 1.0, 0]])  # 1mm spacing

    server.scene.add_line_segments(
        "/grid",
        points=np.array(grid_points),
        colors=(0.3, 0.3, 0.3),  # Single color for all segments
        line_width=1.0,
    )

    # Origin axes
    server.scene.add_line_segments(
        "/axes/x",
        points=np.array([[[0, 0, 0], [0.5, 0, 0]]]),
        colors=(1.0, 0.0, 0.0),
        line_width=3.0,
    )
    server.scene.add_line_segments(
        "/axes/y",
        points=np.array([[[0, 0, 0], [0, 0.5, 0]]]),
        colors=(0.0, 1.0, 0.0),
        line_width=3.0,
    )
    server.scene.add_line_segments(
        "/axes/z",
        points=np.array([[[0, 0, 0], [0, 0, 0.5]]]),
        colors=(0.0, 0.0, 1.0),
        line_width=3.0,
    )

    # Callback to update wall position
    def update_wall():
        nonlocal wall_handle
        if room_mode_enable.value:
            # In room mode, don't update main wall
            return
        try:
            wall_dist = wall_dist_slider.value
            wall_handle.position = (wall_dist / 100.0, 0.0, 0.0)
        except (AttributeError, KeyError):
            # Wall handle doesn't exist, recreate it
            wall_dist = wall_dist_slider.value
            wall_handle = server.scene.add_box(
                "/wall",
                dimensions=(0.01, 2.0, 2.0),
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
    viewing_angle_slider.on_update(lambda _: update_scene())
    rot_front_pos.on_update(lambda _: update_scene())
    rot_front_neg.on_update(lambda _: update_scene())
    rot_side_pos.on_update(lambda _: update_scene())
    rot_side_neg.on_update(lambda _: update_scene())
    rot_y_front_pos.on_update(lambda _: update_scene())
    rot_y_front_neg.on_update(lambda _: update_scene())
    rot_y_side_pos.on_update(lambda _: update_scene())
    rot_y_side_neg.on_update(lambda _: update_scene())
    # Group position offset callbacks
    offset_front_pos_x.on_update(lambda _: update_scene())
    offset_front_pos_y.on_update(lambda _: update_scene())
    offset_front_pos_z.on_update(lambda _: update_scene())
    offset_front_neg_x.on_update(lambda _: update_scene())
    offset_front_neg_y.on_update(lambda _: update_scene())
    offset_front_neg_z.on_update(lambda _: update_scene())
    offset_side_pos_x.on_update(lambda _: update_scene())
    offset_side_pos_y.on_update(lambda _: update_scene())
    offset_side_pos_z.on_update(lambda _: update_scene())
    offset_side_neg_x.on_update(lambda _: update_scene())
    offset_side_neg_y.on_update(lambda _: update_scene())
    offset_side_neg_z.on_update(lambda _: update_scene())
    radius_slider.on_update(lambda _: update_scene())
    circle_center_slider.on_update(lambda _: update_scene())
    ray_length_slider.on_update(lambda _: update_scene())
    led_lumens_slider.on_update(lambda _: None)  # No auto-update, use manual button
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: update_intensity_map())
    row1_chk.on_update(lambda _: update_scene())
    row2_chk.on_update(lambda _: update_scene())
    row3_chk.on_update(lambda _: update_scene())
    row4_chk.on_update(lambda _: update_scene())
    absorbers_enable.on_update(lambda _: (update_scene(), update_ui_visibility()))
    show_camera_fov.on_update(lambda _: update_scene())
    camera_fov_h.on_update(lambda _: update_scene())
    camera_fov_v.on_update(lambda _: update_scene())
    camera_pos_x.on_update(lambda _: update_scene())
    abs0_off_x.on_update(lambda _: update_scene())
    abs0_off_y.on_update(lambda _: update_scene())
    abs0_off_z.on_update(lambda _: update_scene())
    abs1_off_x.on_update(lambda _: update_scene())
    abs1_off_y.on_update(lambda _: update_scene())
    abs1_off_z.on_update(lambda _: update_scene())
    abs2_off_x.on_update(lambda _: update_scene())
    abs2_off_y.on_update(lambda _: update_scene())
    abs2_off_z.on_update(lambda _: update_scene())
    abs2_rot_z.on_update(lambda _: update_scene())
    abs3_off_x.on_update(lambda _: update_scene())
    abs3_off_y.on_update(lambda _: update_scene())
    abs3_off_z.on_update(lambda _: update_scene())
    abs3_rot_z.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: None)  # No auto-update - manual button only
    ray_uniformity_slider.on_update(lambda _: None)  # No auto-update for expensive params
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
            # Restore main wall
            wall_dist = wall_dist_slider.value
            wall_handle = server.scene.add_box(
                "/wall",
                dimensions=(0.01, 2.0, 2.0),
                color=(0.5, 0.5, 0.5),
                position=(wall_dist / 100.0, 0.0, 0.0),
            )
    
    room_mode_enable.on_update(on_room_mode_toggle)
    show_room_walls.on_update(lambda _: draw_room_walls())
    show_room_intensity.on_update(lambda _: (update_room_intensity_map() if (room_mode_enable.value and show_room_intensity.value) else draw_room_walls()) if room_mode_enable.value else None)
    room_front_dist.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    room_side_dist.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    room_top_bottom_dist.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    show_back_wall.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    room_back_dist.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    wall_view_size.on_update(lambda _: None)  # No auto-update for expensive params
    wall_dist_slider.on_update(lambda _: (update_wall(), update_scene()))
    
    # Register LED control button callbacks
    # Group buttons
    for group_idx, btn in group_buttons.items():
        def make_group_handler(g_idx):
            def handler(_):
                start_idx = g_idx * 12
                end_idx = start_idx + 12
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_group_handler(group_idx))
    
    # Row buttons
    for (group_idx, row_idx), btn in row_buttons.items():
        def make_row_handler(g_idx, r_idx):
            def handler(_):
                start_idx = g_idx * 12 + r_idx * 3
                end_idx = start_idx + 3
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} Row {r_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_row_handler(group_idx, row_idx))
    
    # Individual LED buttons
    for led_idx, btn in led_buttons.items():
        def make_led_handler(l_idx):
            def handler(_):
                led_states[l_idx] = not led_states[l_idx]
                print(f"LED {l_idx} toggled to {led_states[l_idx]}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_led_handler(led_idx))
    
    # Button for manual intensity map update
    update_intensity_button.on_click(lambda _: update_intensity_map())
    
    # Button for capturing FOV intensity image
    capture_fov_btn.on_click(lambda _: capture_camera_fov_image())
    
    # Button for room intensity map update
    update_room_button.on_click(lambda _: update_room_intensity_map())

    # Capture default values so reset restores them
    defaults = {
        "viewing_angle": viewing_angle_slider.value,
        "rot_front_pos": rot_front_pos.value,
        "rot_front_neg": rot_front_neg.value,
        "rot_side_pos": rot_side_pos.value,
        "rot_side_neg": rot_side_neg.value,
        "radius": radius_slider.value,
        "circle_center_x": circle_center_slider.value,
        "wall_dist": wall_dist_slider.value,
        "row1": row1_chk.value,
        "row2": row2_chk.value,
        "row3": row3_chk.value,
        "row4": row4_chk.value,
        "absorbers_enable": absorbers_enable.value,
        "abs0_off_x": abs0_off_x.value,
        "abs0_off_y": abs0_off_y.value,
        "abs0_off_z": abs0_off_z.value,
        "abs1_off_x": abs1_off_x.value,
        "abs1_off_y": abs1_off_y.value,
        "abs1_off_z": abs1_off_z.value,
    }

    # Initial draw
    update_scene()
    update_ui_visibility()  # Set initial UI visibility indicators
    if room_mode_enable.value:
        draw_room_walls()

    print("\n" + "=" * 60)
    print("INTERACTIVE LIGHTING DESIGN")
    print("Open http://localhost:8080 in your browser")
    print("Use the sliders on the left to adjust LED parameters")
    print("=" * 60 + "\n")

    # Keep server running; poll the reset button (some Viser button handles
    # don't expose event callbacks). When pressed, restore defaults and redraw.
    try:
        while True:
            time.sleep(0.2)
            try:
                if getattr(reset_button, "value", False):
                    # Restore default slider values
                    viewing_angle_slider.value = defaults["viewing_angle"]
                    rot_front_pos.value = defaults["rot_front_pos"]
                    rot_front_neg.value = defaults["rot_front_neg"]
                    rot_side_pos.value = defaults["rot_side_pos"]
                    rot_side_neg.value = defaults["rot_side_neg"]
                    radius_slider.value = defaults["radius"]
                    circle_center_slider.value = defaults["circle_center_x"]
                    wall_dist_slider.value = defaults["wall_dist"]
                    # Restore row checkbox states
                    try:
                        row1_chk.value = defaults["row1"]
                        row2_chk.value = defaults["row2"]
                        row3_chk.value = defaults["row3"]
                        row4_chk.value = defaults["row4"]
                    except Exception:
                        pass
                    # Restore absorber controls
                    try:
                        absorbers_enable.value = defaults["absorbers_enable"]
                        abs0_off_x.value = defaults["abs0_off_x"]
                        abs0_off_y.value = defaults["abs0_off_y"]
                        abs0_off_z.value = defaults["abs0_off_z"]
                        abs1_off_x.value = defaults["abs1_off_x"]
                        abs1_off_y.value = defaults["abs1_off_y"]
                        abs1_off_z.value = defaults["abs1_off_z"]
                    except Exception:
                        pass

                    # Force wall update and scene redraw
                    update_wall()
                    update_scene()

                    # Clear the button press (Viser button may keep value True)
                    try:
                        reset_button.value = False
                    except Exception:
                        pass
            except Exception:
                # Be defensive: ignore polling errors to keep server alive
                pass
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
