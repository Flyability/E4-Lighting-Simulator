"""
Interactive lighting design tool using Viser.
Allows real-time adjustment of LED parameters with sliders.
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
else:
    # Force line-buffered output so print() messages appear immediately in terminals
    if not sys.stdout.line_buffering:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors='replace', line_buffering=True)
    if not sys.stderr.line_buffering:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors='replace', line_buffering=True)

import numpy as np
import warnings
import viser
import viser.transforms as tf
import time
import multiprocessing
import json
import os
import hashlib
from functools import partial
import trimesh

# GPU acceleration (NVIDIA CUDA via CuPy) — lazy init, no CUDA loaded at import time
try:
    from gpu_raytrace import gpu_process_led_wall_batch, gpu_process_room_batch
    from gpu_raytrace import GPU_AVAILABLE as _GPU_OK, _ensure_gpu_init
    _HAS_GPU_MODULE = True
except ImportError:
    _HAS_GPU_MODULE = False
    print("[GPU] gpu_raytrace module not found, using CPU multiprocessing")

# Suppress viser warnings about removing already-removed nodes
warnings.filterwarnings("ignore", message="Attempted to remove already removed node")


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
                    led.lumens = None  # No per-LED lumens override for base groups
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
            led_beam_tilts = custom_group_config.get('led_beam_tilts', [])
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
                
                # Apply beam_tilt: tilt beam direction relative to square surface
                # (same as individual LED path - Rodrigues rotation around row_dir)
                beam_tilt_deg = led_beam_tilts[i] if i < len(led_beam_tilts) else 0.0
                if abs(beam_tilt_deg) > 0.01:
                    bt_rad = np.radians(beam_tilt_deg)
                    k_bt = row_dir / np.linalg.norm(row_dir)
                    beam_direction = direction * np.cos(bt_rad) + np.cross(k_bt, direction) * np.sin(bt_rad) + k_bt * np.dot(k_bt, direction) * (1 - np.cos(bt_rad))
                    beam_direction = beam_direction / np.linalg.norm(beam_direction)
                else:
                    beam_direction = direction
                
                led = LED(
                    width=size,
                    viewing_angle=led_view_angle,
                    position=(pos_x, pos_y, pos_z),
                    direction=tuple(beam_direction),
                    color=custom_color,
                )
                led.enabled = is_led_on
                led.led_index = led_index
                led.row_direction = row_dir
                led.square_normal = direction  # Store original direction for square mesh orientation
                led.is_custom = True  # Mark as custom group LED
                led.is_dynamic_group = True  # Mark as dynamic group LED
                led.lumens = custom_group_config.get('lumens_override', None)  # Per-group lumens override
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
                    led.lumens = custom_group_config.get('lumens_override', None)  # Per-group lumens override
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
        
        # Apply beam_tilt: tilt beam direction relative to square surface
        # Rotate direction around row_dir axis by beam_tilt angle (Rodrigues)
        beam_tilt_deg = individual_led_config.get('beam_tilt', 0.0)
        if abs(beam_tilt_deg) > 0.01:
            bt_rad = np.radians(beam_tilt_deg)
            k_bt = row_dir / np.linalg.norm(row_dir)
            beam_direction = direction * np.cos(bt_rad) + np.cross(k_bt, direction) * np.sin(bt_rad) + k_bt * np.dot(k_bt, direction) * (1 - np.cos(bt_rad))
            beam_direction = beam_direction / np.linalg.norm(beam_direction)
        else:
            beam_direction = direction
        
        # Create LED - square orientation uses original direction, beam uses tilted direction
        individual_color = (0.0, 1.0, 1.0)  # Cyan for individual LEDs
        led = LED(
            width=size,
            viewing_angle=led_viewing_angle,
            position=(pos_x, pos_y, pos_z),
            direction=tuple(beam_direction),
            color=individual_color,
        )
        led.enabled = led_is_on  # LED is active only if turned on
        led.led_index = led_index
        led.row_direction = row_dir
        led.square_normal = direction  # Store original direction for square mesh orientation
        led.is_individual = True  # Mark as individual LED
        led.lumens = individual_led_config.get('lumens_override', None)  # Per-LED lumens override
        # Apply external collimating lens (per-LED)
        ext_lens_angle = individual_led_config.get('ext_lens_angle', None)
        if ext_lens_angle is not None:
            ext_lens_eff = individual_led_config.get('ext_lens_efficiency', 1.0)
            led.ext_lens_angle = ext_lens_angle  # Store for n calculation (cone stays wide)
            if led.lumens is not None:
                led.lumens = led.lumens * ext_lens_eff
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

def _prepare_mesh_ray_accelerator(mesh_data):
    """
    Pre-build a trimesh ray accelerator and inverse transform for batch ray testing.
    Call once before tracing, then use _batch_ray_mesh_intersection for all rays.
    
    Returns: dict with 'mesh', 'inv_transform', 'inv_rot' or None if no mesh.
    """
    if mesh_data is None:
        return None
    
    import trimesh as _trimesh
    
    # Build trimesh object with BVH (cached on mesh_data dict)
    if 'trimesh_obj' not in mesh_data:
        mesh_data['trimesh_obj'] = _trimesh.Trimesh(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces'],
            process=False
        )
    
    mesh = mesh_data['trimesh_obj']
    transform = mesh_data.get('transform', np.eye(4))
    
    try:
        inv_transform = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        inv_transform = np.eye(4)
    
    # Force BVH construction now (so it's not built lazily during first query)
    _ = mesh.ray
    
    return {
        'mesh': mesh,
        'inv_transform': inv_transform,
        'inv_rot': inv_transform[:3, :3],
        'transform': transform,
    }

# Thread pool for parallel mesh intersection (Embree releases the GIL)
_mesh_intersection_pool = None

import atexit as _atexit
def _cleanup_mesh_pool():
    global _mesh_intersection_pool
    if _mesh_intersection_pool is not None:
        try:
            _mesh_intersection_pool.shutdown(wait=False)
        except Exception:
            pass
        _mesh_intersection_pool = None
_atexit.register(_cleanup_mesh_pool)


def _intersect_chunk_thread(mesh, origins, dirs):
    """Worker function for ThreadPoolExecutor — runs Embree without GIL."""
    N = len(origins)
    if N == 0:
        return np.zeros(0, dtype=bool)
    index_tri, index_ray = mesh.ray.intersects_id(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False,
    )
    chunk_hits = np.zeros(N, dtype=bool)
    if len(index_ray) > 0:
        face_normals = mesh.face_normals[index_tri]
        hit_ray_dirs = dirs[index_ray]
        dots = np.sum(hit_ray_dirs * face_normals, axis=1)
        chunk_hits[index_ray[dots < 0]] = True
    return chunk_hits


def _batch_ray_mesh_intersection(ray_origins, ray_directions, accel):
    """
    Test multiple rays against mesh in a single batched call.
    Uses AABB pre-filter to skip rays that can't hit the mesh,
    then parallelizes across CPU cores for large batches.
    Only counts front-face hits (ray hitting the outer surface of the mesh).
    
    Args:
        ray_origins: (N, 3) array of ray origins
        ray_directions: (N, 3) array of ray directions (normalized)
        accel: accelerator dict from _prepare_mesh_ray_accelerator
    
    Returns:
        boolean array of shape (N,) — True where ray hits the mesh front face
    """
    global _mesh_intersection_pool, _mesh_pool_key
    
    if accel is None:
        return np.zeros(len(ray_origins), dtype=bool)
    
    N = len(ray_origins)
    if N == 0:
        return np.zeros(0, dtype=bool)
    
    mesh = accel['mesh']
    inv_transform = accel['inv_transform']
    inv_rot = accel['inv_rot']
    
    # Transform all ray origins to mesh local space
    origins_h = np.hstack([ray_origins, np.ones((N, 1))])  # (N, 4)
    origins_local = (inv_transform @ origins_h.T).T[:, :3]  # (N, 3)
    
    # Transform ray directions (rotation only, then normalize)
    dirs_local = (inv_rot @ ray_directions.T).T  # (N, 3)
    norms = np.linalg.norm(dirs_local, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    dirs_local = dirs_local / norms
    
    # ---- AABB pre-filter ----
    # Skip rays that can't possibly hit the mesh bounding box (very fast, pure numpy)
    bbox = mesh.bounds  # (2, 3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    pad = np.maximum((bbox[1] - bbox[0]) * 0.001, 1e-6)
    bmin = bbox[0] - pad
    bmax = bbox[1] + pad
    
    eps = 1e-10
    safe_dirs = np.where(np.abs(dirs_local) < eps,
                         np.copysign(eps, dirs_local + 1e-30), dirs_local)
    inv_d = 1.0 / safe_dirs
    t1 = (bmin - origins_local) * inv_d
    t2 = (bmax - origins_local) * inv_d
    t_enter = np.max(np.minimum(t1, t2), axis=1)
    t_exit = np.min(np.maximum(t1, t2), axis=1)
    can_hit = (t_enter <= t_exit) & (t_exit > 0)
    
    candidates = np.where(can_hit)[0]
    n_cand = len(candidates)
    
    hits = np.zeros(N, dtype=bool)
    if n_cand == 0:
        return hits
    
    print(f"    AABB pre-filter: {N:,} → {n_cand:,} rays ({100*n_cand/N:.0f}%) | mesh: {len(mesh.faces):,} faces")
    
    cand_origins = origins_local[candidates]
    cand_dirs = dirs_local[candidates]
    
    # ---- Parallel or single-threaded intersection ----
    PARALLEL_THRESHOLD = 20000
    n_cpus = multiprocessing.cpu_count()
    use_parallel = n_cand >= PARALLEL_THRESHOLD and n_cpus > 1
    cand_hits = None
    
    if use_parallel:
        n_workers = min(n_cpus, 8)
        chunk_size = max(2000, (n_cand + n_workers - 1) // n_workers)
        
        try:
            global _mesh_intersection_pool
            if _mesh_intersection_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                _mesh_intersection_pool = ThreadPoolExecutor(max_workers=n_workers)
                print(f"    Started {n_workers}-thread pool for mesh intersection (Embree releases GIL)")
            
            futures = []
            for i in range(0, n_cand, chunk_size):
                futures.append(_mesh_intersection_pool.submit(
                    _intersect_chunk_thread, mesh,
                    cand_origins[i:i+chunk_size], cand_dirs[i:i+chunk_size]
                ))
            
            results = [f.result() for f in futures]
            cand_hits = np.concatenate(results)
        except Exception as e:
            print(f"    Parallel mesh intersection failed ({e}), falling back to single-thread")
            cand_hits = None
    
    if cand_hits is None:
        # Single-threaded path (small batch or parallel failed)
        index_tri, index_ray = mesh.ray.intersects_id(
            ray_origins=cand_origins,
            ray_directions=cand_dirs,
            multiple_hits=False,
        )
        cand_hits = np.zeros(n_cand, dtype=bool)
        if len(index_ray) > 0:
            face_normals = mesh.face_normals[index_tri]
            hit_ray_dirs = cand_dirs[index_ray]
            dots = np.sum(hit_ray_dirs * face_normals, axis=1)
            cand_hits[index_ray[dots < 0]] = True
    
    hits[candidates[cand_hits]] = True
    return hits

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
    # multiple_hits=False: only get the closest hit (much faster)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin_local],
        ray_directions=[ray_direction_local],
        multiple_hits=False,
    )
    
    if len(locations) == 0:
        return None
    
    # With multiple_hits=False we get only the closest hit — check if it's front-face
    face_normal = mesh.face_normals[index_tri[0]]
    dot_product = np.dot(ray_direction_local, face_normal)
    
    if dot_product >= 0:
        # Closest hit is a back face — ray is inside the mesh, don't absorb
        return None
    
    # Calculate distance for the front-face hit
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

def _ray_box_intersection_batch_np(origins, directions, absorbers):
    """Vectorized ray-box intersection for ALL rays against ALL absorbers (NumPy).

    Mirrors the GPU version ``_gpu_ray_box_intersection_batch`` but runs on CPU
    using pure NumPy – eliminates the O(rays × absorbers) Python loop.

    Parameters
    ----------
    origins : ndarray (N, 3) – ray origin positions
    directions : ndarray (N, 3) – ray direction vectors (assumed normalised)
    absorbers : list[dict] – each dict has 'center', 'half_sizes', optional 'rotation'

    Returns
    -------
    absorbed : ndarray (N,) bool – True where *any* absorber blocks the ray
    """
    N = origins.shape[0]
    absorbed = np.zeros(N, dtype=bool)

    for box in absorbers:
        center = np.asarray(box['center'], dtype=np.float32)
        half = np.asarray(box['half_sizes'], dtype=np.float32)
        rotation = box.get('rotation', None)

        if rotation is not None:
            qw, qx, qy, qz = rotation
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ], dtype=np.float32)
            R_inv = R.T
            local_origins = (origins - center[None, :]) @ R_inv.T
            local_dirs = directions @ R_inv.T
            local_center = np.zeros(3, dtype=np.float32)
        else:
            local_origins = origins - center[None, :]
            local_dirs = directions
            local_center = np.zeros(3, dtype=np.float32)

        tmin = np.full(N, -1e30, dtype=np.float32)
        tmax = np.full(N, 1e30, dtype=np.float32)
        valid = np.ones(N, dtype=bool)

        for k in range(3):
            d_k = local_dirs[:, k]
            o_k = local_origins[:, k]
            lo = local_center[k] - half[k]
            hi = local_center[k] + half[k]

            parallel = np.abs(d_k) < 1e-12
            outside = parallel & ((o_k < lo) | (o_k > hi))
            valid &= ~outside

            inv_d = np.where(parallel, np.float32(1.0),
                             np.float32(1.0) / np.where(parallel, np.float32(1.0), d_k))
            t1 = (lo - o_k) * inv_d
            t2 = (hi - o_k) * inv_d
            t_near = np.minimum(t1, t2)
            t_far = np.maximum(t1, t2)

            mask_np = ~parallel
            tmin = np.where(mask_np & (t_near > tmin), t_near, tmin)
            tmax = np.where(mask_np & (t_far < tmax), t_far, tmax)

        box_hit = valid & (tmin <= tmax) & (tmax > 0)
        absorbed |= box_hit

    return absorbed


def _compute_uniformity_html(grid, fov_bounds=None, wall_size_cm=None):
    """Compute illuminotechnical uniformity metrics from a 2D lux grid.

    If *fov_bounds* is given as (fov_width_cm, fov_height_cm) and
    *wall_size_cm* is provided, only the sub-grid inside the FOV rectangle
    (centred on the wall) is analysed.

    Returns an HTML string to be appended below the intensity legend.
    Metrics computed (excluding zero-lux cells outside the light cone):
        - U0 = Emin / Eavg   (General Uniformity)
        - U1 = Emin / Emax   (Overall Uniformity)
        - CV = sigma / Eavg   (Coefficient of Variation)
        - deltaEV  (perceptual stop range) with diagnostic classification
    """
    # --- Crop to FOV rectangle if requested ---
    if fov_bounds is not None and wall_size_cm is not None:
        fov_w_cm, fov_h_cm = fov_bounds
        gz, gy = grid.shape  # rows=Z, cols=Y
        cell_cm = wall_size_cm / gy  # square grid assumed (gz == gy)
        half_wall = wall_size_cm / 2.0
        # Y-axis limits (columns)
        y_lo = int(max(0, (half_wall - fov_w_cm / 2.0) / cell_cm))
        y_hi = int(min(gy, (half_wall + fov_w_cm / 2.0) / cell_cm))
        # Z-axis limits (rows)
        z_lo = int(max(0, (half_wall - fov_h_cm / 2.0) / cell_cm))
        z_hi = int(min(gz, (half_wall + fov_h_cm / 2.0) / cell_cm))
        grid = grid[z_lo:z_hi, y_lo:y_hi]

    active = grid[grid > 0]
    if active.size == 0:
        return ""

    E_max = float(active.max())
    E_min = float(active.min())
    E_avg = float(active.mean())
    sigma = float(active.std())

    U0 = E_min / E_avg if E_avg > 0 else 0.0   # General Uniformity
    U1 = E_min / E_max if E_max > 0 else 0.0   # Overall Uniformity
    CV = sigma / E_avg if E_avg > 0 else 0.0    # Coefficient of Variation

    # --- Perceptual (EV / photographic stops) ---
    # EV = log2(Lux * 0.4)
    EV_max = np.log2(E_max * 0.4) if E_max * 0.4 > 0 else 0.0
    EV_min = np.log2(E_min * 0.4) if E_min * 0.4 > 0 else 0.0
    delta_EV = EV_max - EV_min

    # Classification thresholds
    if delta_EV < 0.3:
        ev_label = "Absolute uniformity"
        ev_detail = "green-screen / archival grade"
        ev_color = "#4CAF50"  # green
    elif delta_EV <= 1.0:
        ev_label = "Soft wall washing"
        ev_detail = "smooth gradient, great general lighting"
        ev_color = "#2196F3"  # blue
    elif delta_EV <= 2.0:
        ev_label = "Moderate gradient"
        ev_detail = "visible fall-off, acceptable for most uses"
        ev_color = "#FF9800"  # orange
    else:
        ev_label = "Wall grazing"
        ev_detail = "dramatic gradient, good for textures"
        ev_color = "#F44336"  # red

    # Uniformity percentage: U0 expressed as %
    uniformity_pct = U0 * 100.0

    html = (
        "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
        "<div style='font-weight:600;margin-bottom:4px;'>Pattern Uniformity</div>"
        f"<div style='font-size:22px;font-weight:700;color:{ev_color};margin:2px 0 6px;'>{uniformity_pct:.1f}%</div>"
        "<table style='font-size:11px;color:#ccc;border-collapse:collapse;width:100%;'>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>max</sub></td><td>{E_max:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>min</sub></td><td>{E_min:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>avg</sub></td><td>{E_avg:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>U<sub>0</sub> (E<sub>min</sub>/E<sub>avg</sub>)</td><td>{U0:.3f}</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>U<sub>1</sub> (E<sub>min</sub>/E<sub>max</sub>)</td><td>{U1:.3f}</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>CV (&sigma;/E<sub>avg</sub>)</td><td>{CV:.3f}</td></tr>"
        "<tr><td colspan='2' style='padding-top:4px;'></td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>EV<sub>max</sub></td><td>{EV_max:.2f} stops</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>EV<sub>min</sub></td><td>{EV_min:.2f} stops</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>&Delta;EV</td><td><b>{delta_EV:.2f} stops</b></td></tr>"
        "</table>"
        f"<div style='margin-top:6px;padding:4px 6px;background:{ev_color}22;border-left:3px solid {ev_color};border-radius:2px;'>"
        f"<span style='font-weight:600;color:{ev_color};'>{ev_label}</span><br/>"
        f"<span style='font-size:10px;color:#aaa;'>{ev_detail}</span>"
        "</div></div>"
    )
    return html


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


def _get_effective_n(led, ray_uniformity):
    """Get Lambertian exponent, using ext_lens_angle for sharper concentration if present."""
    ext_lens_angle = getattr(led, 'ext_lens_angle', None)
    if ext_lens_angle is not None:
        return _calculate_lambertian_exponent(ext_lens_angle, ray_uniformity)
    return _calculate_lambertian_exponent(led.viewing_angle, ray_uniformity)


def _lens_efficiency(viewing_angle):
    """Optical efficiency of an LED collimating lens.

    A bare LED (~120° beam) needs no lens (eta=1.0).  Narrower beams
    require a lens that redirects Lambertian flux into a tighter cone,
    incurring losses from Fresnel reflections, absorption & scattering.

    Model: fraction of Lambertian flux that must be redirected determines
    the loss.  eta = 1 - max_loss * max(0, 1 - sin^2(a/2) / sin^2(60))
    Typical results: 120->1.00  90->0.93  60->0.87  30->0.82  10->0.80
    """
    MAX_LOSS = 0.20
    sin2_half = np.sin(np.radians(viewing_angle / 2.0)) ** 2
    SIN2_REF = 0.75  # sin^2(60°) – Lambertian reference
    frac = max(0.0, 1.0 - sin2_half / SIN2_REF)
    return 1.0 - MAX_LOSS * frac

# Wall inward normals for room reflection (pointing INTO the room)
_WALL_INWARD_NORMALS = {
    'front':  np.array([-1.0,  0.0,  0.0]),
    'back':   np.array([ 1.0,  0.0,  0.0]),
    'left':   np.array([ 0.0,  1.0,  0.0]),
    'right':  np.array([ 0.0, -1.0,  0.0]),
    'top':    np.array([ 0.0,  0.0, -1.0]),
    'bottom': np.array([ 0.0,  0.0,  1.0]),
}

def _sample_cosine_hemisphere(normal):
    """Sample a cosine-weighted random direction in the hemisphere around normal.
    
    Physics: Lambertian (diffuse) reflection. The reflected radiance is uniform
    in all directions, so sampling proportional to cos(theta) gives correct
    flux weighting (BRDF = rho/pi, importance-sampled).
    """
    u1 = np.random.random()
    u2 = np.random.random()
    # Malley's method: project uniform disk sample to hemisphere
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    x_local = r * np.cos(phi)
    y_local = r * np.sin(phi)
    z_local = np.sqrt(max(0.0, 1.0 - u1))
    # Build tangent frame from normal
    if abs(normal[2]) < 0.999:
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    # Transform local to world
    world_dir = x_local * tangent + y_local * bitangent + z_local * normal
    norm = np.linalg.norm(world_dir)
    if norm > 1e-10:
        world_dir /= norm
    return world_dir

def _process_led_wall_worker(args):
    """Worker function for single wall ray tracing (multiprocessing).
    Batches all rays and tests mesh intersection in a single call for performance."""
    led, params = args
    
    # Unpack parameters
    wall_dist = params['wall_dist']
    rays_per_led = params['rays_per_led']
    grid_size = params['grid_size']
    wall_size = params['wall_size']
    lumens_per_led = getattr(led, 'lumens', None) or params['lumens_per_led']
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
    n = _get_effective_n(led, ray_uniformity)
    max_theta = np.radians(led.viewing_angle / 2.0)
    cos_max = np.cos(max_theta)
    
    # --- Generate ALL rays at once (vectorized) ---
    u = np.random.uniform(0, 1, (rays_per_led, 2))
    cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = 2 * np.pi * u[:, 1]
    
    sin_theta = np.sin(theta)
    local_dirs = np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ])  # (N, 3)
    
    # Transform to world coordinates
    world_dirs = (local_dirs[:, 0:1] * x_axis +
                  local_dirs[:, 1:2] * y_axis +
                  local_dirs[:, 2:3] * z_axis)
    norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
    world_dirs = world_dirs / norms  # (N, 3)
    
    # Calculate lumens per ray with cone normalization and lens efficiency
    cos_max_n1 = cos_max ** (n + 1.0)
    denom = 1.0 - cos_max_n1
    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
    intensity_coefficients = np.power(cos_theta_clamped, n)
    lumens_per_ray = (lumens_per_led / rays_per_led) * intensity_coefficients * norm_factor  # (N,)
    
    # --- Check box absorber intersection (vectorized) ---
    hit_absorbed = np.zeros(rays_per_led, dtype=bool)
    if absorbers:
        ray_origins = np.broadcast_to(led.position, (rays_per_led, 3)).copy().astype(np.float32)
        hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
    
    # --- Batch STL mesh intersection (single call for all non-absorbed rays) ---
    stl_mesh_data = params.get('stl_mesh_data')
    if stl_mesh_data is not None:
        # Only test rays not already absorbed by boxes
        not_absorbed_mask = ~hit_absorbed
        not_absorbed_indices = np.where(not_absorbed_mask)[0]
        
        if len(not_absorbed_indices) > 0:
            origins = np.tile(led.position, (len(not_absorbed_indices), 1)).astype(np.float64)
            directions = world_dirs[not_absorbed_indices]
            
            accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
            mesh_hits = _batch_ray_mesh_intersection(origins, directions, accel)
            hit_absorbed[not_absorbed_indices[mesh_hits]] = True
    
    # --- Compute wall hits for non-absorbed rays ---
    alive = ~hit_absorbed  # rays that weren't absorbed
    
    # Only process rays going towards the wall (world_dir[0] > 0)
    towards_wall = world_dirs[:, 0] > 0
    valid = alive & towards_wall
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) > 0:
        t = (wall_dist - led.position[0]) / world_dirs[valid_indices, 0]
        pos_t = t > 0
        vi = valid_indices[pos_t]
        t = t[pos_t]
        
        hit_y = led.position[1] + world_dirs[vi, 1] * t
        hit_z = led.position[2] + world_dirs[vi, 2] * t
        
        grid_y = ((hit_y + half_size) / cell_size).astype(int)
        grid_z = ((hit_z + half_size) / cell_size).astype(int)
        
        in_bounds = (grid_y >= 0) & (grid_y < grid_size) & (grid_z >= 0) & (grid_z < grid_size)
        bi = np.where(in_bounds)[0]
        
        lux_values = lumens_per_ray[vi[bi]] / cell_area_m2
        np.add.at(local_grid, (grid_z[bi], grid_y[bi]), lux_values)
    
    return local_grid

def _process_led_wall_worker_with_accel(args, accel):
    """Same as _process_led_wall_worker but uses a pre-built BVH accelerator.
    Thread-safe: uses per-LED RandomState instead of global seed."""
    led, params = args
    
    wall_dist = params['wall_dist']
    rays_per_led = params['rays_per_led']
    grid_size = params['grid_size']
    wall_size = params['wall_size']
    lumens_per_led = getattr(led, 'lumens', None) or params['lumens_per_led']
    absorbers = params['absorbers']
    ray_uniformity = params['ray_uniformity']
    led_idx = params['led_idx']
    
    local_grid = np.zeros((grid_size, grid_size))
    cell_size = wall_size / grid_size
    cell_area_m2 = (cell_size * cell_size) / 10000.0
    half_size = wall_size / 2
    
    # Thread-safe RNG (each LED gets its own independent generator)
    rng = np.random.RandomState((42 + led_idx) % (2**32))
    
    z_axis = led.direction
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, [0, 0, 1])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    n = _get_effective_n(led, ray_uniformity)
    max_theta = np.radians(led.viewing_angle / 2.0)
    cos_max = np.cos(max_theta)
    
    # Generate ALL rays (thread-safe)
    u = rng.uniform(0, 1, (rays_per_led, 2))
    cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = 2 * np.pi * u[:, 1]
    
    sin_theta = np.sin(theta)
    local_dirs = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    
    world_dirs = (local_dirs[:, 0:1] * x_axis + local_dirs[:, 1:2] * y_axis + local_dirs[:, 2:3] * z_axis)
    norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
    world_dirs = world_dirs / norms
    
    cos_max_n1 = cos_max ** (n + 1.0)
    denom = 1.0 - cos_max_n1
    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
    lumens_per_ray = (lumens_per_led / rays_per_led) * np.power(cos_theta_clamped, n) * norm_factor
    
    # Check box absorbers (vectorized)
    hit_absorbed = np.zeros(rays_per_led, dtype=bool)
    if absorbers:
        ray_origins = np.broadcast_to(led.position, (rays_per_led, 3)).copy().astype(np.float32)
        hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
    
    # Batch mesh intersection using PRE-BUILT accelerator (no BVH rebuild!)
    if accel is not None:
        not_absorbed_indices = np.where(~hit_absorbed)[0]
        if len(not_absorbed_indices) > 0:
            origins = np.tile(led.position, (len(not_absorbed_indices), 1)).astype(np.float64)
            mesh_hits = _batch_ray_mesh_intersection(origins, world_dirs[not_absorbed_indices], accel)
            hit_absorbed[not_absorbed_indices[mesh_hits]] = True
    
    # Wall hits
    alive = ~hit_absorbed
    towards_wall = world_dirs[:, 0] > 0
    valid_indices = np.where(alive & towards_wall)[0]
    
    if len(valid_indices) > 0:
        t = (wall_dist - led.position[0]) / world_dirs[valid_indices, 0]
        pos_t = t > 0
        vi = valid_indices[pos_t]
        t = t[pos_t]
        
        hit_y = led.position[1] + world_dirs[vi, 1] * t
        hit_z = led.position[2] + world_dirs[vi, 2] * t
        
        grid_y = ((hit_y + half_size) / cell_size).astype(int)
        grid_z = ((hit_z + half_size) / cell_size).astype(int)
        
        in_bounds = (grid_y >= 0) & (grid_y < grid_size) & (grid_z >= 0) & (grid_z < grid_size)
        bi = np.where(in_bounds)[0]
        
        lux_values = lumens_per_ray[vi[bi]] / cell_area_m2
        np.add.at(local_grid, (grid_z[bi], grid_y[bi]), lux_values)
    
    return local_grid

def _process_led_worker(args):
    """Worker function to process rays for a single LED (for multiprocessing).
    Batches all rays and tests mesh intersection in a single call for performance."""
    led, params = args
    
    # Unpack parameters
    front_dist = params['front_dist']
    side_dist = params['side_dist']
    top_bottom_dist = params['top_bottom_dist']
    back_dist = params.get('back_dist')
    led_x_center = params.get('led_x_center', -35)
    num_rays_per_led = params['num_rays_per_led']
    grid_size = params['grid_size']
    lumens_per_led = getattr(led, 'lumens', None) or params['lumens_per_led']
    absorbers = params['absorbers']
    stl_mesh_data = params.get('stl_mesh_data')
    ray_uniformity = params['ray_uniformity']
    grid_shapes = params['grid_shapes']
    wall_specs = params['wall_specs']
    max_bounces = params.get('max_bounces', 0)
    wall_reflectance = params.get('wall_reflectance', 0.0)
    
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
        cell_areas_m2[wall_name] = cell_area_cm2 / 10000.0
    
    # Build local coordinate system from LED direction
    z_axis = led.direction
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, [0, 0, 1])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Calculate rays per LED and emission parameters
    rays_traced = num_rays_per_led * grid_size * grid_size
    n = _get_effective_n(led, ray_uniformity)
    max_theta = np.radians(led.viewing_angle / 2.0)
    cos_max = np.cos(max_theta)
    
    # --- Generate ALL rays at once (vectorized) ---
    u = np.random.uniform(0, 1, (rays_traced, 2))
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
    
    # Transform to world coordinates
    world_dirs = (local_dirs[:, 0:1] * x_axis +
                  local_dirs[:, 1:2] * y_axis +
                  local_dirs[:, 2:3] * z_axis)
    norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
    world_dirs = world_dirs / norms
    
    # Calculate lumens per ray with cone normalization and lens efficiency
    cos_max_n1 = cos_max ** (n + 1.0)
    denom = 1.0 - cos_max_n1
    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
    cos_n_theta = np.power(cos_theta_clamped, n)
    lumens_per_ray = (lumens_per_led / rays_traced) * cos_n_theta * norm_factor
    
    # --- Check box absorber intersection (vectorized) ---
    hit_absorbed = np.zeros(rays_traced, dtype=bool)
    if absorbers:
        ray_origins = np.broadcast_to(led.position, (rays_traced, 3)).copy().astype(np.float32)
        hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
    
    # --- Batch STL mesh intersection ---
    if stl_mesh_data is not None:
        not_absorbed_indices = np.where(~hit_absorbed)[0]
        if len(not_absorbed_indices) > 0:
            origins = np.tile(led.position, (len(not_absorbed_indices), 1)).astype(np.float64)
            directions = world_dirs[not_absorbed_indices]
            accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
            mesh_hits = _batch_ray_mesh_intersection(origins, directions, accel)
            hit_absorbed[not_absorbed_indices[mesh_hits]] = True
    
    # --- Process wall hits for non-absorbed rays ---
    local_total_rays = rays_traced
    alive_indices = np.where(~hit_absorbed)[0]
    
    pos = led.position
    
    for idx in alive_indices:
        ray_dir = world_dirs[idx].copy()
        ray_origin = np.array(pos, dtype=np.float64)
        ray_flux = lumens_per_ray[idx]
        
        for bounce in range(max_bounces + 1):
            # Calculate intersection with each wall
            intersections = []
            
            if ray_dir[0] > 0:
                t = (front_dist - ray_origin[0]) / ray_dir[0]
                if t > 0:
                    intersections.append(('front', t, ray_origin[1] + ray_dir[1] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[1] < 0:
                t = (-side_dist - ray_origin[1]) / ray_dir[1]
                if t > 0:
                    intersections.append(('left', t, ray_origin[0] + ray_dir[0] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[1] > 0:
                t = (side_dist - ray_origin[1]) / ray_dir[1]
                if t > 0:
                    intersections.append(('right', t, ray_origin[0] + ray_dir[0] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[2] > 0:
                t = (top_bottom_dist - ray_origin[2]) / ray_dir[2]
                if t > 0:
                    intersections.append(('top', t, ray_origin[0] + ray_dir[0] * t, ray_origin[1] + ray_dir[1] * t))
            
            if ray_dir[2] < 0:
                t = (-top_bottom_dist - ray_origin[2]) / ray_dir[2]
                if t > 0:
                    intersections.append(('bottom', t, ray_origin[0] + ray_dir[0] * t, ray_origin[1] + ray_dir[1] * t))
            
            if back_dist is not None and ray_dir[0] < 0:
                t = (-back_dist - ray_origin[0]) / ray_dir[0]
                if t > 0:
                    intersections.append(('back', t, ray_origin[1] + ray_dir[1] * t, ray_origin[2] + ray_dir[2] * t))
            
            if not intersections:
                break
            
            wall_name, t_min, coord1, coord2 = min(intersections, key=lambda x: x[1])
            wall_spec = wall_specs[wall_name]
            
            # Map coordinates to grid indices
            if wall_name in ('front', 'back'):
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
            
            lux_contribution = ray_flux / cell_areas_m2[wall_name]
            local_grids[wall_name][grid_i, grid_j] += lux_contribution
            local_ray_hits[wall_name] += 1
            
            # Check if we should generate a reflected bounce
            if bounce >= max_bounces or wall_reflectance <= 0:
                break
            ray_flux *= wall_reflectance
            if ray_flux < 1e-8:  # negligible flux
                break
            # Update ray origin to hit point + small offset along wall normal
            hit_point = ray_origin + ray_dir * t_min
            normal = _WALL_INWARD_NORMALS[wall_name]
            ray_origin = hit_point + normal * 0.01  # 0.01cm offset to avoid self-intersection
            ray_dir = _sample_cosine_hemisphere(normal)
    
    return local_grids, local_ray_hits, local_total_rays


def _process_led_worker_with_accel(args, accel):
    """Same as _process_led_worker but uses a pre-built BVH accelerator.
    Avoids rebuilding BVH per LED in single-process mode."""
    led, params = args
    
    front_dist = params['front_dist']
    side_dist = params['side_dist']
    top_bottom_dist = params['top_bottom_dist']
    back_dist = params.get('back_dist')
    led_x_center = params.get('led_x_center', -35)
    num_rays_per_led = params['num_rays_per_led']
    grid_size = params['grid_size']
    lumens_per_led = getattr(led, 'lumens', None) or params['lumens_per_led']
    absorbers = params['absorbers']
    ray_uniformity = params['ray_uniformity']
    grid_shapes = params['grid_shapes']
    wall_specs = params['wall_specs']
    max_bounces = params.get('max_bounces', 0)
    wall_reflectance = params.get('wall_reflectance', 0.0)
    
    local_grids = {
        'front': np.zeros(grid_shapes['front']),
        'left': np.zeros(grid_shapes['left']),
        'right': np.zeros(grid_shapes['right']),
        'top': np.zeros(grid_shapes['top']),
        'bottom': np.zeros(grid_shapes['bottom'])
    }
    local_ray_hits = {'front': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
    
    if back_dist is not None:
        local_grids['back'] = np.zeros(grid_shapes['back'])
        local_ray_hits['back'] = 0
    
    cell_areas_m2 = {}
    for wall_name, spec in wall_specs.items():
        if wall_name in ('front', 'back'):
            cell_width_cm = spec['size_y'] / spec['grid_y']
            cell_height_cm = spec['size_z'] / spec['grid_z']
        elif wall_name in ['left', 'right']:
            cell_width_cm = spec['size_x'] / spec['grid_x']
            cell_height_cm = spec['size_z'] / spec['grid_z']
        else:
            cell_width_cm = spec['size_x'] / spec['grid_x']
            cell_height_cm = spec['size_y'] / spec['grid_y']
        cell_areas_m2[wall_name] = (cell_width_cm * cell_height_cm) / 10000.0
    
    # Thread-safe RNG (each LED gets its own independent generator)
    rng = np.random.RandomState((42 + int(getattr(led, 'led_idx', id(led)))) % (2**32))
    
    z_axis = led.direction
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross(z_axis, [0, 0, 1])
    else:
        x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    rays_traced = num_rays_per_led * grid_size * grid_size
    n = _get_effective_n(led, ray_uniformity)
    max_theta = np.radians(led.viewing_angle / 2.0)
    cos_max = np.cos(max_theta)
    
    u = rng.uniform(0, 1, (rays_traced, 2))
    cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    phi = 2 * np.pi * u[:, 1]
    
    sin_theta = np.sin(theta)
    local_dirs = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    
    world_dirs = (local_dirs[:, 0:1] * x_axis + local_dirs[:, 1:2] * y_axis + local_dirs[:, 2:3] * z_axis)
    norms = np.linalg.norm(world_dirs, axis=1, keepdims=True)
    world_dirs = world_dirs / norms
    
    cos_max_n1 = cos_max ** (n + 1.0)
    denom = 1.0 - cos_max_n1
    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
    cos_n_theta = np.power(cos_theta_clamped, n)
    lumens_per_ray = (lumens_per_led / rays_traced) * cos_n_theta * norm_factor
    
    hit_absorbed = np.zeros(rays_traced, dtype=bool)
    if absorbers:
        ray_origins = np.broadcast_to(led.position, (rays_traced, 3)).copy().astype(np.float32)
        hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
    
    # Batch mesh intersection using PRE-BUILT accelerator (no BVH rebuild!)
    if accel is not None:
        not_absorbed_indices = np.where(~hit_absorbed)[0]
        if len(not_absorbed_indices) > 0:
            origins = np.tile(led.position, (len(not_absorbed_indices), 1)).astype(np.float64)
            mesh_hits = _batch_ray_mesh_intersection(origins, world_dirs[not_absorbed_indices], accel)
            hit_absorbed[not_absorbed_indices[mesh_hits]] = True
    
    local_total_rays = rays_traced
    alive_indices = np.where(~hit_absorbed)[0]
    pos = led.position
    
    for idx in alive_indices:
        ray_dir = world_dirs[idx].copy()
        ray_origin = np.array(pos, dtype=np.float64)
        ray_flux = lumens_per_ray[idx]
        
        for bounce in range(max_bounces + 1):
            intersections = []
            
            if ray_dir[0] > 0:
                t = (front_dist - ray_origin[0]) / ray_dir[0]
                if t > 0:
                    intersections.append(('front', t, ray_origin[1] + ray_dir[1] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[1] < 0:
                t = (-side_dist - ray_origin[1]) / ray_dir[1]
                if t > 0:
                    intersections.append(('left', t, ray_origin[0] + ray_dir[0] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[1] > 0:
                t = (side_dist - ray_origin[1]) / ray_dir[1]
                if t > 0:
                    intersections.append(('right', t, ray_origin[0] + ray_dir[0] * t, ray_origin[2] + ray_dir[2] * t))
            
            if ray_dir[2] > 0:
                t = (top_bottom_dist - ray_origin[2]) / ray_dir[2]
                if t > 0:
                    intersections.append(('top', t, ray_origin[0] + ray_dir[0] * t, ray_origin[1] + ray_dir[1] * t))
            
            if ray_dir[2] < 0:
                t = (-top_bottom_dist - ray_origin[2]) / ray_dir[2]
                if t > 0:
                    intersections.append(('bottom', t, ray_origin[0] + ray_dir[0] * t, ray_origin[1] + ray_dir[1] * t))
            
            if back_dist is not None and ray_dir[0] < 0:
                t = (-back_dist - ray_origin[0]) / ray_dir[0]
                if t > 0:
                    intersections.append(('back', t, ray_origin[1] + ray_dir[1] * t, ray_origin[2] + ray_dir[2] * t))
            
            if not intersections:
                break
            
            wall_name, t_min, coord1, coord2 = min(intersections, key=lambda x: x[1])
            wall_spec = wall_specs[wall_name]
            
            if wall_name in ('front', 'back'):
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
            else:
                size_x = wall_spec['size_x']
                size_y = wall_spec['size_y']
                grid_size_x = wall_spec['grid_x']
                grid_size_y = wall_spec['grid_y']
                x_min = wall_spec['x_min']
                x_idx = int((coord1 - x_min) / (size_x/grid_size_x))
                y_idx = int((coord2 + size_y/2) / (size_y/grid_size_y))
                grid_i = y_idx
                grid_j = x_idx
            
            grid_shape = local_grids[wall_name].shape
            grid_i = max(0, min(grid_shape[0] - 1, grid_i))
            grid_j = max(0, min(grid_shape[1] - 1, grid_j))
            
            lux_contribution = ray_flux / cell_areas_m2[wall_name]
            local_grids[wall_name][grid_i, grid_j] += lux_contribution
            local_ray_hits[wall_name] += 1
            
            # Check if we should generate a reflected bounce
            if bounce >= max_bounces or wall_reflectance <= 0:
                break
            ray_flux *= wall_reflectance
            if ray_flux < 1e-8:
                break
            hit_point = ray_origin + ray_dir * t_min
            normal = _WALL_INWARD_NORMALS[wall_name]
            ray_origin = hit_point + normal * 0.01
            ray_dir = _sample_cosine_hemisphere(normal)
    
    return local_grids, local_ray_hits, local_total_rays


def main():
    # Create Viser server — bind to 0.0.0.0 so other computers on the LAN can connect
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    
    # Allow users to collapse/hide the control panel
    server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
    
    # Get local IP for LAN access
    import socket as _socket
    _local_ip = "127.0.0.1"
    try:
        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _local_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        pass
    
    # Try to add Windows Firewall rule for LAN access (requires admin, fails silently)
    try:
        import subprocess as _sp
        _sp.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=LightingSim', 'dir=in', 'action=allow',
             'protocol=TCP', 'localport=8080'],
            capture_output=True, timeout=5
        )
    except Exception:
        pass
    
    # Auto-open browser
    import webbrowser as _wb
    _wb.open("http://localhost:8080")
    
    print(f"\n  🖥️  Local:   http://localhost:8080")
    print(f"  🌐  Network: http://{_local_ip}:8080")
    print(f"\n  ⚠️  Se la connessione da rete non funziona, eseguire come Amministratore")
    print(f"      oppure aprire manualmente porta 8080 nel Windows Firewall.")
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
    
    # Panel Configurator state (populated later when UI is created)
    _panel_slot_data = [None, None, None, None]
    _panel_dropdowns = []
    
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
            # Save lumens override settings for custom group
            group_cfg['lumens_override_enabled'] = group.get('lumens_override') and group['lumens_override'].value
            group_cfg['lumens_value'] = group['lumens_value'].value if group.get('lumens_value') else 100
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
        # Reset panel configurator slots
        for si in range(len(_panel_slot_data)):
            _panel_slot_data[si] = None
        # Reset panel configurator dropdowns (if already created)
        for dd in _panel_dropdowns:
            try:
                dd.value = "-- Nessuno --"
            except Exception:
                pass
    
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
            if group_cfg.get('panel_slot') is not None:
                if 'rot_tilt_lr' in group_data:
                    group_data['rot_tilt_lr'].value = int(round(group_cfg.get('rotation_z', 0)))
                if 'rot_tilt_ud' in group_data:
                    group_data['rot_tilt_ud'].value = int(round(group_cfg.get('rotation_y', 0)))
                if 'rot_roll' in group_data:
                    group_data['rot_roll'].value = int(round(group_cfg.get('rotation_x', 0)))
                # Preserve panel slot metadata
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
                        grp_rot_lr = server.gui.add_slider("Sinistra/Destra (\u00b0)", min=-180, max=180, step=1, initial_value=group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0)
                        grp_rot_ud = server.gui.add_slider("Alto/Basso (\u00b0)", min=-180, max=180, step=1, initial_value=group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0)
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
                    grp_rot_lr = server.gui.add_slider("Sinistra/Destra (°)", min=-180, max=180, step=1, initial_value=group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0)
                    grp_rot_ud = server.gui.add_slider("Alto/Basso (°)", min=-180, max=180, step=1, initial_value=group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0)
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
                            group_cfg['led_euler_angles'] = group.get('led_euler_angles', [])
                            group_cfg['led_beam_tilts'] = group.get('led_beam_tilts', [])
                        # Save lumens override for template
                        group_cfg['lumens_override_enabled'] = group.get('lumens_override') and group['lumens_override'].value
                        group_cfg['lumens_value'] = group['lumens_value'].value if group.get('lumens_value') else 100
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
                            'viewing_angle': led['viewing_angle'].value,
                            'square_roll': led['square_roll'].value,
                            'beam_tilt': led['beam_tilt'].value,
                            'lumens_override_enabled': led.get('lumens_override') and led['lumens_override'].value,
                            'lumens_value': led['lumens_value'].value if led.get('lumens_value') else 100,
                        })
                    
                    # Save as template with separate groups
                    save_custom_group_template(name, custom_groups_data, individual_leds_data)
                    # Refresh template dropdown list
                    _fresh = get_available_templates()
                    template_dropdown.options = ["Empty"] + _fresh
                    for _pdd in _panel_dropdowns:
                        _cur = _pdd.value
                        _pdd.options = ["-- Nessuno --"] + _fresh
                        _pdd.value = _cur if _cur in _pdd.options else "-- Nessuno --"
                    
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
            "Wall distance (cm)", min=10, max=1500, step=5, initial_value=50
        )
        circle_center_slider = server.gui.add_slider(
            "Circle center X (cm)", min=-60, max=0, step=5, initial_value=-35
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Global Rotation:</b>")
        global_rotation_z_slider = server.gui.add_slider(
            "Ruota configurazione (°)", min=-180, max=180, step=1, initial_value=0
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Global Position:</b>")
        global_pos_x_slider = server.gui.add_slider(
            "Global offset X (cm)", min=-100, max=100, step=0.5, initial_value=0
        )
        global_pos_y_slider = server.gui.add_slider(
            "Global offset Y (cm)", min=-100, max=100, step=0.5, initial_value=0
        )
        global_pos_z_slider = server.gui.add_slider(
            "Global offset Z (cm)", min=-100, max=100, step=0.5, initial_value=0
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
            "Rays per pixel (↑quality, ↓speed)", min=10, max=500000, step=10, initial_value=10500
        )
        ray_uniformity_slider = server.gui.add_slider(
            "Focus factor (0=Standard, 1=3x focused)", min=0.0, max=1.0, step=0.05, initial_value=0.0
        )
        led_lumens_slider = server.gui.add_slider(
            "LED lumens (lm/LED)", min=10, max=1000, step=10, initial_value=168
        )
        
        server.gui.add_html("<hr style='margin:8px 0;'><b>Diffuser Lens:</b>")
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-bottom:4px;'>"
            "Simula una lente diffusore davanti ai LED: allarga il fascio e lo rende più uniforme. "
            "Trasmittanza tipica 85-95%.</div>"
        )
        diffuser_enable_chk = server.gui.add_checkbox(
            "Enable diffuser lens", initial_value=False
        )
        diffuser_angle_slider = server.gui.add_slider(
            "Diffuser output angle (°)", min=60, max=180, step=5, initial_value=170
        )
        diffuser_transmission_slider = server.gui.add_slider(
            "Diffuser transmission (%)", min=50, max=100, step=1, initial_value=90
        )
        calibration_factor_slider = server.gui.add_slider(
            "Calibration factor", min=0.5, max=1.5, step=0.001, initial_value=1.0
        )
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=5, max=1000 , step=5, initial_value=30
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
        bw_scale_chk = server.gui.add_checkbox("B/W Scale", initial_value=False)
        legend_max_input = server.gui.add_number("Legend max (lux)", initial_value=3500, min=1, max=100000, step=50)
        server.gui.add_html("<div style='color:#888;font-size:10px;margin-top:-4px;'>Fixed cap: colors scale 0–this value. If peak exceeds it, switches to AUTO.</div>")
        wall_view_size = server.gui.add_slider("Wall view size (cm)", min=100, max=2000, step=10, initial_value=100)
        # Manual update button for intensity map (computationally expensive)
        update_intensity_button = server.gui.add_button("Update Intensity Map")
        export_lux_matrix_button = server.gui.add_button("Export Lux Matrix (±40cm)")
        run_benchmark_button = server.gui.add_button("Run Benchmark (multi-distance)")

    # --- CSV Pattern Import (initially collapsed) ---
    with server.gui.add_folder("📊 Import CSV Pattern", expand_by_default=False):
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-bottom:6px;'>"
            "Import a benchmark or FOV intensity CSV and overlay the measured pattern on the wall."
            "</div>"
        )
        csv_import_path = server.gui.add_text("CSV File Path", initial_value="")
        csv_import_btn = server.gui.add_button("📂 Import CSV", color="#4CAF50")
        csv_clear_btn = server.gui.add_button("🗑️ Clear Imported Pattern", color="#FF5555")
        csv_import_status = server.gui.add_html("<div style='font-size:11px;color:#888;'>No file imported</div>")
        csv_legend_max_input = server.gui.add_number("CSV Legend max (lux)", initial_value=523, min=1, max=100000, step=10)
        server.gui.add_html("<div style='color:#888;font-size:10px;margin-top:-4px;'>Fixed cap for CSV pattern legend. AUTO if peak exceeds it.</div>")
        csv_legend_html = server.gui.add_html("")
        csv_diff_html = server.gui.add_html("")

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

    # ---------- Fast pure-numpy rotation matrices (avoid trimesh overhead) ----------
    def _rot4_x(angle_rad):
        """4x4 rotation matrix around X axis (pure numpy, ~50x faster than trimesh)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)

    def _rot4_y(angle_rad):
        """4x4 rotation matrix around Y axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)

    def _rot4_z(angle_rad):
        """4x4 rotation matrix around Z axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    # -------------------------------------------------------------------------------

    def _build_stl_transform(stl_scale_ctrl, stl_rot_x_ctrl, stl_rot_y_ctrl, stl_rot_z_ctrl,
                              stl_pos_x_ctrl, stl_pos_y_ctrl, stl_pos_z_ctrl):
        """Build 4x4 transform matrix from STL GUI controls. Shared by all ray tracing paths."""
        transform = np.eye(4)
        scale = float(stl_scale_ctrl.value)
        if np.isfinite(scale) and scale > 0:
            transform[:3, :3] *= scale
        rx = float(stl_rot_x_ctrl.value) if np.isfinite(float(stl_rot_x_ctrl.value)) else 0.0
        ry = float(stl_rot_y_ctrl.value) if np.isfinite(float(stl_rot_y_ctrl.value)) else 0.0
        rz = float(stl_rot_z_ctrl.value) if np.isfinite(float(stl_rot_z_ctrl.value)) else 0.0
        if rx != 0:
            transform = _rot4_x(np.radians(rx)) @ transform
        if ry != 0:
            transform = _rot4_y(np.radians(ry)) @ transform
        if rz != 0:
            transform = _rot4_z(np.radians(rz)) @ transform
        px = float(stl_pos_x_ctrl.value) if np.isfinite(float(stl_pos_x_ctrl.value)) else 0.0
        py = float(stl_pos_y_ctrl.value) if np.isfinite(float(stl_pos_y_ctrl.value)) else 0.0
        pz = float(stl_pos_z_ctrl.value) if np.isfinite(float(stl_pos_z_ctrl.value)) else 0.0
        transform[:3, 3] = [px, py, pz]
        return transform

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
        
        def _get_stl_cache_path(file_path):
            """Get cache file path for an STL file based on its content hash."""
            # Use file path + modification time as cache key for speed
            stat = os.stat(file_path)
            cache_key = f"{file_path}|{stat.st_size}|{stat.st_mtime_ns}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(os.path.dirname(file_path), '.stl_cache')
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"{cache_hash}.npz")

        def load_stl_file():
            """Load STL file and display in scene. Uses numpy binary cache for fast reloads."""
            file_path = stl_file_path.value.strip()
            if not file_path:
                print("⚠️ Please enter a file path")
                return
            
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return
            
            try:
                t_start = time.perf_counter()
                cache_path = _get_stl_cache_path(file_path)
                
                # Try loading from numpy binary cache first (10-50x faster)
                if os.path.exists(cache_path):
                    print(f"Loading STL from cache: {os.path.basename(file_path)}")
                    cached = np.load(cache_path)
                    mesh = trimesh.Trimesh(
                        vertices=cached['vertices'],
                        faces=cached['faces'],
                        vertex_normals=cached['vertex_normals'],
                        process=False  # Skip expensive validation since we know data is good
                    )
                    t_load = time.perf_counter()
                    print(f"  Cache loaded in {t_load - t_start:.2f}s")
                else:
                    print(f"Loading STL file: {file_path} (first load, will cache)")
                    mesh = trimesh.load(file_path, force='mesh', process=False)
                    
                    # Handle multiple meshes (Scene object)
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(
                            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                        )
                    
                    # Center mesh at origin (move centroid to 0,0,0)
                    mesh.vertices -= mesh.centroid
                    
                    t_load = time.perf_counter()
                    print(f"  STL parsed in {t_load - t_start:.2f}s")
                    
                    # Save to numpy binary cache for fast future loads
                    # np.savez (uncompressed) is much faster than np.savez_compressed
                    try:
                        np.savez(
                            cache_path,
                            vertices=mesh.vertices.astype(np.float32),
                            faces=mesh.faces,
                            vertex_normals=mesh.vertex_normals.astype(np.float32)
                        )
                        print(f"  Cache saved: {os.path.basename(cache_path)}")
                    except Exception as ce:
                        print(f"  Warning: could not save cache: {ce}")
                
                stl_mesh_data[0] = mesh
                
                # Calculate mesh dimensions
                num_vertices = len(mesh.vertices)
                num_faces = len(mesh.faces)
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                
                # Auto-calculate ideal scale
                target_size_cm = 70.0
                max_dimension = np.max(size)
                ideal_scale = 1.0
                
                if max_dimension > 0:
                    ideal_scale = target_size_cm / max_dimension
                    stl_scale.value = ideal_scale
                
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
                
                t_total = time.perf_counter() - t_start
                print(f"✓ STL loaded: {num_vertices:,} vertices, {num_faces:,} faces ({t_total:.2f}s total)")
                update_stl_mesh(skip_lighting=True)
                
            except Exception as e:
                print(f"❌ Error loading STL: {e}")
                import traceback; traceback.print_exc()
                stl_info_html.content = f"<div style='color:#FF5555;'>Error: {str(e)}</div>"
        
        def update_stl_mesh(skip_lighting=False):
            """Update STL mesh visualization in scene.
            
            Args:
                skip_lighting: If True, use uniform base color instead of computing
                    per-vertex lighting. Much faster for initial load.
            """
            nonlocal stl_mesh_handle
            
            try:
                t_start = time.perf_counter()
                
                # Remove existing mesh
                if stl_mesh_handle[0] is not None:
                    try:
                        stl_mesh_handle[0].remove()
                    except:
                        pass
                    stl_mesh_handle[0] = None
                
                # Add mesh if loaded and visible
                if stl_mesh_data[0] is not None and stl_visible.value:
                    orig_mesh = stl_mesh_data[0]
                    
                    # Validate values (protect against NaN)
                    scale = float(stl_scale.value)
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    
                    pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
                    pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
                    pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
                    rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
                    rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0
                    rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
                    opacity = float(stl_opacity.value) if np.isfinite(float(stl_opacity.value)) else 0.8
                    
                    # Update info display
                    bounds = orig_mesh.bounds
                    orig_size = bounds[1] - bounds[0]
                    scaled_size = orig_size * scale
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
                    
                    # --- Build combined 4x4 transform (no mesh.copy() needed) ---
                    orig_vertices = orig_mesh.vertices  # Direct reference, no copy
                    orig_normals = orig_mesh.vertex_normals
                    orig_faces = orig_mesh.faces
                    
                    # Build rotation matrix using pure numpy helpers (much faster than trimesh)
                    T_rot = np.eye(4)
                    if rot_x != 0:
                        T_rot = T_rot @ _rot4_x(np.radians(rot_x))
                    if rot_y != 0:
                        T_rot = T_rot @ _rot4_y(np.radians(rot_y))
                    if rot_z != 0:
                        T_rot = T_rot @ _rot4_z(np.radians(rot_z))
                    
                    # Apply global Z rotation on top of STL-local rotation
                    g_rot_deg = global_rotation_z_slider.value
                    if abs(g_rot_deg) > 0.01:
                        T_rot = _rot4_z(np.radians(g_rot_deg)) @ T_rot

                    # Combined transform: Scale -> Rotate -> Translate -> cm-to-meters
                    # Instead of building 4 separate matrices and multiplying, build directly
                    R = T_rot[:3, :3]
                    scale_m = scale * 0.01  # scale * cm_to_meters
                    
                    # Transform vertices: v' = (R * scale_m) @ v + translate_m
                    RS = R * scale_m  # 3x3 scaled rotation
                    # Also rotate the translation vector by global rotation
                    translate_cm = np.array([pos_x, pos_y, pos_z])
                    if abs(g_rot_deg) > 0.01:
                        cg2, sg2 = np.cos(np.radians(g_rot_deg)), np.sin(np.radians(g_rot_deg))
                        translate_cm = np.array([cg2*pos_x - sg2*pos_y, sg2*pos_x + cg2*pos_y, pos_z])
                    translate_m = translate_cm * 0.01  # cm to meters
                    
                    vertices_transformed = (orig_vertices @ RS.T) + translate_m
                    vertices_transformed = vertices_transformed.astype(np.float32)
                    
                    # --- Compute vertex colors & render ---
                    if skip_lighting:
                        # FAST PATH: use add_mesh_simple (no Trimesh/GLB overhead, no normal transform)
                        stl_mesh_handle[0] = server.scene.add_mesh_simple(
                            name="/stl_model",
                            vertices=vertices_transformed,
                            faces=orig_faces.astype(np.uint32),
                            color=(0, 0, 51),  # dark blue (0.0, 0.0, 0.2) * 255
                            opacity=opacity if opacity < 1.0 else None,
                            flat_shading=False,
                            side="double",
                            wireframe=stl_wireframe.value,
                            visible=True,
                        )
                    else:
                        # Transform normals (rotation only, then normalize) — only needed for lighting
                        normals_transformed = orig_normals @ R.T
                        norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
                        norms[norms < 1e-10] = 1.0
                        normals_transformed = (normals_transformed / norms).astype(np.float32)

                        # Full lighting calculation — need per-vertex colors → Trimesh path
                        vertex_colors = calculate_mesh_lighting(
                            vertices_transformed, 
                            normals_transformed, 
                            current_leds,
                            base_color=(0.7, 0.7, 0.9),
                            led_lumens=led_lumens_slider.value
                        )
                        vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)
                    
                        # Add alpha channel
                        alpha_value = int(opacity * 255)
                        vertex_colors_rgba = np.concatenate([
                            vertex_colors_uint8,
                            np.full((num_vertices, 1), alpha_value, dtype=np.uint8)
                        ], axis=1)
                    
                        # Build trimesh with transformed data (process=False skips expensive validation)
                        mesh_transformed = trimesh.Trimesh(
                            vertices=vertices_transformed,
                            faces=orig_faces,
                            vertex_normals=normals_transformed,
                            process=False
                        )
                        from trimesh.visual import ColorVisuals
                        mesh_transformed.visual = ColorVisuals(mesh=mesh_transformed, vertex_colors=vertex_colors_rgba)
                    
                        # Add to scene
                        stl_mesh_handle[0] = server.scene.add_mesh_trimesh(
                            name="/stl_model",
                            mesh=mesh_transformed,
                            visible=True,
                        )
                    
                    t_total = time.perf_counter() - t_start
                    if t_total > 0.5:
                        print(f"  Mesh update: {t_total:.2f}s {'(no lighting)' if skip_lighting else '(with lighting)'}")
            except Exception as e:
                print(f"Error updating STL mesh: {e}")
                import traceback; traceback.print_exc()
        
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
                    print("Calculating mesh lighting...")
                    update_stl_mesh(skip_lighting=False)
                    print("✓ Mesh lighting updated")
                else:
                    print("⚠️ No mesh loaded")
            except Exception as e:
                print(f"Error updating mesh lighting: {e}")
        
        # Update mesh when parameters change (geometry only, no lighting)
        def safe_update_stl(_):
            try:
                update_stl_mesh(skip_lighting=True)
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
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Wall Reflections</div>")
        reflections_enable = server.gui.add_checkbox("Enable Reflections", initial_value=False)
        wall_material_dropdown = server.gui.add_dropdown(
            "Wall Material",
            options=["White Paint (\u03c1=0.85)", "Light Gray (\u03c1=0.65)", "Concrete (\u03c1=0.30)",
                     "Wood (\u03c1=0.45)", "Brick (\u03c1=0.25)", "Dark Paint (\u03c1=0.15)", "Custom"],
            initial_value="White Paint (\u03c1=0.85)"
        )
        custom_reflectance_slider = server.gui.add_slider(
            "Reflectance (\u03c1)", min=0.0, max=0.99, step=0.01, initial_value=0.85
        )
        max_bounces_slider_room = server.gui.add_slider(
            "Max Bounces", min=1, max=10, step=1, initial_value=3
        )
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-top:-4px;'>More bounces = more accurate but slower.<br>"
            "Reflected flux per bounce: \u03a6 \u00d7 \u03c1<sup>n</sup></div>"
        )
        _MATERIAL_REFLECTANCE = {
            "White Paint (\u03c1=0.85)": 0.85,
            "Light Gray (\u03c1=0.65)": 0.65,
            "Concrete (\u03c1=0.30)": 0.30,
            "Wood (\u03c1=0.45)": 0.45,
            "Brick (\u03c1=0.25)": 0.25,
            "Dark Paint (\u03c1=0.15)": 0.15,
        }
        @wall_material_dropdown.on_update
        def _on_material_change(_):
            mat = wall_material_dropdown.value
            if mat in _MATERIAL_REFLECTANCE:
                custom_reflectance_slider.value = _MATERIAL_REFLECTANCE[mat]

    # Camera FOV visualization
    with server.gui.add_folder("Camera FOV"):
        show_camera_fov = server.gui.add_checkbox("Show Camera FOV", initial_value=True)
        camera_fov_h = server.gui.add_slider(
            "Horizontal FOV (°)", min=10, max=120, step=1, initial_value=75
        )
        camera_fov_v = server.gui.add_slider(
            "Vertical FOV (°)", min=10, max=120, step=1, initial_value=60
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
    imported_csv_handles = []
    
    # Store current LED objects (for reuse in room intensity calculation)
    current_leds = []

    def calculate_mesh_lighting(mesh_vertices, mesh_normals, leds, base_color=(0.7, 0.7, 0.9), led_lumens=100):
        """
        Calculate per-vertex lighting for STL mesh using physical lux calculation.
        Vectorized: processes all LEDs in a single batched computation.
        
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
        
        # Get only enabled LEDs
        active_leds = [led for led in leds if getattr(led, 'enabled', True)]
        
        if len(active_leds) == 0:
            # Just return ambient
            ambient = 0.15
            base = np.array(base_color, dtype=np.float32) * ambient
            return np.tile(base, (num_vertices, 1))
        
        num_leds = len(active_leds)
        
        # Pre-extract LED data into contiguous arrays for vectorized computation
        led_positions = np.array([led.position for led in active_leds], dtype=np.float32) / 100.0  # cm -> m, shape (L, 3)
        led_directions = np.array([led.direction for led in active_leds], dtype=np.float32)  # (L, 3)
        led_half_angles = np.array([np.radians(led.viewing_angle / 2.0) for led in active_leds], dtype=np.float32)  # (L,)
        
        # Per-LED lumens: use override if available, else global
        per_led_lumens = np.array([
            float(getattr(led, 'lumens', None) or led_lumens)
            for led in active_leds
        ], dtype=np.float32)
        
        # Normalize LED directions
        led_dir_norms = np.linalg.norm(led_directions, axis=1, keepdims=True) + 1e-10
        led_directions = led_directions / led_dir_norms  # (L, 3)
        
        # Calculate luminous intensity per LED (candelas)
        solid_angles = 2 * np.pi * (1 - np.cos(led_half_angles))  # (L,)
        solid_angles = np.maximum(solid_angles, 0.001)
        luminous_intensities = per_led_lumens / solid_angles  # (L,)
        cos_half_angles = np.cos(led_half_angles)  # (L,)
        
        # Filter out LEDs with invalid intensity
        valid = np.isfinite(luminous_intensities) & (luminous_intensities > 0)
        if not np.any(valid):
            ambient = 0.15
            base = np.array(base_color, dtype=np.float32) * ambient
            return np.tile(base, (num_vertices, 1))
        
        led_positions = led_positions[valid]
        led_directions = led_directions[valid]
        luminous_intensities = luminous_intensities[valid]
        cos_half_angles = cos_half_angles[valid]
        num_leds = len(led_positions)
        
        # --- Batched computation: all LEDs x all vertices ---
        # Process in chunks to limit memory (L * N * 3 can be large)
        chunk_size = max(1, min(num_leds, 50_000_000 // max(num_vertices, 1)))  # ~200MB limit
        
        total_illuminance = np.zeros(num_vertices, dtype=np.float32)
        
        for led_start in range(0, num_leds, chunk_size):
            led_end = min(led_start + chunk_size, num_leds)
            L = led_end - led_start
            
            # to_vertex[l, v, :] = mesh_vertices[v] - led_positions[l]
            # Shape: (L, N, 3) 
            to_vertex = mesh_vertices[np.newaxis, :, :] - led_positions[led_start:led_end, np.newaxis, :]  # (L, N, 3)
            distances = np.linalg.norm(to_vertex, axis=2) + 1e-6  # (L, N)
            to_vertex_norm = to_vertex / distances[:, :, np.newaxis]  # (L, N, 3)
            
            # Cone check: cos(angle) between LED direction and to_vertex
            cos_angle = np.einsum('ld,lnd->ln', led_directions[led_start:led_end], to_vertex_norm)  # (L, N)
            in_cone = cos_angle > cos_half_angles[led_start:led_end, np.newaxis]  # (L, N)
            
            # Lambert's cosine: angle between surface normal and incoming light
            cos_incident = np.einsum('nd,lnd->ln', mesh_normals, -to_vertex_norm)  # (L, N)
            cos_incident = np.maximum(0, cos_incident)
            
            # Illuminance = I * cos_incident / d^2 (only in cone)
            illuminance = luminous_intensities[led_start:led_end, np.newaxis] * cos_incident / (distances ** 2)  # (L, N)
            
            # Cone edge falloff
            denom = 1.0 - cos_half_angles[led_start:led_end, np.newaxis]
            denom = np.maximum(denom, 1e-10)
            angle_factor = np.maximum(0, (cos_angle - cos_half_angles[led_start:led_end, np.newaxis]) / denom) ** 2
            illuminance *= angle_factor
            
            # Zero out contributions outside cone
            illuminance *= in_cone.astype(np.float32)
            
            # Sum across LEDs in this chunk
            total_illuminance += np.nansum(illuminance, axis=0)
        
        # Convert lux to color intensity
        lux_to_color_scale = 0.002
        intensity_normalized = np.clip(total_illuminance * lux_to_color_scale, 0.0, 1.0)
        
        # Map to blue-to-white gradient
        blue_base = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        t = intensity_normalized[:, np.newaxis]  # (N, 1)
        final_colors = blue_base * (1.0 - t) + white * t
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
                _fresh = get_available_templates()
                template_dropdown.options = ["Empty"] + _fresh
                for _pdd in _panel_dropdowns:
                    _cur = _pdd.value
                    _pdd.options = ["-- Nessuno --"] + _fresh
                    _pdd.value = _cur if _cur in _pdd.options else "-- Nessuno --"

    # =====================================================================
    #  PANEL CONFIGURATOR  – 4 Elios 3 slots with per-slot template choice
    # =====================================================================
    # Pre-computed Elios3 slot data (centroid + outward Z-rotation angle)
    _ELIOS3_SLOTS = [
        {"name": "Front +",  "centroid": [18.06, -7.92, -1.59], "angle_deg": -23.7, "config_rot": 0.7,  "config_roll":  0,  "config_lr":  10},
        {"name": "Front -",  "centroid": [18.03,  7.97, -1.59], "angle_deg":  23.8, "config_rot": -0.7, "config_roll": -3,  "config_lr":  -6},
        {"name": "Side -",   "centroid": [16.07, 12.57, -0.83], "angle_deg":  38.0, "config_rot": -18,  "config_roll":  0,  "config_lr":  21},
        {"name": "Side +",   "centroid": [16.07, -12.63, -0.82], "angle_deg": -38.2, "config_rot": 18,  "config_roll":  0,  "config_lr": -21},
    ]

    def _Rz_matrix(deg):
        """Build a 3×3 rotation matrix around Z axis."""
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

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
        slot_folder = server.gui.add_folder(f"Slot {slot_label}: {template_display} ({mode_label})")

        with slot_folder:
            slot_enable = server.gui.add_checkbox("Enable Slot", initial_value=True)
            slot_pos_x = server.gui.add_slider("Offset X (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_pos_y = server.gui.add_slider("Offset Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_pos_z = server.gui.add_slider("Offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
            slot_rot_x = server.gui.add_slider("Ruota su se stesso (°)", min=-180, max=180, step=1, initial_value=0)
            slot_rot_y = server.gui.add_slider("Inclina Alto/Basso (°)", min=-180, max=180, step=1, initial_value=0)
            slot_rot_z = server.gui.add_slider("Inclina Sinistra/Destra (°)", min=-180, max=180, step=1, initial_value=0)
            server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>")
            slot_lumens_chk = server.gui.add_checkbox("Enable custom lumens", initial_value=False)
            slot_lumens_slider = server.gui.add_slider("Lumens per LED (lm)", min=1, max=900000, step=1, initial_value=100)
            slot_remove_btn = server.gui.add_button("Rimuovi Slot", color="red")

        created_groups = []
        created_individual_leds = []

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
            # slot_rot_x → rot_roll (Ruota su se stesso, around panel forward axis)
            # slot_rot_y → rot_tilt_ud (Inclina Alto/Basso)
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
        }

        # Also register in template_folders for cleanup on new project
        template_folders.append({
            'folder': slot_folder,
            'groups': created_groups,
        })

        loading_in_progress[0] = False
        update_scene()
        print(f"✓ Panel Configurator: Loaded '{template_display}' into slot {slot_label} ({mode_label})")

    def _clear_panel_slot(slot_idx):
        """Remove all content from a panel slot."""
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

    # --- Panel Configurator UI ---
    panel_config_folder = server.gui.add_folder("Panel Configurator (Elios 3 Slots)", expand_by_default=False)

    _panel_dropdowns = []
    _panel_mode_dropdowns = []
    _panel_load_btns = []
    _panel_clear_btns = []

    with panel_config_folder:
        server.gui.add_html(
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>"
            "Assign a template to each of the 4 Elios 3 panel positions.<br>"
            "Positions and rotations are pre-set to match the drone geometry.</div>"
        )

        for _si, _slot_info in enumerate(_ELIOS3_SLOTS):
            with server.gui.add_folder(f"📍 {_slot_info['name']}"):
                _tpl_dd = server.gui.add_dropdown(
                    "Template",
                    options=["-- Nessuno --"] + get_available_templates(),
                    initial_value="-- Nessuno --",
                )
                _mode_dd = server.gui.add_dropdown(
                    "Mode",
                    options=["Solid (Group)", "Individual LEDs"],
                    initial_value="Solid (Group)",
                )
                _load_btn = server.gui.add_button("✅ Carica Pannello", color="green")
                _clear_btn = server.gui.add_button("🗑️ Rimuovi Pannello", color="red")

                _panel_dropdowns.append(_tpl_dd)
                _panel_mode_dropdowns.append(_mode_dd)
                _panel_load_btns.append(_load_btn)
                _panel_clear_btns.append(_clear_btn)

                def _make_load_handler(si, dd, mdd):
                    def handler(_):
                        tpl = dd.value
                        if tpl == "-- Nessuno --":
                            _clear_panel_slot(si)
                            update_scene()
                            print(f"Panel Configurator: Slot {_ELIOS3_SLOTS[si]['name']} cleared.")
                            return
                        as_individual = mdd.value == "Individual LEDs"
                        _load_template_into_slot(si, tpl, as_individual)
                    return handler

                def _make_clear_handler(si, dd):
                    def handler(_):
                        _clear_panel_slot(si)
                        dd.value = "-- Nessuno --"
                        update_scene()
                        print(f"Panel Configurator: Slot {_ELIOS3_SLOTS[si]['name']} cleared.")
                    return handler

                _load_btn.on_click(_make_load_handler(_si, _tpl_dd, _mode_dd))
                _clear_btn.on_click(_make_clear_handler(_si, _tpl_dd))

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
                "square_roll": float(led_data['square_roll'].value),
                "beam_tilt": float(led_data['beam_tilt'].value)
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
    
    def export_leds_to_stl():
        """Export each LED as an editable planar surface in STEP format.
        
        One face per LED (rectangle with filleted corners). The output is a
        true B-Rep STEP file with planar faces, fully editable in SolidWorks
        (selectable as reference plane, extrudable, etc.).
        """
        if len(current_leds) == 0:
            print("⚠️ No LEDs in the scene. Update the scene first.")
            return None
        
        try:
            import cadquery as cq
        except ImportError:
            print("⚠️ 'cadquery' library required for STEP export.  pip install cadquery")
            return None
        try:
            from shapely.geometry import Polygon
        except ImportError:
            print("⚠️ 'shapely' library required.  pip install shapely")
            return None
        
        active_leds = [led for led in current_leds
                       if not (hasattr(led, 'enabled') and not led.enabled)]
        if not active_leds:
            print("⚠️ No active LEDs to export.")
            return None
        
        # ── Parameters (cm) ──
        margin   = 0.05   # 0.5 mm border around each LED
        fillet_r = 0.04   # 0.4 mm fillet on outer corners
        faces    = []
        
        def _normal(led):
            n = np.array(getattr(led, 'square_normal', led.direction), dtype=float)
            nm = np.linalg.norm(n)
            return n / nm if nm > 1e-10 else np.array([1., 0., 0.])
        
        for led in active_leds:
            pos = np.array(led.position, dtype=float)
            nrm = _normal(led)
            hw  = led.width / 2.0
            
            # Local 2-D frame on the LED's plane
            if abs(nrm[2]) < 0.9:
                lx = np.cross(nrm, [0, 0, 1])
            else:
                lx = np.cross(nrm, [0, 1, 0])
            lx /= np.linalg.norm(lx)
            ly = np.cross(nrm, lx)
            ly /= np.linalg.norm(ly)
            
            # Use row_direction for consistent orientation
            row_d = getattr(led, 'row_direction', None)
            if row_d is not None:
                row_d = np.array(row_d, dtype=float)
                r2x = np.dot(row_d, lx)
                r2y = np.dot(row_d, ly)
                n2  = np.hypot(r2x, r2y)
                if n2 > 0.01:
                    r_hat = np.array([r2x, r2y]) / n2
                else:
                    r_hat = np.array([1., 0.])
            else:
                r_hat = np.array([1., 0.])
            p_hat = np.array([-r_hat[1], r_hat[0]])
            
            # ── Outer panel outline (LED square + margin) with filleted corners ──
            m = hw + margin
            outer_corners = [(r_hat[0]*sx*m + p_hat[0]*sy*m,
                              r_hat[1]*sx*m + p_hat[1]*sy*m)
                             for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
            outer = Polygon(outer_corners)
            try:
                sm = outer.buffer(-fillet_r, resolution=8).buffer(fillet_r, resolution=8)
                if sm.is_valid and not sm.is_empty and sm.area > outer.area * 0.5:
                    outer = sm
            except Exception:
                pass
            
            if outer.is_empty:
                continue
            
            polys = (list(outer.geoms)
                     if outer.geom_type == 'MultiPolygon'
                     else [outer])
            
            # Build a CadQuery Workplane on the LED's local plane.
            # Units: cadquery uses mm; our scene is in cm → multiply by 10.
            plane = cq.Plane(
                origin=cq.Vector(float(pos[0])*10, float(pos[1])*10, float(pos[2])*10),
                xDir=cq.Vector(float(lx[0]), float(lx[1]), float(lx[2])),
                normal=cq.Vector(float(nrm[0]), float(nrm[1]), float(nrm[2])),
            )
            
            for poly in polys:
                try:
                    coords = list(poly.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    pts_mm = [(float(x)*10, float(y)*10) for (x, y) in coords]
                    wire = (
                        cq.Workplane(plane)
                        .polyline(pts_mm)
                        .close()
                        .val()
                    )
                    face = cq.Face.makeFromWires(wire)
                    faces.append(face)
                except Exception as e:
                    print(f"   [skip face] {e}")
        
        if not faces:
            print("⚠️ No faces generated.")
            return None
        
        compound = cq.Compound.makeCompound(faces)
        
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"led_panel_{ts}.step"
        filepath = os.path.join(export_dir, filename)
        cq.exporters.export(compound, filepath, exportType='STEP')
        
        print(f"✓ Exported STEP: {filename}")
        print(f"  Planar faces: {len(faces)} (editable in SolidWorks)")
        print(f"  Margin: {margin*10:.1f} mm  Fillet: {fillet_r*10:.1f} mm  Units: mm")
        print(f"  Path: {os.path.abspath(filepath)}")
        return filepath
    
    def export_custom_group_dxf():
        """Export a 2D DXF file for CNC cutting of the custom group LEDs.
        
        Projects all custom-group LEDs onto their unfolded flat plane,
        groups them into rows by Y-coordinate clustering, then places
        horizontal living-hinge slot patterns between rows whose normals
        differ so the flat panel can be bent into the 3-D shape.
        
        Units in the DXF are millimetres.
        
        Layers:
          PANEL_OUTLINE  – outer contour (white)
          LED_HOLES      – square LED apertures (red)
          FLEX_CUTS      – living-hinge slots between rows (green)
        """
        try:
            import ezdxf
        except ImportError:
            print("⚠️ 'ezdxf' library required.  pip install ezdxf")
            return None
        
        # Gather custom-group LEDs that are active
        custom_leds = [
            led for led in current_leds
            if getattr(led, 'is_custom', False)
            and not (hasattr(led, 'enabled') and not led.enabled)
        ]
        if not custom_leds:
            print("⚠️ No active custom-group LEDs in the scene.")
            return None
        
        # --- Helper: normalised normal vector ---
        def _led_normal(led):
            n = np.array(getattr(led, 'square_normal', led.direction), dtype=float)
            nm = np.linalg.norm(n)
            return n / nm if nm > 1e-10 else np.array([1., 0., 0.])
        
        # --- Compute local 2-D frame from the average LED normal ---
        positions_3d = np.array([led.position for led in custom_leds])
        normals_3d = np.array([_led_normal(led) for led in custom_leds])
        
        avg_normal = normals_3d.mean(axis=0)
        n_len = np.linalg.norm(avg_normal)
        avg_normal = avg_normal / n_len if n_len > 1e-10 else np.array([1., 0., 0.])
        
        centroid = positions_3d.mean(axis=0)
        
        # Orthonormal frame on the projection plane
        if abs(avg_normal[2]) < 0.9:
            x_local = np.cross(avg_normal, [0, 0, 1])
        else:
            x_local = np.cross(avg_normal, [0, 1, 0])
        x_local /= np.linalg.norm(x_local)
        y_local = np.cross(avg_normal, x_local)
        y_local /= np.linalg.norm(y_local)
        
        # --- Project each LED onto the 2-D plane (cm → mm) ---
        margin_mm      = 1.5   # margin around each LED hole
        panel_border_mm = 3.0  # extra border around the panel edges
        
        led_data = []  # [(cx_mm, cy_mm, hw_mm, normal_3d), ...]
        for idx, led in enumerate(custom_leds):
            delta = np.array(led.position) - centroid
            cx = np.dot(delta, x_local) * 10.0  # cm → mm
            cy = np.dot(delta, y_local) * 10.0
            hw = (led.width / 2.0) * 10.0
            led_data.append((cx, cy, hw, normals_3d[idx]))
        
        # --- Outer panel bounding rectangle ---
        all_x  = [d[0] for d in led_data]
        all_y  = [d[1] for d in led_data]
        max_hw = max(d[2] for d in led_data)
        border = max_hw + margin_mm + panel_border_mm
        
        x_min = min(all_x) - border
        x_max = max(all_x) + border
        y_min = min(all_y) - border
        y_max = max(all_y) + border
        
        # ================================================================
        #  Cluster LEDs into rows by Y coordinate, then add flex cuts
        #  between adjacent rows whose average normals differ
        # ================================================================
        # Sort LEDs by Y coordinate
        sorted_indices = sorted(range(len(led_data)), key=lambda i: led_data[i][1])
        
        # Cluster into rows: LEDs within cluster_tol mm of each other in Y
        cluster_tol = max_hw * 1.5  # LEDs in same row are close in Y
        rows = []  # list of lists of led_data indices
        current_row = [sorted_indices[0]]
        for k in range(1, len(sorted_indices)):
            prev_y = led_data[sorted_indices[k - 1]][1]
            curr_y = led_data[sorted_indices[k]][1]
            if abs(curr_y - prev_y) < cluster_tol:
                current_row.append(sorted_indices[k])
            else:
                rows.append(current_row)
                current_row = [sorted_indices[k]]
        rows.append(current_row)
        
        # Compute per-row average Y and average normal
        row_info = []  # (avg_y, avg_normal_3d, min_x, max_x)
        for row in rows:
            avg_y = np.mean([led_data[i][1] for i in row])
            avg_n = np.mean([led_data[i][3] for i in row], axis=0)
            nm = np.linalg.norm(avg_n)
            avg_n = avg_n / nm if nm > 1e-10 else np.array([1., 0., 0.])
            r_min_x = min(led_data[i][0] - led_data[i][2] for i in row)
            r_max_x = max(led_data[i][0] + led_data[i][2] for i in row)
            row_info.append((avg_y, avg_n, r_min_x, r_max_x))
        
        # --- Generate flex cuts between adjacent rows ---
        flex_angle_threshold_deg = 2.0
        slot_length_mm  = 4.0   # length of each slot segment
        slot_gap_mm     = 1.5   # gap between consecutive slots in a line
        n_slot_lines    = 3     # parallel lines of slots
        slot_line_gap   = 1.0   # spacing between parallel lines
        
        flex_cuts = []  # ((x1,y1),(x2,y2))
        
        for r in range(len(rows) - 1):
            # Angle between adjacent row normals
            dot = np.clip(np.dot(row_info[r][1], row_info[r + 1][1]), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(dot)))
            if angle_deg < flex_angle_threshold_deg:
                continue
            
            # Y zone: between the bottom of upper row and top of lower row
            # (rows sorted bottom to top, i.e. ascending Y)
            row_top_leds    = rows[r]
            row_bottom_leds = rows[r + 1]
            
            y_top_of_lower = max(led_data[i][1] + led_data[i][2] + margin_mm for i in row_top_leds)
            y_bot_of_upper = min(led_data[i][1] - led_data[i][2] - margin_mm for i in row_bottom_leds)
            
            zone_y_center = (y_top_of_lower + y_bot_of_upper) / 2.0
            zone_y_height = y_bot_of_upper - y_top_of_lower
            
            if zone_y_height < 1.5:
                # Not enough vertical space for flex cuts; place them anyway at midpoint
                zone_y_center = (row_info[r][0] + row_info[r + 1][0]) / 2.0
                zone_y_height = abs(row_info[r + 1][0] - row_info[r][0]) * 0.3
                if zone_y_height < 1.0:
                    continue
            
            # X extent of the flex zone = full panel width minus a small inset
            inset = panel_border_mm * 0.5
            zone_x_min = x_min + inset
            zone_x_max = x_max - inset
            zone_width = zone_x_max - zone_x_min
            if zone_width < slot_length_mm:
                continue
            
            # Place n_slot_lines parallel horizontal lines of staggered slots
            total_lines_span = (n_slot_lines - 1) * slot_line_gap
            
            for line_k in range(n_slot_lines):
                line_y = zone_y_center - total_lines_span / 2.0 + line_k * slot_line_gap
                
                # Stagger odd lines by half a stride
                stride = slot_length_mm + slot_gap_mm
                stagger = (stride / 2.0) if (line_k % 2 == 1) else 0.0
                
                x_pos = zone_x_min + stagger
                while x_pos + slot_length_mm <= zone_x_max:
                    x1 = x_pos
                    x2 = x_pos + slot_length_mm
                    flex_cuts.append(((x1, line_y), (x2, line_y)))
                    x_pos += stride
        
        # --- Build DXF ---
        doc = ezdxf.new(dxfversion='R2010')
        doc.units = ezdxf.units.MM
        msp = doc.modelspace()
        
        # Outer panel contour
        msp.add_lwpolyline(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
            close=True,
            dxfattribs={'layer': 'PANEL_OUTLINE', 'color': 7}
        )
        
        # LED holes (square)
        for cx, cy, hw, _ in led_data:
            msp.add_lwpolyline(
                [(cx - hw, cy - hw), (cx + hw, cy - hw),
                 (cx + hw, cy + hw), (cx - hw, cy + hw)],
                close=True,
                dxfattribs={'layer': 'LED_HOLES', 'color': 1}
            )
        
        # Flex cuts (horizontal living-hinge slots)
        for (x1, y1), (x2, y2) in flex_cuts:
            msp.add_line(
                (x1, y1), (x2, y2),
                dxfattribs={'layer': 'FLEX_CUTS', 'color': 3}
            )
        
        # --- Save ---
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cnc_panel_{ts}.dxf"
        filepath = os.path.join(export_dir, filename)
        doc.saveas(filepath)
        
        panel_w = x_max - x_min
        panel_h = y_max - y_min
        print(f"✓ Exported CNC DXF: {filename}")
        print(f"  Custom LEDs: {len(custom_leds)}  Rows detected: {len(rows)}")
        print(f"  Panel size: {panel_w:.1f} x {panel_h:.1f} mm")
        print(f"  Flex zones: {max(0, len(rows)-1)}  Slots: {len(flex_cuts)}")
        print(f"  Layers: PANEL_OUTLINE, LED_HOLES, FLEX_CUTS")
        print(f"  Path: {os.path.abspath(filepath)}")
        return filepath
    
    with individual_leds_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Add New Individual LED</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Add single LEDs with custom position, rotation, and size</div>")
        
        add_individual_led_btn = server.gui.add_button("➕ Add LED", color="cyan")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export Individual LEDs</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Save exact coordinates without transformations</div>")
        
        export_individual_leds_btn = server.gui.add_button("💾 Export to JSON", color="#4CAF50")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export Cover Panel (STEP)</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Superfici planari editabili in SolidWorks (selezionabili come piano, estrudibili). Una faccia per LED.</div>")
        
        export_stl_btn = server.gui.add_button("📦 Export Panel STEP", color="#FF9800")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export CNC Cutting (DXF)</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>File DXF 2D per taglio CNC del pannello custom. Apribile in AutoCAD, LibreCAD, ecc.</div>")
        
        export_cnc_btn = server.gui.add_button("🔩 Export CNC DXF", color="#2196F3")
        
        @add_individual_led_btn.on_click
        def _(_):
            create_individual_led()
            print("✓ Individual LED added")
        
        @export_individual_leds_btn.on_click
        def _(_):
            export_individual_leds_simple()
        
        @export_stl_btn.on_click
        def _(_):
            export_leds_to_stl()
        
        @export_cnc_btn.on_click
        def _(_):
            export_custom_group_dxf()


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
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
        
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
        
        # === GPU ACCELERATION PATH ===
        # Lazy-init: check GPU availability on first use (not at import)
        use_gpu = False
        if _HAS_GPU_MODULE:
            _ensure_gpu_init()
            from gpu_raytrace import GPU_AVAILABLE
            use_gpu = GPU_AVAILABLE
        
        if use_gpu:
            # Prepare LED data for GPU (shared between both paths)
            leds_data = []
            per_led_lumens_list = []
            for led, led_idx in active_leds:
                leds_data.append({
                    'position': np.array(led.position, dtype=np.float32),
                    'direction': np.array(led.direction, dtype=np.float32),
                    'viewing_angle': float(led.viewing_angle),
                    'ext_lens_angle': getattr(led, 'ext_lens_angle', None),
                    'led_idx': int(led_idx),
                })
                per_led_lumens_list.append(float(getattr(led, 'lumens', None) or lumens_per_led))
            gpu_params = {
                'wall_dist': wall_dist,
                'rays_per_led': rays_per_led_calculated,
                'grid_size': grid_size,
                'wall_size': wall_size,
                'lumens_per_led': lumens_per_led,
                'per_led_lumens': np.array(per_led_lumens_list, dtype=np.float32),
                'absorbers': absorbers if absorbers else [],
                'stl_mesh_data': None,  # GPU always traces without mesh
                'ray_uniformity': ray_uniformity,
            }
            
            if stl_mesh_data is not None:
                # === HYBRID GPU+CPU PATH ===
                # 1) GPU: compute full grid WITHOUT mesh absorber (fast)
                # 2) CPU: per-LED ray generation + mesh intersection (memory-efficient)
                # 3) Subtract absorbed lux from full grid
                print(f"[GPU+Mesh] Hybrid mode: GPU ray trace + CPU mesh subtraction for {len(active_leds)} LEDs...")
                import time as _time
                t0 = _time.perf_counter()
                
                grid_full = gpu_process_led_wall_batch(leds_data, gpu_params)
                t_gpu = _time.perf_counter()
                print(f"  GPU full grid: {t_gpu - t0:.2f}s")
                
                # Build BVH once for mesh intersection
                accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
                t_bvh = _time.perf_counter()
                print(f"  BVH built in {t_bvh - t_gpu:.2f}s")
                
                cell_size = wall_size / grid_size
                cell_area_m2 = (cell_size * cell_size) / 10000.0
                half_size = wall_size / 2
                absorbed_grid = np.zeros((grid_size, grid_size))
                total_candidate_rays = 0
                total_mesh_hits = 0
                
                # Process each LED sequentially (inner _batch_ray_mesh_intersection
                # already parallelizes across CPU cores via ThreadPoolExecutor)
                for led_i, (led, led_idx) in enumerate(active_leds):
                    t_led = _time.perf_counter()
                    led_lumens_val = float(getattr(led, 'lumens', None) or lumens_per_led)
                    np.random.seed((42 + led_idx) % (2**32))
                    
                    z_axis = led.direction
                    if abs(z_axis[2]) < 0.9:
                        x_axis = np.cross(z_axis, [0, 0, 1])
                    else:
                        x_axis = np.cross(z_axis, [0, 1, 0])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    
                    n = _get_effective_n(led, ray_uniformity)
                    max_theta = np.radians(led.viewing_angle / 2.0)
                    cos_max = np.cos(max_theta)
                    
                    u = np.random.uniform(0, 1, (rays_per_led_calculated, 2))
                    cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta = np.arccos(cos_theta)
                    phi = 2 * np.pi * u[:, 1]
                    
                    sin_theta = np.sin(theta)
                    local_dirs = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
                    world_dirs = (local_dirs[:, 0:1] * x_axis + local_dirs[:, 1:2] * y_axis + local_dirs[:, 2:3] * z_axis)
                    norms_d = np.linalg.norm(world_dirs, axis=1, keepdims=True)
                    world_dirs = world_dirs / norms_d
                    
                    cos_max_n1 = cos_max ** (n + 1.0)
                    denom = 1.0 - cos_max_n1
                    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
                    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
                    lumens_per_ray = (led_lumens_val / rays_per_led_calculated) * np.power(cos_theta_clamped, n) * norm_factor
                    del u, cos_theta, theta, phi, sin_theta, local_dirs, cos_theta_clamped
                    
                    towards_wall = world_dirs[:, 0] > 0
                    candidate_mask = towards_wall.copy()
                    if absorbers:
                        candidate_indices = np.where(candidate_mask)[0]
                        if len(candidate_indices) > 0:
                            origins_batch = np.broadcast_to(led.position, (len(candidate_indices), 3)).copy().astype(np.float32)
                            dirs_batch = world_dirs[candidate_indices].astype(np.float32)
                            absorbed = _ray_box_intersection_batch_np(origins_batch, dirs_batch, absorbers)
                            candidate_mask[candidate_indices[absorbed]] = False
                    
                    towards_indices = np.where(candidate_mask)[0]
                    if len(towards_indices) == 0:
                        continue
                    
                    n_cand = len(towards_indices)
                    total_candidate_rays += n_cand
                    
                    origins = np.tile(led.position, (n_cand, 1)).astype(np.float64)
                    cand_dirs = world_dirs[towards_indices].astype(np.float64)
                    cand_lumens = lumens_per_ray[towards_indices]
                    del world_dirs, lumens_per_ray
                    
                    mesh_hits = _batch_ray_mesh_intersection(origins, cand_dirs, accel)
                    n_hits = int(np.sum(mesh_hits))
                    total_mesh_hits += n_hits
                    
                    if n_hits > 0:
                        hit_dirs = cand_dirs[mesh_hits]
                        hit_lumens = cand_lumens[mesh_hits]
                        hit_origins = origins[mesh_hits]
                        t_wall = (wall_dist - led.position[0]) / hit_dirs[:, 0]
                        pos_t = t_wall > 0
                        if np.any(pos_t):
                            t_pos = t_wall[pos_t]
                            hit_y = hit_origins[pos_t, 1] + hit_dirs[pos_t, 1] * t_pos
                            hit_z = hit_origins[pos_t, 2] + hit_dirs[pos_t, 2] * t_pos
                            grid_y = ((hit_y + half_size) / cell_size).astype(int)
                            grid_z = ((hit_z + half_size) / cell_size).astype(int)
                            in_bounds = (grid_y >= 0) & (grid_y < grid_size) & (grid_z >= 0) & (grid_z < grid_size)
                            np.add.at(absorbed_grid, (grid_z[in_bounds], grid_y[in_bounds]),
                                      hit_lumens[pos_t][in_bounds] / cell_area_m2)
                    
                    del origins, cand_dirs, cand_lumens, mesh_hits
                    dt = _time.perf_counter() - t_led
                    if dt > 0.3:
                        print(f"    LED {led_i+1}/{len(active_leds)}: {n_cand:,} rays, {n_hits:,} hits, {dt:.2f}s")
                
                grid = np.maximum(0, grid_full - absorbed_grid)
                t_total = _time.perf_counter() - t0
                absorbed_pct = np.sum(absorbed_grid) / max(np.sum(grid_full), 1e-10) * 100
                print(f"  Mesh subtraction: {t_total - (t_gpu - t0):.2f}s "
                      f"({total_mesh_hits:,} hits / {total_candidate_rays:,} rays, {absorbed_pct:.1f}% absorbed)")
                print(f"  Total hybrid: {t_total:.2f}s\n")
            else:
                # === PURE GPU PATH (no mesh) ===
                print(f"[GPU] Using NVIDIA CUDA acceleration for {len(active_leds)} LEDs...")
                grid = gpu_process_led_wall_batch(leds_data, gpu_params)
                print(f"GPU ray tracing complete!\n")
            return grid, wall_size
        
        # === CPU FALLBACK (no GPU available) ===
        if stl_mesh_data is not None:
            print(f"STL mesh active → sequential mode (inner mesh queries use multi-thread BVH)")
            import time as _time
            t0 = _time.perf_counter()
            
            # Pre-build accelerator ONCE
            accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
            t_bvh = _time.perf_counter()
            print(f"  BVH built in {t_bvh - t0:.2f}s")
            
            # Process LEDs sequentially (inner _batch_ray_mesh_intersection
            # already parallelizes across CPU cores)
            for led, led_idx in active_leds:
                params = {
                    'wall_dist': wall_dist,
                    'rays_per_led': rays_per_led_calculated,
                    'grid_size': grid_size,
                    'wall_size': wall_size,
                    'lumens_per_led': lumens_per_led,
                    'absorbers': absorbers if absorbers else [],
                    'stl_mesh_data': None,
                    'ray_uniformity': ray_uniformity,
                    'led_idx': led_idx,
                }
                local_grid = _process_led_wall_worker_with_accel((led, params), accel)
                grid += local_grid
            
            t_total = _time.perf_counter() - t0
            print(f"  Ray tracing complete in {t_total:.2f}s (BVH: {t_bvh - t0:.2f}s, tracing: {t_total - (t_bvh - t0):.2f}s)\n")
        else:
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
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
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
        
        # Get reflection settings
        _refl_enabled = reflections_enable.value
        _refl_bounces = int(max_bounces_slider_room.value) if _refl_enabled else 0
        _refl_rho = float(custom_reflectance_slider.value) if _refl_enabled else 0.0
        if _refl_enabled:
            print(f"Reflections: ON (\u03c1={_refl_rho:.2f}, max {_refl_bounces} bounces)")
        
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
        
        # === GPU ACCELERATION PATH (room mode) ===
        use_gpu = False
        if _HAS_GPU_MODULE:
            _ensure_gpu_init()
            from gpu_raytrace import GPU_AVAILABLE
            use_gpu = GPU_AVAILABLE
        
        if use_gpu:
            # Prepare LED data for GPU
            leds_data = []
            per_led_lumens_list = []
            for led in active_leds:
                leds_data.append({
                    'position': np.array(led.position, dtype=np.float32),
                    'direction': np.array(led.direction, dtype=np.float32),
                    'viewing_angle': float(led.viewing_angle),
                    'ext_lens_angle': getattr(led, 'ext_lens_angle', None),
                })
                _lm = float(getattr(led, 'lumens', None) or lumens_per_led)
                if diffuser_enable_chk.value and getattr(led, 'lumens', None) is not None:
                    _lm *= float(diffuser_transmission_slider.value) / 100.0
                per_led_lumens_list.append(_lm)
            gpu_params = {
                'front_dist': front_dist,
                'side_dist': side_dist,
                'top_bottom_dist': top_bottom_dist,
                'back_dist': back_dist,
                'led_x_center': circle_center_slider.value,
                'num_rays_per_led': num_rays_per_led,
                'grid_size': grid_size,
                'lumens_per_led': lumens_per_led,
                'per_led_lumens': np.array(per_led_lumens_list, dtype=np.float32),
                'absorbers': absorbers if absorbers else [],
                'stl_mesh_data': None,  # GPU always traces without mesh
                'ray_uniformity': ray_uniformity,
                'grid_shapes': grid_shapes,
                'wall_specs': wall_specs,
                'max_bounces': _refl_bounces,
                'wall_reflectance': _refl_rho,
            }
            
            if stl_mesh_data is not None:
                # === HYBRID GPU+CPU PATH (room mode) ===
                print(f"[GPU+Mesh] Hybrid room mode: GPU ray trace + CPU mesh subtraction for {len(active_leds)} LEDs...")
                import time as _time
                t0 = _time.perf_counter()
                
                gpu_grids, gpu_hits, total_rays = gpu_process_room_batch(leds_data, gpu_params)
                for wall_name in grids.keys():
                    grids[wall_name] = gpu_grids[wall_name]
                    ray_hits[wall_name] = gpu_hits.get(wall_name, 0)
                t_gpu = _time.perf_counter()
                print(f"  GPU full grids: {t_gpu - t0:.2f}s")
                
                # Build BVH once
                accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
                t_bvh = _time.perf_counter()
                print(f"  BVH built in {t_bvh - t_gpu:.2f}s")
                
                # Pre-compute cell areas
                cell_areas_m2 = {}
                for wall_name, spec in wall_specs.items():
                    if wall_name in ('front', 'back'):
                        cw = spec['size_y'] / spec['grid_y']
                        ch = spec['size_z'] / spec['grid_z']
                    elif wall_name in ('left', 'right'):
                        cw = spec['size_x'] / spec['grid_x']
                        ch = spec['size_z'] / spec['grid_z']
                    else:
                        cw = spec['size_x'] / spec['grid_x']
                        ch = spec['size_y'] / spec['grid_y']
                    cell_areas_m2[wall_name] = (cw * ch) / 10000.0
                
                rays_per_led = num_rays_per_led * grid_size * grid_size
                absorbed_grids = {wn: np.zeros_like(g) for wn, g in grids.items()}
                total_candidate_rays = 0
                total_mesh_hits = 0
                
                # Process each LED independently to keep memory low
                for led_i, led in enumerate(active_leds):
                    t_led = _time.perf_counter()
                    led_lumens_val = float(getattr(led, 'lumens', None) or lumens_per_led)
                    np.random.seed(42)
                    
                    z_axis = led.direction
                    if abs(z_axis[2]) < 0.9:
                        x_axis = np.cross(z_axis, [0, 0, 1])
                    else:
                        x_axis = np.cross(z_axis, [0, 1, 0])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                    
                    n = _get_effective_n(led, ray_uniformity)
                    max_theta = np.radians(led.viewing_angle / 2.0)
                    cos_max = np.cos(max_theta)
                    
                    u = np.random.uniform(0, 1, (rays_per_led, 2))
                    cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta = np.arccos(cos_theta)
                    phi = 2 * np.pi * u[:, 1]
                    
                    sin_theta = np.sin(theta)
                    local_dirs = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
                    world_dirs = (local_dirs[:, 0:1] * x_axis + local_dirs[:, 1:2] * y_axis + local_dirs[:, 2:3] * z_axis)
                    norms_d = np.linalg.norm(world_dirs, axis=1, keepdims=True)
                    world_dirs = world_dirs / norms_d
                    
                    cos_max_n1 = cos_max ** (n + 1.0)
                    denom = 1.0 - cos_max_n1
                    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
                    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
                    lumens_per_ray = (led_lumens_val / rays_per_led) * np.power(cos_theta_clamped, n) * norm_factor
                    
                    del u, cos_theta, theta, phi, sin_theta, local_dirs, cos_theta_clamped
                    
                    n_cand = rays_per_led
                    total_candidate_rays += n_cand
                    
                    origins = np.tile(led.position, (n_cand, 1)).astype(np.float64)
                    cand_dirs = world_dirs.astype(np.float64)
                    del world_dirs
                    
                    mesh_hits = _batch_ray_mesh_intersection(origins, cand_dirs, accel)
                    n_hits = int(np.sum(mesh_hits))
                    total_mesh_hits += n_hits
                    
                    if n_hits > 0:
                        hit_dirs = cand_dirs[mesh_hits]
                        hit_lumens = lumens_per_ray[mesh_hits]
                        hit_pos = origins[mesh_hits]
                        
                        dx = hit_dirs[:, 0]
                        dy = hit_dirs[:, 1]
                        dz = hit_dirs[:, 2]
                        ox = hit_pos[:, 0]
                        oy = hit_pos[:, 1]
                        oz = hit_pos[:, 2]
                        
                        INF = 1e30
                        best_t = np.full(n_hits, INF)
                        best_wall = np.full(n_hits, -1, dtype=int)
                        best_c1 = np.zeros(n_hits)
                        best_c2 = np.zeros(n_hits)
                        
                        # Front wall
                        mask_f = dx > 0
                        if np.any(mask_f):
                            t_f = (front_dist - ox[mask_f]) / dx[mask_f]
                            y_f = oy[mask_f] + dy[mask_f] * t_f
                            z_f = oz[mask_f] + dz[mask_f] * t_f
                            better = mask_f.copy()
                            better[mask_f] &= (t_f > 0) & (t_f < best_t[mask_f])
                            best_t[better] = t_f[better[mask_f]]
                            best_wall[better] = 0
                            best_c1[better] = y_f[better[mask_f]]
                            best_c2[better] = z_f[better[mask_f]]
                        
                        # Left wall
                        mask_l = dy < 0
                        if np.any(mask_l):
                            t_l = (-side_dist - oy[mask_l]) / dy[mask_l]
                            x_l = ox[mask_l] + dx[mask_l] * t_l
                            z_l = oz[mask_l] + dz[mask_l] * t_l
                            better = mask_l.copy()
                            better[mask_l] &= (t_l > 0) & (t_l < best_t[mask_l])
                            best_t[better] = t_l[better[mask_l]]
                            best_wall[better] = 1
                            best_c1[better] = x_l[better[mask_l]]
                            best_c2[better] = z_l[better[mask_l]]
                        
                        # Right wall
                        mask_r = dy > 0
                        if np.any(mask_r):
                            t_r = (side_dist - oy[mask_r]) / dy[mask_r]
                            x_r = ox[mask_r] + dx[mask_r] * t_r
                            z_r = oz[mask_r] + dz[mask_r] * t_r
                            better = mask_r.copy()
                            better[mask_r] &= (t_r > 0) & (t_r < best_t[mask_r])
                            best_t[better] = t_r[better[mask_r]]
                            best_wall[better] = 2
                            best_c1[better] = x_r[better[mask_r]]
                            best_c2[better] = z_r[better[mask_r]]
                        
                        # Top wall
                        mask_t = dz > 0
                        if np.any(mask_t):
                            t_t = (top_bottom_dist - oz[mask_t]) / dz[mask_t]
                            x_t = ox[mask_t] + dx[mask_t] * t_t
                            y_t = oy[mask_t] + dy[mask_t] * t_t
                            better = mask_t.copy()
                            better[mask_t] &= (t_t > 0) & (t_t < best_t[mask_t])
                            best_t[better] = t_t[better[mask_t]]
                            best_wall[better] = 3
                            best_c1[better] = x_t[better[mask_t]]
                            best_c2[better] = y_t[better[mask_t]]
                        
                        # Bottom wall
                        mask_b = dz < 0
                        if np.any(mask_b):
                            t_b = (-top_bottom_dist - oz[mask_b]) / dz[mask_b]
                            x_b = ox[mask_b] + dx[mask_b] * t_b
                            y_b = oy[mask_b] + dy[mask_b] * t_b
                            better = mask_b.copy()
                            better[mask_b] &= (t_b > 0) & (t_b < best_t[mask_b])
                            best_t[better] = t_b[better[mask_b]]
                            best_wall[better] = 4
                            best_c1[better] = x_b[better[mask_b]]
                            best_c2[better] = y_b[better[mask_b]]
                        
                        # Back wall
                        if back_dist is not None:
                            mask_bk = dx < 0
                            if np.any(mask_bk):
                                t_bk = (-back_dist - ox[mask_bk]) / dx[mask_bk]
                                y_bk = oy[mask_bk] + dy[mask_bk] * t_bk
                                z_bk = oz[mask_bk] + dz[mask_bk] * t_bk
                                better = mask_bk.copy()
                                better[mask_bk] &= (t_bk > 0) & (t_bk < best_t[mask_bk])
                                best_t[better] = t_bk[better[mask_bk]]
                                best_wall[better] = 5
                                best_c1[better] = y_bk[better[mask_bk]]
                                best_c2[better] = z_bk[better[mask_bk]]
                        
                        # Scatter absorbed lux into per-wall grids
                        wall_configs = [(0, 'front'), (1, 'left'), (2, 'right'), (3, 'top'), (4, 'bottom')]
                        if back_dist is not None:
                            wall_configs.append((5, 'back'))
                        
                        for wall_id, wall_name in wall_configs:
                            w_mask = best_wall == wall_id
                            if not np.any(w_mask):
                                continue
                            c1 = best_c1[w_mask]
                            c2 = best_c2[w_mask]
                            ray_lux = hit_lumens[w_mask] / cell_areas_m2[wall_name]
                            spec = wall_specs[wall_name]
                            shape = absorbed_grids[wall_name].shape
                            if wall_name in ('front', 'back'):
                                gi = ((c1 + spec['size_y'] / 2) / (spec['size_y'] / spec['grid_y'])).astype(int)
                                gj = ((c2 + spec['size_z'] / 2) / (spec['size_z'] / spec['grid_z'])).astype(int)
                            elif wall_name in ('left', 'right'):
                                gi = ((c1 - spec['x_min']) / (spec['size_x'] / spec['grid_x'])).astype(int)
                                gj = ((c2 + spec['size_z'] / 2) / (spec['size_z'] / spec['grid_z'])).astype(int)
                            else:
                                gi = ((c1 - spec['x_min']) / (spec['size_x'] / spec['grid_x'])).astype(int)
                                gj = ((c2 + spec['size_y'] / 2) / (spec['size_y'] / spec['grid_y'])).astype(int)
                            row = np.clip(gj, 0, shape[0] - 1)
                            col = np.clip(gi, 0, shape[1] - 1)
                            np.add.at(absorbed_grids[wall_name], (row, col), ray_lux)
                    
                    del origins, cand_dirs, lumens_per_ray, mesh_hits
                    dt = _time.perf_counter() - t_led
                    if dt > 0.3:
                        print(f"    LED {led_i+1}/{len(active_leds)}: {n_cand:,} rays, {n_hits:,} hits, {dt:.2f}s")
                
                # Subtract absorbed from full grids
                for wall_name in grids.keys():
                    grids[wall_name] = np.maximum(0, grids[wall_name] - absorbed_grids[wall_name])
                
                t_total = _time.perf_counter() - t0
                absorbed_total = sum(np.sum(a) for a in absorbed_grids.values())
                full_total = max(sum(np.sum(g) for g in grids.values()), 1e-10)
                print(f"  Mesh subtraction: {t_total - (t_gpu - t0):.2f}s "
                      f"({total_mesh_hits:,} hits / {total_candidate_rays:,} rays)")
                print(f"  Total hybrid room: {t_total:.2f}s\n")
            else:
                # === PURE GPU PATH (no mesh) ===
                print(f"[GPU] Using NVIDIA CUDA acceleration for room mode ({len(active_leds)} LEDs)...")
                gpu_grids, gpu_hits, total_rays = gpu_process_room_batch(leds_data, gpu_params)
                for wall_name in grids.keys():
                    grids[wall_name] = gpu_grids[wall_name]
                    ray_hits[wall_name] = gpu_hits.get(wall_name, 0)
        else:
            # === CPU FALLBACK ===
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
                'wall_specs': wall_specs,
                'max_bounces': _refl_bounces,
                'wall_reflectance': _refl_rho,
            }
            
            led_args = []
            import copy as _copy
            _diff_trans_cpu = float(diffuser_transmission_slider.value) / 100.0 if diffuser_enable_chk.value else 1.0
            for led in active_leds:
                _led = _copy.copy(led)
                if _diff_trans_cpu < 1.0 and _led.lumens is not None:
                    _led.lumens = _led.lumens * _diff_trans_cpu
                led_args.append((_led, worker_params))
            
            if stl_mesh_data is not None:
                # Sequential mode: build BVH ONCE, reuse for all LEDs
                # (inner _batch_ray_mesh_intersection already parallelizes across CPU cores)
                print(f"[STL] Building BVH once for {len(active_leds)} LEDs...")
                accel = _prepare_mesh_ray_accelerator(stl_mesh_data)
                for led_arg in led_args:
                    led_grids, led_ray_hits, led_total_rays = _process_led_worker_with_accel(led_arg, accel)
                    for wall_name in grids.keys():
                        grids[wall_name] += led_grids[wall_name]
                        ray_hits[wall_name] += led_ray_hits[wall_name]
                    total_rays += led_total_rays
            else:
                # Multiprocessing: no mesh, safe to parallelize
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
        total_emitted = sum(float(getattr(led, 'lumens', None) or lumens_per_led) for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
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
            # Pass lumens override for custom group
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
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
                
                half_w = fov_width_cm / 2
                half_h = fov_height_cm / 2
                
                in_fov = (hit_y >= -half_w) & (hit_y <= half_w) & (hit_z >= -half_h) & (hit_z <= half_h)
                fi = np.where(in_fov)[0]
                
                grid_x = ((hit_y[fi] + half_w) / cell_size_cm).astype(int)
                grid_y = ((hit_z[fi] + half_h) / cell_size_cm).astype(int)
                
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
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fov_intensity_{timestamp}.png"
        img_pil.save(filename)
        
        print(f"FOV image saved to {filename}")
        print(f"Image size: {grid_width} x {grid_height} pixels (1 pixel = 1cm²)")
        print(f"FOV dimensions: {fov_width_cm:.2f} x {fov_height_cm:.2f} cm")
        print(f"Total lumens in FOV: {fov_grid.sum():.2f} lm")
        print(f"Max illuminance: {max_lux:.2f} lux")

    FIXED_LEGEND_MAX = 3500.0  # Default fixed absolute legend cap (overridden by GUI)

    def intensity_to_color(value, max_val):
        """Convert intensity to colormap (inferno-like or black-to-white)."""
        # Handle invalid values
        if max_val == 0 or not np.isfinite(value) or not np.isfinite(max_val):
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

    def _refresh_uniformity():
        """Recalculate FOV-only uniformity from cached intensity grid (cheap)."""
        cache = _last_intensity_cache
        if cache['grid'] is None:
            return  # No intensity data yet
        grid = cache['grid']
        wall_size_cm = cache['wall_size_cm']
        wall_dist = cache['wall_dist']
        cell_area_m2 = cache['cell_area_m2']
        max_lux = cache['max_lux']
        color_scale_max = cache['color_scale_max']

        # Rebuild the legend exactly as update_intensity_map does
        grid_size = grid.shape[0]
        _legend_cap = float(legend_max_input.value)
        if color_scale_max <= _legend_cap:
            _step = max(1, _legend_cap / 8)
            legend_vals_lux = np.arange(0, _legend_cap + 1, _step)
        else:
            legend_vals_lux = np.linspace(0, color_scale_max, 9)
        scale_label = f"(scale 0\u2013{int(color_scale_max)} lx" + (", FIXED)" if color_scale_max <= _legend_cap else ", AUTO)")
        html_lines = ["<div style='font-family: sans-serif;'>",
                      f"<div style='font-weight:600;margin-bottom:2px;'>Intensity legend (lux)</div>",
                      f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} \u2014 peak {max_lux:.0f} lx</div>"]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, color_scale_max)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            lumen_val = lux_val * cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.0f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} lm/cell)</div></div>"
            )
        html_lines.append("</div>")

        # Recalculate FOV crop and uniformity
        _cam_x = camera_pos_x.value
        _fov_h_rad = np.radians(camera_fov_h.value)
        _fov_v_rad = np.radians(camera_fov_v.value)
        _fov_w_cm = 2.0 * (wall_dist - _cam_x) * np.tan(_fov_h_rad / 2.0)
        _fov_h_cm = 2.0 * (wall_dist - _cam_x) * np.tan(_fov_v_rad / 2.0)
        uniformity_html = _compute_uniformity_html(
            grid,
            fov_bounds=(_fov_w_cm, _fov_h_cm),
            wall_size_cm=wall_size_cm,
        )
        legend_html.content = "".join(html_lines) + uniformity_html

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
                roll_deg = group['rot_roll'].value if 'rot_roll' in group else 0
                pitch_deg = group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0
                yaw_deg = group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0
                roll_rad = np.radians(roll_deg)
                pitch_rad = np.radians(pitch_deg)
                yaw_rad = np.radians(yaw_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]
                ])
                Ry = np.array([
                    [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
                ])
                Rz = np.array([
                    [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]
                ])
                R_total = Rz @ Ry @ Rx
                position_offset = np.array([group['pos_x'].value, group['pos_y'].value, group['pos_z'].value])
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
                original_positions = group.get('original_led_positions', group.get('led_positions', []))
                original_rotations = group.get('original_led_rotations', group.get('led_rotations', []))
                translated_positions = []
                for orig_pos in original_positions:
                    rotated_pos = R_total @ np.array(orig_pos)
                    final_pos = rotated_pos + position_offset
                    translated_positions.append(tuple(final_pos))
                rotated_directions = []
                for orig_dir in original_rotations:
                    rotated_dir = R_total @ np.array(orig_dir)
                    rotated_directions.append(tuple(rotated_dir))
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                config['led_beam_tilts'] = group.get('led_beam_tilts', [])
                original_row_dirs = group.get('original_led_row_directions', group.get('led_row_directions', []))
                if original_row_dirs:
                    rotated_row_dirs = [tuple(R_total @ np.array(rd)) for rd in original_row_dirs]
                    config['led_row_directions'] = rotated_row_dirs
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
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
            individual_leds_configs.append(config)

        leds = create_leds(
            front_angle, side_angle, viewing_angle, radius, circle_center_x,
            group_rotations=rotations, group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states, group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )

        _g_rot_z_deg = global_rotation_z_slider.value
        if abs(_g_rot_z_deg) > 0.01:
            _g_rad = np.radians(_g_rot_z_deg)
            _cg, _sg = np.cos(_g_rad), np.sin(_g_rad)
            _Rg = np.array([[_cg, -_sg, 0], [_sg, _cg, 0], [0, 0, 1]], dtype=float)
            for led in leds:
                led.position = _Rg @ led.position
                led.direction = _Rg @ led.direction
                if hasattr(led, 'row_direction') and led.row_direction is not None:
                    led.row_direction = _Rg @ np.asarray(led.row_direction)
                if hasattr(led, 'square_normal') and led.square_normal is not None:
                    led.square_normal = _Rg @ np.asarray(led.square_normal)
        else:
            _g_rad = 0.0
            _cg, _sg = 1.0, 0.0
            _Rg = None

        # Apply global position offset
        _gp_x = global_pos_x_slider.value
        _gp_y = global_pos_y_slider.value
        _gp_z = global_pos_z_slider.value
        if abs(_gp_x) > 0.001 or abs(_gp_y) > 0.001 or abs(_gp_z) > 0.001:
            _gp_offset = np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            for led in leds:
                led.position = led.position + _gp_offset

        if diffuser_enable_chk.value:
            _diff_angle = float(diffuser_angle_slider.value)
            _diff_trans = float(diffuser_transmission_slider.value) / 100.0
            for led in leds:
                led.viewing_angle = max(led.viewing_angle, _diff_angle)
                if led.lumens is not None:
                    led.lumens = led.lumens * _diff_trans

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
        if absorbers_enable.value:
            abs_cx = 0.0 + abs2_off_x.value
            abs_cy = 0.0 + abs2_off_y.value
            abs_cz = 0.0 + abs2_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            angle_rad = np.radians(abs2_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx, qy, qz = 0.0, 0.0, np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
            abs_cx = 0.0 + abs3_off_x.value
            abs_cy = 0.0 + abs3_off_y.value
            abs_cz = 0.0 + abs3_off_z.value
            angle_rad = np.radians(abs3_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx, qy, qz = 0.0, 0.0, np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
        if abs(_g_rot_z_deg) > 0.01:
            for a in absorbers:
                cx, cy, cz = a['center']
                a['center'] = (_cg * cx - _sg * cy, _sg * cx + _cg * cy, cz)
                if a.get('rotation') is not None:
                    qw0, qx0, qy0, qz0 = a['rotation']
                    _half = _g_rad / 2.0
                    gqw, gqz = np.cos(_half), np.sin(_half)
                    a['rotation'] = (
                        gqw*qw0 - gqz*qz0, gqw*qx0 - gqz*qy0,
                        gqw*qy0 + gqz*qx0, gqw*qz0 + gqz*qw0,
                    )
                else:
                    _half = _g_rad / 2.0
                    a['rotation'] = (np.cos(_half), 0.0, 0.0, np.sin(_half))

        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_ref = stl_mesh_data[0]
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            if abs(_g_rot_z_deg) > 0.01:
                T_global = np.eye(4)
                T_global[:3, :3] = _Rg
                transform = T_global @ transform
            stl_mesh_for_raytracing = {
                'vertices': mesh_ref.vertices,
                'faces': mesh_ref.faces,
                'transform': transform
            }

        return leds, absorbers, stl_mesh_for_raytracing

    def update_intensity_map():
        """Update only the intensity map on the wall (expensive operation)."""
        nonlocal intensity_handles, legend_html
        
        import time as _time
        t_total_start = _time.perf_counter()
        
        # Clear previous intensity handles
        for handle in intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        intensity_handles = []
        
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
        # Fixed legend ticks at absolute values up to color_scale_max
        _legend_cap = float(legend_max_input.value)
        if color_scale_max <= _legend_cap:
            _step = max(1, _legend_cap / 8)
            legend_vals_lux = np.arange(0, _legend_cap + 1, _step)
        else:
            # Exceeded cap – uniform 9 ticks up to actual max
            legend_vals_lux = np.linspace(0, color_scale_max, 9)
        scale_label = f"(scale 0–{int(color_scale_max)} lx" + (", FIXED)" if color_scale_max <= _legend_cap else ", AUTO)")
        html_lines = ["<div style='font-family: sans-serif;'>",
                      f"<div style='font-weight:600;margin-bottom:2px;'>Intensity legend (lux)</div>",
                      f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} — peak {max_lux:.0f} lx</div>"]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, color_scale_max)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            # Convert lux to lumens for this cell: Lumen = Lux × Area
            lumen_val = lux_val * cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.0f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} lm/cell)</div></div>"
            )
        html_lines.append("</div>")
        # --- Uniformity metrics (FOV-only) ---
        # Compute FOV footprint on wall so metrics cover only the green rectangle
        _cam_x = camera_pos_x.value
        _fov_h_rad = np.radians(camera_fov_h.value)
        _fov_v_rad = np.radians(camera_fov_v.value)
        _fov_w_cm = 2.0 * (wall_dist - _cam_x) * np.tan(_fov_h_rad / 2.0)
        _fov_h_cm = 2.0 * (wall_dist - _cam_x) * np.tan(_fov_v_rad / 2.0)
        uniformity_html = _compute_uniformity_html(
            intensity_grid,
            fov_bounds=(_fov_w_cm, _fov_h_cm),
            wall_size_cm=actual_wall_size,
        )
        legend_html.content = "".join(html_lines) + uniformity_html

        # Cache grid so FOV changes can recalculate uniformity cheaply
        _last_intensity_cache['grid'] = intensity_grid
        _last_intensity_cache['wall_size_cm'] = actual_wall_size
        _last_intensity_cache['wall_dist'] = wall_dist
        _last_intensity_cache['cell_area_m2'] = cell_area_m2
        _last_intensity_cache['max_lux'] = max_lux
        _last_intensity_cache['color_scale_max'] = color_scale_max

    # ── CSV Pattern Import logic ──────────────────────────────────────────
    def _parse_benchmark_csv(filepath):
        """Parse a benchmark CSV file (NORMAL + UNIFORMITY sections).
        Supports both the internal text export and spreadsheet CSV formats.
        Extracts DFRobot lux values when available, falls back to Simulator.
        Returns (grid_2d, wall_size_cm, description, y_min, y_max, x_min, x_max) or raises ValueError."""
        with open(filepath, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        if not lines:
            raise ValueError("Empty file")

        if lines[0].startswith("Camera FOV Intensity Image"):
            return _parse_fov_intensity_csv(lines)

        # Detect if this is a comma-separated spreadsheet CSV
        is_csv = any(',' in l for l in lines[:20])

        # Helper: extract non-empty cells from a CSV line
        def _csv_cells(line):
            return [c.strip() for c in line.split(',')]

        # Helper: find a cell matching a pattern in a line (returns cell index or -1)
        def _find_cell(cells, pattern):
            pat_lower = pattern.lower()
            for idx, c in enumerate(cells):
                if pat_lower in c.lower():
                    return idx
            return -1

        # Helper: extract numeric values from cells starting at given index
        def _extract_nums(cells, start):
            vals = []
            for c in cells[start:]:
                c = c.strip()
                if not c:
                    continue
                try:
                    vals.append(float(c))
                except ValueError:
                    break
            return vals

        # Find UNIFORMITY section
        unif_start = None
        for i, line in enumerate(lines):
            text = line.strip().replace(',', ' ').strip()
            cells = _csv_cells(line) if is_csv else [line.strip()]
            for c in cells:
                if 'UNIFORMITY' in c.upper():
                    unif_start = i
                    break
            if unif_start is not None:
                break
        if unif_start is None:
            raise ValueError("No UNIFORMITY section found in benchmark file")

        # Parse Y blocks
        y_positions_mm = []
        lux_rows = []
        x_scan_mm = None
        i = unif_start + 1

        while i < len(lines):
            if is_csv:
                cells = _csv_cells(lines[i])
            else:
                cells = [lines[i].strip()]

            # Look for Y label in any cell
            y_cell_idx = -1
            for ci, c in enumerate(cells):
                c_stripped = c.strip()
                if c_stripped.startswith('Y ') or c_stripped.startswith('Y+') or c_stripped.startswith('Y-'):
                    y_cell_idx = ci
                    break
            if y_cell_idx >= 0:
                y_label = cells[y_cell_idx].strip()
                if 'center' in y_label.lower():
                    y_mm = 0
                else:
                    # Extract number: "Y +200", "Y -100", "Y+200"
                    import re as _re
                    nums = _re.findall(r'[+-]?\d+', y_label)
                    if nums:
                        y_mm = int(nums[0])
                    else:
                        i += 1
                        continue
                y_positions_mm.append(y_mm)

                # Scan forward for X scan, Simulator, DFrobot lux lines
                x_line_found = False
                sim_vals = None
                dfr_vals = None
                j = i + 1
                search_limit = min(j + 8, len(lines))
                while j < search_limit:
                    if is_csv:
                        jcells = _csv_cells(lines[j])
                    else:
                        jcells = lines[j].strip().split()

                    joined = ' '.join(c.strip() for c in jcells).lower()

                    if 'x scan' in joined or 'x_scan' in joined:
                        # Extract X positions
                        if is_csv:
                            xci = _find_cell(jcells, 'scan')
                            if xci >= 0:
                                x_vals = _extract_nums(jcells, xci + 1)
                            else:
                                x_vals = _extract_nums(jcells, 0)
                        else:
                            x_vals = []
                            for t in jcells:
                                try:
                                    x_vals.append(int(t))
                                except ValueError:
                                    continue
                        if x_vals and x_scan_mm is None:
                            x_scan_mm = [int(v) for v in x_vals]
                        x_line_found = True

                    elif 'simulator' in joined:
                        if is_csv:
                            sci = _find_cell(jcells, 'simulator')
                            if sci >= 0:
                                sim_vals = _extract_nums(jcells, sci + 1)
                        else:
                            sv = []
                            for t in jcells:
                                try:
                                    sv.append(float(t))
                                except ValueError:
                                    continue
                            sim_vals = sv

                    elif 'dfrobot' in joined or 'df robot' in joined or 'dfr' in joined.replace(' ', ''):
                        if is_csv:
                            dci = _find_cell(jcells, 'lux')
                            if dci < 0:
                                dci = _find_cell(jcells, 'dfrobot')
                                if dci < 0:
                                    dci = _find_cell(jcells, 'DFR')
                            if dci >= 0:
                                dfr_vals = _extract_nums(jcells, dci + 1)
                            else:
                                dfr_vals = _extract_nums(jcells, 0)
                        else:
                            dv = []
                            for t in jcells:
                                try:
                                    dv.append(float(t))
                                except ValueError:
                                    continue
                            dfr_vals = dv

                    # Stop when we have both or hit next Y block or empty block
                    if dfr_vals is not None and sim_vals is not None:
                        break
                    j += 1

                # Prefer DFRobot lux; fall back to Simulator
                chosen = dfr_vals if dfr_vals else sim_vals
                if chosen:
                    lux_rows.append(chosen)
                else:
                    # Remove the Y position since we found no data
                    y_positions_mm.pop()

            i += 1

        if not lux_rows or x_scan_mm is None:
            raise ValueError("Could not parse UNIFORMITY data")

        # Build dense grid via interpolation
        from scipy.interpolate import RegularGridInterpolator

        y_cm = [y / 10.0 for y in y_positions_mm]  # e.g. [20, 10, 0, -10, -20]
        x_cm = [x / 10.0 for x in x_scan_mm]       # e.g. [-40, -30, ..., 40]

        # Sort y ascending for interpolator
        sorted_pairs = sorted(zip(y_cm, lux_rows), key=lambda p: p[0])
        y_sorted = [p[0] for p in sorted_pairs]
        data_sorted = [p[1] for p in sorted_pairs]
        data_2d = np.array(data_sorted, dtype=np.float64)  # shape (ny, nx)

        # Target dense grid covering the data range
        y_min, y_max = y_sorted[0], y_sorted[-1]
        x_min, x_max = x_cm[0], x_cm[-1]
        wall_size_cm = max(abs(x_max - x_min), abs(y_max - y_min))
        dense_n = max(50, int(wall_size_cm))  # ~1cm resolution

        dense_y = np.linspace(y_min, y_max, dense_n)
        dense_x = np.linspace(x_min, x_max, dense_n)

        interp = RegularGridInterpolator(
            (np.array(y_sorted), np.array(x_cm)), data_2d,
            method='linear', bounds_error=False, fill_value=0.0
        )
        yy, xx = np.meshgrid(dense_y, dense_x, indexing='ij')
        grid = interp((yy, xx))
        grid = np.clip(grid, 0, None)

        desc = f"DFRobot lux – UNIFORMITY ({len(y_positions_mm)} Y × {len(x_scan_mm)} X)"
        return grid, wall_size_cm, desc, y_min, y_max, x_min, x_max

    def _parse_fov_intensity_csv(lines):
        """Parse a Camera FOV Intensity Image CSV.
        Returns (grid_2d, wall_size_cm, description, y_min, y_max, x_min, x_max)."""
        # Header parsing
        fov_w_cm = None
        fov_h_cm = None
        grid_start = None
        for i, line in enumerate(lines):
            if line.startswith("FOV Width"):
                fov_w_cm = float(line.split(",")[1])
            elif line.startswith("FOV Height"):
                fov_h_cm = float(line.split(",")[1])
            elif line.startswith("Intensity Grid"):
                grid_start = i + 2  # skip blank line after header
                break

        if fov_w_cm is None or fov_h_cm is None or grid_start is None:
            raise ValueError("Cannot parse FOV intensity CSV header")

        # Parse grid data
        rows = []
        for i in range(grid_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",") if v.strip()]
            if vals:
                rows.append(vals)

        grid = np.array(rows, dtype=np.float64)
        wall_size_cm = max(fov_w_cm, fov_h_cm)
        half_w = fov_w_cm / 2.0
        half_h = fov_h_cm / 2.0
        # FOV grid stores lumens per cell — convert to lux (lm/m²)
        cell_w_cm = fov_w_cm / grid.shape[1]
        cell_h_cm = fov_h_cm / grid.shape[0]
        cell_area_m2 = (cell_w_cm / 100.0) * (cell_h_cm / 100.0)
        if cell_area_m2 > 0:
            grid = grid / cell_area_m2

        desc = f"FOV Intensity ({grid.shape[0]}×{grid.shape[1]})"
        return grid, wall_size_cm, desc, -half_h, half_h, -half_w, half_w

    def _render_imported_csv_on_wall(grid, wall_size_cm, y_min, y_max, x_min, x_max):
        """Render an imported lux grid on the wall, similar to update_intensity_map."""
        nonlocal imported_csv_handles

        # Clear previous
        for h in imported_csv_handles:
            try:
                h.remove()
            except (KeyError, Exception):
                pass
        imported_csv_handles = []

        if room_mode_enable.value:
            wall_dist = room_front_dist.value
        else:
            wall_dist = wall_dist_slider.value

        _csv_cap = float(csv_legend_max_input.value)

        max_lux = float(np.max(grid))
        if max_lux <= 0:
            print("[CSV Import] All values are zero — nothing to display.")
            return

        color_scale_max = _csv_cap if max_lux <= _csv_cap else max_lux
        nrows, ncols = grid.shape

        cell_h_cm = (y_max - y_min) / nrows
        cell_w_cm = (x_max - x_min) / ncols

        x_pos = wall_dist / 100.0 - 0.006  # slightly in front (different offset from sim)

        vertices_list = []
        faces_list = []
        colors_list = []
        vert_idx = 0
        gap = 0.025

        for gz in range(nrows):
            for gy in range(ncols):
                intensity = grid[gz, gy]
                if intensity > 0:
                    color = intensity_to_color(intensity, color_scale_max)
                    color_uint8 = [int(c * 255) for c in color] + [255]

                    # Map grid cell to wall coordinates (in meters)
                    y_center = (x_min + gy * cell_w_cm + cell_w_cm / 2) / 100.0
                    z_center = (y_min + gz * cell_h_cm + cell_h_cm / 2) / 100.0
                    half_cell_y = (cell_w_cm / 100.0) * 0.5 * (1.0 - gap)
                    half_cell_z = (cell_h_cm / 100.0) * 0.5 * (1.0 - gap)

                    v0 = [x_pos, y_center - half_cell_y, z_center - half_cell_z]
                    v1 = [x_pos, y_center + half_cell_y, z_center - half_cell_z]
                    v2 = [x_pos, y_center + half_cell_y, z_center + half_cell_z]
                    v3 = [x_pos, y_center - half_cell_y, z_center + half_cell_z]

                    vertices_list.extend([v0, v1, v2, v3])
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

            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
            from trimesh.visual import ColorVisuals
            mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=colors_np)

            handle = server.scene.add_mesh_trimesh(
                name="/imported_csv_pattern",
                mesh=mesh,
                visible=True,
            )
            imported_csv_handles.append(handle)

        print(f"[CSV Import] Rendered {vert_idx // 4} cells, max {max_lux:.1f} lux")
        return max_lux

    def import_csv_pattern():
        """Import and render a CSV pattern file on the wall."""
        nonlocal imported_csv_handles
        filepath = csv_import_path.value.strip()
        if not filepath:
            csv_import_status.content = "<div style='font-size:11px;color:#f44;'>⚠ Enter a file path first</div>"
            return
        if not os.path.isfile(filepath):
            csv_import_status.content = f"<div style='font-size:11px;color:#f44;'>⚠ File not found: {filepath}</div>"
            return
        try:
            result = _parse_benchmark_csv(filepath)
            grid, wall_size_cm, desc = result[0], result[1], result[2]
            y_min, y_max, x_min, x_max = result[3], result[4], result[5], result[6]
            max_lux = _render_imported_csv_on_wall(grid, wall_size_cm, y_min, y_max, x_min, x_max)
            csv_import_status.content = (
                f"<div style='font-size:11px;color:#4CAF50;'>"
                f"✓ {desc}<br>Peak: {max_lux:.1f} lux | Grid: {grid.shape[0]}×{grid.shape[1]}"
                f"</div>"
            )
            # Compute stats from the imported CSV values
            csv_vals = grid[grid > 0] if np.any(grid > 0) else grid.ravel()
            csv_min = float(np.min(csv_vals))
            csv_max = float(np.max(csv_vals))
            csv_avg = float(np.mean(csv_vals))
            csv_diff_html.content = (
                "<div style='font-family:sans-serif;margin-top:6px;padding:6px;background:#1a1a2e;border-radius:4px;'>"
                "<div style='font-weight:600;font-size:12px;color:#e0e0e0;margin-bottom:4px;'>CSV Stats (lux)</div>"
                f"<div style='font-size:11px;color:#90caf9;'>Min: <b>{csv_min:.1f}</b></div>"
                f"<div style='font-size:11px;color:#ef9a9a;'>Max: <b>{csv_max:.1f}</b></div>"
                f"<div style='font-size:11px;color:#fff59d;'>Avg: <b>{csv_avg:.1f}</b></div>"
                "</div>"
            )
            # Build color/value legend (fixed / auto, like main intensity legend)
            _csv_cap = float(csv_legend_max_input.value)
            csv_color_scale = _csv_cap if max_lux <= _csv_cap else max_lux
            if csv_color_scale <= _csv_cap:
                _csv_step = max(1, _csv_cap / 10)
                legend_vals = np.arange(0, _csv_cap + 1, _csv_step)
                mode_label = "FIXED"
            else:
                legend_vals = np.linspace(0, csv_color_scale, 11)
                mode_label = "AUTO"
            scale_label = f"(scale 0–{int(csv_color_scale)} lx, {mode_label})"
            legend_lines = [
                "<div style='font-family:sans-serif;margin-top:8px;'>",
                "<div style='font-weight:600;margin-bottom:2px;font-size:12px;'>CSV Pattern Legend (lux)</div>",
                f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} — peak {max_lux:.0f} lx</div>",
            ]
            for lux_val in reversed(legend_vals):
                color = intensity_to_color(lux_val, csv_color_scale)
                hex_c = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
                legend_lines.append(
                    f"<div style='display:flex;align-items:center;margin:1px 0;'>"
                    f"<div style='width:18px;height:10px;background:{hex_c};margin-right:6px;"
                    f"border:1px solid #333;'></div>"
                    f"<span style='font-size:11px;'>{lux_val:.0f} lx</span></div>"
                )
            legend_lines.append("</div>")
            csv_legend_html.content = "".join(legend_lines)
        except Exception as e:
            csv_import_status.content = f"<div style='font-size:11px;color:#f44;'>⚠ Error: {e}</div>"
            csv_legend_html.content = ""
            csv_diff_html.content = ""
            print(f"[CSV Import] Error: {e}")

    def clear_csv_pattern():
        """Remove imported CSV pattern from the wall."""
        nonlocal imported_csv_handles
        for h in imported_csv_handles:
            try:
                h.remove()
            except (KeyError, Exception):
                pass
        imported_csv_handles = []
        csv_import_status.content = "<div style='font-size:11px;color:#888;'>No file imported</div>"
        csv_legend_html.content = ""
        csv_diff_html.content = ""
        print("[CSV Import] Cleared imported pattern.")

    def export_lux_matrix():
        """Export a text file with lux values sampled every 10cm in ±40cm range on Y and Z axes."""
        cache = _last_intensity_cache
        if cache['grid'] is None:
            print("[Export] No intensity data available. Run 'Update Intensity Map' first.")
            return

        grid = cache['grid']
        wall_size_cm = cache['wall_size_cm']
        wall_dist = cache['wall_dist']
        grid_size = grid.shape[0]
        cell_size = wall_size_cm / grid_size
        half_size = wall_size_cm / 2.0

        # Sample positions: -40 to +40 cm, step 10 cm
        sample_positions = list(range(-40, 41, 10))  # [-40, -30, ..., 0, ..., 30, 40]

        # Build the lux matrix: rows = Z (top to bottom), cols = Y (left to right)
        lux_matrix = []
        for z_cm in reversed(sample_positions):  # top to bottom
            row = []
            for y_cm in sample_positions:
                # Convert cm to grid index
                gy = int((y_cm + half_size) / cell_size)
                gz = int((z_cm + half_size) / cell_size)
                if 0 <= gy < grid_size and 0 <= gz < grid_size:
                    row.append(grid[gz, gy])
                else:
                    row.append(0.0)
            lux_matrix.append(row)

        # Write to file
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"lux_matrix_{ts}.txt"
        filepath = os.path.join(export_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"Lux Matrix Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Wall distance: {wall_dist:.1f} cm\n")
            f.write(f"Grid resolution: {grid_size}x{grid_size}\n")
            f.write(f"Wall size: {wall_size_cm:.1f} cm\n")
            f.write(f"Sample range: +-40 cm, step 10 cm\n")
            f.write(f"Rows: Z axis (top +40 to bottom -40)\n")
            f.write(f"Columns: Y axis (left -40 to right +40)\n")
            f.write("\n")

            # Header row with Y positions
            header = "Z\\Y(cm)" + "\t" + "\t".join(f"{y:+d}" for y in sample_positions)
            f.write(header + "\n")

            # Data rows
            for i, z_cm in enumerate(reversed(sample_positions)):
                row_str = f"{z_cm:+d}" + "\t" + "\t".join(f"{v:.1f}" for v in lux_matrix[i])
                f.write(row_str + "\n")

        print(f"[Export] Lux matrix saved to {filepath}")

    def run_benchmark():
        """Run benchmark: center lux at multiple distances + uniformity grid at 50cm.
        Exports a CSV file matching the NORMAL + UNIFORMITY format."""
        import time as _time
        from datetime import datetime

        print("\n=== STARTING BENCHMARK ===")
        t0 = _time.perf_counter()

        leds, absorbers, stl_mesh_for_raytracing = _build_current_leds_and_absorbers()
        grid_size = int(intensity_grid_size.value)
        rays_per_pixel = int(intensity_rays_slider.value)

        # --- NORMAL: center lux at multiple distances ---
        normal_distances = [10, 20, 30, 50, 80]
        normal_center_lux = {}
        for dist in normal_distances:
            print(f"  [Benchmark] Computing center lux at {dist} cm...")
            # Wall size must cover at least +-40cm
            w_size = max(80, int(wall_view_size.value))
            grid, actual_ws = compute_wall_intensity(
                leds, dist, rays_per_pixel, grid_size, w_size,
                absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
            )
            grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
            center_idx = grid_size // 2
            normal_center_lux[dist] = grid[center_idx, center_idx]
            print(f"    Center lux: {normal_center_lux[dist]:.1f}")

        # --- UNIFORMITY at 50cm: grid scan ---
        unif_dist = 50
        print(f"  [Benchmark] Computing uniformity grid at {unif_dist} cm...")
        w_size = max(80, int(wall_view_size.value))
        unif_grid, actual_ws = compute_wall_intensity(
            leds, unif_dist, rays_per_pixel, grid_size, w_size,
            absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        unif_grid = np.nan_to_num(unif_grid, nan=0.0, posinf=0.0, neginf=0.0)

        cell_size = actual_ws / grid_size
        half_size = actual_ws / 2.0

        # X scan positions (horizontal on wall = Y axis): -400 to +400 mm step 100 = -40 to +40 cm step 10
        x_scan_mm = list(range(-400, 401, 100))
        x_scan_cm = [x / 10.0 for x in x_scan_mm]
        # Y positions (vertical on wall = Z axis): +200, +100, 0, -100, -200 mm
        y_positions_mm = [400, 300, 200, 100, 0, -100, -200, -300, -400]
        y_positions_cm = [y / 10.0 for y in y_positions_mm]

        uniformity_data = {}
        for y_mm, y_cm in zip(y_positions_mm, y_positions_cm):
            row_lux = []
            for x_mm, x_cm in zip(x_scan_mm, x_scan_cm):
                gy = min(int(round((x_cm + half_size) / cell_size)), grid_size - 1)
                gz = min(int(round((y_cm + half_size) / cell_size)), grid_size - 1)
                if 0 <= gy < grid_size and 0 <= gz < grid_size:
                    row_lux.append(unif_grid[gz, gy])
                else:
                    row_lux.append(0.0)
            uniformity_data[y_mm] = row_lux

        # --- Write CSV ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"benchmark_{ts}.txt"
        filepath = os.path.join(export_dir, filename)

        COL1 = 16  # first column width
        COLN = 10  # data column width

        with open(filepath, 'w') as f:
            # NORMAL section
            f.write("NORMAL\n")
            f.write("DISTANCE cm".ljust(COL1) + "".join(str(d).rjust(COLN) for d in normal_distances) + "\n")
            f.write("Simulator".ljust(COL1) + "".join(f"{normal_center_lux[d]:.1f}".rjust(COLN) for d in normal_distances) + "\n")
            f.write("DFRobot lux\n")
            f.write("\n")

            # UNIFORMITY section
            f.write(f"UNIFORMITY\n")
            f.write(f"DISTANCE {unif_dist}CM\n")
            f.write("\n")

            for y_mm in y_positions_mm:
                if y_mm == 0:
                    label = "Y center"
                else:
                    label = f"Y {y_mm:+d}"
                f.write(f"{label}\n")
                f.write("X scan".ljust(COL1) + "".join(str(x).rjust(COLN) for x in x_scan_mm) + "\n")
                f.write("Simulator".ljust(COL1) + "".join(f"{v:.1f}".rjust(COLN) for v in uniformity_data[y_mm]) + "\n")
                f.write("DFRobot lux\n")
                f.write("\n")

        t1 = _time.perf_counter()
        print(f"=== BENCHMARK COMPLETE ({t1 - t0:.1f}s) ===")
        print(f"[Export] Benchmark saved to {filepath}")

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
        
        # Update legend (grid stores lux)
        # Fixed legend ticks at absolute values up to color_scale_max
        _legend_cap = float(legend_max_input.value)
        if color_scale_max <= _legend_cap:
            _step = max(1, _legend_cap / 8)
            legend_vals_lux = np.arange(0, _legend_cap + 1, _step)
        else:
            legend_vals_lux = np.linspace(0, color_scale_max, 9)
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
        
        scale_label = f"(scale 0–{int(color_scale_max)} lx" + (", FIXED)" if color_scale_max <= _legend_cap else ", AUTO)")
        html_lines = ["<div style='font-family: sans-serif;'>",
                      f"<div style='font-weight:600;margin-bottom:2px;'>Intensity legend (lux)</div>",
                      f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} — peak {max_lux:.0f} lx</div>"]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, color_scale_max)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            # Convert lux to lumens using average cell area: Lumen = Lux × Area
            lumen_val = lux_val * avg_cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.0f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} lm/cell avg)</div></div>"
            )
        html_lines.append("</div>")
        # --- Uniformity metrics (combine all room wall grids) ---
        all_active = np.concatenate([g[g > 0] for g in grids.values() if g[g > 0].size > 0]) if grids else np.array([])
        room_uniformity_grid = all_active.reshape(-1) if all_active.size > 0 else np.zeros(1)
        # Build a pseudo-2D array so the helper works (it just needs grid > 0)
        uniformity_html = _compute_uniformity_html(room_uniformity_grid.reshape(1, -1))
        legend_html.content = "".join(html_lines) + uniformity_html
    
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
            # Pass lumens override for custom group
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
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
        
        # ── Apply global Z rotation to all LEDs ──
        global_rot_z_deg = global_rotation_z_slider.value
        if abs(global_rot_z_deg) > 0.01:
            g_rad = np.radians(global_rot_z_deg)
            cg, sg = np.cos(g_rad), np.sin(g_rad)
            Rg = np.array([[cg, -sg, 0],
                           [sg,  cg, 0],
                           [0,   0,  1]], dtype=float)
            for led in leds:
                led.position = Rg @ led.position
                led.direction = Rg @ led.direction
                if hasattr(led, 'row_direction') and led.row_direction is not None:
                    led.row_direction = Rg @ np.asarray(led.row_direction)
                if hasattr(led, 'square_normal') and led.square_normal is not None:
                    led.square_normal = Rg @ np.asarray(led.square_normal)

        # ── Apply global position offset to all LEDs ──
        _gp_x = global_pos_x_slider.value
        _gp_y = global_pos_y_slider.value
        _gp_z = global_pos_z_slider.value
        if abs(_gp_x) > 0.001 or abs(_gp_y) > 0.001 or abs(_gp_z) > 0.001:
            _gp_offset = np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            for led in leds:
                led.position = led.position + _gp_offset

        # ── Apply diffuser lens effect ──
        # A diffuser lens scatters light, widening the viewing angle toward
        # a near-Lambertian distribution.  The transmission loss is applied
        # separately in the simulation functions (lumens_per_led *= transmission).
        if diffuser_enable_chk.value:
            diff_angle = float(diffuser_angle_slider.value)
            for led in leds:
                # Widen to diffuser output angle (only if wider than native)
                led.viewing_angle = max(led.viewing_angle, diff_angle)

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

        # ── Apply global Z rotation to absorbers ──
        if abs(global_rot_z_deg) > 0.01:
            for a in absorbers:
                cx, cy, cz = a['center']
                new_cx = cg * cx - sg * cy
                new_cy = sg * cx + cg * cy
                a['center'] = (new_cx, new_cy, cz)
                # Compose global rotation with existing quaternion rotation
                if a.get('rotation') is not None:
                    qw0, qx0, qy0, qz0 = a['rotation']
                    # Quaternion for Rg around Z: (cos(a/2), 0, 0, sin(a/2))
                    half = g_rad / 2.0
                    gqw, gqx, gqy, gqz = np.cos(half), 0.0, 0.0, np.sin(half)
                    # q_new = q_global * q_existing
                    nw = gqw*qw0 - gqx*qx0 - gqy*qy0 - gqz*qz0
                    nx = gqw*qx0 + gqx*qw0 + gqy*qz0 - gqz*qy0
                    ny = gqw*qy0 - gqx*qz0 + gqy*qw0 + gqz*qx0
                    nz = gqw*qz0 + gqx*qy0 - gqy*qx0 + gqz*qw0
                    a['rotation'] = (nw, nx, ny, nz)
                else:
                    half = g_rad / 2.0
                    a['rotation'] = (np.cos(half), 0.0, 0.0, np.sin(half))

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
                # Use square_normal (original direction) for mesh if beam_tilt is applied
                square_dir = getattr(led, 'square_normal', led.direction)
                z_axis = square_dir / np.linalg.norm(square_dir)
                
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
            mesh_ref = stl_mesh_data[0]
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            # Apply global Z rotation to STL transform
            if abs(global_rot_z_deg) > 0.01:
                T_global = np.eye(4)
                T_global[:3, :3] = Rg
                transform = T_global @ transform
            stl_mesh_for_raytracing = {
                'vertices': mesh_ref.vertices,
                'faces': mesh_ref.faces,
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
                        
                        uniformity = float(ray_uniformity_slider.value)
                        n = _get_effective_n(led, uniformity)
                        
                        _vis_angle = getattr(led, 'ext_lens_angle', None) or led.viewing_angle
                        max_theta = np.radians(_vis_angle / 2.0)
                        cos_max = np.cos(max_theta)
                        cos_theta_sampled = 1.0 - u1 * (1.0 - cos_max)
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
            wall_x = wall_dist / 100.0 - 0.008  # Slightly in front of wall

            # In room mode, clamp the FOV rectangle to the front wall boundaries
            if room_mode_enable.value:
                max_half_w = room_side_dist.value / 100.0        # wall half-width in m
                max_half_h = room_top_bottom_dist.value / 100.0  # wall half-height in m
                half_w = min(half_w, max_half_w)
                half_h = min(half_h, max_half_h)
            
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
    wall_size_init = wall_view_size.value / 100.0  # cm -> m
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, wall_size_init, wall_size_init),
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

    # Callback to update wall position and size
    def update_wall():
        nonlocal wall_handle
        if room_mode_enable.value:
            # In room mode, don't update main wall
            return
        wall_dist = wall_dist_slider.value
        wall_size_m = wall_view_size.value / 100.0  # cm -> m
        try:
            wall_handle.remove()
        except (AttributeError, KeyError):
            pass
        wall_handle = server.scene.add_box(
            "/wall",
            dimensions=(0.01, wall_size_m, wall_size_m),
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
    diffuser_enable_chk.on_update(lambda _: update_scene())
    diffuser_angle_slider.on_update(lambda _: update_scene())
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
    def _on_global_rotation_change(_):
        """Handle global rotation slider: update LEDs immediately, debounce mesh update."""
        update_scene()
        # Schedule mesh update with debounce to avoid recomputing on every tick
        import threading
        if hasattr(_on_global_rotation_change, '_timer') and _on_global_rotation_change._timer is not None:
            _on_global_rotation_change._timer.cancel()
        def _deferred_mesh():
            try:
                update_stl_mesh(skip_lighting=True)
            except Exception:
                pass
        _on_global_rotation_change._timer = threading.Timer(0.3, _deferred_mesh)
        _on_global_rotation_change._timer.start()
    _on_global_rotation_change._timer = None
    global_rotation_z_slider.on_update(_on_global_rotation_change)
    global_pos_x_slider.on_update(_on_global_rotation_change)
    global_pos_y_slider.on_update(_on_global_rotation_change)
    global_pos_z_slider.on_update(_on_global_rotation_change)
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
    show_camera_fov.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_h.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_v.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
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
            wall_size_m = wall_view_size.value / 100.0
            wall_handle = server.scene.add_box(
                "/wall",
                dimensions=(0.01, wall_size_m, wall_size_m),
                color=(0.5, 0.5, 0.5),
                position=(wall_dist / 100.0, 0.0, 0.0),
            )
    
    room_mode_enable.on_update(on_room_mode_toggle)
    show_room_walls.on_update(lambda _: draw_room_walls())
    show_room_intensity.on_update(lambda _: (update_room_intensity_map() if (room_mode_enable.value and show_room_intensity.value) else draw_room_walls()) if room_mode_enable.value else None)
    room_front_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_side_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_top_bottom_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    show_back_wall.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    room_back_dist.on_update(lambda _: draw_room_walls() if room_mode_enable.value else None)
    wall_view_size.on_update(lambda _: (update_wall(), update_cell_area_info()))  # Update wall size and cell area

    def on_wall_dist_change(_):
        """Clear stale intensity map when wall distance changes."""
        # Remove old intensity visualization (values are no longer valid)
        for h in intensity_handles:
            try:
                h.remove()
            except KeyError:
                pass
        intensity_handles.clear()
        # Update legend to inform user that recalculation is needed
        legend_html.content = (
            "<div style='font-family: sans-serif;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
            "<div style='color:#F0AD4E;font-size:12px;'>⚠ Wall distance changed.<br>Click 'Update Intensity Map' to recalculate.</div>"
            "</div>"
        )
        update_wall()
        update_scene()

    wall_dist_slider.on_update(on_wall_dist_change)
    
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
    
    # Button for exporting lux matrix
    export_lux_matrix_button.on_click(lambda _: export_lux_matrix())
    
    # Button for running benchmark (multi-distance)
    run_benchmark_button.on_click(lambda _: run_benchmark())
    
    # Buttons for CSV pattern import
    csv_import_btn.on_click(lambda _: import_csv_pattern())
    csv_clear_btn.on_click(lambda _: clear_csv_pattern())

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
