import numpy as np

from .geometry import default_row_direction, rotate_vector
from .led import LED
from .placement import LEDPlacement



def create_leds(
    front_angle_deg,
    side_angle_deg,
    viewing_angle,
    radius,
    circle_center_x,
    default_lumens=100.0,
    group_rotations=(0.0, 0.0, 0.0, 0.0),
    group_rotations_y=(0.0, 0.0, 0.0, 0.0),
    row_enabled=None,
    led_states=None,
    group_offsets=(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ),
    custom_groups_configs=(),
    individual_leds_configs=(),
    create_base_groups=True,
):
    """
    TEMPORARY: This function recreates Elios 3' LED configuration and allows for custom groups and individual LEDs to be added.
    It is intended for testing and visualization purposes. It should be removed or renamed for this specific case.
    
    FULL OF MAGIC NUMBERS AND HARD-CODED VALUES.  USE WITH CAUTION.
    
    Create 4 LED groups; each group contains 3 LEDs spaced by 3 mm.
    Optionally add multiple custom groups at specified positions.
    
    Args:
        create_base_groups: If False, skip creating the 4 base LED groups (front+/-, side+/-)
    """
    
    leds = []
    led_index = 0  # Track global LED index
    
    # Only create base groups if requested
    if create_base_groups:
        base_leds = _create_base_leds(
            front_angle_deg=front_angle_deg,
            side_angle_deg=side_angle_deg,
            viewing_angle=viewing_angle,
            radius=radius,
            circle_center_x=circle_center_x,
            group_rotations=group_rotations,
            group_rotations_y=group_rotations_y,
            row_enabled=row_enabled,
            led_states=led_states,
            group_offsets=group_offsets,
            default_lumens=default_lumens,
            start_index=led_index,
        )
        leds.extend(base_leds)
        led_index += len(base_leds)
        

    # Add custom groups if any are enabled
    for custom_group_config in custom_groups_configs:
        if not custom_group_config.get('enabled', False):
            continue
        
        # Check if this is a dynamic group (from individual LEDs)
        is_dynamic = 'num_leds' in custom_group_config and 'led_positions' in custom_group_config
        
        if is_dynamic:
            created = _create_dynamic_group_leds(
                custom_group_config,
                viewing_angle=viewing_angle,
                default_lumens=default_lumens,
                start_index=led_index,
            )
            leds.extend(created)
            led_index += len(created)
        else:
            created = _create_standard_group_leds(
                custom_group_config,
                viewing_angle=viewing_angle,
                default_lumens=default_lumens,
                start_index=led_index,
            )
            leds.extend(created)
            led_index += len(created)

    # Add individual LEDs
    for individual_led_config in individual_leds_configs:
        created = _create_individual_leds(
            [individual_led_config],
            viewing_angle=viewing_angle,
            default_lumens=default_lumens,
            start_index=led_index,
        )
        leds.extend(created)
        led_index += len(created)

    return leds


def _placement(led, index, *, row_direction=None, square_normal=None,
               owner=None, is_custom=False, is_dynamic_group=False,
               is_individual=False):
    return LEDPlacement(
        led=led,
        index=index,
        row_direction=row_direction,
        square_normal=square_normal,
        owner=owner,
        is_custom=is_custom,
        is_dynamic_group=is_dynamic_group,
        is_individual=is_individual,
    )

def _create_standard_group_leds(config, viewing_angle, default_lumens, start_index):
    position = np.asarray(config.get('position', (0.0, 0.0, 0.0)), dtype=float)
    roll = np.radians(config.get('rotation_x', 0.0))
    pitch = np.radians(config.get('rotation_y', 0.0))
    yaw = np.radians(config.get('rotation_z', 0.0))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    roll_matrix = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    tangent_rotation = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    yaw_rotation = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation = yaw_rotation @ tangent_rotation
    rolled_z = roll_matrix @ np.array([0.0, 0.0, 1.0])
    row_direction = rotation @ roll_matrix @ np.array([0.0, -1.0, 0.0])
    radial = rotation @ np.array([1.0, 0.0, 0.0])
    led_states = config.get('led_states', [True] * 12)
    row_enabled = config.get('row_enabled', [True] * 4)
    inclinations = [90, 30, -30, -90]
    row_offsets = [-0.85, -0.55, 0.55, 0.85]
    leds = []
    for row_index, inclination in enumerate(inclinations):
        row_center = position + rolled_z * row_offsets[row_index]
        direction = np.cos(np.radians(inclination)) * radial
        direction -= np.sin(np.radians(inclination)) * (rotation @ rolled_z)
        direction /= np.linalg.norm(direction)
        if row_index in (0, 3):
            row_center -= radial * 0.5
        for led_in_row, offset in enumerate((-0.8, 0.0, 0.8)):
            index = row_index * 3 + led_in_row
            enabled = (
                row_enabled[row_index] if row_index < len(row_enabled) else True
            ) and (led_states[index] if index < len(led_states) else True)
            led = LED(
                width=0.5,
                viewing_angle=viewing_angle,
                position=tuple(row_center + row_direction * offset),
                direction=tuple(direction),
                color=(1.0, 0.0, 1.0),
                lumens=_resolve_lumens(config, index, default_lumens),
                enabled=enabled,
            )
            leds.append(_placement(
                led,
                start_index + len(leds),
                row_direction=row_direction,
                owner=config.get('owner'),
                is_custom=True,
            ))
    return leds


def _create_individual_leds(configs, viewing_angle, default_lumens, start_index):
    leds = []
    for config in configs:
        if not config.get('enabled', True):
            continue
        rx, ry, rz = np.radians([
            config.get('rot_x', 0.0),
            config.get('rot_y', 0.0),
            config.get('rot_z', 0.0),
        ])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rotation = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rotation = rotation @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rotation = rotation @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        direction = rotation @ np.array([1.0, 0.0, 0.0])
        row_direction = default_row_direction(direction)
        led = LED(
            position=(config.get('pos_x', 0.0), config.get('pos_y', 0.0), config.get('pos_z', 0.0)),
            direction=tuple(direction),
            lumens=_resolve_lumens(config, 0, default_lumens),
            color=(0.0, 1.0, 1.0),
            width=config.get('size', 0.5),
            viewing_angle=config.get('viewing_angle', viewing_angle),
            enabled=config.get('led_on', True),
        )
        placement = _placement(
            led,
            start_index + len(leds),
            row_direction=row_direction,
            square_normal=direction,
            owner=config.get('owner'),
            is_individual=True,
        )
        lens_angle = config.get('ext_lens_angle')
        if lens_angle is not None:
            placement.ext_lens_angle = lens_angle
            placement.lumens *= config.get('ext_lens_efficiency', 1.0)
        leds.append(placement)
    return leds


def _create_dynamic_group_leds(config, viewing_angle, default_lumens, start_index):
    positions = config.get('led_positions') or []
    rotations = config.get('led_rotations') or []
    sizes = config.get('led_sizes') or []
    viewing_angles = config.get('led_viewing_angles') or []
    row_directions = config.get('led_row_directions') or []
    beam_tilts = config.get('led_beam_tilts') or []
    led_states = config.get('led_states') or []
    leds = []
    color = (1.0, 0.0, 1.0)

    for index in range(config.get('num_leds', 0)):
        if index >= len(positions):
            break

        direction = np.asarray(
            rotations[index] if index < len(rotations) else (1.0, 0.0, 0.0),
            dtype=float,
        )
        if np.linalg.norm(direction) < 1e-10:
            direction = np.array([1.0, 0.0, 0.0])

        if index < len(row_directions):
            row_direction = np.asarray(row_directions[index], dtype=float)
        else:
            row_direction = default_row_direction(direction)

        beam_tilt = beam_tilts[index] if index < len(beam_tilts) else 0.0
        if abs(beam_tilt) > 0.01:
            beam_direction = rotate_vector(direction, row_direction, beam_tilt)
        else:
            beam_direction = direction

        led = LED(
            width=sizes[index] if index < len(sizes) else 0.5,
            viewing_angle=(
                viewing_angles[index]
                if index < len(viewing_angles)
                else viewing_angle
            ),
            position=positions[index],
            direction=tuple(beam_direction),
            color=color,
            lumens=_resolve_lumens(config, index, default_lumens),
            enabled=led_states[index] if index < len(led_states) else True,
        )
        leds.append(_placement(
            led,
            start_index + len(leds),
            row_direction=row_direction,
            square_normal=direction,
            owner=config.get('owner'),
            is_custom=True,
            is_dynamic_group=True,
        ))

    return leds


def _resolve_lumens(config, index, default_lumens):
    led_lumens = config.get("led_lumens", [])

    if index < len(led_lumens) and led_lumens[index] is not None:
        return float(led_lumens[index])

    override = config.get("lumens_override")
    if override is not None:
        return float(override)

    return float(default_lumens)

def _create_base_leds(
    front_angle_deg,
    side_angle_deg,
    viewing_angle,
    radius,
    circle_center_x,
    group_rotations,
    group_rotations_y,
    row_enabled,
    led_states,
    group_offsets,
    default_lumens,
    start_index,
):
    angles_deg = [
        front_angle_deg,
        -front_angle_deg,
        side_angle_deg,
        -side_angle_deg,
    ]
    colors = [
        (1.0, 0.2, 0.2),
        (0.2, 1.0, 0.2),
        (0.2, 0.2, 1.0),
        (1.0, 1.0, 0.2),
    ]
    leds = []
    led_index = start_index
    front_x_positions = {}  # Store X positions of front groups for side groups alignment
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
                    lumens=default_lumens,
                    enabled=is_row_enabled and is_led_enabled,
                )
                leds.append(_placement(
                    led,
                    led_index,
                    row_direction=rotated_row_dir.copy(),
                    owner=('base_group', i),
                ))
                led_index += 1

    return leds

