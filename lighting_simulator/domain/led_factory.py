import numpy as np

from .geometry import default_row_direction, rotate_vector
from .led import LED, normalize_role, normalize_roles
from .placement import LEDPlacement



def create_leds(
    viewing_angle=120.0,
    default_lumens=100.0,
    custom_groups_configs=(),
    individual_leds_configs=(),
):
    """LED placements of the schema-v1 runtime configs (custom groups + individual LEDs).

    ``viewing_angle`` is the fallback beam angle for LEDs without their own.
    """
    leds = []
    led_index = 0

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
    led_roles = normalize_roles(config.get('led_roles'), 12)
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
                role=led_roles[index],
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
            role=normalize_role(config.get('role')),
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
    n_leds = int(config.get('num_leds', 0))
    led_roles = normalize_roles(config.get('led_roles'), n_leds)
    leds = []
    color = (1.0, 0.0, 1.0)

    for index in range(n_leds):
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
            role=led_roles[index],
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
