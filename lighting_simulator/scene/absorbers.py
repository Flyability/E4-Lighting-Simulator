"""Fixed box absorbers of the Elios 3 rig (arms / cage blocks that shadow the LEDs)."""

import numpy as np

from lighting_simulator.domain.geometry import quaternion_about_z, quaternion_multiply

# Half extents (cm) of every absorber block: 5 × 1.5 × 3 cm.
ABSORBER_HALF_SIZES = (5.0 / 2.0, 1.5 / 2.0, 3.0 / 2.0)
# Y offsets (cm) of the two front LED groups relative to the ring.
FRONT_GROUP_Y_OFFSETS = (6.5, -6.5)
FRONT_ABSORBER_Y_OFFSETS = (-4.2, 4.2)


def _front_absorber_base(index, radius, circle_center_x, front_angle_deg=0.0):
    angle = np.radians(front_angle_deg if index == 0 else -front_angle_deg)
    gx = circle_center_x + radius * np.cos(angle)
    gy = radius * np.sin(angle) + FRONT_GROUP_Y_OFFSETS[index]
    radial = np.array((gx - circle_center_x, gy, 0.0), dtype=float)
    radial_unit = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > 0 else np.array((1.0, 0.0, 0.0))
    return (
        gx + radial_unit[0] * 5.0 - 5.0,
        gy + radial_unit[1] * 5.0 + FRONT_ABSORBER_Y_OFFSETS[index],
        0.0,
    )


def build_elios_absorbers(absorber_cfg, radius, circle_center_x, front_angle_deg=0.0):
    """Absorber dicts from a config section shaped like ``get_current_config()['absorbers']``.

    ``absorber_cfg = {'enabled': bool, 'abs0': {x,y,z}, 'abs1': {x,y,z},
    'abs2': {x,y,z,rot_z}, 'abs3': {x,y,z,rot_z}}`` (offsets in cm / degrees).
    """
    if not absorber_cfg or not absorber_cfg.get('enabled', False):
        return []
    absorbers = []
    for i in (0, 1):
        off = absorber_cfg.get(f'abs{i}', {})
        bx, by, bz = _front_absorber_base(i, radius, circle_center_x, front_angle_deg)
        absorbers.append({
            'center': (bx + off.get('x', 0.0), by + off.get('y', 0.0), bz + off.get('z', 0.0)),
            'half_sizes': ABSORBER_HALF_SIZES,
            'rotation': None,
        })
    for i in (2, 3):
        off = absorber_cfg.get(f'abs{i}', {})
        absorbers.append({
            'center': (off.get('x', 0.0), off.get('y', 0.0), off.get('z', 0.0)),
            'half_sizes': ABSORBER_HALF_SIZES,
            'rotation': quaternion_about_z(off.get('rot_z', 0.0)),
        })
    return absorbers


def rotate_absorbers_z(absorbers, angle_deg):
    """Rotate absorber centres and orientations about the world Z axis (in place)."""
    if abs(angle_deg) <= 0.01:
        return absorbers
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    q_global = quaternion_about_z(angle_deg)
    for a in absorbers:
        cx, cy, cz = a['center']
        a['center'] = (c * cx - s * cy, s * cx + c * cy, cz)
        a['rotation'] = quaternion_multiply(q_global, a['rotation']) if a.get('rotation') is not None else q_global
    return absorbers
