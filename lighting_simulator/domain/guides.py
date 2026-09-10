"""Circular construction guides for dynamic LED panels.

A *guide* is a cylinder fitted through the LED rows of a panel. When enabled,
the panel can be slid ("orbited") around the cylinder axis by ``theta_deg``.

Group dicts may hold either plain numbers or GUI handles exposing ``.value``
for the pose keys (``pos_x``, ``pos_y``, ``pos_z``, ``rot_roll``,
``rot_tilt_ud``, ``rot_tilt_lr``); :func:`_val` normalises both.
"""

import numpy as np

from .geometry import (
    as_vec3,
    circle_polyline,
    euler_xyz_matrix,
    fit_circle_npoints,
    mirror_xz_vec,
    rodrigues_rotation,
)

GUIDE_NORMAL_SPREAD_WARN_DEG = 8.0
GUIDE_RADIUS_REL_WARN = 0.20


def _val(x, default=0.0):
    """Unwrap a GUI handle (``.value``) or return the scalar itself."""
    if x is None:
        return default
    return float(getattr(x, "value", x))


def fit_panel_cylinder_from_rows(positions, led_rows):
    """Fit one circle per LED row and reconcile a common cylinder axis.

    Returns (guide_dict, error_string). error_string is None on success.
    """
    positions = list(positions)
    circles = []
    for row in led_rows or []:
        pts = [as_vec3(positions[idx]) for idx in row
               if isinstance(idx, int) and 0 <= idx < len(positions)]
        if len(pts) < 3:
            continue
        fit = fit_circle_npoints(pts)
        if fit is None:
            continue
        center, radius, normal = fit
        circles.append({'center': center, 'radius': radius, 'normal': normal})

    if not circles:
        return None, "Need at least one LED row with 3 non-collinear points."

    ref = circles[0]['normal']
    for circ in circles[1:]:
        if np.dot(circ['normal'], ref) < 0.0:
            circ['normal'] = -circ['normal']

    axis = np.mean([circ['normal'] for circ in circles], axis=0)
    axis_n = np.linalg.norm(axis)
    if axis_n < 1e-12:
        return None, "Could not reconcile a common duct axis from the row circles."
    axis = axis / axis_n

    dots = [float(np.clip(np.dot(circ['normal'], axis), -1.0, 1.0)) for circ in circles]
    spread = max(float(np.degrees(np.arccos(abs(d)))) for d in dots)
    radii = [circ['radius'] for circ in circles]
    r_mean = float(np.mean(radii))
    r_span = (max(radii) - min(radii)) / r_mean if r_mean > 1e-9 else 0.0
    origin = np.mean([circ['center'] for circ in circles], axis=0)

    warning = None
    if spread > GUIDE_NORMAL_SPREAD_WARN_DEG or r_span > GUIDE_RADIUS_REL_WARN:
        warning = (
            f"Rows are not a tight cylinder (axis spread {spread:.1f}°, "
            f"radius span {100.0 * r_span:.0f}%). Sliding still uses one common axis."
        )

    guide = {
        'type': 'cylinder',
        'enabled': True,
        'origin': [float(v) for v in origin],
        'axis': [float(v) for v in axis],
        'theta_deg': 0.0,
        'circles': [
            {
                'center': [float(v) for v in c['center']],
                'radius': float(c['radius']),
                'normal': [float(v) for v in c['normal']],
            }
            for c in circles
        ],
        'normal_spread_deg': spread,
        'warning': warning,
    }
    return guide, None


def serialize_guide(guide):
    """JSON-safe copy of a guide dict, or None."""
    if not guide or not isinstance(guide, dict):
        return None
    circles = [
        {
            'center': [float(x) for x in circ.get('center', (0, 0, 0))],
            'radius': float(circ.get('radius', 0.0)),
            'normal': [float(x) for x in circ.get('normal', (0, 0, 1))],
        }
        for circ in guide.get('circles') or []
    ]
    out = {
        'type': guide.get('type', 'cylinder'),
        'enabled': bool(guide.get('enabled', False)),
        'origin': [float(x) for x in guide.get('origin', (0, 0, 0))],
        'axis': [float(x) for x in guide.get('axis', (0, 0, 1))],
        'theta_deg': float(guide.get('theta_deg', 0.0)),
        'circles': circles,
        'normal_spread_deg': float(guide.get('normal_spread_deg', 0.0)),
    }
    if guide.get('warning'):
        out['warning'] = str(guide['warning'])
    return out


def mirror_guide_xz(guide):
    """Reflect a construction guide across the XZ plane and negate theta."""
    g = serialize_guide(guide)
    if g is None:
        return None
    g['origin'] = list(mirror_xz_vec(g['origin']))
    ax = as_vec3(g['axis'])
    ax[1] = -ax[1]
    n = np.linalg.norm(ax)
    g['axis'] = [float(v) for v in ax / n] if n > 1e-12 else [0.0, 0.0, 1.0]
    g['theta_deg'] = -float(g.get('theta_deg', 0.0))
    mirrored = []
    for circ in g.get('circles') or []:
        nrm = as_vec3(circ['normal'])
        nrm[1] = -nrm[1]
        nn = np.linalg.norm(nrm)
        if nn > 1e-12:
            nrm = nrm / nn
        mirrored.append({
            'center': list(mirror_xz_vec(circ['center'])),
            'radius': float(circ['radius']),
            'normal': [float(v) for v in nrm],
        })
    g['circles'] = mirrored
    return g


def guide_is_enabled(group):
    g = group.get('guide') if isinstance(group, dict) else None
    return bool(isinstance(g, dict) and g.get('enabled'))


def group_euler_matrix(group):
    """Extrinsic X-Y-Z matrix from a group's roll / tilt-ud / tilt-lr pose."""
    roll = _val(group.get('rot_roll'))
    pitch = _val(group.get('rot_tilt_ud'))
    yaw = _val(group.get('rot_tilt_lr'))
    return euler_xyz_matrix(roll, pitch, yaw)


def group_position_offset(group):
    return np.array([
        _val(group.get('pos_x')),
        _val(group.get('pos_y')),
        _val(group.get('pos_z')),
    ], dtype=float)


def dynamic_group_world_geometry(group):
    """World-space LED positions, directions and row directions of a dynamic group.

    Applies extrinsic Euler + group offset, then the circular-guide orbit if
    enabled. Lazily caches the rest pose in ``original_led_*`` keys.
    """
    position_offset = group_position_offset(group)
    if not group.get('original_led_positions') and group.get('led_positions'):
        group['original_led_positions'] = [
            tuple(np.asarray(p, dtype=float) - position_offset)
            for p in group.get('led_positions', [])
        ]
    if not group.get('original_led_rotations') and group.get('led_rotations'):
        group['original_led_rotations'] = [tuple(r) for r in group.get('led_rotations', [])]
    if not group.get('original_led_row_directions') and group.get('led_row_directions'):
        group['original_led_row_directions'] = [tuple(rd) for rd in group.get('led_row_directions', [])]

    R_total = group_euler_matrix(group)
    original_positions = group.get('original_led_positions') or group.get('led_positions') or []
    original_rotations = group.get('original_led_rotations') or group.get('led_rotations') or []
    original_row_dirs = group.get('original_led_row_directions') or group.get('led_row_directions') or []

    positions = [tuple(R_total @ as_vec3(p) + position_offset) for p in original_positions]
    directions = [tuple(R_total @ as_vec3(d)) for d in original_rotations]
    row_dirs = [tuple(R_total @ as_vec3(rd)) for rd in original_row_dirs] if original_row_dirs else []

    guide = group.get('guide')
    if isinstance(guide, dict) and guide.get('enabled'):
        theta = float(guide.get('theta_deg', 0.0))
        if abs(theta) > 1e-12:
            R_g = rodrigues_rotation(guide.get('axis', (0.0, 0.0, 1.0)), np.radians(theta))
            O = as_vec3(guide.get('origin', (0.0, 0.0, 0.0)))
            positions = [tuple(O + R_g @ (as_vec3(p) - O)) for p in positions]
            directions = [tuple(R_g @ as_vec3(d)) for d in directions]
            row_dirs = [tuple(R_g @ as_vec3(rd)) for rd in row_dirs]
    return positions, directions, row_dirs


def enable_circular_guide(group):
    """Fit a cylinder from the group's current rest-pose LED rows. Returns (ok, error)."""
    if not group.get('is_dynamic', False):
        return False, "Anchor requires a dynamic panel with LED rows."
    rows = group.get('led_rows')
    if not rows:
        return False, "This panel has no LED rows to fit a circle to."
    prev = group.get('guide')
    group['guide'] = None
    try:
        positions, _, _ = dynamic_group_world_geometry(group)
    finally:
        group['guide'] = prev
    if len(positions) < 3:
        return False, "Need at least 3 LED positions to fit a guide."
    guide, err = fit_panel_cylinder_from_rows(positions, rows)
    if err:
        return False, err
    group['guide'] = guide
    group['guide_error'] = None
    return True, None


def bake_and_disable_guide(group):
    """Fold the current slide into ``original_led_*`` so disabling does not jump."""
    if not guide_is_enabled(group):
        group['guide'] = None
        group['guide_error'] = None
        return
    positions, directions, row_dirs = dynamic_group_world_geometry(group)
    R_inv = group_euler_matrix(group).T
    offset = group_position_offset(group)
    orig_pos = [tuple(float(x) for x in (R_inv @ (as_vec3(p) - offset))) for p in positions]
    orig_dir = [tuple(float(x) for x in (R_inv @ as_vec3(d))) for d in directions]
    orig_row = [tuple(float(x) for x in (R_inv @ as_vec3(rd))) for rd in row_dirs]
    group['original_led_positions'] = orig_pos
    group['original_led_rotations'] = orig_dir
    group['led_positions'] = list(orig_pos)
    group['led_rotations'] = list(orig_dir)
    if orig_row:
        group['original_led_row_directions'] = orig_row
        group['led_row_directions'] = list(orig_row)
    apply_rot = group.get('apply_rotation')
    if callable(apply_rot):
        apply_rot()
    group['guide'] = None
    group['guide_error'] = None


def restore_group_guide(group, group_cfg):
    """Attach a saved construction guide (if any) onto a runtime group."""
    group['guide'] = serialize_guide(group_cfg.get('guide') if isinstance(group_cfg, dict) else None)
    group['guide_error'] = None


def circle_line_segments_m(center_cm, radius_cm, normal, n_seg=64):
    """Polyline segments for a 3D circle given in cm, returned in metres."""
    return circle_polyline(center_cm, radius_cm, normal, n_seg) / 100.0
