"""Room geometry: wall specifications, grid mapping and cell centres.

Wall grids are indexed ``[row, col]`` as follows:
front/back ``[Z, Y]``, left/right ``[Z, X]``, top/bottom ``[Y, X]``.
"""

import numpy as np

WALL_NAMES = ('front', 'left', 'right', 'top', 'bottom')
WALL_IDS = {'front': 0, 'left': 1, 'right': 2, 'top': 3, 'bottom': 4, 'back': 5}

# Inward normals (pointing INTO the room) used for diffuse bounces.
WALL_INWARD_NORMALS = {
    'front':  np.array([-1.0,  0.0,  0.0]),
    'back':   np.array([ 1.0,  0.0,  0.0]),
    'left':   np.array([ 0.0,  1.0,  0.0]),
    'right':  np.array([ 0.0, -1.0,  0.0]),
    'top':    np.array([ 0.0,  0.0, -1.0]),
    'bottom': np.array([ 0.0,  0.0,  1.0]),
}

# Side/top/bottom walls extend behind the LED rig by this factor of the base depth.
LATERAL_DEPTH_FACTOR = 2.5


def build_wall_specs(front_dist, side_dist, top_bottom_dist, grid_size,
                     led_x_center=-35.0, back_dist=None):
    """Return ``{wall_name: spec}`` describing each wall's size and grid.

    Every wall gets a ``grid_size × grid_size`` grid whose cells adapt to the
    wall's physical dimensions (cm).
    """
    wall_width_x = front_dist + abs(led_x_center)
    wall_width_y = 2 * side_dist
    wall_height_z = 2 * top_bottom_dist
    lateral_depth_x = wall_width_x * LATERAL_DEPTH_FACTOR
    extended_x_min = front_dist - lateral_depth_x

    front_spec = {
        'size_y': wall_width_y, 'size_z': wall_height_z, 'dims': ('y', 'z'),
        'grid_y': grid_size, 'grid_z': grid_size,
    }
    side_spec = {
        'size_x': lateral_depth_x, 'size_z': wall_height_z, 'dims': ('x', 'z'),
        'x_min': extended_x_min, 'grid_x': grid_size, 'grid_z': grid_size,
    }
    top_spec = {
        'size_x': lateral_depth_x, 'size_y': wall_width_y, 'dims': ('x', 'y'),
        'x_min': extended_x_min, 'grid_x': grid_size, 'grid_y': grid_size,
    }
    specs = {
        'front': dict(front_spec),
        'left': dict(side_spec),
        'right': dict(side_spec),
        'top': dict(top_spec),
        'bottom': dict(top_spec),
    }
    if back_dist is not None:
        specs['back'] = dict(front_spec)
    return specs


def wall_grid_shape(spec, wall_name):
    if wall_name in ('front', 'back'):
        return (spec['grid_z'], spec['grid_y'])
    if wall_name in ('left', 'right'):
        return (spec['grid_z'], spec['grid_x'])
    return (spec['grid_y'], spec['grid_x'])


def empty_wall_grids(wall_specs):
    return {name: np.zeros(wall_grid_shape(spec, name)) for name, spec in wall_specs.items()}


def wall_cell_areas_m2(wall_specs):
    """Per-wall cell area in m²."""
    areas = {}
    for name, spec in wall_specs.items():
        if name in ('front', 'back'):
            cw = spec['size_y'] / spec['grid_y']
            ch = spec['size_z'] / spec['grid_z']
        elif name in ('left', 'right'):
            cw = spec['size_x'] / spec['grid_x']
            ch = spec['size_z'] / spec['grid_z']
        else:
            cw = spec['size_x'] / spec['grid_x']
            ch = spec['size_y'] / spec['grid_y']
        areas[name] = (cw * ch) / 10000.0
    return areas


def wall_grid_indices(wall_name, spec, coord1, coord2):
    """Map wall-plane hit coordinates to (row, col) grid indices (unclamped).

    ``coord1``/``coord2`` follow the ray tracers' convention:
    front/back -> (y, z); left/right -> (x, z); top/bottom -> (x, y).
    Works on scalars or arrays.
    """
    if wall_name in ('front', 'back'):
        col = (coord1 + spec['size_y'] / 2) / (spec['size_y'] / spec['grid_y'])
        row = (coord2 + spec['size_z'] / 2) / (spec['size_z'] / spec['grid_z'])
    elif wall_name in ('left', 'right'):
        col = (coord1 - spec['x_min']) / (spec['size_x'] / spec['grid_x'])
        row = (coord2 + spec['size_z'] / 2) / (spec['size_z'] / spec['grid_z'])
    else:
        col = (coord1 - spec['x_min']) / (spec['size_x'] / spec['grid_x'])
        row = (coord2 + spec['size_y'] / 2) / (spec['size_y'] / spec['grid_y'])
    return row, col


def room_wall_cell_centers(wall_name, spec, front_dist, side_dist, top_bottom_dist, back_dist=None):
    """World cell-centre coordinates (cm) for a room-wall grid, shape ``grid + (3,)``."""
    if wall_name in ('front', 'back'):
        size_y, size_z = spec['size_y'], spec['size_z']
        grid_y, grid_z = spec['grid_y'], spec['grid_z']
        z_c = -size_z / 2.0 + (np.arange(grid_z) + 0.5) * (size_z / grid_z)
        y_c = -size_y / 2.0 + (np.arange(grid_y) + 0.5) * (size_y / grid_y)
        zz, yy = np.meshgrid(z_c, y_c, indexing='ij')
        pts = np.empty(zz.shape + (3,), dtype=float)
        pts[..., 0] = front_dist if wall_name == 'front' else -float(back_dist or 0.0)
        pts[..., 1] = yy
        pts[..., 2] = zz
        return pts

    if wall_name in ('left', 'right'):
        size_x, size_z = spec['size_x'], spec['size_z']
        grid_x, grid_z = spec['grid_x'], spec['grid_z']
        z_c = -size_z / 2.0 + (np.arange(grid_z) + 0.5) * (size_z / grid_z)
        x_c = spec['x_min'] + (np.arange(grid_x) + 0.5) * (size_x / grid_x)
        zz, xx = np.meshgrid(z_c, x_c, indexing='ij')
        pts = np.empty(zz.shape + (3,), dtype=float)
        pts[..., 0] = xx
        pts[..., 1] = -side_dist if wall_name == 'left' else side_dist
        pts[..., 2] = zz
        return pts

    size_x, size_y = spec['size_x'], spec['size_y']
    grid_x, grid_y = spec['grid_x'], spec['grid_y']
    y_c = -size_y / 2.0 + (np.arange(grid_y) + 0.5) * (size_y / grid_y)
    x_c = spec['x_min'] + (np.arange(grid_x) + 0.5) * (size_x / grid_x)
    yy, xx = np.meshgrid(y_c, x_c, indexing='ij')
    pts = np.empty(yy.shape + (3,), dtype=float)
    pts[..., 0] = xx
    pts[..., 1] = yy
    pts[..., 2] = top_bottom_dist if wall_name == 'top' else -top_bottom_dist
    return pts


def wall_grid_cell_centers_cm(grid_shape, wall_size_cm, wall_dist):
    """Cell centres (cm) of the single-wall grid at ``x = wall_dist``; rows = Z, cols = Y."""
    gz, gy = grid_shape
    cell_cm = wall_size_cm / gy
    half = wall_size_cm / 2.0
    z_centers = -half + (np.arange(gz) + 0.5) * cell_cm
    y_centers = -half + (np.arange(gy) + 0.5) * cell_cm
    zz, yy = np.meshgrid(z_centers, y_centers, indexing='ij')
    pts = np.empty(tuple(grid_shape) + (3,), dtype=float)
    pts[..., 0] = wall_dist
    pts[..., 1] = yy
    pts[..., 2] = zz
    return pts


def nearest_wall_hits(origins, dirs, front_dist, side_dist, top_bottom_dist, back_dist=None):
    """Vectorised first-hit of rays against the room walls.

    Returns (wall_id (N,) int, -1 for miss; coord1 (N,); coord2 (N,); t (N,)).
    """
    n = origins.shape[0]
    ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    best_t = np.full(n, 1e30)
    best_wall = np.full(n, -1, dtype=int)
    best_c1 = np.zeros(n)
    best_c2 = np.zeros(n)

    def _consider(mask, wall_id, plane_value, o_axis, d_axis, c1_o, c1_d, c2_o, c2_d):
        if not np.any(mask):
            return
        t = (plane_value - o_axis[mask]) / d_axis[mask]
        better = mask.copy()
        better[mask] &= (t > 0) & (t < best_t[mask])
        sel = better[mask]
        best_t[better] = t[sel]
        best_wall[better] = wall_id
        best_c1[better] = c1_o[mask][sel] + c1_d[mask][sel] * t[sel]
        best_c2[better] = c2_o[mask][sel] + c2_d[mask][sel] * t[sel]

    _consider(dx > 0, WALL_IDS['front'], front_dist, ox, dx, oy, dy, oz, dz)
    _consider(dy < 0, WALL_IDS['left'], -side_dist, oy, dy, ox, dx, oz, dz)
    _consider(dy > 0, WALL_IDS['right'], side_dist, oy, dy, ox, dx, oz, dz)
    _consider(dz > 0, WALL_IDS['top'], top_bottom_dist, oz, dz, ox, dx, oy, dy)
    _consider(dz < 0, WALL_IDS['bottom'], -top_bottom_dist, oz, dz, ox, dx, oy, dy)
    if back_dist is not None:
        _consider(dx < 0, WALL_IDS['back'], -back_dist, ox, dx, oy, dy, oz, dz)
    return best_wall, best_c1, best_c2, best_t
