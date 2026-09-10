"""STL occluder placement: pose → 4x4 transform and ray-tracer payload."""

import numpy as np


def _rot4_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot4_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def _rot4_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _finite(value, default=0.0):
    value = float(value)
    return value if np.isfinite(value) else default


def stl_transform(scale, rotation_deg, position_cm):
    """4x4 matrix: uniform scale, then extrinsic X→Y→Z rotation, then translation."""
    transform = np.eye(4)
    scale = _finite(scale, 1.0)
    if scale > 0:
        transform[:3, :3] *= scale
    rx, ry, rz = (_finite(v) for v in rotation_deg)
    if rx != 0:
        transform = _rot4_x(np.radians(rx)) @ transform
    if ry != 0:
        transform = _rot4_y(np.radians(ry)) @ transform
    if rz != 0:
        transform = _rot4_z(np.radians(rz)) @ transform
    transform[:3, 3] = [_finite(v) for v in position_cm]
    return transform


def global_z_rotation_4x4(angle_deg):
    return _rot4_z(np.radians(float(angle_deg)))


def stl_mesh_data(mesh, transform):
    """Payload consumed by the ray tracers for a loaded ``trimesh.Trimesh``."""
    return {'vertices': mesh.vertices, 'faces': mesh.faces, 'transform': transform}
