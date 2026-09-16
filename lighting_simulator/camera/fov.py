"""Camera field-of-view geometry (pinhole main camera and VD66GY fisheye VIO cameras).

World convention: optical axis +X, +Y horizontal (left), +Z up. Pitch > 0 looks
up, yaw > 0 rotates toward +Y. All distances in centimetres unless noted.
"""

import numpy as np

# VD66GY native resolution (short × long). Equidistant fisheye: r = f·θ.
VD66GY_SHORT_PX = 1124
VD66GY_LONG_PX = 1364

# For an axis-aligned plane with fixed world axis a, the patch spans the two
# remaining axes (u, v) in ascending axis order.
_PLANE_UV_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def vio_hfov_vfov_deg(long_fov_deg, landscape):
    """Return (HFOV, VFOV) in degrees for a VD66GY given FOV on the long side."""
    short_fov = float(long_fov_deg) * VD66GY_SHORT_PX / VD66GY_LONG_PX
    if landscape:
        return float(long_fov_deg), short_fov
    return short_fov, float(long_fov_deg)


def vio_optical_axis(pitch_deg, yaw_deg):
    """World-space optical-axis unit vector (camera +X after pitch then yaw)."""
    p = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    axis = np.array([np.cos(p), 0.0, np.sin(p)])
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([cy * axis[0] - sy * axis[1], sy * axis[0] + cy * axis[1], axis[2]])


def fisheye_intrinsics(hfov_deg, vfov_deg):
    """Equidistant fisheye: f in px/rad, sensor half-extents, image-circle radius."""
    long_fov = max(float(hfov_deg), float(vfov_deg))
    long_half_px = VD66GY_LONG_PX / 2.0
    f = long_half_px / np.radians(long_fov / 2.0)
    half_u = f * np.radians(float(hfov_deg) / 2.0)
    half_v = f * np.radians(float(vfov_deg) / 2.0)
    return f, half_u, half_v, long_half_px


def camera_rotations(pitch_deg, yaw_deg):
    """Return R (cam→world) and R_inv (world→cam). Pitch then yaw."""
    p = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(yaw), np.sin(yaw)
    r_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    r = r_yaw @ r_pitch
    return r, r.T


def compute_fisheye_fov_wall_segments(
    cam_pos, pitch_deg, yaw_deg, hfov_deg, vfov_deg,
    wall_dist, wall_half_y, wall_half_z, n_samples=720,
):
    """Project an equidistant-fisheye FOV border onto the wall plane x = wall_dist.

    Returns an array of shape (K, 2, 3) with wall hit points in cm, or (0, 2, 3).
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    f, half_u, half_v, r_circle = fisheye_intrinsics(hfov_deg, vfov_deg)

    n_side = max(2, int(n_samples) // 4)
    u = np.concatenate([
        np.linspace(-half_u, half_u, n_side, endpoint=False),
        np.full(n_side, half_u),
        np.linspace(half_u, -half_u, n_side, endpoint=False),
        np.full(n_side, -half_u),
    ])
    v = np.concatenate([
        np.full(n_side, -half_v),
        np.linspace(-half_v, half_v, n_side, endpoint=False),
        np.full(n_side, half_v),
        np.linspace(half_v, -half_v, n_side, endpoint=False),
    ])

    r = np.hypot(u, v)
    scale = np.minimum(1.0, r_circle / np.maximum(r, 1e-12))
    u = u * scale
    v = v * scale
    r = np.hypot(u, v)
    theta = r / f
    st, ct = np.sin(theta), np.cos(theta)
    inv_r = 1.0 / np.maximum(r, 1e-12)
    dir_cam = np.stack([ct, -st * u * inv_r, st * v * inv_r], axis=1)

    r_cam_to_world, _ = camera_rotations(pitch_deg, yaw_deg)
    dir_w = (r_cam_to_world @ dir_cam.T).T

    dx = dir_w[:, 0]
    valid_dir = dx > 1e-8
    t = np.full(len(dx), np.nan)
    t[valid_dir] = (wall_dist - cam_pos[0]) / dx[valid_dir]
    pts = cam_pos[None, :] + t[:, None] * dir_w
    in_wall = (
        valid_dir
        & (t > 1e-8)
        & (np.abs(pts[:, 1]) <= wall_half_y + 1e-6)
        & (np.abs(pts[:, 2]) <= wall_half_z + 1e-6)
    )

    n = len(pts)
    segs = [[pts[i], pts[(i + 1) % n]] for i in range(n) if in_wall[i] and in_wall[(i + 1) % n]]
    if not segs:
        return np.zeros((0, 2, 3))
    return np.asarray(segs, dtype=float)


def points_in_fisheye_fov(cam_pos, pitch_deg, yaw_deg, hfov_deg, vfov_deg, points):
    """Boolean mask: world points (..., 3) inside the equidistant-fisheye FOV."""
    cam_pos = np.asarray(cam_pos, dtype=float)
    pts = np.asarray(points, dtype=float)
    dir_w = pts.reshape(-1, 3) - cam_pos[None, :]
    dir_w = dir_w / np.maximum(np.linalg.norm(dir_w, axis=-1, keepdims=True), 1e-12)

    _, r_inv = camera_rotations(pitch_deg, yaw_deg)
    dir_cam = dir_w @ r_inv.T
    cx, cy_c, cz_c = dir_cam[:, 0], dir_cam[:, 1], dir_cam[:, 2]

    f, half_u, half_v, r_circle = fisheye_intrinsics(hfov_deg, vfov_deg)
    rho = np.hypot(cy_c, cz_c)
    # arctan2 handles theta up to 180 deg, so walls beside/behind the optical
    # axis are still tested correctly (r_px <= r_circle caps at the FOV edge).
    theta = np.arctan2(rho, cx)
    r_px = f * theta
    inv_rho = 1.0 / np.maximum(rho, 1e-12)
    u = np.where(rho < 1e-12, 0.0, r_px * (-cy_c) * inv_rho)
    v = np.where(rho < 1e-12, 0.0, r_px * cz_c * inv_rho)
    mask = (
        (np.abs(u) <= half_u + 1e-6)
        & (np.abs(v) <= half_v + 1e-6)
        & (r_px <= r_circle + 1e-6)
    )
    return mask.reshape(pts.shape[:-1])


def points_in_pinhole_fov(cam_pos, pitch_deg, hfov_deg, vfov_deg, points):
    """Boolean mask: world points (..., 3) inside a pitched pinhole-camera frustum (yaw = 0)."""
    cam_pos = np.asarray(cam_pos, dtype=float)
    pts = np.asarray(points, dtype=float)
    dir_w = pts.reshape(-1, 3) - cam_pos[None, :]
    _, r_inv = camera_rotations(pitch_deg, 0.0)
    dir_cam = dir_w @ r_inv.T
    cx = dir_cam[:, 0]
    inv_x = 1.0 / np.maximum(cx, 1e-12)
    tan_h = np.tan(np.radians(float(hfov_deg) / 2.0))
    tan_v = np.tan(np.radians(float(vfov_deg) / 2.0))
    mask = (
        (cx > 1e-8)
        & (np.abs(dir_cam[:, 1] * inv_x) <= tan_h + 1e-9)
        & (np.abs(dir_cam[:, 2] * inv_x) <= tan_v + 1e-9)
    )
    return mask.reshape(pts.shape[:-1])


def camera_fov_wall_trapezoid(dist_cm, pitch_deg, fov_h_deg, fov_v_deg):
    """Footprint of a pitched pinhole camera on the wall plane.

    The camera sits on the X axis at ``dist_cm`` from the wall. Returns
    ``(z_bottom_cm, z_top_cm, half_width_bottom_cm, half_width_top_cm)``;
    with pitch = 0 this reduces to the usual centred rectangle.
    """
    half_v = fov_v_deg / 2.0
    half_h_tan = np.tan(np.radians(fov_h_deg / 2.0))
    cos_half_v = np.cos(np.radians(half_v))
    # Clamp elevation so boundary rays always intersect the wall plane.
    v_lo = np.clip(pitch_deg - half_v, -89.0, 89.0)
    v_hi = np.clip(pitch_deg + half_v, -89.0, 89.0)
    z_bot = dist_cm * np.tan(np.radians(v_lo))
    z_top = dist_cm * np.tan(np.radians(v_hi))
    half_w_bot = dist_cm * half_h_tan * cos_half_v / np.cos(np.radians(v_lo))
    half_w_top = dist_cm * half_h_tan * cos_half_v / np.cos(np.radians(v_hi))
    return float(z_bot), float(z_top), float(half_w_bot), float(half_w_top)


def rasterize_fisheye_fov_on_plane(
    cam_pos, pitch_deg, yaw_deg, hfov_deg, vfov_deg,
    axis, plane_coord, u_min, u_max, v_min, v_max, n_grid=100,
):
    """Cell-centered FOV occupancy on an axis-aligned plane patch (cm units).

    axis: fixed world axis (0=x, 1=y, 2=z); plane_coord is its value in cm.
    Returns (mask, us, vs) with cell-center coordinates along the two free axes.
    """
    n_grid = int(n_grid)
    us = np.linspace(u_min, u_max, n_grid, endpoint=False) + (u_max - u_min) / (2 * n_grid)
    vs = np.linspace(v_min, v_max, n_grid, endpoint=False) + (v_max - v_min) / (2 * n_grid)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    u_axis, v_axis = _PLANE_UV_AXES[axis]
    pts = np.empty(uu.shape + (3,), dtype=float)
    pts[..., axis] = float(plane_coord)
    pts[..., u_axis] = uu
    pts[..., v_axis] = vv
    mask = points_in_fisheye_fov(cam_pos, pitch_deg, yaw_deg, hfov_deg, vfov_deg, pts)
    return mask, us, vs


def rasterize_pinhole_fov_on_plane(
    cam_pos, pitch_deg, hfov_deg, vfov_deg,
    axis, plane_coord, u_min, u_max, v_min, v_max, n_grid=100,
):
    """Pinhole-camera counterpart of :func:`rasterize_fisheye_fov_on_plane` (yaw = 0)."""
    n_grid = int(n_grid)
    us = np.linspace(u_min, u_max, n_grid, endpoint=False) + (u_max - u_min) / (2 * n_grid)
    vs = np.linspace(v_min, v_max, n_grid, endpoint=False) + (v_max - v_min) / (2 * n_grid)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    u_axis, v_axis = _PLANE_UV_AXES[axis]
    pts = np.empty(uu.shape + (3,), dtype=float)
    pts[..., axis] = float(plane_coord)
    pts[..., u_axis] = uu
    pts[..., v_axis] = vv
    mask = points_in_pinhole_fov(cam_pos, pitch_deg, hfov_deg, vfov_deg, pts)
    return mask, us, vs


def fov_plane_mask_to_quads_and_contour(mask, us, vs, axis, plane_m):
    """Build plane quads (metres) and silhouette segments from a U×V occupancy mask.

    axis is the fixed world axis (0=x, 1=y, 2=z); plane_m its coordinate in
    metres (already offset off the wall to avoid z-fighting).
    Returns (vertices, faces, segments).
    """
    n_u, n_v = mask.shape
    du = (us[1] - us[0]) if n_u > 1 else 0.0
    dv = (vs[1] - vs[0]) if n_v > 1 else 0.0
    hu_m, hv_m = du / 200.0, dv / 200.0
    u_axis, v_axis = _PLANE_UV_AXES[axis]

    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.uint32),
            np.zeros((0, 2, 3), np.float32),
        )

    u0 = us[ii] / 100.0
    v0 = vs[jj] / 100.0
    n = ii.size
    verts = np.empty((n * 4, 3), dtype=np.float32)
    verts[:, axis] = plane_m
    verts[0::4, u_axis] = u0 - hu_m
    verts[0::4, v_axis] = v0 - hv_m
    verts[1::4, u_axis] = u0 + hu_m
    verts[1::4, v_axis] = v0 - hv_m
    verts[2::4, u_axis] = u0 + hu_m
    verts[2::4, v_axis] = v0 + hv_m
    verts[3::4, u_axis] = u0 - hu_m
    verts[3::4, v_axis] = v0 + hv_m
    faces = np.empty((n * 2, 3), dtype=np.uint32)
    base = np.arange(n, dtype=np.uint32) * 4
    faces[0::2, 0] = base
    faces[0::2, 1] = base + 1
    faces[0::2, 2] = base + 2
    faces[1::2, 0] = base
    faces[1::2, 1] = base + 2
    faces[1::2, 2] = base + 3

    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    def _seg(ua, va, ub, vb):
        pa = [0.0, 0.0, 0.0]
        pb = [0.0, 0.0, 0.0]
        pa[axis] = pb[axis] = plane_m
        pa[u_axis], pa[v_axis] = ua, va
        pb[u_axis], pb[v_axis] = ub, vb
        return [pa, pb]

    segs = []
    for k in range(n):
        i, j = int(ii[k]), int(jj[k])
        ui, vi = u0[k], v0[k]
        pi, pj = i + 1, j + 1
        if not padded[pi - 1, pj]:
            segs.append(_seg(ui - hu_m, vi - hv_m, ui - hu_m, vi + hv_m))
        if not padded[pi + 1, pj]:
            segs.append(_seg(ui + hu_m, vi - hv_m, ui + hu_m, vi + hv_m))
        if not padded[pi, pj - 1]:
            segs.append(_seg(ui - hu_m, vi - hv_m, ui + hu_m, vi - hv_m))
        if not padded[pi, pj + 1]:
            segs.append(_seg(ui - hu_m, vi + hv_m, ui + hu_m, vi + hv_m))
    segs_np = np.asarray(segs, dtype=np.float32) if segs else np.zeros((0, 2, 3), np.float32)
    return verts, faces, segs_np
