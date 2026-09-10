"""Ray / axis-aligned (optionally rotated) box intersection for absorber blocks.

An absorber is a dict: ``{'center': (x,y,z), 'half_sizes': (hx,hy,hz),
'rotation': (w,x,y,z) | None}``.
"""

import numpy as np

from lighting_simulator.domain.geometry import quaternion_to_matrix


def ray_box_intersection(pos, direction, box):
    """Distance to the first hit of a single ray against ``box``, or None."""
    center = np.array(box['center'], dtype=float)
    half = np.array(box['half_sizes'], dtype=float)
    rotation = box.get('rotation', None)

    if rotation is not None:
        R_inv = quaternion_to_matrix(rotation).T
        pos = R_inv @ (pos - center)
        direction = R_inv @ direction
        center = np.zeros(3)

    tmin, tmax = -np.inf, np.inf
    for k in range(3):
        if abs(direction[k]) < 1e-12:
            if pos[k] < center[k] - half[k] or pos[k] > center[k] + half[k]:
                return None
        else:
            t1 = (center[k] - half[k] - pos[k]) / direction[k]
            t2 = (center[k] + half[k] - pos[k]) / direction[k]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
            if tmin > tmax:
                return None
    if tmax < 0:
        return None
    return tmin if tmin > 0 else (tmax if tmax > 0 else None)


def ray_box_intersection_batch(origins, directions, absorbers):
    """Vectorised test of all rays against all absorbers.

    Parameters
    ----------
    origins : (N, 3) ray origins
    directions : (N, 3) unit ray directions
    absorbers : list of absorber dicts

    Returns
    -------
    (N,) bool -- True where *any* absorber blocks the ray.
    """
    N = origins.shape[0]
    absorbed = np.zeros(N, dtype=bool)

    for box in absorbers:
        center = np.asarray(box['center'], dtype=np.float32)
        half = np.asarray(box['half_sizes'], dtype=np.float32)
        rotation = box.get('rotation', None)

        if rotation is not None:
            R_inv = quaternion_to_matrix(rotation).astype(np.float32).T
            local_origins = (origins - center[None, :]) @ R_inv.T
            local_dirs = directions @ R_inv.T
        else:
            local_origins = origins - center[None, :]
            local_dirs = directions

        tmin = np.full(N, -1e30, dtype=np.float32)
        tmax = np.full(N, 1e30, dtype=np.float32)
        valid = np.ones(N, dtype=bool)

        for k in range(3):
            d_k = local_dirs[:, k]
            o_k = local_origins[:, k]
            lo, hi = -half[k], half[k]

            parallel = np.abs(d_k) < 1e-12
            valid &= ~(parallel & ((o_k < lo) | (o_k > hi)))

            inv_d = np.where(parallel, np.float32(1.0),
                             np.float32(1.0) / np.where(parallel, np.float32(1.0), d_k))
            t1 = (lo - o_k) * inv_d
            t2 = (hi - o_k) * inv_d
            t_near = np.minimum(t1, t2)
            t_far = np.maximum(t1, t2)

            mask_np = ~parallel
            tmin = np.where(mask_np & (t_near > tmin), t_near, tmin)
            tmax = np.where(mask_np & (t_far < tmax), t_far, tmax)

        absorbed |= valid & (tmin <= tmax) & (tmax > 0)

    return absorbed
