"""Ray / triangle-mesh intersection (STL occluders) via trimesh BVH.

``mesh_data`` is a dict ``{'vertices', 'faces', 'transform' (4x4)}``; a
``trimesh_obj`` entry is cached on it after first use.
"""

import atexit
import multiprocessing

import numpy as np

_mesh_intersection_pool = None
_PARALLEL_THRESHOLD = 20000


def _cleanup_mesh_pool():
    global _mesh_intersection_pool
    if _mesh_intersection_pool is not None:
        try:
            _mesh_intersection_pool.shutdown(wait=False)
        except Exception:
            pass
        _mesh_intersection_pool = None


atexit.register(_cleanup_mesh_pool)


def ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2):
    """Möller–Trumbore: distance t to the triangle, or None."""
    epsilon = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
        return None
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
    return t if t > epsilon else None


def _ensure_trimesh(mesh_data):
    if 'trimesh_obj' not in mesh_data:
        import trimesh
        mesh_data['trimesh_obj'] = trimesh.Trimesh(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces'],
            process=False,
        )
    return mesh_data['trimesh_obj']


def prepare_mesh_ray_accelerator(mesh_data):
    """Pre-build the BVH and inverse transform for batched ray queries.

    Returns a dict with 'mesh', 'inv_transform', 'inv_rot', 'transform', or None.
    """
    if mesh_data is None:
        return None
    mesh = _ensure_trimesh(mesh_data)
    transform = mesh_data.get('transform', np.eye(4))
    try:
        inv_transform = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        inv_transform = np.eye(4)
    _ = mesh.ray  # force BVH construction now
    return {
        'mesh': mesh,
        'inv_transform': inv_transform,
        'inv_rot': inv_transform[:3, :3],
        'transform': transform,
    }


def _front_face_hits(mesh, origins, dirs):
    """Boolean hits where the closest intersection is a front face."""
    N = len(origins)
    hits = np.zeros(N, dtype=bool)
    if N == 0:
        return hits
    index_tri, index_ray = mesh.ray.intersects_id(
        ray_origins=origins, ray_directions=dirs, multiple_hits=False,
    )
    if len(index_ray) > 0:
        face_normals = mesh.face_normals[index_tri]
        dots = np.sum(dirs[index_ray] * face_normals, axis=1)
        hits[index_ray[dots < 0]] = True
    return hits


def batch_ray_mesh_intersection(ray_origins, ray_directions, accel, verbose=True):
    """Front-face hit test of many rays against the mesh.

    Uses an AABB pre-filter, then a thread pool for large batches (Embree
    releases the GIL). Returns a (N,) bool array.
    """
    global _mesh_intersection_pool

    if accel is None:
        return np.zeros(len(ray_origins), dtype=bool)
    N = len(ray_origins)
    if N == 0:
        return np.zeros(0, dtype=bool)

    mesh = accel['mesh']
    inv_transform = accel['inv_transform']
    inv_rot = accel['inv_rot']

    origins_h = np.hstack([ray_origins, np.ones((N, 1))])
    origins_local = (inv_transform @ origins_h.T).T[:, :3]
    dirs_local = (inv_rot @ ray_directions.T).T
    norms = np.linalg.norm(dirs_local, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    dirs_local = dirs_local / norms

    # AABB pre-filter
    bbox = mesh.bounds
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

    if verbose:
        print(f"    AABB pre-filter: {N:,} → {n_cand:,} rays ({100*n_cand/N:.0f}%) | mesh: {len(mesh.faces):,} faces")

    cand_origins = origins_local[candidates]
    cand_dirs = dirs_local[candidates]

    n_cpus = multiprocessing.cpu_count()
    cand_hits = None
    if n_cand >= _PARALLEL_THRESHOLD and n_cpus > 1:
        n_workers = min(n_cpus, 8)
        chunk_size = max(2000, (n_cand + n_workers - 1) // n_workers)
        try:
            if _mesh_intersection_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                _mesh_intersection_pool = ThreadPoolExecutor(max_workers=n_workers)
                if verbose:
                    print(f"    Started {n_workers}-thread pool for mesh intersection (Embree releases GIL)")
            futures = [
                _mesh_intersection_pool.submit(
                    _front_face_hits, mesh,
                    cand_origins[i:i + chunk_size], cand_dirs[i:i + chunk_size],
                )
                for i in range(0, n_cand, chunk_size)
            ]
            cand_hits = np.concatenate([f.result() for f in futures])
        except Exception as e:
            print(f"    Parallel mesh intersection failed ({e}), falling back to single-thread")
            cand_hits = None

    if cand_hits is None:
        cand_hits = _front_face_hits(mesh, cand_origins, cand_dirs)

    hits[candidates[cand_hits]] = True
    return hits


def ray_mesh_intersection(ray_origin, ray_direction, mesh_data):
    """Distance to the closest front-face hit of one ray, or None."""
    if mesh_data is None:
        return None
    mesh = _ensure_trimesh(mesh_data)
    transform = mesh_data.get('transform', np.eye(4))
    try:
        inv_transform = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        inv_transform = np.eye(4)

    ray_origin_local = (inv_transform @ np.append(ray_origin, 1))[:3]
    ray_direction_local = inv_transform[:3, :3] @ ray_direction
    ray_direction_local = ray_direction_local / np.linalg.norm(ray_direction_local)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[ray_origin_local],
        ray_directions=[ray_direction_local],
        multiple_hits=False,
    )
    if len(locations) == 0:
        return None
    face_normal = mesh.face_normals[index_tri[0]]
    if np.dot(ray_direction_local, face_normal) >= 0:
        return None  # back face: ray starts inside the mesh
    hit_point_world = (transform @ np.append(locations[0], 1))[:3]
    return float(np.linalg.norm(hit_point_world - ray_origin))
