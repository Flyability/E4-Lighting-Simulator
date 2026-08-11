"""
Parallel mesh ray intersection workers.
Used by interactive_lighting.py via ProcessPoolExecutor.
Each worker process builds its own trimesh + BVH once (via initializer),
then processes ray chunks without re-building.
"""
import numpy as np

# Per-worker global: trimesh object with pre-built BVH
_worker_mesh = None


def init_mesh_worker(mesh_verts, mesh_faces):
    """Called once per worker process to build trimesh + BVH."""
    global _worker_mesh
    import trimesh
    _worker_mesh = trimesh.Trimesh(
        vertices=mesh_verts, faces=mesh_faces, process=False
    )
    # Force BVH construction
    _ = _worker_mesh.ray


def intersect_chunk(args):
    """
    Test a chunk of rays (already in mesh local space) against the pre-built mesh.
    Returns boolean array: True where ray hits front face of mesh.
    """
    origins_local, dirs_local = args
    N = len(origins_local)
    if N == 0:
        return np.zeros(0, dtype=bool)

    mesh = _worker_mesh
    index_tri, index_ray = mesh.ray.intersects_id(
        ray_origins=origins_local,
        ray_directions=dirs_local,
        multiple_hits=False,
    )

    hits = np.zeros(N, dtype=bool)
    if len(index_ray) > 0:
        face_normals = mesh.face_normals[index_tri]
        hit_ray_dirs = dirs_local[index_ray]
        dots = np.sum(hit_ray_dirs * face_normals, axis=1)
        hits[index_ray[dots < 0]] = True

    return hits
