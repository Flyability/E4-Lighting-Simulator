"""STEP (ISO 10303) import → triangle mesh, via CadQuery / OCP (optional dependency).

STEP files carry their length unit, and OpenCascade converts everything to
millimetres on read, so the returned mesh is always in **mm** with the modelled
origin preserved. Install with ``pip install cadquery`` (or ``uv sync --extra step``).
"""

import numpy as np
import trimesh

STEP_EXTENSIONS = ('.step', '.stp')
STEP_MM_TO_CM = 0.1


def is_step_file(path):
    return str(path).lower().endswith(STEP_EXTENSIONS)


def load_step_mesh(path, tolerance_mm=0.2, angular_tolerance_rad=0.3):
    """Tessellate every solid/shell of a STEP file into one ``trimesh.Trimesh`` (mm).

    ``tolerance_mm`` is the max chord deviation; 0.2 mm keeps a drone frame at a
    few tens of thousands of faces.
    """
    try:
        import cadquery as cq
    except ImportError as e:
        raise ImportError(
            "STEP import needs CadQuery: `pip install cadquery` (or `uv sync --extra step`)."
        ) from e

    shape = cq.importers.importStep(str(path))
    vertices, faces = [], []
    n = 0
    for solid in shape.vals():
        verts, tris = solid.tessellate(tolerance_mm, angular_tolerance_rad)
        if not tris:
            continue
        vertices.append(np.array([(v.x, v.y, v.z) for v in verts], dtype=np.float64))
        faces.append(np.asarray(tris, dtype=np.int64) + n)
        n += len(verts)
    if not faces:
        raise ValueError("STEP file contains no tessellatable geometry")
    mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)
    mesh.merge_vertices()
    return mesh
