"""Low-level CPU ray intersection primitives (boxes, triangle meshes)."""

from .boxes import ray_box_intersection, ray_box_intersection_batch
from .mesh import (
    batch_ray_mesh_intersection,
    prepare_mesh_ray_accelerator,
    ray_mesh_intersection,
    ray_triangle_intersection,
)

__all__ = [
    "batch_ray_mesh_intersection",
    "prepare_mesh_ray_accelerator",
    "ray_box_intersection",
    "ray_box_intersection_batch",
    "ray_mesh_intersection",
    "ray_triangle_intersection",
]
