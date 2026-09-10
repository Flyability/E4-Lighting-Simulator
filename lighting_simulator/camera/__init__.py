"""Camera models (main pinhole camera, VIO fisheye cameras)."""

from .fov import (
    VD66GY_LONG_PX,
    VD66GY_SHORT_PX,
    camera_fov_wall_trapezoid,
    camera_rotations,
    compute_fisheye_fov_wall_segments,
    fisheye_intrinsics,
    fov_plane_mask_to_quads_and_contour,
    points_in_fisheye_fov,
    points_in_pinhole_fov,
    rasterize_fisheye_fov_on_plane,
    vio_hfov_vfov_deg,
    vio_optical_axis,
)

__all__ = [
    "VD66GY_LONG_PX",
    "VD66GY_SHORT_PX",
    "camera_fov_wall_trapezoid",
    "camera_rotations",
    "compute_fisheye_fov_wall_segments",
    "fisheye_intrinsics",
    "fov_plane_mask_to_quads_and_contour",
    "points_in_fisheye_fov",
    "points_in_pinhole_fov",
    "rasterize_fisheye_fov_on_plane",
    "vio_hfov_vfov_deg",
    "vio_optical_axis",
]
