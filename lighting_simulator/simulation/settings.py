"""Plain-data settings objects consumed by the simulation engine."""

from dataclasses import dataclass


@dataclass
class EmissionSettings:
    """Global emission parameters shared by every tracing mode."""

    default_lumens: float = 100.0
    """Flux used for LEDs whose own ``lumens`` is unset/zero (already calibrated)."""
    ray_uniformity: float = 0.0
    """0 = pure Lambertian; >0 sharpens the beam (see optics.lambertian_exponent)."""


@dataclass
class WallSettings:
    """Single-wall target: a square wall at ``x = wall_dist`` centred on the X axis."""

    wall_dist: float = 100.0
    grid_size: int = 50
    wall_size: float = 80.0
    rays_per_pixel: int = 2
    """Target ray budget per grid cell, spread across active LEDs."""

    @property
    def cell_size_cm(self):
        return self.wall_size / self.grid_size

    @property
    def cell_area_m2(self):
        return (self.cell_size_cm ** 2) / 10000.0


@dataclass
class RoomSettings:
    """Closed room around the rig: front, left/right, top/bottom and optional back wall."""

    front_dist: float = 100.0
    side_dist: float = 100.0
    top_bottom_dist: float = 100.0
    back_dist: float | None = None
    grid_size: int = 20
    rays_per_pixel: int = 2
    led_x_center: float = -35.0
    lateral_depth: float | None = None
    """Depth (cm) of the side/top/bottom walls behind the front wall; None = 2.5× the rig-to-wall width."""
    max_bounces: int = 0
    wall_reflectance: float = 0.0

    @property
    def rays_per_led(self):
        return int(self.rays_per_pixel) * int(self.grid_size) ** 2
