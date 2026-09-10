"""Headless end-to-end pipeline: config → scene → illuminance → metrics.

This is the entry point intended for scripted / iterative use (e.g. LED
placement optimisation): no Viser, no GUI state.

Example::

    from lighting_simulator.pipeline import simulate_wall
    from lighting_simulator.scene import load_config
    from lighting_simulator.simulation import WallSettings

    result = simulate_wall(load_config("configs/elios3.json"),
                           WallSettings(wall_dist=100, grid_size=50, rays_per_pixel=2))
    print(result.metrics.u0)
"""

from dataclasses import dataclass

import numpy as np

from .analysis.uniformity import UniformityMetrics, select_region, uniformity_metrics
from .scene.builder import Scene, build_scene_from_config
from .simulation.room import compute_room_intensity
from .simulation.settings import EmissionSettings, RoomSettings, WallSettings
from .simulation.wall import compute_wall_intensity


@dataclass
class WallResult:
    scene: Scene
    settings: WallSettings
    grid: np.ndarray
    """Lux, shape (grid_size, grid_size); rows = Z (bottom→top), cols = Y."""
    metrics: UniformityMetrics | None

    @property
    def total_wall_lumens(self):
        return float(np.sum(self.grid) * self.settings.cell_area_m2)


@dataclass
class RoomResult:
    scene: Scene
    settings: RoomSettings
    grids: dict
    wall_specs: dict
    metrics: UniformityMetrics | None
    """Uniformity over all lit cells of every wall."""


def simulate_wall(cfg, settings: WallSettings, emission: EmissionSettings | None = None,
                  stl_mesh=None, diffuser=None, fov_trapezoid=None, use_gpu=None, verbose=False,
                  parallel=True):
    """Build the scene from ``cfg`` and trace it onto the single wall.

    ``fov_trapezoid`` optionally restricts the metrics to a camera footprint
    (see ``camera.fov.camera_fov_wall_trapezoid``; append the camera Y).
    """
    emission = emission or EmissionSettings()
    scene = build_scene_from_config(cfg, default_lumens=emission.default_lumens,
                                    stl_mesh=stl_mesh, diffuser=diffuser)
    grid = compute_wall_intensity(scene.leds, settings, emission, absorbers=scene.absorbers,
                                  stl_mesh_data=scene.stl_mesh_data, use_gpu=use_gpu, verbose=verbose,
                                  parallel=parallel)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    region = select_region(grid, wall_size_cm=settings.wall_size, fov_trapezoid=fov_trapezoid)
    return WallResult(scene=scene, settings=settings, grid=grid, metrics=uniformity_metrics(region))


def simulate_room(cfg, settings: RoomSettings, emission: EmissionSettings | None = None,
                  stl_mesh=None, diffuser=None, use_gpu=None, verbose=False):
    """Build the scene from ``cfg`` and trace it inside the room."""
    emission = emission or EmissionSettings()
    scene = build_scene_from_config(cfg, default_lumens=emission.default_lumens,
                                    stl_mesh=stl_mesh, diffuser=diffuser)
    grids, wall_specs = compute_room_intensity(scene.leds, settings, emission, absorbers=scene.absorbers,
                                               stl_mesh_data=scene.stl_mesh_data, use_gpu=use_gpu, verbose=verbose)
    all_lux = np.concatenate([np.asarray(g).reshape(-1) for g in grids.values()])
    return RoomResult(scene=scene, settings=settings, grids=grids, wall_specs=wall_specs,
                      metrics=uniformity_metrics(all_lux))
