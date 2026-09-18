"""Engine regression tests (CPU path only; fast)."""

import numpy as np
import pytest

from lighting_simulator.analysis.uniformity import uniformity_metrics
from lighting_simulator.domain.led import LED
from lighting_simulator.scene import build_scene_from_config, load_config
from lighting_simulator.simulation import (
    EmissionSettings,
    RoomSettings,
    WallSettings,
    compute_room_intensity,
    compute_wall_intensity,
    trace_led_to_wall,
)
from lighting_simulator.simulation.emission import generate_led_rays, led_rng
from lighting_simulator.simulation.room import trace_led_in_room
from lighting_simulator.simulation.room_geometry import build_wall_specs, wall_cell_areas_m2


@pytest.fixture(scope="module")
def elios3():
    return load_config("layouts/Elios3.json")


def test_ray_weights_conserve_flux():
    led = LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=120.0)
    dirs, lumens = generate_led_rays(led, 50_000, 100.0, 0.0, led_rng(0))
    assert dirs.shape == (50_000, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-9)
    assert lumens.sum() == pytest.approx(100.0, rel=0.02)


def test_wall_trace_is_deterministic():
    led = LED(position=(-5, 2, 0), direction=(1, 0.1, -0.2), viewing_angle=120.0)
    a = trace_led_to_wall(led, 3, 10_000, 100.0, 0.0, 100.0, 40, 80.0)
    b = trace_led_to_wall(led, 3, 10_000, 100.0, 0.0, 100.0, 40, 80.0)
    np.testing.assert_array_equal(a, b)


def test_wall_flux_below_emitted_and_box_absorber_reduces_it():
    led = LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=60.0)
    grid = trace_led_to_wall(led, 0, 40_000, 100.0, 0.0, 50.0, 40, 400.0)
    cell_area_m2 = (400.0 / 40) ** 2 / 10_000
    on_wall = grid.sum() * cell_area_m2
    assert 95.0 < on_wall <= 100.5  # narrow cone fully lands on a huge wall
    blocker = [{'center': (10.0, 0.0, 0.0), 'half_sizes': (0.5, 3.0, 3.0), 'rotation': None}]
    blocked = trace_led_to_wall(led, 0, 40_000, 100.0, 0.0, 50.0, 40, 400.0, absorbers=blocker)
    assert blocked.sum() < 0.5 * grid.sum()


def test_room_direct_hits_conserve_flux():
    led = LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=120.0)
    settings = RoomSettings(front_dist=100, side_dist=80, top_bottom_dist=60, back_dist=50,
                            grid_size=10, rays_per_pixel=100)
    specs = build_wall_specs(100, 80, 60, 10, -35.0, 50)
    grids, hits, n_rays = trace_led_in_room(led, 0, 100.0, 0.0, settings, specs)
    areas = wall_cell_areas_m2(specs)
    lumens = sum(g.sum() * areas[name] for name, g in grids.items())
    assert n_rays == 10_000
    assert sum(hits.values()) == n_rays
    assert lumens == pytest.approx(100.0, rel=0.02)


def test_scene_from_config_and_wall_pipeline(elios3):
    scene = build_scene_from_config(elios3)
    assert len(scene.leds) == 48
    assert len(scene.active_leds) == 24
    grid = compute_wall_intensity(
        scene.leds, WallSettings(wall_dist=100, grid_size=20, wall_size=80, rays_per_pixel=2),
        EmissionSettings(default_lumens=100.0), use_gpu=False, verbose=False,
    )
    assert grid.shape == (20, 20) and grid.max() > 0
    metrics = uniformity_metrics(grid)
    assert 0 < metrics.u0 <= 1 and 0 < metrics.u1 <= metrics.u0


def test_room_pipeline_with_bounces(elios3):
    scene = build_scene_from_config(elios3)
    grids, specs = compute_room_intensity(
        scene.leds,
        RoomSettings(front_dist=100, side_dist=100, top_bottom_dist=100, grid_size=8,
                     rays_per_pixel=2, max_bounces=1, wall_reflectance=0.5),
        use_gpu=False, verbose=False,
    )
    assert set(grids) == {'front', 'left', 'right', 'top', 'bottom'}
    assert all(g.shape == (8, 8) for g in grids.values())
    assert grids['front'].sum() > 0


def test_direct_illuminance_matches_monte_carlo_mean():
    """Analytic gather == expected value of the MC room tracer on lit cells (no bounces)."""
    from lighting_simulator.simulation import direct_illuminance
    from lighting_simulator.simulation.room_geometry import WALL_INWARD_NORMALS, room_wall_cell_centers
    leds = [LED(position=(0, 0, 0), direction=(1, 0, 0.3), viewing_angle=120.0),
            LED(position=(2, -3, 1), direction=(0.5, 0.5, -0.4), viewing_angle=90.0)]
    d, g = 100.0, 12
    settings = RoomSettings(front_dist=d, side_dist=d, top_bottom_dist=d, grid_size=g, rays_per_pixel=4000)
    emission = EmissionSettings(default_lumens=150.0, ray_uniformity=0.3)
    grids, specs = compute_room_intensity(leds, settings, emission, use_gpu=False, verbose=False)
    for name in ('front', 'top', 'right'):
        pts = room_wall_cell_centers(name, specs[name], d, d, d, None).reshape(-1, 3)
        nrm = np.tile(WALL_INWARD_NORMALS[name], (len(pts), 1))
        e = direct_illuminance(pts, nrm, leds, emission).reshape(grids[name].shape)
        mc = grids[name]
        bright = mc > 0.2 * mc.max()  # skip cone-edge cells where the MC bin is half-filled
        rel = np.abs(e[bright] - mc[bright]) / mc[bright]
        assert np.median(rel) < 0.05, (name, np.median(rel))
        # the analytic value is a point sample at the cell centre, MC a bin average: they only
        # differ on cone-edge cells, so the wall totals agree loosely at this coarse resolution
        assert e.sum() == pytest.approx(mc.sum(), rel=0.10)
    blocker = [{'center': (10.0, 0.0, 3.0), 'half_sizes': (0.5, 4.0, 4.0), 'rotation': None}]
    pts = room_wall_cell_centers('front', specs['front'], d, d, d, None).reshape(-1, 3)
    nrm = np.tile(WALL_INWARD_NORMALS['front'], (len(pts), 1))
    shaded = direct_illuminance(pts, nrm, leds, emission, absorbers=blocker)
    assert shaded.sum() < 0.9 * direct_illuminance(pts, nrm, leds, emission).sum()
    base = direct_illuminance(pts, nrm, leds, emission).sum()  # LED() defaults to 100 lm
    assert direct_illuminance(pts, nrm, leds, emission, lumens=[200.0, 200.0]).sum() == pytest.approx(2 * base)
