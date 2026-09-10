"""Optimisation layer tests (fast, CPU, tiny ray budgets)."""

import numpy as np
import pytest

from lighting_simulator.optimization import (
    CameraSpec,
    ConstraintSpec,
    Duct,
    DuctRingLayout,
    LedStates,
    OptimizerSpec,
    PanelPose,
    Problem,
    load_spec,
    problem_from_spec,
    run,
)
from lighting_simulator.scene import build_scene_from_config, load_config
from lighting_simulator.simulation import WallSettings

WALL = WallSettings(wall_dist=100, grid_size=12, wall_size=250, rays_per_pixel=2)
CAM = CameraSpec()


def test_duct_led_pose_lies_on_cylinder_and_points_outward():
    duct = Duct(center=(5, 2, 0), axis=(0, 0, 1), radius=6.0, mount_offset=0.0)
    for theta in (-90, -30, 0, 45, 120):
        p, d, rd = duct.led_pose(theta, 0.7)
        radial = p - np.array([5, 2, 0.7])
        assert np.linalg.norm(radial) == pytest.approx(6.0)
        assert np.dot(d, radial / 6.0) == pytest.approx(1.0)
        assert abs(np.dot(rd, d)) < 1e-9
    _, tilted, _ = duct.led_pose(0, 0, tilt_axial_deg=30)
    assert tilted[2] == pytest.approx(np.sin(np.radians(30)))


def test_duct_ring_arc_generates_lattice_and_mirror():
    cfg = load_config("configs/elios3.json")
    var = DuctRingLayout(name="d", duct=Duct(center=(8, 9, 0), radius=7), n_leds=6, n_rows=2,
                         placement="arc", beam_angle_range=(80, 130), optimize_enabled=True, mirror_xz=True)
    assert var.size == 4 + 1 + 1 + 6  # theta_c, span, axial_c, pitch | tilt | beam | 6 on-flags
    problem = Problem(cfg, [var], WALL, CAM, clear_base=True)
    x = problem.x0.copy()
    x[-1] = 0  # switch one LED off
    out = problem.decode(x)
    groups = out['custom_groups']
    assert len(groups) == 2 and groups[1]['name'] == "d_mirror"
    assert groups[0]['led_rows'] == [[0, 1, 2], [3, 4, 5]]
    assert sum(groups[0]['led_states']) == 5
    pos = np.array(groups[0]['led_positions'])
    mirrored = np.array(groups[1]['led_positions'])
    np.testing.assert_allclose(mirrored[:, 1], -pos[:, 1])
    scene = build_scene_from_config(out)
    assert len(scene.active_leds) == 10


def test_panel_pose_and_led_states_modify_base_groups():
    cfg = load_config("configs/Elios3.json")
    pose = PanelPose(group_index=0, pos_delta=(1, 0, 1), rot_delta=(0, 0, 5))
    states = LedStates(group_index=1).bind(cfg)
    problem = Problem(cfg, [pose, states], WALL, CAM)
    assert pose.names == ["group0.dx", "group0.dz", "group0.drot_z"]
    x = problem.x0.copy()
    x[:3] = [0.5, -0.5, 3.0]
    x[3:] = 0
    out = problem.decode(x)
    base = cfg['custom_groups'][0]
    assert out['custom_groups'][0]['position'][0] == pytest.approx(base['position'][0] + 0.5)
    # Plain dynamic group: the UI ignores rotation_*, so the delta is baked into the LED arrays.
    assert out['custom_groups'][0]['rotation_z'] == base.get('rotation_z', 0)
    p_base = np.array(base['led_positions'][0])
    p_out = np.array(out['custom_groups'][0]['led_positions'][0])
    assert np.linalg.norm(p_out) == pytest.approx(np.linalg.norm(p_base))
    ang = np.degrees(np.arctan2(p_out[1], p_out[0]) - np.arctan2(p_base[1], p_base[0]))
    assert ang == pytest.approx(3.0, abs=1e-6)
    assert not any(out['custom_groups'][1]['led_states'])


def test_evaluate_scores_and_penalties():
    cfg = load_config("configs/elios3.json")
    var = DuctRingLayout(name="d", duct=Duct(center=(8, 9, 0), radius=7), n_leds=4, placement="arc",
                         tilt_axial_range=None)
    problem = Problem(cfg, [var], WALL, CAM, clear_base=True,
                      constraints=ConstraintSpec(max_leds=2, max_leds_weight=0.1, min_led_spacing_cm=50.0))
    ev = problem.evaluate(problem.x0)
    assert ev.n_active == 4
    assert ev.penalties['max_leds'] == pytest.approx(0.2)
    assert ev.penalties['spacing'] > 0
    assert 0 <= ev.uniformity_pct <= 100 and ev.score > 0
    assert problem(problem.x0) == pytest.approx(ev.score)


@pytest.mark.parametrize("method", ["differential_evolution", "nelder_mead", "random_search"])
def test_run_methods_produce_outputs(tmp_path, method):
    spec, spec_dir = load_spec("optimization_specs/elios3_duct_rings.json")
    spec['wall'] = {"wall_dist": 100, "grid_size": 10, "wall_size": 250, "rays_per_pixel": 1}
    problem = problem_from_spec(spec, spec_dir)
    summary, best = run(problem, OptimizerSpec(method=method, max_evals=12, population=6, log_every=0),
                        output_dir=tmp_path)
    out = tmp_path / problem.name
    assert (out / "best_config.json").exists() and (out / "history.csv").exists()
    assert summary['best_score'] == pytest.approx(best.score)
    assert len(build_scene_from_config(load_config(out / "best_config.json")).leds) > 0
