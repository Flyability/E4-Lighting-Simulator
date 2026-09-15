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
                        output_dir=tmp_path, report=False)
    out = tmp_path / problem.name
    assert (out / "best_config.json").exists() and (out / "history.csv").exists()
    assert summary['best_score'] == pytest.approx(best.score)
    assert summary['run_dir'] == str(out)
    assert len(build_scene_from_config(load_config(out / "best_config.json")).leds) > 0
    # a second run with the same name must not overwrite the first
    summary2, _ = run(problem, OptimizerSpec(method=method, max_evals=12, population=6, log_every=0),
                      output_dir=tmp_path, report=False)
    assert summary2['run_dir'] == str(tmp_path / f"{problem.name}_002")
    assert (out / "summary.json").exists() and (tmp_path / f"{problem.name}_002" / "summary.json").exists()


def test_duct_center_delta_shifts_all_leds():
    cfg = load_config("configs/elios3.json")
    var = DuctRingLayout(name="d", duct=Duct(center=(8, 9, 0), radius=7), n_leds=4, placement="arc",
                         tilt_axial_range=None, center_delta=(2.0, 0.0, 1.0))
    assert var.names[-2:] == ["d.dcx", "d.dcz"]
    problem = Problem(cfg, [var], WALL, CAM, clear_base=True)
    x = problem.x0.copy()
    base = np.array(problem.decode(x)['custom_groups'][0]['led_positions'])
    x[-2:] = [1.5, -0.5]
    moved = np.array(problem.decode(x)['custom_groups'][0]['led_positions'])
    np.testing.assert_allclose(moved - base, np.tile([1.5, 0.0, -0.5], (4, 1)), atol=1e-9)


def test_wall_size_per_distance_and_auto_fit():
    cfg = load_config("configs/elios3.json")
    var = DuctRingLayout(name="d", duct=Duct(center=(8, 9, 0), radius=7), n_leds=2, placement="arc",
                         tilt_axial_range=None)
    p = Problem(cfg, [var], WALL, CAM, clear_base=True, wall_dists=[50, 100, 200], wall_sizes="auto")
    sizes = [w.wall_size for w in p.walls]
    assert sizes[0] < sizes[1] < sizes[2]
    for w, m in zip(p.walls, p._fov_masks):
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        assert 0 < rows[0] and rows[-1] < w.grid_size - 1  # footprint inside the grid with a margin
        assert 0 < cols[0] and cols[-1] < w.grid_size - 1
        assert m.sum() > 0.3 * m.size  # ...but filling most of it (4:3 footprint in a square grid)
    p2 = Problem(cfg, [var], WALL, CAM, clear_base=True, wall_dists=[50, 100], wall_sizes=[120, 240])
    assert [w.wall_size for w in p2.walls] == [120.0, 240.0]
    with pytest.raises(ValueError):
        Problem(cfg, [var], WALL, CAM, clear_base=True, wall_dists=[50, 100], wall_sizes=[1, 2, 3])


def test_requirements_spec_modes_drivers_and_report(tmp_path):
    spec, spec_dir = load_spec("optimization_specs/elios4_ducts_flash_vio.json")
    spec['wall'].update({"grid_size": 10, "rays_per_pixel": 1})
    spec['vio'].update({"grid_size": 8})
    problem = problem_from_spec(spec, spec_dir)
    names = problem.names
    assert names[:2] == ["front_duct.n_rows", "front_duct.n_cols"] and problem.integrality[:2].all()
    assert "front_duct.radial" in names and names[-1] == "front_duct.current_a"

    x = problem.x0.copy()
    x[0], x[1] = 2, 3  # 2 rows x 3 cols, mirrored -> 12 LEDs
    ev = problem.evaluate(x)
    assert ev.n_active == 12 and ev.n_drivers == 3  # 4 LEDs per driver
    cfg = problem.decode(x)
    g = cfg['custom_groups'][0]
    assert g['num_leds'] == 6 and g['lumens_override_enabled']
    assert g['lumens_value'] == pytest.approx(g['drive_current_a'] * 6.0 * 180.0)
    assert ev.total_current_a == pytest.approx(12 * g['drive_current_a'])
    assert set(ev.modes) == {"normal", "flash"}
    assert ev.modes['flash']['scale'] == pytest.approx(13.0 / g['drive_current_a'])
    assert 0.0 <= ev.modes['normal']['vio_fraction'] <= 1.0

    summary, best = run(problem, OptimizerSpec(method="random_search", max_evals=8, population=4, log_every=0),
                        output_dir=tmp_path)
    out = tmp_path / problem.name
    assert summary['report'] and (out / "report.pdf").stat().st_size > 10_000
    assert (out / "initial_config.json").exists()
    with open(out / "report.pdf", "rb") as f:
        assert f.read(5) == b"%PDF-"
