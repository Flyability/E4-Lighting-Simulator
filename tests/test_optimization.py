"""Optimisation layer tests (fast, CPU, tiny ray budgets)."""

import numpy as np
import pytest

from lighting_simulator.optimization import (
    CameraSpec,
    ConstraintSpec,
    Duct,
    DuctRingLayout,
    LedStates,
    ModeSpec,
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
    cfg = load_config("tests/data/v1_configs/elios3.json")
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


def test_duct_ring_variable_counts_keep_per_led_variables_in_place():
    """role[i] must address the same lattice cell whether the lattice is 3x5 or 2x3."""
    var = DuctRingLayout(name="d", duct=Duct(radius=7), n_leds=15, n_rows=3, n_rows_range=(1, 3), n_cols_range=(1, 5),
                         optimize_enabled=True, optimize_roles=True, tilt_axial_range=None)
    assert not var.optimize_enabled  # counts are variables -> on/off dropped
    assert not var.roles_allow_off
    role_idx = [i for i, n in enumerate(var.names) if '.role[' in n]
    assert len(role_idx) == 15 and all(var.bounds[i] == (1.0, 3.0) for i in role_idx)
    x = list(var.x0)
    x[role_idx[7]] = 1   # centre of the 3x5 max lattice (row 1, col 2) -> vio
    x[role_idx[10]] = 2  # row 2, col 0 -> flash

    cfg = {'custom_groups': []}
    var.apply(x, cfg)
    g = cfg['custom_groups'][0]
    assert g['num_leds'] == 15 and g['led_roles'][7] == 'vio' and g['led_roles'][10] == 'flash'
    assert all(g['led_states'])

    x[0], x[1] = 3, 3  # shrink to 3x3: centred sub-block, columns 1..3 of the max lattice
    cfg = {'custom_groups': []}
    var.apply(x, cfg)
    g = cfg['custom_groups'][0]
    assert g['num_leds'] == 9 and g['led_rows'] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert g['led_roles'][4] == 'vio'          # row 1, centre column is still the vio LED
    assert 'flash' not in g['led_roles']       # row 2 col 0 of the max lattice was trimmed away
    assert all(g['led_states'])

    fixed = DuctRingLayout(name="f", duct=Duct(radius=7), n_leds=6, n_rows=2, optimize_roles=True, tilt_axial_range=None)
    assert fixed.roles_allow_off and all(fixed.bounds[i] == (0.0, 3.0) for i, n in enumerate(fixed.names) if '.role[' in n)


def test_duct_panel_pose_slides_existing_layout_over_the_duct():
    """x0 reproduces the scene exactly; a theta step is a rigid rotation about the duct axis."""
    from lighting_simulator.optimization import DuctPanelPose
    from lighting_simulator.scene.builder import group_config_to_factory
    cfg = load_config("tests/data/v1_configs/ludos_panels_n4.json")
    gi = 0
    duct = Duct(center=(6.3, 12.0, 0.0), axis=(0, 0, 1), radius=8.5, mount_offset=0.0)
    var = DuctPanelPose(group_index=gi, duct=duct, theta_range=(-30, 30), axial_range=(-2, 2),
                        tilt_axial_range=(-20, 20), mirror_group_index=1).bind(cfg)
    assert var.names == [f"group{gi}.duct_theta", f"group{gi}.duct_axial", f"group{gi}.duct_tilt"]
    assert var.x0[0] == pytest.approx(var.theta0) and var.x0[2] == 0.0

    base = np.asarray(group_config_to_factory(cfg['custom_groups'][gi])['led_positions'], float)
    base_dirs = np.asarray(group_config_to_factory(cfg['custom_groups'][gi])['led_rotations'], float)
    problem = Problem(cfg, [var], WALL, CAM)
    same = problem.decode(problem.x0)
    out = np.asarray(group_config_to_factory(same['custom_groups'][gi])['led_positions'], float)
    np.testing.assert_allclose(out, base, atol=1e-6)
    np.testing.assert_allclose(np.asarray(group_config_to_factory(same['custom_groups'][gi])['led_rotations'], float),
                               base_dirs, atol=1e-6)
    mirror = np.asarray(group_config_to_factory(same['custom_groups'][1])['led_positions'], float)
    np.testing.assert_allclose(mirror, out * np.array([1.0, -1.0, 1.0]), atol=1e-6)

    x = problem.x0.copy()
    x[0] += 20.0  # rotate 20 deg around the duct
    moved = np.asarray(group_config_to_factory(problem.decode(x)['custom_groups'][gi])['led_positions'], float)
    r_before = np.linalg.norm((base - duct.center)[:, :2], axis=1)
    r_after = np.linalg.norm((moved - duct.center)[:, :2], axis=1)
    np.testing.assert_allclose(r_after, r_before, atol=1e-6)   # distance to the axis preserved
    np.testing.assert_allclose(moved[:, 2], base[:, 2], atol=1e-6)  # nothing moved along the axis
    d_before = np.linalg.norm(base[:, None] - base[None], axis=-1)
    d_after = np.linalg.norm(moved[:, None] - moved[None], axis=-1)
    np.testing.assert_allclose(d_after, d_before, atol=1e-6)   # rigid: pairwise distances kept
    ang = np.degrees(np.arctan2(*(moved - duct.center)[0, [1, 0]]) - np.arctan2(*(base - duct.center)[0, [1, 0]]))
    assert ang == pytest.approx(20.0, abs=1e-6)
    assert len(build_scene_from_config(problem.decode(x)).active_leds) > 0

    # per-LED slide: x0 still reproduces the scene, one LED's delta moves only that LED along the surface
    var2 = DuctPanelPose(group_index=gi, duct=duct, theta_range=(-30, 30), axial_range=None,
                         led_theta_range=(-10, 10), led_axial_range=(-1, 1)).bind(cfg)
    n = len(base)
    assert len(var2.names) == 1 + 2 * n and var2.names[1] == f"group{gi}.led0.dtheta"
    p2 = Problem(cfg, [var2], WALL, CAM)
    out0 = np.asarray(group_config_to_factory(p2.decode(p2.x0)['custom_groups'][gi])['led_positions'], float)
    np.testing.assert_allclose(out0, base, atol=1e-6)
    x = p2.x0.copy()
    x[1] = 10.0           # LED 0 rotates 10 deg more around the duct
    x[1 + n] = 0.7        # ... and slides 0.7 cm along the axis
    outp = np.asarray(group_config_to_factory(p2.decode(x)['custom_groups'][gi])['led_positions'], float)
    np.testing.assert_allclose(outp[1:], base[1:], atol=1e-6)
    assert np.linalg.norm((outp[0] - duct.center)[:2]) == pytest.approx(np.linalg.norm((base[0] - duct.center)[:2]))
    assert outp[0, 2] - base[0, 2] == pytest.approx(0.7)
    ang0 = np.degrees(np.arctan2(*(outp - duct.center)[0, [1, 0]]) - np.arctan2(*(base - duct.center)[0, [1, 0]]))
    assert ang0 == pytest.approx(10.0, abs=1e-6)


def test_panel_pose_and_led_states_modify_base_groups():
    cfg = load_config("tests/data/v1_configs/Elios3.json")
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
    cfg = load_config("tests/data/v1_configs/elios3.json")
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


def test_lumens_only_modes_without_electrical_model():
    """Default: fluxes come from the modes in lumens; no driver / current bookkeeping."""
    cfg = load_config("tests/data/v1_configs/elios3.json")
    var = DuctRingLayout(name="d", duct=Duct(center=(8, 9, 0), radius=7), n_leds=4, placement="arc",
                         tilt_axial_range=None, lumens=250.0)
    assert not any("current" in n for n in var.names)
    modes = [ModeSpec(name="flight", lumens=500.0), ModeSpec(name="flash", flash=True, lumens=20000.0,
                                                               min_avg_lux=1000.0)]
    problem = Problem(cfg, [var], WALL, CAM, clear_base=True, modes=modes,
                      constraints=ConstraintSpec(driver_cost=0.5, max_total_current_a=0.1))
    assert problem.flash_lumens == 20000.0 and problem.flight_lumens == 500.0
    cfgd = problem.decode(problem.x0)
    assert cfgd['custom_groups'][0]['lumens_value'] == 250.0  # group override (ignored: flight lumens wins)
    ev = problem.evaluate(problem.x0, keep_grid=True)
    assert ev.n_drivers == 0 and ev.total_current_a == 0.0
    assert set(ev.electrical) == {'n_vio', 'n_flash', 'n_both'} and ev.electrical['n_both'] == 4
    assert not {'driver_cost', 'current', 'peak_current'} & set(ev.penalties)
    assert ev.modes['flight']['lumens'] == 500.0 and ev.modes['flash']['lumens'] == 20000.0
    grid = ev.grid if not isinstance(ev.grid, list) else ev.grid[0]
    fgrid = ev.flight_grid if not isinstance(ev.flight_grid, list) else ev.flight_grid[0]
    assert float(grid.sum()) == pytest.approx(float(fgrid.sum()) * 40.0, rel=0.15)
    # the same design with the electrical model on reports drivers and currents
    prob_el = Problem(cfg, [var], WALL, CAM, clear_base=True, modes=modes, electrical=True,
                      constraints=ConstraintSpec(driver_cost=0.5))
    ev2 = prob_el.evaluate(problem.x0)
    assert ev2.n_drivers == 4 and ev2.penalties['driver_cost'] == pytest.approx(2.0)
    assert ev2.electrical['peak_current_a'] == pytest.approx(4 * 20000.0 / (6.0 * 180.0))
    with pytest.raises(ValueError, match="needs 'lumens'"):
        Problem(cfg, [var], WALL, CAM, modes=[ModeSpec(name="flash", flash=True)])
    spec, spec_dir = load_spec("optimization_specs/elios4_ducts_flash_vio.json")
    spec.pop('electrical')
    with pytest.raises(ValueError, match="electrical"):
        problem_from_spec(spec, spec_dir)


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
    cfg = load_config("tests/data/v1_configs/elios3.json")
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
    cfg = load_config("tests/data/v1_configs/elios3.json")
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
    spec['vio'].update({"room_grid_size": 8, "geometry": "room"})  # obsolete key must be ignored
    spec['objective'].update({"tilt_room_grid_size": 12})
    problem = problem_from_spec(spec, spec_dir)
    names = problem.names
    assert names[:2] == ["front_duct.n_rows", "front_duct.n_cols"] and problem.integrality[:2].all()
    assert "front_duct.radial" in names and names[-1] == "front_duct.current_a"
    assert len(problem._tilt_rooms) == 3 and set(problem._vio_masks) == {'front', 'left', 'right', 'top', 'bottom'}

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
    # T1 judges the flash image: all LEDs are 'both', so it is the flight image at 13 A instead of I_design
    scale = 13.0 / g['drive_current_a']
    evg = problem.evaluate(x, keep_grid=True)
    grids = evg.grid if isinstance(evg.grid, list) else [evg.grid]
    fgrids = evg.flight_grid if isinstance(evg.flight_grid, list) else [evg.flight_grid]
    for gflash, gflight in zip(grids, fgrids):
        assert float(gflash.sum()) == pytest.approx(float(gflight.sum()) * scale, rel=0.1)
    assert ev.modes['flash']['uniformity_pct'] == pytest.approx(ev.uniformity_pct)
    assert ev.modes['flash']['n_leds'] == ev.modes['normal']['n_leds'] == 12
    assert ev.electrical['n_pulse_drivers'] == 3 and ev.electrical['n_cont_drivers'] == 0
    assert ev.electrical['peak_current_a'] == pytest.approx(12 * 13.0)
    assert 0.0 <= ev.modes['normal']['vio_fraction'] <= 1.0
    # T2 room grids (MC) and T3 room grids (analytic, one per wall distance)
    assert set(evg.vio_grid) == {'front', 'left', 'right', 'top', 'bottom'}
    assert len(evg.tilt_grids) == 3 and all(gr['front'].shape == (12, 12) for gr in evg.tilt_grids)
    assert 'tilt_uniformity' in ev.penalties and len(ev.tilt_walls) == 3 and set(ev.tilt) == {'up', 'down'}
    ev2 = problem.evaluate(x)
    assert ev2.tilt == ev.tilt  # analytic T3 carries no Monte-Carlo noise

    summary, best = run(problem, OptimizerSpec(method="random_search", max_evals=8, population=4, log_every=0),
                        output_dir=tmp_path)
    out = tmp_path / problem.name
    assert summary['report'] and (out / "report.pdf").stat().st_size > 10_000
    assert (out / "initial_config.json").exists()
    with open(out / "report.pdf", "rb") as f:
        assert f.read(5) == b"%PDF-"

    # sensitivity analysis on synthetic records: score driven by variable 0 only
    from lighting_simulator.optimization import sensitivity as S
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in problem.bounds]); hi = np.array([b[1] for b in problem.bounds])
    xs = lo + rng.random((40, problem.dim)) * (hi - lo)
    xs[:, :2] = np.round(xs[:, :2])
    recs = []
    for i, x in enumerate(xs):
        e = problem.evaluate(x)
        e.score = float(x[2] * 10 + rng.normal(0, 0.1))
        recs.append((i, e, x))
    res = S.analyse(problem, recs)
    assert res.corr.shape == (problem.dim + 6, len(res.outcome_names))
    assert res.outcome_names[0] == 'score' and {'T2 VIO %', 'T3 U0 up %'} <= set(res.outcome_names)
    assert res.corr[2, 0] > 0.95 and res.ranking()[0] == 2 and res.surrogate_r2 > 0.9
    assert np.isfinite(res.X[:, problem.dim]).all()  # active LEDs decoded for every record
    assert S.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert S.spearman([1, 1, 1], [1, 2, 3]) != S.spearman([1, 1, 1], [1, 2, 3])  # NaN on constant input
