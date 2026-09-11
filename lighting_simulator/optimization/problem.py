"""Optimisation problem: decision vector → config → scene → score.

Score is minimised. It combines the uniformity metric inside the main-camera
footprint with soft penalties for the configurable constraints. All knobs
come from a JSON spec (see ``problem_from_spec`` and ``optimization_specs/``).

Operating modes (``ModeSpec``) share the geometry and differ only by drive
current, so the wall is traced once and the lux grids are rescaled per mode
(``lm ∝ I``): e.g. a *flash* mode that must reach 41 klx at 50 cm and a
*normal* mode that must light ≥ 50 % of the VIO field of view at 3 m.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lighting_simulator.analysis.uniformity import trapezoid_mask, uniformity_metrics
from lighting_simulator.camera.fov import camera_fov_wall_trapezoid, points_in_fisheye_fov, vio_hfov_vfov_deg
from lighting_simulator.scene.builder import build_scene_from_config, load_config
from lighting_simulator.simulation.room_geometry import wall_grid_cell_centers_cm
from lighting_simulator.simulation.settings import EmissionSettings, WallSettings
from lighting_simulator.simulation.wall import compute_wall_intensity

from .electrical import DriverModel
from .variables import variable_from_spec


@dataclass
class CameraSpec:
    """Main camera used to crop the wall grid (same convention as the UI 'FOV' tab)."""

    pos_x: float = 10.0
    pos_y: float = -4.0
    pitch: float = 0.0
    fov_h: float = 75.0
    fov_v: float = 60.0

    def trapezoid(self, wall_dist):
        trap = camera_fov_wall_trapezoid(wall_dist - self.pos_x, self.pitch, self.fov_h, self.fov_v)
        return (*trap, self.pos_y)


@dataclass
class ObjectiveSpec:
    metric: str = "u0"
    """'u0' (Emin/Eavg), 'u1' (Emin/Emax) or 'cv' (σ/Eavg, minimised directly)."""
    min_percentile: float = 0.0
    """Use this percentile of lit FOV cells as Emin (0 = hard minimum, as in the UI)."""
    coverage_weight: float = 1.0
    """Penalty weight for the fraction of FOV cells that receive no light."""
    min_avg_lux: float | None = None
    lux_weight: float = 1.0
    """Penalty weight for (min_avg_lux − Eavg)/min_avg_lux when below target."""


@dataclass
class VioSpec:
    """Two VD66GY fisheye VIO cameras (same convention as the UI 'VIO Cameras' folder).

    Their footprint is evaluated on a dedicated far wall (``wall_dist`` cm) so
    the *fraction of VIO-visible wall cells above a lux threshold* can be scored.
    """

    position: tuple = (10.0, 0.0, 0.0)
    cam1_pitch: float = 45.0
    cam1_yaw: float = 0.0
    cam2_pitch: float = -45.0
    cam2_yaw: float = 0.0
    long_fov: float = 170.0
    landscape: bool = True
    wall_dist: float = 300.0
    wall_size: float = 1200.0
    grid_size: int = 40

    def wall_settings(self, rays_per_pixel):
        return WallSettings(wall_dist=self.wall_dist, grid_size=self.grid_size, wall_size=self.wall_size,
                            rays_per_pixel=rays_per_pixel)

    def mask(self):
        """Union of both cameras' footprints on the VIO wall grid."""
        pts = wall_grid_cell_centers_cm((self.grid_size, self.grid_size), self.wall_size, self.wall_dist)
        hfov, vfov = vio_hfov_vfov_deg(self.long_fov, self.landscape)
        pos = np.asarray(self.position, float)
        m1 = points_in_fisheye_fov(pos, self.cam1_pitch, self.cam1_yaw, hfov, vfov, pts)
        m2 = points_in_fisheye_fov(pos, self.cam2_pitch, self.cam2_yaw, hfov, vfov, pts)
        return m1 | m2


@dataclass
class ModeSpec:
    """An operating point of the same hardware (e.g. 'normal' vs 'flash').

    Exactly one of ``current_a`` (absolute per-LED current, scaled against the
    design current implied by the LED lumens) or ``lumens_scale`` sets the flux
    multiplier relative to the decoded config.
    """

    name: str = "normal"
    current_a: float | None = None
    lumens_scale: float = 1.0
    min_avg_lux: float | None = None
    """Average lux target inside the main-camera FOV (e.g. 41000 for flash)."""
    min_avg_lux_dist: float | None = None
    """Wall distance (cm) the lux target refers to; nearest entry of ``wall_dists`` is used."""
    lux_weight: float = 1.0
    vio_min_lux: float | None = None
    """Lux threshold on the VIO wall (e.g. 120 at 3 m)."""
    vio_min_fraction: float = 0.5
    """Required share of VIO-visible cells above ``vio_min_lux``."""
    vio_weight: float = 1.0


@dataclass
class ConstraintSpec:
    max_leds: int | None = None
    max_leds_weight: float = 0.05
    """Penalty per LED above ``max_leds``."""
    led_cost: float = 0.0
    """Penalty per active LED (prefer fewer lights)."""
    max_drivers: int | None = None
    max_drivers_weight: float = 0.05
    driver_cost: float = 0.0
    """Penalty per driver IC (``ceil(n_leds / driver.leds_per_driver)``)."""
    max_total_current_a: float | None = None
    current_weight: float = 1.0
    """Penalty for the relative excess of summed LED current (thermal / supply budget)."""
    min_led_spacing_cm: float | None = None
    spacing_weight: float = 1.0
    keep_out: list = field(default_factory=list)
    """Axis-aligned boxes ``{'center': [x,y,z], 'half_sizes': [hx,hy,hz]}`` LEDs must avoid."""
    keep_out_weight: float = 1.0
    min_beam_angle_deg: float | None = None
    """Prefer beams tilted at least this far from ``beam_angle_axis`` (e.g. 30° off the camera axis)."""
    beam_angle_axis: tuple = (1.0, 0.0, 0.0)
    beam_angle_weight: float = 1.0
    symmetry_weight: float = 0.0
    """Penalty on the share of LEDs without an XZ-mirror partner within ``symmetry_tol_cm``."""
    symmetry_tol_cm: float = 0.5


@dataclass
class Evaluation:
    score: float
    uniformity_pct: float
    coverage: float
    e_avg: float
    n_active: int
    penalties: dict
    metrics: object = None
    grid: np.ndarray | None = None
    vio_grid: np.ndarray | None = None
    n_drivers: int = 0
    total_current_a: float = 0.0
    modes: dict = field(default_factory=dict)
    """Per-mode diagnostics: ``{name: {'e_avg', 'vio_fraction', 'scale'}}``."""

    def summary(self):
        pen = ", ".join(f"{k}={v:.3f}" for k, v in self.penalties.items() if v)
        s = (f"score={self.score:.4f} U={self.uniformity_pct:.1f}% cov={self.coverage*100:.0f}% "
             f"Eavg={self.e_avg:.0f}lx LEDs={self.n_active}")
        if self.n_drivers:
            s += f" drv={self.n_drivers} I={self.total_current_a:.1f}A"
        for name, m in self.modes.items():
            parts = [f"Eavg={m['e_avg']:.0f}lx"]
            if m.get('vio_fraction') is not None:
                parts.append(f"VIO={m['vio_fraction']*100:.0f}%")
            s += f" {name}[{' '.join(parts)}]"
        return s + (f" [{pen}]" if pen else "")


class Problem:
    def __init__(self, base_cfg, variables, wall: WallSettings, camera: CameraSpec,
                 emission: EmissionSettings | None = None, objective: ObjectiveSpec | None = None,
                 constraints: ConstraintSpec | None = None, clear_base=False, name="optim",
                 wall_dists=None, use_gpu=False, stl_mesh=None, diffuser=None,
                 driver: DriverModel | None = None, vio: VioSpec | None = None, modes=None):
        """``wall_dists``: optional list of distances (cm); the score is averaged over them
        so a layout is optimised for a range instead of a single wall distance.

        ``use_gpu`` traces on the GPU backend (single process only). ``stl_mesh`` /
        ``diffuser`` are forwarded to ``build_scene_from_config`` so the UI scene is
        reproduced exactly. ``driver`` converts LED flux to current and counts driver
        ICs; ``vio`` + ``modes`` add the per-operating-point lux / VIO-coverage targets."""
        self.name = name
        self.use_gpu = bool(use_gpu)
        self.stl_mesh = stl_mesh
        self.diffuser = diffuser
        self.driver = driver or DriverModel()
        self.vio = vio
        self.modes = list(modes or [])
        self.base_cfg = copy.deepcopy(base_cfg)
        if clear_base:
            self.base_cfg['custom_groups'] = []
            self.base_cfg['individual_leds'] = []
            self.base_cfg['led_states'] = [False] * len(self.base_cfg.get('led_states', [False] * 48))
            self.base_cfg['mirror_primary'] = None
        self.variables = list(variables)
        self.wall = wall
        self.camera = camera
        self.emission = emission or EmissionSettings()
        self.objective = objective or ObjectiveSpec()
        self.constraints = constraints or ConstraintSpec()
        dists = [float(d) for d in (wall_dists or [wall.wall_dist])]
        self.walls = [WallSettings(wall_dist=d, grid_size=wall.grid_size, wall_size=wall.wall_size,
                                   rays_per_pixel=wall.rays_per_pixel) for d in dists]
        self._fov_masks = [
            trapezoid_mask((w.grid_size, w.grid_size), w.wall_size, camera.trapezoid(w.wall_dist))
            for w in self.walls
        ]
        self._needs_vio = self.vio is not None and any(m.vio_min_lux for m in self.modes)
        if self._needs_vio:
            self._vio_wall = self.vio.wall_settings(wall.rays_per_pixel)
            self._vio_mask = self.vio.mask()
            if not np.any(self._vio_mask):
                raise ValueError("VIO cameras do not see the VIO wall: enlarge vio.wall_size or check the poses")
        self.n_evals = 0

    # -- decision vector -------------------------------------------------
    @property
    def names(self):
        return [n for v in self.variables for n in v.names]

    @property
    def bounds(self):
        return [b for v in self.variables for b in v.bounds]

    @property
    def integrality(self):
        return np.array([i for v in self.variables for i in v.integrality], dtype=bool)

    @property
    def x0(self):
        return np.array([x for v in self.variables for x in v.x0], dtype=float)

    @property
    def dim(self):
        return len(self.names)

    def decode(self, x):
        """Config dict for a decision vector."""
        x = np.asarray(x, dtype=float)
        cfg = copy.deepcopy(self.base_cfg)
        offset = 0
        for var in self.variables:
            var.apply(x[offset:offset + var.size], cfg)
            offset += var.size
        return cfg

    # -- evaluation ------------------------------------------------------
    def evaluate(self, x, keep_grid=False) -> Evaluation:
        return self.evaluate_config(self.decode(x), keep_grid=keep_grid)

    def evaluate_config(self, cfg, keep_grid=False) -> Evaluation:
        """Score any saved config under this problem's objective (variables not applied).

        Lets a hand-made design be compared with optimiser output on identical
        wall / camera / emission settings.
        """
        scene = self.build_scene(cfg)
        active = scene.active_leds
        currents = self.driver.led_currents(active)
        n_drivers = self.driver.n_drivers(len(active))
        total_current = float(currents.sum()) if len(active) else 0.0
        penalties = self._geometry_penalties(active, n_drivers, total_current)

        if not active:
            return Evaluation(score=10.0 + sum(penalties.values()), uniformity_pct=0.0, coverage=0.0,
                              e_avg=0.0, n_active=0, penalties=penalties)

        scores, unis, e_avgs, covs, grids, last_metrics = [], [], [], [], [], None
        for wall, fov_mask in zip(self.walls, self._fov_masks):
            grid = self._trace(scene, wall)
            fov = grid[fov_mask]
            coverage = float(np.count_nonzero(fov > 0) / max(1, fov.size))
            metrics = uniformity_metrics(fov, self.objective.min_percentile)
            if metrics is None:
                scores.append(5.0)
                unis.append(0.0)
                e_avgs.append(0.0)
            else:
                m = self.objective.metric
                scores.append({'u0': 1.0 - metrics.u0, 'u1': 1.0 - metrics.u1, 'cv': metrics.cv}[m])
                unis.append(metrics.uniformity_pct)
                e_avgs.append(metrics.e_avg)
                last_metrics = metrics
            covs.append(coverage)
            grids.append(grid)
        self.n_evals += 1

        score = float(np.mean(scores))
        uniformity_pct = float(np.mean(unis))
        e_avg = float(np.mean(e_avgs))
        coverage = float(min(covs))
        if self.objective.min_avg_lux and e_avg > 0:
            short = max(0.0, (self.objective.min_avg_lux - min(e_avgs)) / self.objective.min_avg_lux)
            penalties['lux'] = self.objective.lux_weight * short
        penalties['coverage'] = self.objective.coverage_weight * (1.0 - coverage)

        modes = {}
        vio_grid = None
        if self.modes:
            vio_lit = None
            if self._needs_vio:
                vio_grid = self._trace(scene, self._vio_wall)
                vio_lit = vio_grid[self._vio_mask]
            design_current = float(currents.mean())
            for mode in self.modes:
                modes[mode.name] = self._mode_penalties(mode, design_current, e_avgs, vio_lit, penalties)

        score += sum(penalties.values())
        return Evaluation(score=score, uniformity_pct=uniformity_pct, coverage=coverage,
                          e_avg=e_avg, n_active=len(active), penalties=penalties, metrics=last_metrics,
                          grid=(grids[0] if len(grids) == 1 else grids) if keep_grid else None,
                          vio_grid=vio_grid if keep_grid else None,
                          n_drivers=n_drivers, total_current_a=total_current, modes=modes)

    def __call__(self, x):
        return self.evaluate(x).score

    def build_scene(self, cfg):
        return build_scene_from_config(cfg, default_lumens=self.emission.default_lumens,
                                       stl_mesh=self.stl_mesh, diffuser=self.diffuser)

    def _trace(self, scene, wall):
        grid = compute_wall_intensity(scene.leds, wall, self.emission, absorbers=scene.absorbers,
                                      stl_mesh_data=scene.stl_mesh_data, use_gpu=self.use_gpu,
                                      verbose=False, parallel=False)
        return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

    def _mode_penalties(self, mode: ModeSpec, design_current, e_avgs, vio_lit, penalties):
        """Lux / VIO targets of one operating point; flux scales linearly with current."""
        if mode.current_a is not None:
            scale = mode.current_a / design_current if design_current > 0 else 0.0
        else:
            scale = float(mode.lumens_scale)
        info = {'scale': scale, 'e_avg': 0.0, 'vio_fraction': None}
        if mode.min_avg_lux_dist is not None:
            i = int(np.argmin([abs(w.wall_dist - mode.min_avg_lux_dist) for w in self.walls]))
            e_ref = e_avgs[i]
        else:
            e_ref = min(e_avgs)
        info['e_avg'] = e_ref * scale
        if mode.min_avg_lux:
            short = max(0.0, (mode.min_avg_lux - info['e_avg']) / mode.min_avg_lux)
            if short > 0:
                penalties[f'{mode.name}.lux'] = mode.lux_weight * short
        if mode.vio_min_lux and vio_lit is not None:
            frac = float(np.count_nonzero(vio_lit * scale >= mode.vio_min_lux) / max(1, vio_lit.size))
            info['vio_fraction'] = frac
            short = max(0.0, (mode.vio_min_fraction - frac) / max(1e-9, mode.vio_min_fraction))
            if short > 0:
                penalties[f'{mode.name}.vio'] = mode.vio_weight * short
        return info

    def _geometry_penalties(self, active, n_drivers=0, total_current=0.0):
        c = self.constraints
        pen = {}
        n = len(active)
        if c.led_cost:
            pen['led_cost'] = c.led_cost * n
        if c.max_leds is not None and n > c.max_leds:
            pen['max_leds'] = c.max_leds_weight * (n - c.max_leds)
        if c.driver_cost:
            pen['driver_cost'] = c.driver_cost * n_drivers
        if c.max_drivers is not None and n_drivers > c.max_drivers:
            pen['max_drivers'] = c.max_drivers_weight * (n_drivers - c.max_drivers)
        if c.max_total_current_a and total_current > c.max_total_current_a:
            pen['current'] = c.current_weight * (total_current - c.max_total_current_a) / c.max_total_current_a
        if n and c.min_beam_angle_deg:
            dirs = np.array([led.direction for led in active], dtype=float)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
            axis = np.asarray(c.beam_angle_axis, float)
            axis /= max(np.linalg.norm(axis), 1e-12)
            ang = np.degrees(np.arccos(np.clip(dirs @ axis, -1.0, 1.0)))
            short = np.clip(c.min_beam_angle_deg - ang, 0.0, None) / c.min_beam_angle_deg
            if np.any(short > 0):
                pen['beam_angle'] = c.beam_angle_weight * float(short.mean())
        if n and (c.min_led_spacing_cm or c.keep_out or c.symmetry_weight):
            pos = np.array([led.position for led in active], dtype=float)
            if c.symmetry_weight and n > 1:
                mirrored = pos * np.array([1.0, -1.0, 1.0])
                d = np.linalg.norm(pos[:, None, :] - mirrored[None, :, :], axis=-1)
                unmatched = np.count_nonzero(d.min(axis=1) > c.symmetry_tol_cm)
                if unmatched:
                    pen['symmetry'] = c.symmetry_weight * unmatched / n
            if c.min_led_spacing_cm and n > 1:
                d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                iu = np.triu_indices(n, k=1)
                viol = np.clip(c.min_led_spacing_cm - d[iu], 0.0, None) / c.min_led_spacing_cm
                if np.any(viol > 0):
                    pen['spacing'] = c.spacing_weight * float(viol.sum())
            for i, box in enumerate(c.keep_out):
                center = np.asarray(box['center'], float)
                half = np.asarray(box['half_sizes'], float)
                inside = np.all(np.abs(pos - center) < half, axis=1)
                if np.any(inside):
                    # depth of penetration, normalised by the box half size
                    depth = np.min((half - np.abs(pos[inside] - center)) / half, axis=1)
                    pen[f'keep_out{i}'] = c.keep_out_weight * float(depth.sum())
        return pen


def problem_from_spec(spec, spec_dir: Path | None = None, base_cfg=None, **problem_kwargs) -> Problem:
    """Build a Problem from a JSON-like spec dict (see optimization_specs/*.json).

    ``base_cfg`` overrides ``spec['base_config']`` (e.g. the live UI scene); extra
    keyword arguments are forwarded to ``Problem`` (``stl_mesh``, ``diffuser``...).
    """
    spec_dir = Path(spec_dir or ".")
    if base_cfg is None:
        base_path = Path(spec['base_config'])
        if not base_path.is_absolute() and not base_path.exists():
            base_path = spec_dir / base_path
        base_cfg = load_config(base_path)
        default_name = base_path.stem + "_optim"
    else:
        base_cfg = copy.deepcopy(base_cfg)
        default_name = str(base_cfg.get('name') or 'scene').lower().replace(' ', '_') + "_optim"

    wall_spec = dict(spec.get('wall', {}))
    dist = wall_spec.get('wall_dist', 100.0)
    wall_dists = [float(d) for d in dist] if isinstance(dist, (list, tuple)) else [float(dist)]
    wall_spec['wall_dist'] = wall_dists[0]
    wall = WallSettings(**wall_spec)
    camera = CameraSpec(**spec.get('camera', {}))
    emission = EmissionSettings(**spec.get('emission', {}))
    objective = ObjectiveSpec(**spec.get('objective', {}))
    constraints = ConstraintSpec(**spec.get('constraints', {}))
    clear_base = bool(spec.get('clear_base', False))
    driver = DriverModel(**spec.get('driver', {}))
    vio = VioSpec(**spec['vio']) if spec.get('vio') else None
    modes = [ModeSpec(**m) for m in spec.get('modes', [])]

    working = copy.deepcopy(base_cfg)
    if clear_base:
        working['custom_groups'] = []
    variables = []
    for v in spec['variables']:
        v = dict(v)
        if v.get('type') in ('duct_ring', 'group_current'):
            v.setdefault('driver', copy.deepcopy(spec.get('driver', {})))
        variables.append(variable_from_spec(v, working))
    problem_kwargs.setdefault('use_gpu', bool(spec.get('use_gpu', False)))
    return Problem(base_cfg, variables, wall, camera, emission, objective, constraints,
                   clear_base=clear_base, name=spec.get('name', default_name),
                   wall_dists=wall_dists, driver=driver, vio=vio, modes=modes, **problem_kwargs)


def load_spec(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path.parent
