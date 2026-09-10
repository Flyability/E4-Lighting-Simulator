"""Optimisation problem: decision vector → config → scene → score.

Score is minimised. It combines the uniformity metric inside the main-camera
footprint with soft penalties for the configurable constraints. All knobs
come from a JSON spec (see ``problem_from_spec`` and ``optimization_specs/``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lighting_simulator.analysis.uniformity import trapezoid_mask, uniformity_metrics
from lighting_simulator.camera.fov import camera_fov_wall_trapezoid
from lighting_simulator.scene.builder import build_scene_from_config, load_config
from lighting_simulator.simulation.settings import EmissionSettings, WallSettings
from lighting_simulator.simulation.wall import compute_wall_intensity

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
class ConstraintSpec:
    max_leds: int | None = None
    max_leds_weight: float = 0.05
    """Penalty per LED above ``max_leds``."""
    led_cost: float = 0.0
    """Penalty per active LED (prefer fewer lights)."""
    min_led_spacing_cm: float | None = None
    spacing_weight: float = 1.0
    keep_out: list = field(default_factory=list)
    """Axis-aligned boxes ``{'center': [x,y,z], 'half_sizes': [hx,hy,hz]}`` LEDs must avoid."""
    keep_out_weight: float = 1.0


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

    def summary(self):
        pen = ", ".join(f"{k}={v:.3f}" for k, v in self.penalties.items() if v)
        return (f"score={self.score:.4f} U={self.uniformity_pct:.1f}% cov={self.coverage*100:.0f}% "
                f"Eavg={self.e_avg:.0f}lx LEDs={self.n_active}" + (f" [{pen}]" if pen else ""))


class Problem:
    def __init__(self, base_cfg, variables, wall: WallSettings, camera: CameraSpec,
                 emission: EmissionSettings | None = None, objective: ObjectiveSpec | None = None,
                 constraints: ConstraintSpec | None = None, clear_base=False, name="optim",
                 wall_dists=None):
        """``wall_dists``: optional list of distances (cm); the score is averaged over them
        so a layout is optimised for a range instead of a single wall distance."""
        self.name = name
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
        scene = build_scene_from_config(cfg, default_lumens=self.emission.default_lumens)
        active = scene.active_leds
        penalties = self._geometry_penalties(active)

        if not active:
            return Evaluation(score=10.0 + sum(penalties.values()), uniformity_pct=0.0, coverage=0.0,
                              e_avg=0.0, n_active=0, penalties=penalties)

        scores, unis, e_avgs, covs, grids, last_metrics = [], [], [], [], [], None
        for wall, fov_mask in zip(self.walls, self._fov_masks):
            grid = compute_wall_intensity(scene.leds, wall, self.emission, absorbers=scene.absorbers,
                                          stl_mesh_data=scene.stl_mesh_data, use_gpu=False, verbose=False,
                                          parallel=False)
            grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
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
        score += sum(penalties.values())
        return Evaluation(score=score, uniformity_pct=uniformity_pct, coverage=coverage,
                          e_avg=e_avg, n_active=len(active), penalties=penalties, metrics=last_metrics,
                          grid=(grids[0] if len(grids) == 1 else grids) if keep_grid else None)

    def __call__(self, x):
        return self.evaluate(x).score

    def _geometry_penalties(self, active):
        c = self.constraints
        pen = {}
        n = len(active)
        if c.led_cost:
            pen['led_cost'] = c.led_cost * n
        if c.max_leds is not None and n > c.max_leds:
            pen['max_leds'] = c.max_leds_weight * (n - c.max_leds)
        if n and (c.min_led_spacing_cm or c.keep_out):
            pos = np.array([led.position for led in active], dtype=float)
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


def problem_from_spec(spec, spec_dir: Path | None = None) -> Problem:
    """Build a Problem from a JSON-like spec dict (see optimization_specs/*.json)."""
    spec_dir = Path(spec_dir or ".")
    base_path = Path(spec['base_config'])
    if not base_path.is_absolute() and not base_path.exists():
        base_path = spec_dir / base_path
    base_cfg = load_config(base_path)

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

    working = copy.deepcopy(base_cfg)
    if clear_base:
        working['custom_groups'] = []
    variables = [variable_from_spec(v, working) for v in spec['variables']]
    return Problem(base_cfg, variables, wall, camera, emission, objective, constraints,
                   clear_base=clear_base, name=spec.get('name', base_path.stem + "_optim"),
                   wall_dists=wall_dists)


def load_spec(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path.parent
