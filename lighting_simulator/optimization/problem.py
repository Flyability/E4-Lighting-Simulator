"""Optimisation problem: decision vector → config → scene → score.

Score is minimised. Every design is judged on three independent *test cases* plus
geometry / electrical penalties (see ``Problem.evaluate_config``):

* **T1 – inspection image**: flat wall at every ``wall_dists`` distance, main camera
  untilted, U-metric + coverage + min-lux inside the FOV trapezoid. Uses the *flash*
  image when a flash mode is defined (the flight image is then only reported), the
  flight image otherwise. This is the base score (weight 1).
* **T2 – VIO coverage**: 5-wall room (no back wall) ``vio.room_dist`` cm away, flight
  image, the two fisheye VIO cameras only; fraction of visible cells ≥ ``vio_min_lux``.
* **T3 – tilted inspection image**: 5-wall room at each ``wall_dists`` distance, same
  image rule as T1, main camera pitched ±``tilt_fov_deg``; U-metric + coverage of the
  cells inside each tilted FOV, computed analytically (``simulation.direct``) on those
  cells only. Weight ``tilt_fov_weight`` (secondary objective).

Operating modes (``ModeSpec``) share the geometry. Each LED has a *role*
(``domain.led``): 'vio' LEDs are on continuously, 'flash' LEDs only fire during the
photogrammetry pulse, 'both' do both. The flash image is "vio LEDs at their continuous
flux + flash/both LEDs at the pulse flux", the flight image "vio + both at their
continuous flux". Each mode sets its per-LED flux directly (``ModeSpec.lumens``); the
electrical model (currents, drivers, ``DriverModel``) is optional (``Problem(electrical=True)``).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from lighting_simulator.analysis.uniformity import trapezoid_mask, uniformity_metrics
from lighting_simulator.camera.fov import (
    camera_fov_wall_trapezoid, points_in_fisheye_fov, points_in_pinhole_fov, vio_hfov_vfov_deg,
)
from lighting_simulator.domain.led import split_by_role
from lighting_simulator.raytracing.mesh import prepare_mesh_ray_accelerator
from lighting_simulator.scene.builder import build_scene_from_config, load_config
from lighting_simulator.simulation.direct import direct_illuminance
from lighting_simulator.simulation.emission import led_lumens
from lighting_simulator.simulation.room import compute_room_intensity
from lighting_simulator.simulation.room_geometry import (
    WALL_INWARD_NORMALS, build_wall_specs, room_wall_cell_centers,
)
from lighting_simulator.simulation.settings import EmissionSettings, RoomSettings, WallSettings
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

    def trapezoid(self, wall_dist, pitch_offset=0.0):
        trap = camera_fov_wall_trapezoid(wall_dist - self.pos_x, self.pitch + pitch_offset, self.fov_h, self.fov_v)
        return (*trap, self.pos_y)

    def fit_wall_size(self, wall_dist, margin=1.15, step=10.0):
        """Smallest centred square wall (cm) containing the footprint at ``wall_dist`` × ``margin``."""
        z_bot, z_top, w_bot, w_top, y_c = self.trapezoid(wall_dist)
        extent = max(abs(z_bot), abs(z_top), abs(y_c) + max(w_bot, w_top))
        size = 2.0 * extent * float(margin)
        return float(max(step, np.ceil(size / step) * step))


@dataclass
class ObjectiveSpec:
    metric: str = "u0"
    """'u0' (Emin/Eavg), 'u1' (Emin/Emax) or 'cv' (σ/Eavg, minimised directly)."""
    min_percentile: float = 0.0
    """Use this percentile of lit FOV cells as Emin (0 = hard minimum, as in the UI)."""
    coverage_weight: float = 1.0
    """Penalty weight for the fraction of FOV cells that receive no light (T1 and T3)."""
    min_avg_lux: float | None = None
    lux_weight: float = 1.0
    """Penalty weight for (min_avg_lux − Eavg)/min_avg_lux when below target (T1 image)."""
    tilt_fov_deg: float | None = None
    """T3: also score the main camera pitched ±this many degrees inside a room at each wall distance; None = off."""
    tilt_fov_weight: float = 0.3
    """Weight of T3 (mean over distances × {up, down} of the U-metric + coverage term), penalty 'tilt_uniformity'."""
    tilt_room_grid_size: int = 32
    """T3 room grid per wall (independent of the VIO room resolution)."""


@dataclass
class VioSpec:
    """Two VD66GY fisheye VIO cameras (same convention as the UI 'VIO Cameras' folder).

    T2 evaluates the *flight* image on the five walls of a room ``room_dist`` cm from the rig
    in every direction (front, left, right, top, bottom — no back wall; coarse
    ``room_grid_size``² grid per wall). The score is the fraction of VIO-visible cells above
    a lux threshold (``ModeSpec.vio_min_lux`` on the flight mode).
    """

    position: tuple = (10.0, 0.0, 0.0)
    cam1_pitch: float = 45.0
    cam1_yaw: float = 0.0
    cam2_pitch: float = -45.0
    cam2_yaw: float = 0.0
    long_fov: float = 170.0
    landscape: bool = True
    room_dist: float = 300.0
    """Distance (cm) to each of the five room walls (opposite walls are 2× this apart)."""
    room_grid_size: int = 20

    def room_settings(self, rays_per_pixel=1):
        return _room_settings(self.room_dist, self.room_grid_size, rays_per_pixel)

    def room_wall_specs(self):
        return _room_wall_specs(self.room_settings())

    def _fisheye_union(self, pts):
        hfov, vfov = vio_hfov_vfov_deg(self.long_fov, self.landscape)
        pos = np.asarray(self.position, float)
        m1 = points_in_fisheye_fov(pos, self.cam1_pitch, self.cam1_yaw, hfov, vfov, pts)
        m2 = points_in_fisheye_fov(pos, self.cam2_pitch, self.cam2_yaw, hfov, vfov, pts)
        return m1 | m2

    def room_masks(self):
        """``{wall: bool grid}`` of cells seen by at least one VIO camera."""
        s = self.room_settings()
        return {name: self._fisheye_union(room_wall_cell_centers(name, spec, s.front_dist, s.side_dist,
                                                                 s.top_bottom_dist))
                for name, spec in self.room_wall_specs().items()}


def _room_settings(dist, grid_size, rays_per_pixel=1):
    """Cube room ``dist`` cm from the rig in every direction, no bounces.

    The back wall bounds the geometry (rays flying backwards stop there) but its grid is dropped
    from every result: the optimiser only judges the front, side, top and bottom walls.
    """
    d = float(dist)
    return RoomSettings(front_dist=d, side_dist=d, top_bottom_dist=d, back_dist=d, lateral_depth=2 * d,
                        grid_size=int(grid_size), rays_per_pixel=int(rays_per_pixel),
                        max_bounces=0, wall_reflectance=0.0)


def _room_wall_specs(s: RoomSettings):
    specs = build_wall_specs(s.front_dist, s.side_dist, s.top_bottom_dist, s.grid_size, s.led_x_center,
                             s.back_dist, s.lateral_depth)
    specs.pop('back', None)
    return specs


class TiltRoom:
    """T3 geometry at one wall distance: the room cells and the ±tilt main-camera footprints on them.

    ``pts`` / ``nrm`` are all cell centres (cm) and inward normals, concatenated wall by wall;
    ``masks['up'|'down']`` are flat booleans over ``pts``; ``layout`` maps a wall to its slice + shape.
    """

    def __init__(self, dist, grid_size, camera: CameraSpec, tilt_deg):
        self.dist = float(dist)
        self.settings = _room_settings(dist, grid_size)
        self.specs = _room_wall_specs(self.settings)
        s = self.settings
        cam_pos = np.array([camera.pos_x, camera.pos_y, 0.0])
        pts, nrm, up, down, self.layout = [], [], [], [], {}
        start = 0
        for name, spec in self.specs.items():
            p = room_wall_cell_centers(name, spec, s.front_dist, s.side_dist, s.top_bottom_dist)
            flat = p.reshape(-1, 3)
            self.layout[name] = (start, start + len(flat), p.shape[:2])
            start += len(flat)
            pts.append(flat)
            nrm.append(np.tile(WALL_INWARD_NORMALS[name], (len(flat), 1)))
            up.append(points_in_pinhole_fov(cam_pos, camera.pitch + tilt_deg, camera.fov_h, camera.fov_v, flat))
            down.append(points_in_pinhole_fov(cam_pos, camera.pitch - tilt_deg, camera.fov_h, camera.fov_v, flat))
        self.pts = np.concatenate(pts)
        self.nrm = np.concatenate(nrm)
        self.masks = {'up': np.concatenate(up), 'down': np.concatenate(down)}
        self.scored = self.masks['up'] | self.masks['down']

    def grids(self, flat):
        """``{wall: grid}`` from a flat per-cell array."""
        return {name: flat[a:b].reshape(shape) for name, (a, b, shape) in self.layout.items()}

    def wall_masks(self, key):
        return self.grids(self.masks[key])


@dataclass
class ModeSpec:
    """An operating point of the same hardware (e.g. 'flight' vs 'flash').

    A *flash* (pulse) mode (``flash=True`` or ``current_a`` set): 'flash' and 'both' LEDs run at
    the pulse flux while 'vio' LEDs keep their continuous flux. Any other mode is a *flight*
    mode ('vio' + 'both' at continuous flux, ``lumens_scale`` applied).
    """

    name: str = "normal"
    lumens: float | None = None
    """Per-LED flux (lm) in this mode. Flight: replaces the scene flux of every LED (panel
    overrides included); flash: the pulse flux. None = scene flux (flight) / ``current_a`` via
    the driver model (flash)."""
    current_a: float | None = None
    """Pulse current (A); only converted to flux when ``lumens`` is not given."""
    lumens_scale: float = 1.0
    flash: bool | None = None
    """Force pulse (True) / flight (False) semantics; None = pulse iff ``current_a`` is set."""
    min_avg_lux: float | None = None
    """Average lux target inside the main-camera FOV (e.g. 41000 for flash)."""
    min_avg_lux_dist: float | None = None
    """Wall distance (cm) the lux target refers to; nearest entry of ``wall_dists`` is used."""
    lux_weight: float = 1.0
    vio_min_lux: float | None = None
    """T2: lux threshold on the VIO room walls (e.g. 120 at 3 m); flight modes only."""
    vio_min_fraction: float = 0.5
    """Required share of VIO-visible cells above ``vio_min_lux``."""
    vio_weight: float = 1.0

    @property
    def is_flash(self):
        return self.flash if self.flash is not None else self.current_a is not None


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
    """Penalty per driver IC (all classes)."""
    max_pulse_drivers: int | None = None
    pulse_driver_cost: float = 0.0
    """Per pulse-class driver (flash / both LEDs) when a flash mode exists."""
    max_cont_drivers: int | None = None
    cont_driver_cost: float = 0.0
    """Per continuous-class driver ('vio' LEDs) when a flash mode exists."""
    max_total_current_a: float | None = None
    current_weight: float = 1.0
    """Penalty for the relative excess of summed continuous LED current (thermal / supply budget)."""
    max_peak_current_a: float | None = None
    peak_current_weight: float = 1.0
    """Penalty for the relative excess of the summed current during the flash pulse."""
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
    """T1 U-metric (%) of the scored image, mean over wall distances."""
    coverage: float
    e_avg: float
    n_active: int
    penalties: dict
    metrics: object = None
    grid: np.ndarray | None = None
    """T1 image per wall (flash image when a flash mode exists, else flight): array or list."""
    vio_grid: np.ndarray | None = None
    """T2 room grids ``{wall: grid}`` (flight image)."""
    n_drivers: int = 0
    total_current_a: float = 0.0
    """Summed continuous current (flight)."""
    modes: dict = field(default_factory=dict)
    """Per-mode diagnostics: ``{name: {'e_avg', 'vio_fraction', 'scale', 'uniformity_pct', 'n_leds', 'lumens'}}``."""
    tilt: dict = field(default_factory=dict)
    """T3 U-metric (%) ``{'up': .., 'down': ..}``, mean over wall distances, when ``tilt_fov_deg`` is set."""
    tilt_walls: list = field(default_factory=list)
    """T3 per wall distance: ``[{'up': U%, 'down': U%, 'cov_up': f, 'cov_down': f}, ...]``."""
    electrical: dict = field(default_factory=dict)
    """``{'n_vio', 'n_flash', 'n_both'}`` with a flash mode, plus ``{'n_pulse_drivers', 'n_cont_drivers',
    'peak_current_a'}`` when the electrical model is on."""
    flight_grid: np.ndarray | None = None
    """Flight image per wall when a flash mode exists (reported, not scored)."""
    tilt_grids: list | None = None
    """T3 room grids per wall distance ``[{wall: grid}, ...]`` (analytic, full room) when kept."""

    def summary(self):
        pen = ", ".join(f"{k}={v:.3f}" for k, v in self.penalties.items() if v)
        s = (f"score={self.score:.4f} T1[U={self.uniformity_pct:.1f}% cov={self.coverage*100:.0f}% "
             f"Eavg={self.e_avg:.0f}lx] LEDs={self.n_active}")
        if self.tilt:
            s += f" T3[↑{self.tilt.get('up', 0):.0f}% ↓{self.tilt.get('down', 0):.0f}%]"
        if self.n_drivers:
            s += f" drv={self.n_drivers} I={self.total_current_a:.1f}A"
        el = self.electrical
        if el:
            s += f" [vio {el.get('n_vio', 0)} / flash {el.get('n_flash', 0)} / both {el.get('n_both', 0)}"
            if 'n_pulse_drivers' in el:
                s += (f"; pulse drv {el['n_pulse_drivers']}, cont drv {el['n_cont_drivers']},"
                      f" peak {el['peak_current_a']:.1f}A")
            s += "]"
        for name, m in self.modes.items():
            parts = [f"Eavg={m['e_avg']:.0f}lx"]
            if m.get('uniformity_pct') is not None:
                parts.append(f"U={m['uniformity_pct']:.0f}%")
            if m.get('vio_fraction') is not None:
                parts.append(f"T2 VIO={m['vio_fraction']*100:.0f}%")
            s += f" {name}[{' '.join(parts)}]"
        return s + (f" [{pen}]" if pen else "")

    def averaged_with(self, other: "Evaluation") -> "Evaluation":
        """Mean of two independent Monte-Carlo evaluations of the same design."""
        keys = set(self.penalties) | set(other.penalties)
        pen = {k: 0.5 * (self.penalties.get(k, 0.0) + other.penalties.get(k, 0.0)) for k in keys}
        modes = {}
        for name, m in self.modes.items():
            o = other.modes.get(name, m)
            modes[name] = dict(m)
            modes[name]['e_avg'] = 0.5 * (m['e_avg'] + o['e_avg'])
            for k in ('vio_fraction', 'uniformity_pct'):
                if m.get(k) is not None:
                    modes[name][k] = 0.5 * (m[k] + (o.get(k) if o.get(k) is not None else m[k]))
        return Evaluation(
            score=0.5 * (self.score + other.score),
            uniformity_pct=0.5 * (self.uniformity_pct + other.uniformity_pct),
            coverage=0.5 * (self.coverage + other.coverage),
            e_avg=0.5 * (self.e_avg + other.e_avg),
            n_active=self.n_active, penalties=pen, metrics=self.metrics, grid=self.grid,
            vio_grid=self.vio_grid, n_drivers=self.n_drivers, total_current_a=self.total_current_a,
            modes=modes,
            tilt={k: 0.5 * (v + other.tilt.get(k, v)) for k, v in self.tilt.items()},
            tilt_walls=list(self.tilt_walls),  # T3 is analytic: identical in both evaluations
            electrical=dict(self.electrical), flight_grid=self.flight_grid, tilt_grids=self.tilt_grids,
        )


class Problem:
    def __init__(self, base_cfg, variables, wall: WallSettings, camera: CameraSpec,
                 emission: EmissionSettings | None = None, objective: ObjectiveSpec | None = None,
                 constraints: ConstraintSpec | None = None, clear_base=False, name="optim",
                 wall_dists=None, use_gpu=False, stl_mesh=None, diffuser=None,
                 driver: DriverModel | None = None, vio: VioSpec | None = None, modes=None,
                 wall_sizes=None, cont_driver: DriverModel | None = None, electrical=False):
        """``wall_dists``: optional list of distances (cm); the score is averaged over them
        so a layout is optimised for a range instead of a single wall distance.
        ``wall_sizes``: matching list of wall extents (cm), or ``"auto"`` to fit each wall
        to the camera footprint (keeps the FOV at full grid resolution at every distance).

        ``use_gpu`` traces on the GPU backend (single process only). ``stl_mesh`` /
        ``diffuser`` are forwarded to ``build_scene_from_config`` so the UI scene is
        reproduced exactly. ``vio`` + ``modes`` add the per-operating-point lux / VIO-coverage
        targets. ``electrical`` turns on the current / driver model: ``driver`` is the
        pulse-class driver (flash / both LEDs), ``cont_driver`` the continuous class ('vio'
        LEDs; defaults to ``driver``); off, fluxes come straight from the modes and no
        driver / current penalty is computed."""
        self.name = name
        self.use_gpu = bool(use_gpu)
        self.stl_mesh = stl_mesh
        self.diffuser = diffuser
        self.electrical = bool(electrical)
        self.driver = driver or DriverModel()
        self.cont_driver = cont_driver or self.driver
        self.vio = vio
        self.modes = list(modes or [])
        self.flash_modes = [m for m in self.modes if m.is_flash]
        if len(self.flash_modes) > 1:
            raise ValueError("at most one flash (pulse) mode is supported")
        if self.flash_modes and self.flash_modes[0].lumens is None and self.flash_modes[0].current_a is None:
            raise ValueError(f"flash mode '{self.flash_modes[0].name}' needs 'lumens' (or 'current_a')")
        self.flight_mode = next((m for m in self.modes if not m.is_flash), None)
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
        t = self.objective.tilt_fov_deg
        if isinstance(wall_sizes, str) and wall_sizes == "auto":
            sizes = [camera.fit_wall_size(d) for d in dists]
        elif wall_sizes is None:
            sizes = [float(wall.wall_size)] * len(dists)
        else:
            sizes = [float(s) for s in wall_sizes]
            if len(sizes) == 1:
                sizes = sizes * len(dists)
            if len(sizes) != len(dists):
                raise ValueError(f"wall_sizes has {len(sizes)} entries for {len(dists)} wall distances")
        # T1: flat walls + main-camera footprint
        self.walls = [WallSettings(wall_dist=d, grid_size=wall.grid_size, wall_size=s,
                                   rays_per_pixel=wall.rays_per_pixel) for d, s in zip(dists, sizes)]
        self._fov_masks = [
            trapezoid_mask((w.grid_size, w.grid_size), w.wall_size, camera.trapezoid(w.wall_dist))
            for w in self.walls
        ]
        # T3: a room at every wall distance, main camera pitched ±t
        self._tilt_rooms = []
        if t:
            self._tilt_rooms = [TiltRoom(d, self.objective.tilt_room_grid_size, camera, float(t)) for d in dists]
            if not all(r.scored.any() for r in self._tilt_rooms):
                raise ValueError("the ±tilt camera footprints miss the T3 room walls: check the camera pose")
            print(f"[optim] T3: ±{t:g}° tilt FOVs on 5-wall rooms at {', '.join(f'{d:g}' for d in dists)} cm "
                  f"({self.objective.tilt_room_grid_size}² cells/wall, analytic), w={self.objective.tilt_fov_weight:g}")
        # T2: VIO room, flight image, fisheye footprints
        self._needs_vio = self.vio is not None and any(m.vio_min_lux for m in self.modes if not m.is_flash)
        self._vio_masks = None
        if self._needs_vio:
            self._vio_masks = self.vio.room_masks()
            if not any(np.any(m) for m in self._vio_masks.values()):
                raise ValueError("VIO cameras do not see the VIO room walls: check the poses")
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

    @property
    def flash_lumens(self):
        """Pulse flux per flash / both LED (lm), or None without a flash mode."""
        if not self.flash_modes:
            return None
        m = self.flash_modes[0]
        return float(m.lumens) if m.lumens is not None else self.driver.lumens(m.current_a)

    @property
    def flight_lumens(self):
        """Continuous flux forced on every LED (lm), or None to keep the scene's fluxes."""
        m = self.flight_mode
        return float(m.lumens) if (m is not None and m.lumens is not None) else None

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
        if self.flight_lumens is not None:
            active = _with_lumens(active, self.flight_lumens)
        flash_mode = self.flash_modes[0] if self.flash_modes else None
        flash_lumens = self.flash_lumens
        by_role = split_by_role(active)
        flight_leds = by_role['vio'] + by_role['both']
        pulse_leds = by_role['flash'] + by_role['both']
        electrical = {}
        n_drivers, total_current, peak_current = 0, 0.0, 0.0
        if flash_mode is not None:
            electrical = {'n_vio': len(by_role['vio']), 'n_flash': len(by_role['flash']), 'n_both': len(by_role['both'])}
            if self.electrical:
                # Two driver classes: 'vio' LEDs on continuous drivers, flash/both on pulse drivers.
                cont_current = float(self.cont_driver.led_currents(flight_leds).sum()) if flight_leds else 0.0
                vio_current = float(self.cont_driver.led_currents(by_role['vio']).sum()) if by_role['vio'] else 0.0
                peak_current = vio_current + len(pulse_leds) * self.driver.current(flash_lumens)
                n_pulse_drv = self.driver.n_drivers(len(pulse_leds))
                n_cont_drv = self.cont_driver.n_drivers(len(by_role['vio']))
                n_drivers = n_pulse_drv + n_cont_drv
                total_current = cont_current
                electrical.update({'n_pulse_drivers': n_pulse_drv, 'n_cont_drivers': n_cont_drv,
                                   'peak_current_a': peak_current})
        else:
            flight_leds = active  # no pulse mode: every LED is continuous, roles are irrelevant
            if self.electrical:
                currents = self.driver.led_currents(active)
                n_drivers = self.driver.n_drivers(len(active))
                total_current = float(currents.sum()) if len(active) else 0.0
                peak_current = total_current
        penalties = self._geometry_penalties(active, n_drivers, total_current, electrical, peak_current)

        if not active:
            return Evaluation(score=10.0 + sum(penalties.values()), uniformity_pct=0.0, coverage=0.0,
                              e_avg=0.0, n_active=0, penalties=penalties)

        m = self.objective.metric
        cov_w = float(self.objective.coverage_weight)

        def _score_of(metrics):
            return {'u0': 1.0 - metrics.u0, 'u1': 1.0 - metrics.u1, 'cv': metrics.cv}[m]

        def _judge(values):
            """(score term, U %, E_avg, coverage) of the cells inside one camera footprint."""
            values = np.asarray(values)
            cov = float(np.count_nonzero(values > 0) / max(1, values.size))
            mt = uniformity_metrics(values, self.objective.min_percentile)
            if mt is None:
                return 5.0, 0.0, 0.0, cov, None
            return _score_of(mt), mt.uniformity_pct, mt.e_avg, cov, mt

        # ---- T1: flat wall × untilted main camera; the flash image is the one judged when it exists
        scores, unis, e_avgs, covs, grids, last_metrics = [], [], [], [], [], None
        fl_unis, fl_e_avgs, flight_grids = [], [], []
        for wall, fov_mask in zip(self.walls, self._fov_masks):
            flight_grid, flash_grid = self._trace_modes(scene, wall, active, by_role, flash_lumens)
            primary = flash_grid if flash_grid is not None else flight_grid
            sc, u, ea, cov, mt = _judge(primary[fov_mask])
            scores.append(sc); unis.append(u); e_avgs.append(ea); covs.append(cov); grids.append(primary)
            last_metrics = mt or last_metrics
            if flash_grid is not None:  # flight image: diagnostics + flight-mode lux target only
                _, fu, fe, _, _ = _judge(flight_grid[fov_mask])
                fl_unis.append(fu); fl_e_avgs.append(fe); flight_grids.append(flight_grid)
            else:
                fl_unis.append(u); fl_e_avgs.append(ea)
        self.n_evals += 1

        score = float(np.mean(scores))
        uniformity_pct = float(np.mean(unis))
        e_avg = float(np.mean(e_avgs))
        coverage = float(min(covs))
        if self.objective.min_avg_lux and e_avg > 0:
            short = max(0.0, (self.objective.min_avg_lux - min(e_avgs)) / self.objective.min_avg_lux)
            penalties['lux'] = self.objective.lux_weight * short
        penalties['coverage'] = cov_w * (1.0 - coverage)

        # ---- T3: rooms at the wall distances × main camera pitched ±t, same image rule as T1 (analytic)
        tilt, tilt_walls, tilt_grids = {}, [], []
        if self._tilt_rooms:
            if flash_mode is not None:
                t3_leds = by_role['vio'] + pulse_leds
                t3_lumens = ([led_lumens(p, self.emission.default_lumens) for p in by_role['vio']]
                             + [float(flash_lumens)] * len(pulse_leds))
            else:
                t3_leds, t3_lumens = flight_leds, None
            accel = prepare_mesh_ray_accelerator(scene.stl_mesh_data) if scene.stl_mesh_data is not None else None
            tilt_scores, tilt_unis = [], {'up': [], 'down': []}
            for room in self._tilt_rooms:
                sel = np.ones(len(room.pts), bool) if keep_grid else room.scored
                e = np.zeros(len(room.pts))
                e[sel] = direct_illuminance(room.pts[sel], room.nrm[sel], t3_leds, self.emission, lumens=t3_lumens,
                                            absorbers=scene.absorbers, accel=accel)
                rec = {}
                for key in ('up', 'down'):
                    sc, u, _, cov, _ = _judge(e[room.masks[key]])
                    tilt_scores.append(sc + cov_w * (1.0 - cov))
                    tilt_unis[key].append(u)
                    rec[key], rec[f'cov_{key}'] = u, cov
                tilt_walls.append(rec)
                if keep_grid:
                    tilt_grids.append(room.grids(e))
            penalties['tilt_uniformity'] = self.objective.tilt_fov_weight * float(np.mean(tilt_scores))
            tilt = {k: float(np.mean(v)) for k, v in tilt_unis.items()}

        # ---- T2: VIO room (flight image) × fisheye footprints; per-mode lux / coverage targets
        modes = {}
        room_grid = None
        if self.modes:
            vio_lit = None
            if self._needs_vio:
                room_grid = self._trace_room(scene, flight_leds, len(active))
                vio_lit = np.concatenate([room_grid[n][mk] for n, mk in self._vio_masks.items() if n in room_grid])
            for mode in self.modes:
                if mode.is_flash:
                    info = self._flash_mode_penalties(mode, unis, e_avgs, penalties)
                    info['n_leds'] = len(pulse_leds)
                else:
                    info = self._flight_mode_penalties(mode, fl_e_avgs, fl_unis, vio_lit, penalties)
                    info['n_leds'] = len(flight_leds)
                modes[mode.name] = info

        score += sum(penalties.values())
        return Evaluation(score=score, uniformity_pct=uniformity_pct, coverage=coverage,
                          e_avg=e_avg, n_active=len(active), penalties=penalties, metrics=last_metrics,
                          grid=(grids[0] if len(grids) == 1 else grids) if keep_grid else None,
                          vio_grid=room_grid if keep_grid else None,
                          n_drivers=n_drivers, total_current_a=total_current, modes=modes, tilt=tilt,
                          tilt_walls=tilt_walls, electrical=electrical,
                          flight_grid=((flight_grids[0] if len(flight_grids) == 1 else flight_grids)
                                       if (keep_grid and flight_grids) else None),
                          tilt_grids=tilt_grids if (keep_grid and tilt_grids) else None)

    def __call__(self, x):
        return self.evaluate(x).score

    def build_scene(self, cfg):
        return build_scene_from_config(cfg, default_lumens=self.emission.default_lumens,
                                       stl_mesh=self.stl_mesh, diffuser=self.diffuser)

    def _trace_leds(self, leds, wall, budget_frac, scene):
        """Trace a subset of LEDs with ``budget_frac`` of the wall's ray budget (zeros if empty)."""
        if not leds:
            return np.zeros((wall.grid_size, wall.grid_size))
        settings = wall if budget_frac >= 1.0 else replace(wall, rays_per_pixel=max(1, int(round(
            wall.rays_per_pixel * budget_frac))))
        grid = compute_wall_intensity(leds, settings, self.emission, absorbers=scene.absorbers,
                                      stl_mesh_data=scene.stl_mesh_data, use_gpu=self.use_gpu,
                                      verbose=False, parallel=False)
        return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

    def _trace_modes(self, scene, wall, active, by_role, flash_lumens):
        """(flight grid, flash grid or None) for one wall.

        With a flash mode the three role sets are traced separately, sharing the wall's
        ray budget in proportion to their LED counts, and superposed:
        flight = vio + both, flash = vio + (flash + both at the pulse flux).
        """
        if flash_lumens is None:
            return self._trace_leds(active, wall, 1.0, scene), None
        n = max(1, len(active))
        g_vio = self._trace_leds(by_role['vio'], wall, len(by_role['vio']) / n, scene)
        g_both = self._trace_leds(by_role['both'], wall, len(by_role['both']) / n, scene)
        pulse = by_role['flash'] + by_role['both']
        g_pulse = self._trace_leds(_with_lumens(pulse, flash_lumens), wall, len(pulse) / n, scene)
        return g_vio + g_both, g_vio + g_pulse

    def _trace(self, scene, wall):
        return self._trace_leds(scene.leds, wall, 1.0, scene)

    def _trace_room(self, scene, leds, n_active):
        """T2: VIO room trace (flight LEDs) with roughly the ray budget of one main-wall trace."""
        w = self.walls[0]
        room_cells = int(self.vio.room_grid_size) ** 2
        rpp = max(1, int(round(w.rays_per_pixel * w.grid_size ** 2 / max(1, n_active * room_cells))))
        if not leds:
            return {name: np.zeros(mask.shape) for name, mask in self._vio_masks.items()}
        grids, _ = compute_room_intensity(leds, self.vio.room_settings(rpp), self.emission,
                                          absorbers=scene.absorbers, stl_mesh_data=scene.stl_mesh_data,
                                          use_gpu=self.use_gpu, verbose=False)
        return {n: np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for n, g in grids.items() if n != 'back'}

    def _pick_wall(self, mode, values):
        if mode.min_avg_lux_dist is not None:
            i = int(np.argmin([abs(w.wall_dist - mode.min_avg_lux_dist) for w in self.walls]))
            return values[i]
        return min(values)

    def _flight_mode_penalties(self, mode: ModeSpec, e_avgs, unis, vio_lit, penalties):
        """Continuous operating point: 'vio' + 'both' LEDs at their flux (× ``lumens_scale``)."""
        scale = float(mode.lumens_scale)
        info = {'scale': scale, 'e_avg': self._pick_wall(mode, e_avgs) * scale, 'vio_fraction': None,
                'uniformity_pct': float(np.mean(unis)) if unis else None, 'lumens': self.flight_lumens}
        if mode.min_avg_lux:
            short = max(0.0, (mode.min_avg_lux - info['e_avg']) / mode.min_avg_lux)
            if short > 0:
                penalties[f'{mode.name}.lux'] = mode.lux_weight * short
        if mode.vio_min_lux and vio_lit is not None:
            lit = vio_lit * scale
            frac = float(np.count_nonzero(lit >= mode.vio_min_lux) / max(1, lit.size))
            info['vio_fraction'] = frac
            # "≥ f of cells above L" ⇔ "(1−f) percentile ≥ L": penalise the lux shortfall of that
            # percentile so the penalty keeps a gradient even when no cell reaches L yet.
            e_req = float(np.percentile(lit, 100.0 * (1.0 - mode.vio_min_fraction))) if lit.size else 0.0
            short = max(0.0, (mode.vio_min_lux - e_req) / mode.vio_min_lux)
            if short > 0:
                penalties[f'{mode.name}.vio'] = mode.vio_weight * short
        return info

    def _flash_mode_penalties(self, mode: ModeSpec, unis, e_avgs, penalties):
        """Pulse operating point: its image IS the T1 image, so only the lux target is added here."""
        info = {'scale': None, 'e_avg': self._pick_wall(mode, e_avgs) if e_avgs else 0.0,
                'vio_fraction': None, 'uniformity_pct': float(np.mean(unis)) if unis else None,
                'lumens': self.flash_lumens}
        if mode.min_avg_lux:
            short = max(0.0, (mode.min_avg_lux - info['e_avg']) / mode.min_avg_lux)
            if short > 0:
                penalties[f'{mode.name}.lux'] = mode.lux_weight * short
        return info

    def _geometry_penalties(self, active, n_drivers=0, total_current=0.0, electrical=None, peak_current=0.0):
        c = self.constraints
        pen = {}
        n = len(active)
        el = electrical or {}
        if c.led_cost:
            pen['led_cost'] = c.led_cost * n
        if c.max_leds is not None and n > c.max_leds:
            pen['max_leds'] = c.max_leds_weight * (n - c.max_leds)
        if self.electrical:
            if c.driver_cost:
                pen['driver_cost'] = c.driver_cost * n_drivers
            if c.max_drivers is not None and n_drivers > c.max_drivers:
                pen['max_drivers'] = c.max_drivers_weight * (n_drivers - c.max_drivers)
            if 'n_pulse_drivers' in el:
                if c.pulse_driver_cost:
                    pen['pulse_driver_cost'] = c.pulse_driver_cost * el['n_pulse_drivers']
                if c.max_pulse_drivers is not None and el['n_pulse_drivers'] > c.max_pulse_drivers:
                    pen['max_pulse_drivers'] = c.max_drivers_weight * (el['n_pulse_drivers'] - c.max_pulse_drivers)
                if c.cont_driver_cost:
                    pen['cont_driver_cost'] = c.cont_driver_cost * el['n_cont_drivers']
                if c.max_cont_drivers is not None and el['n_cont_drivers'] > c.max_cont_drivers:
                    pen['max_cont_drivers'] = c.max_drivers_weight * (el['n_cont_drivers'] - c.max_cont_drivers)
            if c.max_total_current_a and total_current > c.max_total_current_a:
                pen['current'] = c.current_weight * (total_current - c.max_total_current_a) / c.max_total_current_a
            if c.max_peak_current_a and peak_current > c.max_peak_current_a:
                pen['peak_current'] = c.peak_current_weight * (peak_current - c.max_peak_current_a) / c.max_peak_current_a
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


def _with_lumens(leds, lumens):
    """Copies of LED placements with their flux replaced (the scene's own LEDs are untouched)."""
    out = []
    for p in leds:
        p2 = copy.copy(p)
        p2.led = copy.copy(p.led)
        p2.led.lumens = float(lumens)
        out.append(p2)
    return out


_LEGACY_KEYS = {
    'objective': ('flight_weight', 'tilt_geometry'),
    'vio': ('geometry', 'wall_dist', 'wall_size', 'grid_size'),
    'mode': ('uniformity_weight',),
}


def _drop_legacy(section: dict, kind: str):
    """Remove pre-test-case keys from a spec section (their behaviour is now fixed by T1/T2/T3)."""
    dropped = [k for k in _LEGACY_KEYS[kind] if k in section]
    for k in dropped:
        section.pop(k)
    if dropped:
        print(f"[optim] ignoring obsolete {kind} key(s) {dropped}: T1 always scores the flash image when a flash "
              "mode exists, T2/T3 always use rooms")
    return section


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
    wall_sizes = wall_spec.get('wall_size')
    if isinstance(wall_sizes, (list, tuple)):
        wall_spec['wall_size'] = float(wall_sizes[0])
    elif wall_sizes == "auto":
        wall_spec.pop('wall_size')
    else:
        wall_sizes = None  # scalar: same size at every distance
    wall = WallSettings(**wall_spec)
    camera = CameraSpec(**spec.get('camera', {}))
    emission = EmissionSettings(**spec.get('emission', {}))
    objective = ObjectiveSpec(**_drop_legacy(dict(spec.get('objective', {})), 'objective'))
    constraints = ConstraintSpec(**spec.get('constraints', {}))
    clear_base = bool(spec.get('clear_base', False))
    electrical = bool(spec.get('electrical', False))
    driver = DriverModel(**spec.get('driver', {}))
    cont_driver = DriverModel(**{**spec.get('driver', {}), **spec['cont_driver']}) if spec.get('cont_driver') else None
    vio = VioSpec(**_drop_legacy(dict(spec['vio']), 'vio')) if spec.get('vio') else None
    modes = [ModeSpec(**_drop_legacy(dict(m), 'mode')) for m in spec.get('modes', [])]

    working = copy.deepcopy(base_cfg)
    if clear_base:
        working['custom_groups'] = []
    variables = []
    for v in spec['variables']:
        v = dict(v)
        if v.get('type') == 'group_current' or (v.get('type') == 'duct_ring' and v.get('current_range')):
            if not electrical:
                raise ValueError(f"variable '{v.get('name', v['type'])}' optimises a drive current: set "
                                 "\"electrical\": true or give the flux in lumens instead")
            v.setdefault('driver', copy.deepcopy(spec.get('driver', {})))
        variables.append(variable_from_spec(v, working))
    problem_kwargs.setdefault('use_gpu', bool(spec.get('use_gpu', False)))
    return Problem(base_cfg, variables, wall, camera, emission, objective, constraints,
                   clear_base=clear_base, name=spec.get('name', default_name),
                   wall_dists=wall_dists, wall_sizes=wall_sizes, driver=driver, vio=vio, modes=modes,
                   cont_driver=cont_driver, electrical=electrical, **problem_kwargs)


def load_spec(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path.parent
