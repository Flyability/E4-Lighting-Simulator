"""Design-variable groups: map slices of a decision vector onto a config dict.

Each group exposes ``names``, ``bounds``, ``integrality`` and ``x0`` (initial
guess), and ``apply(x, cfg)`` which mutates a *deep-copied* saved config so the
regular scene builder can consume it. Everything is expressed in the saved
config schema (``configs/*.json``), so any optimised result is directly
loadable in the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lighting_simulator.domain.geometry import euler_xyz_matrix, normalize, rodrigues_rotation
from lighting_simulator.domain.mirroring import mirror_group_config_xz
from lighting_simulator.scene.builder import euler_applies


def _pairs(values, n):
    """Broadcast a scalar / 2-list / list-of-2-lists to n (lo, hi) pairs."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1 and arr.shape == (2,):
        return [(float(arr[0]), float(arr[1]))] * n
    if arr.ndim == 2 and arr.shape == (n, 2):
        return [(float(a), float(b)) for a, b in arr]
    raise ValueError(f"expected a [lo, hi] pair or {n} pairs, got shape {arr.shape}")


@dataclass
class Duct:
    """Cylinder surface the LEDs are mounted on (cm). ``theta=0`` points along ``reference``."""

    center: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    radius: float = 6.0
    reference: tuple = (1.0, 0.0, 0.0)
    mount_offset: float = 0.3
    """Stand-off of the LED face from the cylinder surface, along the outward normal."""

    def frame(self):
        a = normalize(self.axis)
        u = np.asarray(self.reference, dtype=float)
        u = u - a * np.dot(u, a)
        if np.linalg.norm(u) < 1e-9:
            u = np.cross(a, [0.0, 1.0, 0.0])
        u = normalize(u)
        v = np.cross(a, u)
        return a, u, v

    def led_pose(self, theta_deg, axial, tilt_axial_deg=0.0, tilt_tangential_deg=0.0):
        """(position, direction, row_direction) of an LED on the surface."""
        a, u, v = self.frame()
        th = math.radians(theta_deg)
        normal = math.cos(th) * u + math.sin(th) * v
        tangent = np.cross(a, normal)
        position = np.asarray(self.center, float) + (self.radius + self.mount_offset) * normal + axial * a
        direction = normal
        if abs(tilt_axial_deg) > 1e-9:  # tilt toward +axis
            direction = rodrigues_rotation(tangent, -math.radians(tilt_axial_deg)) @ direction
        if abs(tilt_tangential_deg) > 1e-9:  # swing around the duct axis
            direction = rodrigues_rotation(a, math.radians(tilt_tangential_deg)) @ direction
        return position, normalize(direction), tangent


class VariableGroup:
    names: list
    bounds: list
    integrality: list
    x0: list

    def apply(self, x, cfg):  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def size(self):
        return len(self.names)


@dataclass
class DuctRingLayout(VariableGroup):
    """LEDs mounted on a duct cylinder; generates one dynamic group in the config.

    placement:
      - ``"free"``: every LED has its own (theta, axial) — arbitrary pattern.
      - ``"arc"``: ``n_rows × n_cols`` regular lattice; variables are the arc
        centre/span and axial centre/pitch — the panel keeps a rectangular shape.
    Optional per-LED or shared tilt, beam angle and on/off state.
    """

    name: str
    duct: Duct
    n_leds: int = 6
    placement: str = "arc"
    n_rows: int = 1
    theta_range: tuple = (-60.0, 60.0)
    axial_range: tuple = (-1.5, 1.5)
    arc_span_range: tuple = (10.0, 120.0)
    row_pitch_range: tuple = (0.8, 2.0)
    tilt_axial_range: tuple | None = (-30.0, 30.0)
    tilt_tangential_range: tuple | None = None
    shared_tilt: bool = True
    beam_angle_range: tuple | None = None
    shared_beam_angle: bool = True
    default_beam_angle: float = 120.0
    optimize_enabled: bool = False
    led_size: float = 0.5
    color: tuple = (1.0, 0.0, 1.0)
    mirror_xz: bool = False
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if isinstance(self.duct, dict):
            self.duct = Duct(**self.duct)
        self.n_rows = max(1, int(self.n_rows))
        if self.placement == "arc" and self.n_leds % self.n_rows:
            raise ValueError(f"{self.name}: n_leds={self.n_leds} not divisible by n_rows={self.n_rows}")

        if self.placement == "free":
            for i in range(self.n_leds):
                self._add(f"theta[{i}]", *self.theta_range,
                          x0=np.interp(i, [0, max(1, self.n_leds - 1)], self.theta_range))
                self._add(f"axial[{i}]", *self.axial_range)
        elif self.placement == "arc":
            self._add("theta_center", *self.theta_range)
            self._add("arc_span", *self.arc_span_range)
            self._add("axial_center", *self.axial_range)
            if self.n_rows > 1:
                self._add("row_pitch", *self.row_pitch_range)
        else:
            raise ValueError(f"unknown placement {self.placement!r}")

        n_tilt = 1 if self.shared_tilt else self.n_leds
        if self.tilt_axial_range is not None:
            for i in range(n_tilt):
                self._add(f"tilt_axial[{i}]" if n_tilt > 1 else "tilt_axial", *self.tilt_axial_range, x0=0.0)
        if self.tilt_tangential_range is not None:
            for i in range(n_tilt):
                self._add(f"tilt_tan[{i}]" if n_tilt > 1 else "tilt_tan", *self.tilt_tangential_range, x0=0.0)
        if self.beam_angle_range is not None:
            n_beam = 1 if self.shared_beam_angle else self.n_leds
            for i in range(n_beam):
                self._add(f"beam[{i}]" if n_beam > 1 else "beam", *self.beam_angle_range,
                          x0=min(max(self.default_beam_angle, self.beam_angle_range[0]), self.beam_angle_range[1]))
        if self.optimize_enabled:
            for i in range(self.n_leds):
                self._add(f"on[{i}]", 0, 1, integer=True, x0=1)

    def _add(self, name, lo, hi, integer=False, x0=None):
        self.names.append(f"{self.name}.{name}")
        self.bounds.append((float(lo), float(hi)))
        self.integrality.append(integer)
        self.x0.append(float((lo + hi) / 2 if x0 is None else x0))

    def _layout(self, x):
        """Return per-LED (theta, axial) and row membership."""
        it = iter(x)
        if self.placement == "free":
            thetas, axials = [], []
            for _ in range(self.n_leds):
                thetas.append(next(it))
                axials.append(next(it))
            rows = [list(range(self.n_leds))]
        else:
            tc, span, ac = next(it), next(it), next(it)
            pitch = next(it) if self.n_rows > 1 else 0.0
            n_cols = self.n_leds // self.n_rows
            col_th = np.linspace(tc - span / 2, tc + span / 2, n_cols) if n_cols > 1 else np.array([tc])
            row_ax = (np.arange(self.n_rows) - (self.n_rows - 1) / 2) * pitch + ac
            thetas, axials, rows = [], [], []
            for r in range(self.n_rows):
                rows.append(list(range(len(thetas), len(thetas) + n_cols)))
                for th in col_th:
                    thetas.append(float(th))
                    axials.append(float(row_ax[r]))
        return thetas, axials, rows, list(it)

    def apply(self, x, cfg):
        thetas, axials, rows, rest = self._layout(x)
        it = iter(rest)
        n_tilt = 1 if self.shared_tilt else self.n_leds
        tilt_ax = [next(it) for _ in range(n_tilt)] if self.tilt_axial_range is not None else [0.0]
        tilt_tan = [next(it) for _ in range(n_tilt)] if self.tilt_tangential_range is not None else [0.0]
        if self.beam_angle_range is not None:
            beams = [next(it) for _ in range(1 if self.shared_beam_angle else self.n_leds)]
        else:
            beams = [self.default_beam_angle]
        states = [bool(round(next(it))) for _ in range(self.n_leds)] if self.optimize_enabled else [True] * self.n_leds

        positions, directions, row_dirs = [], [], []
        for i, (th, ax) in enumerate(zip(thetas, axials)):
            p, d, rd = self.duct.led_pose(
                th, ax,
                tilt_ax[i if len(tilt_ax) > 1 else 0],
                tilt_tan[i if len(tilt_tan) > 1 else 0],
            )
            positions.append([float(v) for v in p])
            directions.append([float(v) for v in d])
            row_dirs.append([float(v) for v in rd])

        group = {
            'enabled': True,
            'name': self.name,
            'position': [0.0, 0.0, 0.0],
            'rotation_x': 0.0, 'rotation_y': 0.0, 'rotation_z': 0.0,
            'is_dynamic': True,
            'num_leds': self.n_leds,
            'led_positions': positions,
            'led_rotations': directions,
            'led_row_directions': row_dirs,
            'led_sizes': [self.led_size] * self.n_leds,
            'led_viewing_angles': [float(beams[i if len(beams) > 1 else 0]) for i in range(self.n_leds)],
            'led_beam_tilts': [0.0] * self.n_leds,
            'led_states': states,
            'led_rows': rows,
            'led_euler_angles': [],
            'led_lumens': [],
            'lumens_override_enabled': False,
            'lumens_value': 100,
            'template_name': None,
            'initial_pos': [0.0, 0.0, 0.0],
            'initial_rot': [0, 0, 0],
            'generated_by': 'DuctRingLayout',
        }
        cfg.setdefault('custom_groups', []).append(group)
        if self.mirror_xz:
            m = mirror_group_config_xz(group)
            m['name'] = f"{self.name}_mirror"
            m['position'] = [0.0, 0.0, 0.0]
            cfg['custom_groups'].append(m)


@dataclass
class PanelPose(VariableGroup):
    """Rigid-body deltas (cm / deg) on an existing custom group of the base config."""

    group_index: int
    pos_delta: tuple = (2.0, 2.0, 2.0)
    """Max ± translation per axis; 0 freezes that axis."""
    rot_delta: tuple = (10.0, 10.0, 10.0)
    """Max ± roll/pitch/yaw (rotation_x/y/z); 0 freezes that angle."""
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._keys = []
        for axis, d in zip("xyz", self.pos_delta):
            if d > 0:
                self._keys.append(('pos', axis))
                self.names.append(f"group{self.group_index}.d{axis}")
                self.bounds.append((-float(d), float(d)))
                self.integrality.append(False)
                self.x0.append(0.0)
        for axis, d in zip("xyz", self.rot_delta):
            if d > 0:
                self._keys.append(('rot', axis))
                self.names.append(f"group{self.group_index}.drot_{axis}")
                self.bounds.append((-float(d), float(d)))
                self.integrality.append(False)
                self.x0.append(0.0)

    def apply(self, x, cfg):
        group = cfg['custom_groups'][self.group_index]
        pos = list(group.get('position', [0.0, 0.0, 0.0]))
        rot = [0.0, 0.0, 0.0]
        for (kind, axis), value in zip(self._keys, x):
            i = "xyz".index(axis)
            if kind == 'pos':
                pos[i] = float(pos[i] + value)
            else:
                rot[i] = float(value)
        group['position'] = pos
        if not any(rot):
            return
        if group.get('is_dynamic') and not euler_applies(group):
            # UI ignores rotation_* for plain dynamic groups: bake the delta into the
            # LED arrays (rigid rotation about the group origin, like the UI sliders).
            R = euler_xyz_matrix(*rot)
            for key in ('led_positions', 'led_rotations', 'led_row_directions'):
                vecs = group.get(key)
                if vecs:
                    group[key] = [[float(v) for v in R @ np.asarray(p, dtype=float)] for p in vecs]
        else:
            for axis, value in zip("xyz", rot):
                key = f'rotation_{axis}'
                group[key] = float(group.get(key, 0.0) + value)


@dataclass
class LedStates(VariableGroup):
    """Binary on/off per LED of an existing group (``group_index``) or the base rig (``None``)."""

    group_index: int | None = None
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)
    _n: int = field(default=0, init=False)

    def bind(self, base_cfg):
        if self.group_index is None:
            states = base_cfg.get('led_states', [])
            label = "base"
        else:
            g = base_cfg['custom_groups'][self.group_index]
            states = g.get('led_states', [True] * g.get('num_leds', 12))
            label = f"group{self.group_index}"
        self._n = len(states)
        self.names = [f"{label}.on[{i}]" for i in range(self._n)]
        self.bounds = [(0, 1)] * self._n
        self.integrality = [True] * self._n
        self.x0 = [1.0 if s else 0.0 for s in states]
        return self

    def apply(self, x, cfg):
        states = [bool(round(v)) for v in x]
        if self.group_index is None:
            cfg['led_states'] = states
        else:
            cfg['custom_groups'][self.group_index]['led_states'] = states


@dataclass
class BeamAngle(VariableGroup):
    """Viewing angle (deg) shared by all LEDs of a group, or of the base rig (``None``)."""

    group_index: int | None = None
    angle_range: tuple = (60.0, 130.0)
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)

    def __post_init__(self):
        label = "base" if self.group_index is None else f"group{self.group_index}"
        self.names = [f"{label}.beam_angle"]
        self.bounds = [tuple(float(v) for v in self.angle_range)]
        self.integrality = [False]
        self.x0 = [sum(self.bounds[0]) / 2]

    def apply(self, x, cfg):
        angle = float(x[0])
        if self.group_index is None:
            cfg['viewing_angle'] = angle
        else:
            g = cfg['custom_groups'][self.group_index]
            g['led_viewing_angles'] = [angle] * g.get('num_leds', len(g.get('led_positions', [])))


@dataclass
class BeamTilts(VariableGroup):
    """Per-LED beam tilt (deg, about the row direction) inside an existing dynamic group."""

    group_index: int
    tilt_range: tuple = (-20.0, 20.0)
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)
    _n: int = field(default=0, init=False)

    def bind(self, base_cfg):
        g = base_cfg['custom_groups'][self.group_index]
        self._n = g.get('num_leds', len(g.get('led_positions', [])))
        self.names = [f"group{self.group_index}.tilt[{i}]" for i in range(self._n)]
        self.bounds = [tuple(float(v) for v in self.tilt_range)] * self._n
        self.integrality = [False] * self._n
        base = g.get('led_beam_tilts') or [0.0] * self._n
        self.x0 = [float(t) for t in base] + [0.0] * (self._n - len(base))
        return self

    def apply(self, x, cfg):
        cfg['custom_groups'][self.group_index]['led_beam_tilts'] = [float(v) for v in x]


VARIABLE_TYPES = {
    'duct_ring': DuctRingLayout,
    'panel_pose': PanelPose,
    'led_states': LedStates,
    'beam_angle': BeamAngle,
    'beam_tilts': BeamTilts,
}


def variable_from_spec(spec, base_cfg):
    spec = dict(spec)
    kind = spec.pop('type')
    cls = VARIABLE_TYPES[kind]
    var = cls(**spec)
    if hasattr(var, 'bind'):
        var.bind(base_cfg)
    return var
