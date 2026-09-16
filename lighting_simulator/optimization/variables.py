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
from lighting_simulator.domain.led import CODE_ROLES, ROLE_CODES, normalize_roles
from lighting_simulator.domain.mirroring import mirror_group_config_xz
from lighting_simulator.scene.builder import euler_applies

from .electrical import DriverModel


def arc_cm_to_deg(arc_cm, radius_cm):
    """Angle subtended by ``arc_cm`` along the circumference of a duct of ``radius_cm``."""
    return math.degrees(float(arc_cm) / max(1e-6, float(radius_cm)))


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
    rotation_deg: tuple | None = None
    """Extrinsic X-Y-Z Euler rotation (deg) of ``axis`` and ``reference`` about ``center`` (tilted ducts)."""

    def frame(self):
        a = np.asarray(self.axis, dtype=float)
        u = np.asarray(self.reference, dtype=float)
        if self.rotation_deg is not None and any(abs(float(r)) > 1e-9 for r in self.rotation_deg):
            R = euler_xyz_matrix(*(float(r) for r in self.rotation_deg))
            a, u = R @ a, R @ u
        a = normalize(a)
        u = u - a * np.dot(u, a)
        if np.linalg.norm(u) < 1e-9:
            u = np.cross(a, [0.0, 1.0, 0.0])
        u = normalize(u)
        v = np.cross(a, u)
        return a, u, v

    def led_pose(self, theta_deg, axial, tilt_axial_deg=0.0, tilt_tangential_deg=0.0, radial=0.0,
                 center_offset=(0.0, 0.0, 0.0)):
        """(position, direction, row_direction) of an LED on the surface.

        ``radial`` is an extra stand-off (cm) on top of ``mount_offset`` and
        ``center_offset`` shifts the duct itself — the mechanical tolerances."""
        a, u, v = self.frame()
        th = math.radians(theta_deg)
        normal = math.cos(th) * u + math.sin(th) * v
        tangent = np.cross(a, normal)
        position = (np.asarray(self.center, float) + np.asarray(center_offset, float)
                    + (self.radius + self.mount_offset + radial) * normal + axial * a)
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
        With ``n_rows_range`` / ``n_cols_range`` the counts themselves become
        integer variables (``n_leds`` / ``n_rows`` are then the maxima).
    Optional per-LED or shared tilt, beam angle, on/off state, radial stand-off
    tolerance (``radial_range``) and drive current (``current_range`` → lumens
    via ``driver``).
    """

    name: str
    duct: Duct
    n_leds: int = 6
    placement: str = "arc"
    n_rows: int = 1
    n_rows_range: tuple | None = None
    n_cols_range: tuple | None = None
    theta_range: tuple = (-60.0, 60.0)
    axial_range: tuple = (-1.5, 1.5)
    arc_span_range: tuple = (10.0, 120.0)
    row_pitch_range: tuple = (0.8, 2.0)
    radial_range: tuple | None = None
    """Stand-off tolerance (cm) added to ``duct.mount_offset``, e.g. ``[-1, 1]`` for ±10 mm."""
    center_delta: tuple | None = None
    """± shift (cm) of the duct centre per axis; 0 freezes that axis."""
    tilt_axial_range: tuple | None = (-30.0, 30.0)
    tilt_tangential_range: tuple | None = None
    shared_tilt: bool = True
    symmetric_tilt: bool = False
    """Axial tilt mirrored about the panel's middle row: one variable per row pair (outer → inner);
    the top row of a pair looks +t along the duct axis, the bottom row −t, a middle row stays at 0.
    Overrides ``shared_tilt`` for the axial tilt."""
    beam_angle_range: tuple | None = None
    shared_beam_angle: bool = True
    default_beam_angle: float = 120.0
    optimize_enabled: bool = False
    """Per-LED on/off. Ignored when the row/column counts are variables (the counts already set
    how many LEDs there are; a second knob for the same thing only confuses the search)."""
    optimize_roles: bool = False
    """Per-LED role variable (1 vio, 2 flash, 3 both; 0 = off when ``roles_allow_off``); supersedes ``optimize_enabled``."""
    roles_allow_off: bool | None = None
    """Whether the role variable may switch an LED off. Default: only when the counts are fixed."""
    default_role: str = "both"
    current_range: tuple | None = None
    """Shared drive current (A); converted to lumens with ``driver``."""
    driver: DriverModel = field(default_factory=DriverModel)
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
        if isinstance(self.driver, dict):
            self.driver = DriverModel(**self.driver)
        self.n_rows = max(1, int(self.n_rows))
        self._variable_counts = self.placement == "arc" and (self.n_rows_range or self.n_cols_range)
        if self._variable_counts:
            r_lo, r_hi = (int(v) for v in (self.n_rows_range or (self.n_rows, self.n_rows)))
            max_cols = self.n_leds // self.n_rows
            c_lo, c_hi = (int(v) for v in (self.n_cols_range or (max_cols, max_cols)))
            self.n_rows, self.n_leds = r_hi, r_hi * c_hi
            self._count_bounds = ((r_lo, r_hi), (c_lo, c_hi))
        if self.placement == "arc" and self.n_leds % self.n_rows:
            raise ValueError(f"{self.name}: n_leds={self.n_leds} not divisible by n_rows={self.n_rows}")
        if self._variable_counts and self.optimize_enabled:
            if not self.optimize_roles:
                print(f"[optim] {self.name}: per-LED on/off ignored because the row/column counts are optimised")
            self.optimize_enabled = False
        if self.roles_allow_off is None:
            self.roles_allow_off = not self._variable_counts

        if self.placement == "free":
            for i in range(self.n_leds):
                self._add(f"theta[{i}]", *self.theta_range,
                          x0=np.interp(i, [0, max(1, self.n_leds - 1)], self.theta_range))
                self._add(f"axial[{i}]", *self.axial_range)
        elif self.placement == "arc":
            if self._variable_counts:
                (r_lo, r_hi), (c_lo, c_hi) = self._count_bounds
                self._add("n_rows", r_lo, r_hi, integer=True, x0=r_hi)
                self._add("n_cols", c_lo, c_hi, integer=True, x0=c_hi)
            self._add("theta_center", *self.theta_range)
            self._add("arc_span", *self.arc_span_range)
            self._add("axial_center", *self.axial_range)
            if self.n_rows > 1:
                self._add("row_pitch", *self.row_pitch_range)
        else:
            raise ValueError(f"unknown placement {self.placement!r}")

        if self.radial_range is not None:
            self._add("radial", *self.radial_range, x0=0.0)
        self._center_axes = []
        if self.center_delta is not None:
            for axis, d in zip("xyz", self.center_delta):
                if d > 0:
                    self._center_axes.append("xyz".index(axis))
                    self._add(f"dc{axis}", -float(d), float(d), x0=0.0)
        n_tilt = 1 if self.shared_tilt else self.n_leds
        self._n_tilt_pairs = (self.n_rows + 1) // 2 if self.symmetric_tilt else 0
        if self.tilt_axial_range is not None:
            if self.symmetric_tilt:
                lo, hi = self.tilt_axial_range
                lo, hi = min(abs(lo), abs(hi)) if lo * hi > 0 else 0.0, max(abs(lo), abs(hi))
                for k in range(self._n_tilt_pairs):
                    self._add(f"tilt_axial_pair[{k}]" if self._n_tilt_pairs > 1 else "tilt_axial_pair", lo, hi, x0=lo)
            else:
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
        if self.optimize_roles:
            lo = 0 if self.roles_allow_off else 1
            for i in range(self.n_leds):
                self._add(f"role[{i}]", lo, 3, integer=True, x0=max(lo, ROLE_CODES.get(self.default_role, 3)))
        elif self.optimize_enabled:
            for i in range(self.n_leds):
                self._add(f"on[{i}]", 0, 1, integer=True, x0=1)
        if self.current_range is not None:
            lo, hi = self.current_range
            hi = min(float(hi), self.driver.max_current_a)
            self._add("current_a", lo, hi)

    def _add(self, name, lo, hi, integer=False, x0=None):
        self.names.append(f"{self.name}.{name}")
        self.bounds.append((float(lo), float(hi)))
        self.integrality.append(integer)
        self.x0.append(float((lo + hi) / 2 if x0 is None else x0))

    def _per_led_slots(self, rows):
        """Index into the per-LED variable arrays for each generated LED.

        The arrays are sized for the maximum lattice; a smaller ``n_rows x n_cols`` lattice
        reads the centred sub-block so each variable keeps its lattice position when the
        counts change (a flat ``[:n_out]`` slice would shift every row).
        """
        n_out = sum(len(m) for m in rows)
        if self.placement != "arc":
            return list(range(n_out))
        r_hi, c_hi = self.n_rows, self.n_leds // self.n_rows
        n_rows, n_cols = len(rows), (len(rows[0]) if rows else 0)
        r0, c0 = (r_hi - n_rows) // 2, (c_hi - n_cols) // 2
        slots = [0] * n_out
        for r, members in enumerate(rows):
            for c, i in enumerate(members):
                slots[i] = (r0 + r) * c_hi + (c0 + c)
        return slots

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
            n_rows = self.n_rows
            n_cols = self.n_leds // self.n_rows
            if self._variable_counts:
                n_rows = int(round(next(it)))
                n_cols = int(round(next(it)))
            tc, span, ac = next(it), next(it), next(it)
            pitch = next(it) if self.n_rows > 1 else 0.0
            col_th = np.linspace(tc - span / 2, tc + span / 2, n_cols) if n_cols > 1 else np.array([tc])
            row_ax = (np.arange(n_rows) - (n_rows - 1) / 2) * pitch + ac
            thetas, axials, rows = [], [], []
            for r in range(n_rows):
                rows.append(list(range(len(thetas), len(thetas) + n_cols)))
                for th in col_th:
                    thetas.append(float(th))
                    axials.append(float(row_ax[r]))
        return thetas, axials, rows, list(it)

    def apply(self, x, cfg):
        thetas, axials, rows, rest = self._layout(x)
        n_out = len(thetas)  # ≤ self.n_leds when counts are variables
        it = iter(rest)
        radial = next(it) if self.radial_range is not None else 0.0
        center_offset = np.zeros(3)
        for ax in self._center_axes:
            center_offset[ax] = next(it)
        n_tilt = 1 if self.shared_tilt else self.n_leds
        if self.tilt_axial_range is not None and self.symmetric_tilt:
            pair_tilt = [next(it) for _ in range(self._n_tilt_pairs)]
            n_rows = len(rows)
            tilt_ax = [0.0] * n_out
            for r, members in enumerate(rows):
                side = (r - (n_rows - 1) / 2.0)  # < 0 bottom half, > 0 top half, 0 middle row
                sign = 0.0 if abs(side) < 1e-9 else (1.0 if side > 0 else -1.0)
                t = sign * pair_tilt[min(r, n_rows - 1 - r)]
                for i in members:
                    tilt_ax[i] = t
        elif self.tilt_axial_range is not None:
            tilt_ax = [next(it) for _ in range(n_tilt)]
        else:
            tilt_ax = [0.0]
        tilt_tan = [next(it) for _ in range(n_tilt)] if self.tilt_tangential_range is not None else [0.0]
        if self.beam_angle_range is not None:
            beams = [next(it) for _ in range(1 if self.shared_beam_angle else self.n_leds)]
        else:
            beams = [self.default_beam_angle]
        if self.optimize_roles:
            codes = [int(round(next(it))) for _ in range(self.n_leds)]
            all_states = [c != 0 for c in codes]
            all_roles = [CODE_ROLES.get(c, self.default_role) if c else self.default_role for c in codes]
        else:
            all_states = [bool(round(next(it))) for _ in range(self.n_leds)] if self.optimize_enabled else [True] * self.n_leds
            all_roles = [self.default_role] * self.n_leds
        slots = self._per_led_slots(rows)
        states = [all_states[s] for s in slots]
        roles = [all_roles[s] for s in slots]
        current_a = float(next(it)) if self.current_range is not None else None

        positions, directions, row_dirs = [], [], []
        for i, (th, ax) in enumerate(zip(thetas, axials)):
            p, d, rd = self.duct.led_pose(
                th, ax,
                tilt_ax[i if len(tilt_ax) > 1 else 0],
                tilt_tan[i if len(tilt_tan) > 1 else 0],
                radial=radial, center_offset=center_offset,
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
            'num_leds': n_out,
            'led_positions': positions,
            'led_rotations': directions,
            'led_row_directions': row_dirs,
            'led_sizes': [self.led_size] * n_out,
            'led_viewing_angles': [float(beams[i if len(beams) > 1 else 0]) for i in range(n_out)],
            'led_beam_tilts': [0.0] * n_out,
            'led_states': states,
            'led_roles': roles,
            'led_rows': rows,
            'led_euler_angles': [],
            'led_lumens': [],
            'lumens_override_enabled': current_a is not None,
            'lumens_value': self.driver.lumens(current_a) if current_a is not None else 100,
            'drive_current_a': current_a,
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
class LedRoles(VariableGroup):
    """Per-LED operating role of an existing group: 0 off, 1 vio, 2 flash, 3 both (supersedes on/off)."""

    group_index: int = 0
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)
    _n: int = field(default=0, init=False)

    def bind(self, base_cfg):
        g = base_cfg['custom_groups'][self.group_index]
        states = g.get('led_states', [True] * g.get('num_leds', 12))
        roles = normalize_roles(g.get('led_roles'), len(states))
        self._n = len(states)
        self.names = [f"group{self.group_index}.role[{i}]" for i in range(self._n)]
        self.bounds = [(0, 3)] * self._n
        self.integrality = [True] * self._n
        self.x0 = [float(ROLE_CODES[r]) if s else 0.0 for s, r in zip(states, roles)]
        return self

    def apply(self, x, cfg):
        codes = [int(round(v)) for v in x]
        g = cfg['custom_groups'][self.group_index]
        g['led_states'] = [c != 0 for c in codes]
        g['led_roles'] = [CODE_ROLES.get(c, 'both') if c else 'both' for c in codes]


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


@dataclass
class GroupCurrent(VariableGroup):
    """Shared drive current (A) of an existing group, applied as a lumens override."""

    group_index: int
    current_range: tuple = (0.5, 3.0)
    driver: DriverModel = field(default_factory=DriverModel)
    names: list = field(default_factory=list, init=False)
    bounds: list = field(default_factory=list, init=False)
    integrality: list = field(default_factory=list, init=False)
    x0: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if isinstance(self.driver, dict):
            self.driver = DriverModel(**self.driver)
        lo, hi = self.current_range
        hi = min(float(hi), self.driver.max_current_a)
        self.names = [f"group{self.group_index}.current_a"]
        self.bounds = [(float(lo), hi)]
        self.integrality = [False]
        self.x0 = [(float(lo) + hi) / 2]

    def apply(self, x, cfg):
        g = cfg['custom_groups'][self.group_index]
        g['lumens_override_enabled'] = True
        g['lumens_value'] = self.driver.lumens(x[0])
        g['drive_current_a'] = float(x[0])


VARIABLE_TYPES = {
    'duct_ring': DuctRingLayout,
    'panel_pose': PanelPose,
    'led_states': LedStates,
    'led_roles': LedRoles,
    'beam_angle': BeamAngle,
    'beam_tilts': BeamTilts,
    'group_current': GroupCurrent,
}


def variable_from_spec(spec, base_cfg):
    spec = dict(spec)
    kind = spec.pop('type')
    cls = VARIABLE_TYPES[kind]
    var = cls(**spec)
    if hasattr(var, 'bind'):
        var.bind(base_cfg)
    return var
