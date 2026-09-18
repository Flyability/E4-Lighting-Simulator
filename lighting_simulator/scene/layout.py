"""Config schema v2: a *layout* (the lighting design) linked to a *platform* (the drone).

Layout file (``layouts/<name>.json``)::

    {"schema_version": 2, "name": ..., "description": ...,
     "platform": "<platform name>" | {inline platform} | null,
     "flux": {"vio_lumens": 168, "flash_lumens": 14040},
     "panels": [{"name", "enabled", "mirror", "position", "rotation", "rows", "source",
                 "leds": [{"position", "direction", "row_dir", "beam_angle", "tilt", "size", "on", "role"}]}]}

Units cm / degrees; frame +X forward (camera axis), +Y left, +Z up. LED ``position`` /
``direction`` / ``row_dir`` are *panel-local*; the panel ``rotation`` (extrinsic X-Y-Z) and
``position`` place the panel on the rig. ``mirror: true`` generates the XZ-mirrored twin at
build time so a symmetric rig only stores one half. ``direction`` is the LED square normal;
``tilt`` rotates the beam about ``row_dir``. Flux is a property of the operating point, not of
a panel: ``flux.vio_lumens`` is every LED's continuous flux, ``flux.flash_lumens`` the pulse
flux of 'flash' / 'both' LEDs.

Platform file (``platforms/<name>.json``)::

    {"schema_version": 2, "name": ...,
     "stl": {"file", "scale", "position", "rotation", "occludes", "opacity", "wireframe"} | null,
     "vio_cameras": {"position", "cam1_pitch", "cam1_yaw", "cam2_pitch", "cam2_yaw", "long_fov", "landscape"}}

``convert_v1`` upgrades the pre-v2 UI documents (base ring, custom_groups, individual_leds,
mirror_primary, absorbers, global transform) into this schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lighting_simulator.domain.geometry import (
    default_row_direction, euler_xyz_matrix, normalize, rodrigues_rotation, rotate_vector, rotation_matrix_z,
)
from lighting_simulator.domain.led import LED, normalize_role
from lighting_simulator.domain.placement import LEDPlacement

SCHEMA_VERSION = 2
XZ_MIRROR = np.diag([1.0, -1.0, 1.0])


def _v3(v, default=(0.0, 0.0, 0.0)):
    return tuple(float(x) for x in (v if v is not None else default))


@dataclass
class LedSpec:
    position: tuple
    direction: tuple
    row_dir: tuple | None = None
    beam_angle: float = 120.0
    tilt: float = 0.0
    size: float = 1.0
    on: bool = True
    role: str = 'both'

    def __post_init__(self):
        self.position = _v3(self.position)
        self.direction = tuple(float(x) for x in normalize(self.direction))
        self.row_dir = tuple(float(x) for x in normalize(self.row_dir)) if self.row_dir is not None else None
        self.beam_angle, self.tilt, self.size = float(self.beam_angle), float(self.tilt), float(self.size)
        self.on, self.role = bool(self.on), normalize_role(self.role)

    def resolved_row_dir(self):
        return np.asarray(self.row_dir if self.row_dir is not None else default_row_direction(self.direction), float)


@dataclass
class Panel:
    name: str
    leds: list = field(default_factory=list)
    enabled: bool = True
    mirror: bool = False
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)
    """Extrinsic X-Y-Z Euler (deg) of the panel frame."""
    rows: list | None = None
    """Optional LED index groups per row (construction aid for guides / lattices)."""
    source: dict | None = None
    """Provenance only (template name, generator parameters...)."""

    def __post_init__(self):
        self.leds = [l if isinstance(l, LedSpec) else LedSpec(**l) for l in self.leds]
        self.position, self.rotation = _v3(self.position), _v3(self.rotation)
        self.enabled, self.mirror = bool(self.enabled), bool(self.mirror)
        if self.rows is not None:
            self.rows = [[int(i) for i in r] for r in self.rows]

    def matrix(self):
        return euler_xyz_matrix(*self.rotation)

    def world_geometry(self):
        """(positions, square normals, row dirs, beam dirs) of this panel's LEDs in the rig frame."""
        R = self.matrix()
        p0 = np.asarray(self.position, float)
        pos, nrm, row, beam = [], [], [], []
        for l in self.leds:
            n = R @ np.asarray(l.direction, float)
            r = R @ l.resolved_row_dir()
            pos.append(R @ np.asarray(l.position, float) + p0)
            nrm.append(n)
            row.append(r)
            beam.append(normalize(rotate_vector(n, r, l.tilt)) if abs(l.tilt) > 1e-9 else n)
        return pos, nrm, row, beam


@dataclass
class Flux:
    vio_lumens: float = 168.0
    flash_lumens: float | None = None

    def __post_init__(self):
        self.vio_lumens = float(self.vio_lumens)
        self.flash_lumens = float(self.flash_lumens) if self.flash_lumens is not None else None


@dataclass
class StlModel:
    file: str
    scale: float = 1.0
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)
    occludes: bool = True
    """Blocks rays (shadow test) — otherwise display only."""
    opacity: float = 0.8
    wireframe: bool = False

    def __post_init__(self):
        self.position, self.rotation = _v3(self.position), _v3(self.rotation)
        self.scale, self.opacity = float(self.scale), float(self.opacity)
        self.occludes, self.wireframe = bool(self.occludes), bool(self.wireframe)


@dataclass
class VioCameras:
    position: tuple = (10.0, 0.0, 0.0)
    cam1_pitch: float = 45.0
    cam1_yaw: float = 0.0
    cam2_pitch: float = -45.0
    cam2_yaw: float = 0.0
    long_fov: float = 170.0
    landscape: bool = True

    def __post_init__(self):
        self.position = _v3(self.position)


@dataclass
class Platform:
    name: str = "platform"
    stl: StlModel | None = None
    vio_cameras: VioCameras = field(default_factory=VioCameras)

    def __post_init__(self):
        if isinstance(self.stl, dict):
            self.stl = StlModel(**self.stl) if self.stl.get('file') else None
        if isinstance(self.vio_cameras, dict):
            self.vio_cameras = VioCameras(**self.vio_cameras)

    def to_dict(self):
        d = asdict(self)
        d = {'schema_version': SCHEMA_VERSION, **d}
        return d


@dataclass
class Layout:
    name: str = "layout"
    description: str = ""
    platform: str | Platform | None = None
    """Platform name (``platforms/<name>.json``) or an inline :class:`Platform`."""
    flux: Flux = field(default_factory=Flux)
    panels: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.platform, dict):
            self.platform = Platform(**{k: v for k, v in self.platform.items() if k != 'schema_version'})
        if isinstance(self.flux, dict):
            self.flux = Flux(**self.flux)
        self.panels = [p if isinstance(p, Panel) else Panel(**p) for p in self.panels]

    def to_dict(self):
        plat = self.platform.to_dict() if isinstance(self.platform, Platform) else self.platform
        return {
            'schema_version': SCHEMA_VERSION, 'name': self.name, 'description': self.description,
            'platform': plat, 'flux': asdict(self.flux),
            'panels': [asdict(p) for p in self.panels],
        }

    def resolve_platform(self, platforms_dir="platforms") -> Platform:
        if isinstance(self.platform, Platform):
            return self.platform
        if self.platform:
            return load_platform(Path(platforms_dir) / f"{self.platform}.json")
        return Platform()


# --------------------------------------------------------------------------- io
def is_v2(cfg):
    return isinstance(cfg, dict) and int(cfg.get('schema_version', 1) or 1) >= 2


def layout_from_dict(d) -> Layout:
    d = {k: v for k, v in d.items() if k != 'schema_version'}
    return Layout(**d)


def load_layout(path, default_lumens=168.0) -> Layout:
    """Read a layout file; pre-v2 documents are converted on the fly."""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if is_v2(d):
        return layout_from_dict(d)
    lay = convert_v1(d, default_lumens=default_lumens)
    if not lay.name:
        lay.name = Path(path).stem
    return lay


def load_platform(path) -> Platform:
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return Platform(**{k: v for k, v in d.items() if k != 'schema_version'})


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj.to_dict(), f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------- build
def _mirror_leds(placements, start_index):
    out = []
    for p in placements:
        led = LED(position=XZ_MIRROR @ np.asarray(p.led.position, float),
                  direction=XZ_MIRROR @ np.asarray(p.led.direction, float),
                  lumens=p.led.lumens, color=p.led.color, width=p.led.width,
                  viewing_angle=p.led.viewing_angle, enabled=p.led.enabled, role=p.led.role)
        out.append(LEDPlacement(led=led, index=start_index + len(out),
                                row_direction=XZ_MIRROR @ np.asarray(p.row_direction, float),
                                square_normal=XZ_MIRROR @ np.asarray(p.square_normal, float),
                                owner=None, is_custom=True, is_dynamic_group=True))
    return out


def build_panel_leds(panel: Panel, lumens, owner=None, start_index=0):
    """World-space placements of one panel (mirror twin appended when ``panel.mirror``)."""
    pos, nrm, row, beam = panel.world_geometry()
    out = []
    for l, p, n, r, b in zip(panel.leds, pos, nrm, row, beam):
        led = LED(position=np.asarray(p, float), direction=np.asarray(b, float), lumens=float(lumens),
                  color=(1.0, 0.0, 1.0), width=l.size, viewing_angle=l.beam_angle, enabled=l.on, role=l.role)
        out.append(LEDPlacement(led=led, index=start_index + len(out), row_direction=np.asarray(r, float),
                                square_normal=np.asarray(n, float), owner=owner, is_custom=True,
                                is_dynamic_group=True))
    if panel.mirror:
        out += _mirror_leds(out, start_index + len(out))
    return out


def build_leds_from_layout(layout: Layout, lumens=None):
    """Placements of every enabled panel; ``lumens`` overrides ``layout.flux.vio_lumens``."""
    lm = float(lumens if lumens is not None else layout.flux.vio_lumens)
    leds = []
    for i, panel in enumerate(layout.panels):
        if panel.enabled:
            leds += build_panel_leds(panel, lm, owner=('panel', i), start_index=len(leds))
    return leds


# --------------------------------------------------------------------------- v1 conversion
def _v1_group_to_panel(g, index, mirror, global_R, global_offset):
    """A v1 custom group → panel with panel-local LEDs (world geometry reproduced exactly)."""
    from lighting_simulator.domain.guides import dynamic_group_world_geometry
    from lighting_simulator.scene.builder import euler_applies, group_runtime_state

    if not g.get('is_dynamic', False):
        raise ValueError(f"custom group {index}: non-dynamic (Elios-3 template) groups are not supported in v2")
    positions, directions, row_dirs = dynamic_group_world_geometry(group_runtime_state(g))
    rot = (g.get('rotation_x', 0.0), g.get('rotation_y', 0.0), g.get('rotation_z', 0.0)) if euler_applies(g) else (0, 0, 0)
    R = global_R @ euler_xyz_matrix(*rot)
    p0 = global_R @ np.asarray(g.get('position', (0, 0, 0)), float) + global_offset
    roll, pitch, yaw = rot
    yaw += float(np.degrees(np.arctan2(global_R[1, 0], global_R[0, 0])))
    n = int(g.get('num_leds', len(positions)))
    states = list(g.get('led_states') or [])
    roles = list(g.get('led_roles') or [])
    sizes, angles, tilts = g.get('led_sizes') or [], g.get('led_viewing_angles') or [], g.get('led_beam_tilts') or []
    leds = []
    for i in range(min(n, len(positions))):
        d = np.asarray(directions[i] if i < len(directions) else (1.0, 0.0, 0.0), float)
        if np.linalg.norm(d) < 1e-10:
            d = np.array([1.0, 0.0, 0.0])
        r = np.asarray(row_dirs[i], float) if i < len(row_dirs) else default_row_direction(d)
        pw = global_R @ np.asarray(positions[i], float) + global_offset
        leds.append(LedSpec(position=R.T @ (pw - p0), direction=R.T @ (global_R @ d), row_dir=R.T @ (global_R @ r),
                            beam_angle=angles[i] if i < len(angles) else 120.0,
                            tilt=tilts[i] if i < len(tilts) else 0.0,
                            size=sizes[i] if i < len(sizes) else 0.5,
                            on=states[i] if i < len(states) else True,
                            role=roles[i] if i < len(roles) else 'both'))
    src = {}
    if g.get('template_name'):
        src['template'] = g['template_name']
    if g.get('generated_by'):
        src['generator'] = g['generated_by']
    name = g.get('name') or g.get('panel_slot_name') or g.get('template_name') or f"panel{index}"
    return Panel(name=str(name), leds=leds, enabled=bool(g.get('enabled', True)), mirror=mirror,
                 position=p0, rotation=(roll, pitch, yaw), rows=g.get('led_rows'), source=src or None)


def _v1_individual_to_panel(cfg, index, global_R, global_offset):
    """A v1 individual LED → one-LED panel (square_roll and beam_tilt honoured)."""
    rx, ry, rz = (float(cfg.get(k, 0.0)) for k in ('rot_x', 'rot_y', 'rot_z'))
    cx, sx, cy, sy, cz, sz = (f(np.radians(a)) for a in (rx, ry, rz) for f in (np.cos, np.sin))
    Rm = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
          @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    d = global_R @ (Rm @ np.array([1.0, 0.0, 0.0]))
    r = default_row_direction(d)
    roll = float(cfg.get('square_roll', 0.0))
    if abs(roll) > 1e-9:
        r = rodrigues_rotation(d, np.radians(roll)) @ r
    p = global_R @ np.array([cfg.get('pos_x', 0.0), cfg.get('pos_y', 0.0), cfg.get('pos_z', 0.0)], float) + global_offset
    led = LedSpec(position=(0, 0, 0), direction=d, row_dir=r, beam_angle=cfg.get('viewing_angle', 120),
                  tilt=cfg.get('beam_tilt', 0.0), size=cfg.get('size', 0.5), on=cfg.get('led_on', True),
                  role=cfg.get('role', 'both'))
    return Panel(name=f"led{index}", leds=[led], enabled=bool(cfg.get('enabled', True)), position=p)


def convert_v1(cfg, default_lumens=168.0) -> Layout:
    """Upgrade a pre-v2 UI config dict. Drops the base ring, absorbers and lumens overrides;
    bakes the global transform and construction guides into the panels."""
    if is_v2(cfg):
        return layout_from_dict(cfg)
    if any(cfg.get('led_states', [])):
        print("[layout] v1 config lights base-ring LEDs: they are dropped in schema v2")
    rot_z = float(cfg.get('global_rotation_z', 0.0))
    global_R = rotation_matrix_z(rot_z) if abs(rot_z) > 1e-9 else np.eye(3)
    global_offset = np.array([cfg.get('global_pos_x', 0.0), cfg.get('global_pos_y', 0.0),
                              cfg.get('global_pos_z', 0.0)], float)
    mp = cfg.get('mirror_primary')
    primary = (mp['kind'], int(mp['key'])) if isinstance(mp, dict) and 'kind' in mp else None

    panels = []
    for i, g in enumerate(cfg.get('custom_groups', [])):
        slot = g.get('panel_slot')
        owner = ('slot', slot) if slot is not None else ('custom_group', i)
        if g.get('lumens_override_enabled'):
            print(f"[layout] custom group {i}: lumens override {g.get('lumens_value')} lm dropped (flux is per mode now)")
        panels.append(_v1_group_to_panel(g, i, owner == primary, global_R, global_offset))
    for i, l in enumerate(cfg.get('individual_leds', [])):
        panels.append(_v1_individual_to_panel(l, i, global_R, global_offset))

    stl = cfg.get('stl_model') or None
    platform = None
    if (stl and stl.get('file_path')) or cfg.get('vio_cameras'):
        vio = dict(cfg.get('vio_cameras') or {})
        vio.pop('show', None); vio.pop('fill', None)
        platform = Platform(
            name=str(cfg.get('name') or 'platform'),
            stl=StlModel(file=stl['file_path'], scale=stl.get('scale', 1.0), position=stl.get('position', (0, 0, 0)),
                         rotation=stl.get('rotation', (0, 0, 0)), occludes=stl.get('absorber_enable', True),
                         opacity=stl.get('opacity', 0.8), wireframe=stl.get('wireframe', False))
            if stl and stl.get('file_path') else None,
            vio_cameras=VioCameras(**vio) if vio else VioCameras(),
        )
    return Layout(name=str(cfg.get('name') or ''), description=str(cfg.get('description') or ''),
                  platform=platform, flux=Flux(vio_lumens=default_lumens), panels=panels)
