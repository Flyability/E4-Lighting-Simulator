"""Build a simulation-ready scene (LEDs, STL occluder) from a saved config.

The config dict is either a schema-v2 layout (``scene.layout``) or the pre-v2 JSON document
produced by the UI's *Save Configuration*. This module is UI-free so optimisation loops can
mutate a config, rebuild the scene and re-simulate without Viser.
"""

from dataclasses import dataclass
import json

import numpy as np

from lighting_simulator.domain.geometry import rotation_matrix_z
from lighting_simulator.domain.guides import dynamic_group_world_geometry
from lighting_simulator.domain.led_factory import create_leds
from lighting_simulator.domain.mirroring import expand_mirror_configs, mirror_group_config_xz

from .layout import build_leds_from_layout, is_v2, layout_from_dict
from .stl import global_z_rotation_4x4, stl_mesh_data, stl_transform

DEFAULT_ROWS = [True] * 4


@dataclass
class Scene:
    leds: list
    stl_mesh_data: dict | None = None

    @property
    def active_leds(self):
        return [led for led in self.leds if getattr(led, 'enabled', True)]


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _lumens_override(cfg):
    if cfg.get('lumens_override_enabled'):
        return float(cfg.get('lumens_value', 100))
    return None


def euler_applies(group_cfg):
    """UI convention: saved rotation_x/y/z are honoured only for panel-slot groups
    or groups with an active construction guide; for every other dynamic group the
    orientation is expected to be baked into led_positions / led_rotations."""
    guide = group_cfg.get('guide')
    return group_cfg.get('panel_slot') is not None or bool(isinstance(guide, dict) and guide.get('enabled'))


def group_runtime_state(group_cfg):
    """Pose/geometry dict accepted by ``domain.guides.dynamic_group_world_geometry``."""
    pos = group_cfg.get('position', (0.0, 0.0, 0.0))
    use_euler = euler_applies(group_cfg)
    state = {
        'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': pos[2],
        'rot_roll': group_cfg.get('rotation_x', 0.0) if use_euler else 0.0,
        'rot_tilt_ud': group_cfg.get('rotation_y', 0.0) if use_euler else 0.0,
        'rot_tilt_lr': group_cfg.get('rotation_z', 0.0) if use_euler else 0.0,
        'is_dynamic': bool(group_cfg.get('is_dynamic', False)),
        'led_rows': group_cfg.get('led_rows'),
        'guide': group_cfg.get('guide'),
        # Saved led_* are the rest pose (relative to `position`).
        'original_led_positions': [tuple(p) for p in group_cfg.get('led_positions', [])],
        'original_led_rotations': [tuple(r) for r in group_cfg.get('led_rotations', [])],
        'original_led_row_directions': [tuple(r) for r in group_cfg.get('led_row_directions', [])],
    }
    return state


def group_config_to_factory(group_cfg, owner=None, row_enabled=None):
    """Convert one saved custom-group entry into a ``create_leds`` group config."""
    pos = group_cfg.get('position', (0.0, 0.0, 0.0))
    config = {
        'enabled': bool(group_cfg.get('enabled', True)),
        'position': tuple(pos),
        'rotation_x': group_cfg.get('rotation_x', 0.0),
        'rotation_y': group_cfg.get('rotation_y', 0.0),
        'rotation_z': group_cfg.get('rotation_z', 0.0),
        'led_states': list(group_cfg.get('led_states', [True] * 12)),
        'led_roles': list(group_cfg.get('led_roles') or []),
        'row_enabled': list(row_enabled or DEFAULT_ROWS),
        'lumens_override': _lumens_override(group_cfg),
        'owner': owner,
    }
    if group_cfg.get('is_dynamic', False):
        positions, directions, row_dirs = dynamic_group_world_geometry(group_runtime_state(group_cfg))
        config.update({
            'num_leds': group_cfg.get('num_leds', len(positions)),
            'led_positions': positions,
            'led_rotations': directions,
            'led_viewing_angles': list(group_cfg.get('led_viewing_angles', [])),
            'led_beam_tilts': list(group_cfg.get('led_beam_tilts') or []),
            'led_profiles': list(group_cfg.get('led_profiles') or []),
            'led_sizes': list(group_cfg.get('led_sizes', [])),
            'led_lumens': list(group_cfg.get('led_lumens', [])),
        })
        if row_dirs:
            config['led_row_directions'] = row_dirs
    return config


def individual_led_config_to_factory(led_cfg, owner=None):
    """Convert one saved individual-LED entry into a ``create_leds`` LED config."""
    config = {
        'enabled': bool(led_cfg.get('enabled', True)),
        'led_on': bool(led_cfg.get('led_on', True)),
        'pos_x': led_cfg.get('pos_x', 0.0), 'pos_y': led_cfg.get('pos_y', 0.0), 'pos_z': led_cfg.get('pos_z', 0.0),
        'rot_x': led_cfg.get('rot_x', 0.0), 'rot_y': led_cfg.get('rot_y', 0.0), 'rot_z': led_cfg.get('rot_z', 0.0),
        'size': led_cfg.get('size', 0.5),
        'viewing_angle': led_cfg.get('viewing_angle', 120),
        'square_roll': led_cfg.get('square_roll', 0.0),
        'beam_tilt': led_cfg.get('beam_tilt', 0.0),
        'role': led_cfg.get('role', 'both'),
        'lumens_override': _lumens_override(led_cfg),
        'owner': owner,
    }
    if led_cfg.get('ext_lens_enabled'):
        config['ext_lens_angle'] = float(led_cfg.get('ext_lens_angle', 30))
        config['ext_lens_efficiency'] = float(led_cfg.get('ext_lens_efficiency', 80)) / 100.0
    return config


def apply_global_transform(leds, rotation_z_deg=0.0, offset_cm=(0.0, 0.0, 0.0)):
    """Rotate the whole rig about Z, then translate (in place). Returns the 3x3 rotation or None."""
    R = None
    if abs(rotation_z_deg) > 0.01:
        R = rotation_matrix_z(rotation_z_deg)
        for led in leds:
            led.position = R @ led.position
            led.direction = R @ led.direction
            if getattr(led, 'row_direction', None) is not None:
                led.row_direction = R @ np.asarray(led.row_direction)
            if getattr(led, 'square_normal', None) is not None:
                led.square_normal = R @ np.asarray(led.square_normal)
    offset = np.asarray(offset_cm, dtype=float)
    if np.any(np.abs(offset) > 0.001):
        for led in leds:
            led.position = led.position + offset
    return R


def apply_diffuser(leds, angle_deg, transmission):
    """Widen every LED's beam to at least ``angle_deg`` and scale flux by ``transmission``.

    A diffuser redefines the beam, so any measured ``beam_profile`` is dropped in favour of the cosⁿ cone.
    """
    for led in leds:
        led.viewing_angle = max(led.viewing_angle, float(angle_deg))
        led.beam_profile = None
        if led.lumens is not None:
            led.lumens = led.lumens * float(transmission)


def build_leds_from_config(cfg, default_lumens=100.0, mirror_primary=None):
    """LED placements from a v1 config (no global transform / diffuser applied)."""
    row_enabled = list(cfg.get('row_enabled', DEFAULT_ROWS))

    custom_groups_configs = []
    mirrored = []
    for idx, group_cfg in enumerate(cfg.get('custom_groups', [])):
        slot = group_cfg.get('panel_slot')
        owner = ('slot', slot) if slot is not None else ('custom_group', idx)
        config = group_config_to_factory(group_cfg, owner=owner, row_enabled=row_enabled)
        custom_groups_configs.append(config)
        if group_cfg.get('mirror'):
            mirrored.append(config)
    individual_leds_configs = [
        individual_led_config_to_factory(led_cfg) for led_cfg in cfg.get('individual_leds', [])
    ]
    primary = _decode_mirror_primary(cfg, mirror_primary)
    expand_mirror_configs(custom_groups_configs, individual_leds_configs, primary)
    # per-group XZ twins (skip the legacy mirror primary, already expanded above)
    custom_groups_configs += [mirror_group_config_xz(c) for c in mirrored if c.get('owner') != primary]

    return create_leds(
        cfg.get('viewing_angle', 120),
        default_lumens=float(default_lumens),
        custom_groups_configs=custom_groups_configs,
        individual_leds_configs=individual_leds_configs,
    )


def _decode_mirror_primary(cfg, override):
    if override is not None:
        return override
    mp = cfg.get('mirror_primary')
    if isinstance(mp, dict) and 'kind' in mp:
        return (mp['kind'], int(mp['key']))
    return None


def build_scene_from_config(cfg, default_lumens=100.0, stl_mesh=None,
                            diffuser=None, mirror_primary=None):
    """Full scene from a saved config (schema v1 dict or v2 layout dict).

    ``stl_mesh`` is an optional loaded ``trimesh.Trimesh`` for the config's STL model.
    ``diffuser`` is an optional ``(angle_deg, transmission)`` tuple. For v2 layouts the
    flux comes from ``cfg['flux']`` and ``default_lumens`` is ignored.
    """
    if is_v2(cfg):
        return build_scene_from_layout(layout_from_dict(cfg), stl_mesh=stl_mesh, diffuser=diffuser)
    leds = build_leds_from_config(cfg, default_lumens=default_lumens, mirror_primary=mirror_primary)
    rot_z = float(cfg.get('global_rotation_z', 0.0))
    offset = (cfg.get('global_pos_x', 0.0), cfg.get('global_pos_y', 0.0), cfg.get('global_pos_z', 0.0))
    R = apply_global_transform(leds, rot_z, offset)
    if diffuser is not None:
        apply_diffuser(leds, *diffuser)

    mesh_data = None
    stl_cfg = cfg.get('stl_model')
    if stl_mesh is not None and stl_cfg and stl_cfg.get('absorber_enable', True):
        transform = stl_transform(stl_cfg.get('scale', 1.0), stl_cfg.get('rotation', (0, 0, 0)),
                                  stl_cfg.get('position', (0, 0, 0)))
        if R is not None:
            transform = global_z_rotation_4x4(rot_z) @ transform
        mesh_data = stl_mesh_data(stl_mesh, transform)

    return Scene(leds=leds, stl_mesh_data=mesh_data)


def build_scene_from_layout(layout, platform=None, stl_mesh=None, diffuser=None, lumens=None, platforms_dir="platforms"):
    """Scene from a v2 :class:`Layout`; ``platform`` defaults to the layout's (inline or by name)."""
    leds = build_leds_from_layout(layout, lumens=lumens)
    if diffuser is not None:
        apply_diffuser(leds, *diffuser)
    mesh_data = None
    if stl_mesh is not None:
        plat = platform if platform is not None else layout.resolve_platform(platforms_dir)
        if plat.stl is not None and plat.stl.occludes:
            mesh_data = stl_mesh_data(stl_mesh, stl_transform(plat.stl.scale, plat.stl.rotation, plat.stl.position))
    return Scene(leds=leds, stl_mesh_data=mesh_data)
