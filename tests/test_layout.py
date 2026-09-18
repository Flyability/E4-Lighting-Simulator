"""Schema v2 layouts: v1 conversion reproduces the scene, round-trips, mirror and transforms."""

import copy
import glob
import json

import numpy as np
import pytest

from lighting_simulator.scene import (
    Flux, Layout, LedSpec, Panel, Platform, StlModel, build_leds_from_layout, build_scene_from_config,
    build_scene_from_layout, convert_v1, layout_from_dict, load_config, load_layout, save_json,
)

CONFIGS = sorted(glob.glob("configs/*.json"))


def _signature(leds, with_lumens=True):
    rows = []
    for p in leds:
        if not p.enabled:
            continue
        rows.append((tuple(np.round(p.position, 5)), tuple(np.round(p.direction, 6)),
                     tuple(np.round(p.row_direction, 6)), tuple(np.round(p.square_normal, 6)),
                     round(float(p.width), 6), round(float(p.viewing_angle), 6), p.role,
                     round(float(p.lumens), 3) if with_lumens else None))
    return sorted(rows)


def _v1_without_engine_gaps(cfg):
    """v1 reference minus what v2 drops on purpose: the base ring, and the v1 engine's
    ignoring of individual-LED square_roll / beam_tilt (v2 honours them)."""
    cfg = copy.deepcopy(cfg)
    cfg['led_states'] = [False] * len(cfg.get('led_states', []))
    for l in cfg.get('individual_leds', []):
        l['square_roll'] = 0.0
        l['beam_tilt'] = 0.0
    return cfg


@pytest.mark.parametrize("path", CONFIGS, ids=[p.split("/")[-1] for p in CONFIGS])
def test_v1_config_converts_to_identical_scene(path):
    cfg = load_config(path)
    v1 = build_scene_from_config(_v1_without_engine_gaps(cfg), default_lumens=168.0)
    layout = convert_v1(cfg, default_lumens=168.0)
    v2 = build_scene_from_layout(layout)
    has_override = any(g.get('lumens_override_enabled') for g in cfg.get('custom_groups', [])) or any(
        l.get('lumens_override_enabled') for l in cfg.get('individual_leds', []))
    assert _signature(v2.leds, not has_override) == _signature(v1.leds, not has_override)
    assert len(v2.leds) == len(v1.leds)
    # round trip through JSON
    again = layout_from_dict(json.loads(json.dumps(layout.to_dict())))
    assert _signature(build_leds_from_layout(again)) == _signature(v2.leds)
    if cfg.get('stl_model', {}) and cfg['stl_model'].get('file_path'):
        assert isinstance(layout.platform, Platform) and layout.platform.stl.file == cfg['stl_model']['file_path']


def test_convert_drops_legacy_parts_and_bakes_global_transform():
    cfg = load_config("configs/elios3v1_l.json")
    assert cfg['global_rotation_z'] == 33
    layout = convert_v1(cfg)
    d = layout.to_dict()
    for k in ('led_states', 'group_rotations', 'absorbers', 'individual_leds', 'custom_groups', 'global_rotation_z'):
        assert k not in d
    assert d['schema_version'] == 2 and d['flux'] == {'vio_lumens': 168.0, 'flash_lumens': None}
    # the baked yaw shows up in the panel rotation, not in the LED coordinates
    assert all(abs(p.rotation[2] - 33) < 1e-9 or abs(p.rotation[2] - 33) % 360 < 1e-6 for p in layout.panels)


def test_mirror_flag_generates_xz_twin_and_halves_the_file():
    cfg = load_config("configs/ludos_panels_n4.json")
    layout = convert_v1(cfg)
    assert layout.panels[0].mirror and not layout.panels[1].mirror and not layout.panels[1].enabled
    leds = build_leds_from_layout(layout)
    active = [l for l in leds if l.enabled]
    assert len(active) == 8 and sum(1 for p in layout.panels if p.enabled) == 1
    P = np.array([l.position for l in active])
    D = np.array([l.direction for l in active])
    M = np.array([1, -1, 1])
    for i in range(len(active)):
        d = np.linalg.norm(P - P[i] * M, axis=1) + np.linalg.norm(D - D[i] * M, axis=1)
        assert d.min() < 1e-9
    assert [l.owner for l in leds[:9]] == [('panel', 0)] * 9 and all(l.owner is None for l in leds[9:])


def test_panel_transform_and_tilt():
    led = LedSpec(position=(1, 0, 0), direction=(1, 0, 0), row_dir=(0, 1, 0), tilt=30.0, beam_angle=100, size=0.7,
                  role='flash', on=False)
    panel = Panel(name="p", leds=[led], position=(10, 0, 0), rotation=(0, 0, 90))
    pos, nrm, row, beam = panel.world_geometry()
    assert np.allclose(pos[0], (10, 1, 0)) and np.allclose(nrm[0], (0, 1, 0)) and np.allclose(row[0], (-1, 0, 0))
    # tilt about the row direction: beam rotates in the plane ⟂ row_dir
    assert abs(np.dot(beam[0], row[0])) < 1e-9 and abs(np.degrees(np.arccos(np.dot(beam[0], nrm[0]))) - 30) < 1e-9
    leds = build_leds_from_layout(Layout(panels=[panel], flux=Flux(vio_lumens=500)))
    assert leds[0].lumens == 500 and leds[0].viewing_angle == 100 and leds[0].width == 0.7
    assert leds[0].role == 'flash' and not leds[0].enabled
    assert np.allclose(leds[0].square_normal, nrm[0]) and np.allclose(leds[0].direction, beam[0])


def test_layout_and_platform_files(tmp_path):
    plat = Platform(name="rig", stl=StlModel(file="frame.STEP", scale=0.1, occludes=False))
    save_json(plat, tmp_path / "platforms" / "rig.json")
    layout = Layout(name="lay", platform="rig", panels=[Panel(name="a", leds=[LedSpec((0, 0, 0), (1, 0, 0))])])
    save_json(layout, tmp_path / "layouts" / "lay.json")
    back = load_layout(tmp_path / "layouts" / "lay.json")
    assert back.platform == "rig" and back.panels[0].leds[0].direction == (1.0, 0.0, 0.0)
    resolved = back.resolve_platform(tmp_path / "platforms")
    assert resolved.stl.file == "frame.STEP" and resolved.stl.scale == 0.1 and not resolved.stl.occludes
    # a v1 file loads through the same entry point
    lay1 = load_layout("configs/ludos_panels_n4.json")
    assert lay1.name == "ludos_panels_n4" and isinstance(lay1.platform, Platform)
    # build_scene_from_config accepts a v2 dict directly
    scene = build_scene_from_config(layout.to_dict())
    assert len(scene.leds) == 1 and scene.leds[0].lumens == 168.0
