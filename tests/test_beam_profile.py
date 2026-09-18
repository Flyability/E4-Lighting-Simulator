"""Measured beam profiles: normalisation, analytic vs Monte-Carlo agreement, persistence."""

import numpy as np
import pytest

from lighting_simulator.domain.beam_profile import LAMBERTIAN, BeamProfile, builtin_profiles, get_profile, profile_names
from lighting_simulator.domain.led import LED
from lighting_simulator.scene.layout import Layout, LedSpec, Panel, build_leds_from_layout, layout_from_dict, layout_to_v1
from lighting_simulator.scene.builder import build_leds_from_config
from lighting_simulator.simulation import EmissionSettings, WallSettings, compute_wall_intensity
from lighting_simulator.simulation.direct import direct_wall_intensity
from lighting_simulator.simulation.emission import generate_led_rays, led_rng
from lighting_simulator.simulation.sphere import SphereSettings, compute_sphere_illuminance, sphere_cells


def test_xfl12k_profile_shape_and_normalisation():
    p = builtin_profiles()["XFL12K HD"]
    assert p.relative(0.0) == pytest.approx(1.0)
    assert p.relative(60.0) == pytest.approx(0.80)
    assert p.relative(90.0) == pytest.approx(0.18)
    assert p.relative(95.0) == 0.0
    assert 150 < p.half_intensity_angle_deg() < 170  # much flatter than a 120° cos^n beam
    # ∫ I dΩ over the hemisphere == flux
    th = np.radians(np.linspace(0, 90, 20001))
    integral = 2 * np.pi * np.trapz(p.intensity_cd(np.degrees(th), 500.0) * np.sin(th), th)
    assert integral == pytest.approx(500.0, rel=1e-4)


def test_profile_flux_conserved_on_sphere_and_in_rays():
    p = builtin_profiles()["XFL12K HD"]
    led = LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=120.0, lumens=168.0, beam_profile=p)
    s = SphereSettings(radius_cm=200.0, n_phi=180)
    lux = compute_sphere_illuminance([led], s, EmissionSettings(default_lumens=168.0))
    _, _, areas = sphere_cells(s)
    assert (lux * areas).sum() == pytest.approx(168.0, rel=0.01)
    _, w = generate_led_rays(led, 200_000, 168.0, 0.0, led_rng(0))
    assert w.sum() == pytest.approx(168.0, rel=0.01)


def test_profile_analytic_matches_cpu_tracer():
    p = builtin_profiles()["XFL12K HD"]
    leds = [LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=120.0, lumens=168.0, beam_profile=p)]
    ws = WallSettings(wall_dist=60, grid_size=24, wall_size=300, rays_per_pixel=3000)
    em = EmissionSettings(default_lumens=168.0)
    mc = compute_wall_intensity(leds, ws, em, use_gpu=False, verbose=False)
    an = direct_wall_intensity(leds, ws, em)
    assert an.sum() == pytest.approx(mc.sum(), rel=0.03)
    # the flat profile lights the wall edge (±68° off axis) far more than a cos^n 120° beam would
    plain = direct_wall_intensity([LED(position=(0, 0, 0), direction=(1, 0, 0), viewing_angle=120.0, lumens=168.0)], ws, em)
    assert an[12, 0] > 3 * plain[12, 0]


def test_profile_survives_layout_and_v1_round_trip():
    lay = Layout(name="t", panels=[Panel(name="p", leds=[
        LedSpec(position=(0, 0, 0), direction=(1, 0, 0), profile="XFL12K HD"),
        LedSpec(position=(0, 1, 0), direction=(1, 0, 0)),
    ], mirror=True)])
    back = layout_from_dict(lay.to_dict())
    assert [l.profile for l in back.panels[0].leds] == ["XFL12K HD", None]
    leds = build_leds_from_layout(back, lumens=100.0)
    assert [getattr(l.beam_profile, 'name', None) for l in leds] == ["XFL12K HD", None, "XFL12K HD", None]
    v1 = build_leds_from_config(layout_to_v1(back), default_lumens=100.0)
    assert sorted(getattr(l.beam_profile, 'name', '') for l in v1) == ["", "", "XFL12K HD", "XFL12K HD"]
    assert get_profile(LAMBERTIAN) is None and "XFL12K HD" in profile_names()


def test_bad_profiles_rejected():
    with pytest.raises(ValueError):
        BeamProfile("x", [0, 10, 5], [1, 1, 1])
    with pytest.raises(ValueError):
        BeamProfile("x", [0, 10], [0, 0])
