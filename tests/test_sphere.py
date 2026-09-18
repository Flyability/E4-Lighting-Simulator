"""Sphere mode: geometry and flux conservation."""

import numpy as np
import pytest

from lighting_simulator.domain.led import LED
from lighting_simulator.simulation import EmissionSettings
from lighting_simulator.simulation.sphere import (
    SphereSettings, compute_sphere_illuminance, sphere_cell_vertices, sphere_cells,
)


def test_sphere_cells_cover_the_sphere():
    s = SphereSettings(radius_cm=300.0, n_phi=72)
    centers, normals, areas = sphere_cells(s)
    assert centers.shape == (36, 72, 3) and areas.shape == (36, 72)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=-1), 300.0)
    np.testing.assert_allclose(np.einsum('ijk,ijk->ij', normals, centers / 300.0), -1.0)
    assert areas.sum() == pytest.approx(4 * np.pi * 3.0 ** 2, rel=1e-9)
    quads = sphere_cell_vertices(s)
    assert quads.shape == (36, 72, 4, 3)
    # quad normal (v1-v0)x(v2-v0) must point towards the centre
    n = np.cross(quads[..., 1, :] - quads[..., 0, :], quads[..., 2, :] - quads[..., 0, :])
    assert np.all(np.einsum('ijk,ijk->ij', n, quads[..., 0, :]) <= 1e-12)


def test_sphere_illuminance_conserves_flux():
    leds = [LED(position=(5, 2, -1), direction=(1, 0, 0), viewing_angle=120.0),
            LED(position=(-3, -4, 2), direction=(0, 0, 1), viewing_angle=90.0)]
    s = SphereSettings(radius_cm=300.0, n_phi=180)
    lux = compute_sphere_illuminance(leds, s, EmissionSettings(default_lumens=100.0, ray_uniformity=0.0))
    _, _, areas = sphere_cells(s)
    assert lux.shape == (90, 180)
    assert (lux * areas).sum() == pytest.approx(200.0, rel=0.01)
    assert lux[:, :90].sum() > 0  # forward-facing LED lights the +X half
