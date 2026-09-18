"""Sphere mode: direct illuminance on a sphere of radius R centred on the drone.

Every cell of the sphere is at the same distance from the platform, so the map isolates the
angular light distribution (VIO coverage all around) from the 1/d² fall-off of flat walls.
Cells are a latitude/longitude grid in the drone frame (+Z up, +X forward); lux values are the
analytic expectation of the tracer (``direct_illuminance``), optionally shadowed by the frame.
"""
from dataclasses import dataclass

import numpy as np

from lighting_simulator.simulation.direct import direct_illuminance


@dataclass
class SphereSettings:
    radius_cm: float = 300.0
    n_phi: int = 72
    """Cells around the Z axis (longitude); latitude rows = n_phi // 2."""
    center_cm: tuple = (0.0, 0.0, 0.0)

    @property
    def n_theta(self):
        return max(2, int(self.n_phi) // 2)


def sphere_cell_edges(settings: SphereSettings):
    """(theta_edges (n_theta+1,), phi_edges (n_phi+1,)) in radians; theta from +Z, phi from +X towards +Y."""
    return (np.linspace(0.0, np.pi, settings.n_theta + 1),
            np.linspace(-np.pi, np.pi, int(settings.n_phi) + 1))


def sphere_cells(settings: SphereSettings):
    """Cell centres (n_theta, n_phi, 3) cm, inward unit normals (same shape) and cell areas (n_theta, n_phi) m²."""
    th_e, ph_e = sphere_cell_edges(settings)
    th = 0.5 * (th_e[:-1] + th_e[1:])
    ph = 0.5 * (ph_e[:-1] + ph_e[1:])
    T, P = np.meshgrid(th, ph, indexing='ij')
    n_out = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
    centers = np.asarray(settings.center_cm, dtype=float) + settings.radius_cm * n_out
    r_m = settings.radius_cm / 100.0
    band = (np.cos(th_e[:-1]) - np.cos(th_e[1:])) * r_m ** 2  # per latitude row
    areas = np.repeat(band[:, None], int(settings.n_phi), axis=1) * (ph_e[1] - ph_e[0])
    return centers, -n_out, areas


def sphere_cell_vertices(settings: SphereSettings):
    """Corner points (n_theta, n_phi, 4, 3) in metres, ordered so the quad normal points INWARD."""
    th_e, ph_e = sphere_cell_edges(settings)
    r_m = settings.radius_cm / 100.0
    c = np.asarray(settings.center_cm, dtype=float) / 100.0

    def pt(t, p):
        return c + r_m * np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
    T0, P0 = np.meshgrid(th_e[:-1], ph_e[:-1], indexing='ij')
    T1, P1 = np.meshgrid(th_e[1:], ph_e[1:], indexing='ij')
    # (t0,p0) -> (t0,p1) -> (t1,p1) -> (t1,p0) is counter-clockwise seen from inside
    return np.stack([pt(T0, P0), pt(T0, P1), pt(T1, P1), pt(T1, P0)], axis=2)


def compute_sphere_illuminance(leds, settings: SphereSettings, emission, stl_mesh_data=None):
    """Lux on every cell (n_theta, n_phi)."""
    centers, normals, _ = sphere_cells(settings)
    lux = direct_illuminance(centers.reshape(-1, 3), normals.reshape(-1, 3), leds, emission,
                             stl_mesh_data=stl_mesh_data)
    return lux.reshape(settings.n_theta, int(settings.n_phi))
