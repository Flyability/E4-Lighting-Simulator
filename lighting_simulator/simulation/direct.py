"""Analytic direct illuminance: the expected value of the Monte-Carlo tracer, per receiving point.

Same emission model as ``emission.generate_led_rays`` (uniform directions inside the cone,
weight cosⁿθ, total flux Φ), so on a lit cell this equals the MC mean without ray noise:

    E = Σ_LEDs  Φ (n+1) cosⁿθ / (2π (1 − cos^{n+1}θ_max)) · cosφ / d²

with θ the angle off the LED axis, φ the incidence angle on the surface and d the LED–point
distance. Occlusion by absorber boxes / the STL mesh is a per-(LED, point) shadow test; wall
reflections are NOT modelled (use the MC tracers when ``max_bounces > 0``).
"""

import numpy as np

from lighting_simulator.domain.optics import effective_lambertian_exponent
from lighting_simulator.raytracing.boxes import ray_box_intersection_batch
from lighting_simulator.raytracing.mesh import batch_ray_mesh_intersection, prepare_mesh_ray_accelerator
from lighting_simulator.simulation.emission import led_lumens


def direct_illuminance(points_cm, inward_normals, leds, emission, lumens=None, absorbers=(),
                       stl_mesh_data=None, accel=None):
    """Lux at ``points_cm`` (N, 3) whose surfaces face ``inward_normals`` (N, 3), from ``leds``.

    ``lumens``: optional per-LED flux overriding each LED's own (e.g. the flash pulse flux).
    """
    pts = np.asarray(points_cm, dtype=float).reshape(-1, 3)
    nrm = np.asarray(inward_normals, dtype=float).reshape(-1, 3)
    out = np.zeros(len(pts))
    if len(pts) == 0 or not leds:
        return out
    absorbers = list(absorbers or [])
    if accel is None and stl_mesh_data is not None:
        accel = prepare_mesh_ray_accelerator(stl_mesh_data)
    for i, led in enumerate(leds):
        flux = float(lumens[i]) if lumens is not None else led_lumens(led, emission.default_lumens)
        if flux <= 0:
            continue
        pos = np.asarray(led.position, dtype=float)
        axis = np.asarray(led.direction, dtype=float)
        axis /= max(np.linalg.norm(axis), 1e-12)
        profile = getattr(led, 'beam_profile', None)
        if profile is not None:
            n, cos_max, denom = 0.0, profile.cos_max, 1.0
        else:
            n = effective_lambertian_exponent(led, emission.ray_uniformity)
            cos_max = np.cos(np.radians(float(led.viewing_angle) / 2.0))
            denom = 1.0 - cos_max ** (n + 1.0)
            if denom <= 1e-12:
                continue
        vec = pts - pos
        d_cm = np.linalg.norm(vec, axis=1)
        ok = d_cm > 1e-6
        dirs = np.zeros_like(vec)
        dirs[ok] = vec[ok] / d_cm[ok, None]
        cos_theta = dirs @ axis
        cos_phi = -np.einsum('ij,ij->i', dirs, nrm)
        lit = ok & (cos_theta >= cos_max) & (cos_theta > 0) & (cos_phi > 0)
        if not np.any(lit):
            continue
        if absorbers:
            origins = np.broadcast_to(pos, (int(lit.sum()), 3)).astype(np.float32)
            blocked = ray_box_intersection_batch(origins, dirs[lit].astype(np.float32), absorbers)
            idx = np.where(lit)[0]
            lit[idx[blocked]] = False
        if accel is not None and np.any(lit):
            idx = np.where(lit)[0]
            origins = np.tile(pos, (len(idx), 1)).astype(np.float64)
            lit[idx[batch_ray_mesh_intersection(origins, dirs[idx], accel, verbose=False)]] = False
        if not np.any(lit):
            continue
        if profile is not None:
            intensity = profile.intensity_cd(np.degrees(np.arccos(np.clip(cos_theta[lit], -1.0, 1.0))), flux)
        else:
            intensity = flux * (n + 1.0) * np.power(cos_theta[lit], n) / (2.0 * np.pi * denom)  # cd
        out[lit] += intensity * cos_phi[lit] / (d_cm[lit] / 100.0) ** 2
    return out


def _block_mean(grid, k):
    """Average k×k blocks of a (k·R, k·C) grid down to (R, C)."""
    if k <= 1:
        return grid
    r, c = grid.shape[0] // k, grid.shape[1] // k
    return grid.reshape(r, k, c, k).mean(axis=(1, 3))


def direct_wall_intensity(leds, settings, emission, absorbers=(), stl_mesh_data=None, supersample=1, accel=None):
    """Analytic counterpart of ``wall.compute_wall_intensity``: lux grid (rows = Z, cols = Y).

    ``supersample`` = k evaluates k×k points per cell and averages them, matching the tracer's
    area integration at cone / shadow edges (cost ∝ k²). ``accel``: prebuilt mesh BVH (optional).
    """
    from lighting_simulator.simulation.room_geometry import wall_grid_cell_centers_cm
    k = max(1, int(supersample))
    g = int(settings.grid_size) * k
    pts = wall_grid_cell_centers_cm((g, g), float(settings.wall_size), float(settings.wall_dist))
    normals = np.tile([-1.0, 0.0, 0.0], (pts.size // 3, 1))
    active = [led for led in leds if getattr(led, 'enabled', True)]
    fine = direct_illuminance(pts.reshape(-1, 3), normals, active, emission, absorbers=absorbers,
                              stl_mesh_data=stl_mesh_data, accel=accel).reshape(g, g)
    return _block_mean(fine, k)


def direct_room_intensity(leds, settings, emission, absorbers=(), stl_mesh_data=None, supersample=1, accel=None):
    """Analytic counterpart of ``room.compute_room_intensity`` (direct light only): ``(grids, wall_specs)``."""
    from lighting_simulator.raytracing.mesh import prepare_mesh_ray_accelerator
    from lighting_simulator.simulation.room_geometry import (
        WALL_INWARD_NORMALS, build_wall_specs, room_wall_cell_centers, wall_grid_shape,
    )
    k = max(1, int(supersample))
    args = (settings.front_dist, settings.side_dist, settings.top_bottom_dist)
    wall_specs = build_wall_specs(*args, int(settings.grid_size), settings.led_x_center,
                                  settings.back_dist, settings.lateral_depth)
    fine_specs = build_wall_specs(*args, int(settings.grid_size) * k, settings.led_x_center,
                                  settings.back_dist, settings.lateral_depth)
    active = [led for led in leds if getattr(led, 'enabled', True)]
    if accel is None and stl_mesh_data is not None:
        accel = prepare_mesh_ray_accelerator(stl_mesh_data)
    grids = {}
    for name, spec in fine_specs.items():
        pts = room_wall_cell_centers(name, spec, *args, settings.back_dist)
        shape = wall_grid_shape(spec, name)
        normals = np.tile(WALL_INWARD_NORMALS[name], (pts.size // 3, 1))
        fine = direct_illuminance(pts.reshape(-1, 3), normals, active, emission, absorbers=absorbers,
                                  stl_mesh_data=stl_mesh_data, accel=accel).reshape(shape)
        grids[name] = _block_mean(fine, k)
    return grids, wall_specs
