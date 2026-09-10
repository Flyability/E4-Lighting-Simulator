"""Closed-room illuminance simulation (5 or 6 walls, optional diffuse bounces)."""

import copy
import multiprocessing
import time

import numpy as np

from lighting_simulator.domain.optics import sample_cosine_hemisphere
from lighting_simulator.raytracing.boxes import ray_box_intersection_batch
from lighting_simulator.raytracing.mesh import (
    batch_ray_mesh_intersection,
    prepare_mesh_ray_accelerator,
)

from . import gpu_backend
from .emission import generate_led_rays, led_lumens, led_rng
from .room_geometry import (
    WALL_IDS,
    WALL_INWARD_NORMALS,
    build_wall_specs,
    empty_wall_grids,
    nearest_wall_hits,
    wall_cell_areas_m2,
    wall_grid_indices,
)
from .settings import EmissionSettings, RoomSettings
from .wall import active_leds_with_index

_WALL_BY_ID = {v: k for k, v in WALL_IDS.items()}
# Offset (cm) applied along the wall normal after a bounce to avoid self-hits.
_BOUNCE_EPSILON_CM = 0.01


def _deposit(grids, cell_areas, wall_name, spec, c1, c2, flux):
    row, col = wall_grid_indices(wall_name, spec, c1, c2)
    shape = grids[wall_name].shape
    row = np.clip(np.asarray(row).astype(int), 0, shape[0] - 1)
    col = np.clip(np.asarray(col).astype(int), 0, shape[1] - 1)
    np.add.at(grids[wall_name], (row, col), np.asarray(flux) / cell_areas[wall_name])


def trace_led_in_room(led, led_idx, lumens, ray_uniformity, settings: RoomSettings,
                      wall_specs, absorbers=(), accel=None, rng=None):
    """Per-wall lux grids, per-wall ray-hit counts and rays traced for one LED."""
    grids = empty_wall_grids(wall_specs)
    ray_hits = {name: 0 for name in wall_specs}
    cell_areas = wall_cell_areas_m2(wall_specs)
    n_rays = settings.rays_per_led
    rng = rng if rng is not None else led_rng(led_idx)

    dirs, lumens_per_ray = generate_led_rays(led, n_rays, lumens, ray_uniformity, rng)

    absorbed = np.zeros(n_rays, dtype=bool)
    if absorbers:
        origins = np.broadcast_to(led.position, (n_rays, 3)).astype(np.float32)
        absorbed = ray_box_intersection_batch(origins, dirs.astype(np.float32), list(absorbers))
    if accel is not None:
        alive = np.where(~absorbed)[0]
        if len(alive) > 0:
            origins = np.tile(led.position, (len(alive), 1)).astype(np.float64)
            absorbed[alive[batch_ray_mesh_intersection(origins, dirs[alive], accel)]] = True

    alive = np.where(~absorbed)[0]
    origins = np.tile(np.asarray(led.position, dtype=np.float64), (len(alive), 1))
    ray_dirs = dirs[alive].copy()
    flux = lumens_per_ray[alive].copy()

    # Direct hits (vectorised), then optional diffuse bounces.
    for bounce in range(settings.max_bounces + 1):
        if len(flux) == 0:
            break
        wall_id, c1, c2, t = nearest_wall_hits(
            origins, ray_dirs, settings.front_dist, settings.side_dist,
            settings.top_bottom_dist, settings.back_dist,
        )
        hit = wall_id >= 0
        for wid in np.unique(wall_id[hit]):
            name = _WALL_BY_ID[int(wid)]
            sel = wall_id == wid
            _deposit(grids, cell_areas, name, wall_specs[name], c1[sel], c2[sel], flux[sel])
            ray_hits[name] += int(np.count_nonzero(sel))

        if bounce >= settings.max_bounces or settings.wall_reflectance <= 0:
            break
        flux = flux[hit] * settings.wall_reflectance
        keep = flux >= 1e-8
        flux = flux[keep]
        hit_idx = np.where(hit)[0][keep]
        hit_points = origins[hit_idx] + ray_dirs[hit_idx] * t[hit_idx][:, None]
        normals = np.array([WALL_INWARD_NORMALS[_WALL_BY_ID[int(w)]] for w in wall_id[hit_idx]])
        origins = hit_points + normals * _BOUNCE_EPSILON_CM
        ray_dirs = np.array([sample_cosine_hemisphere(nrm, rng) for nrm in normals]).reshape(-1, 3)

    return grids, ray_hits, n_rays


def _room_worker(args):
    """Multiprocessing entry point: ``args = (led, params)``."""
    led, p = args
    accel = prepare_mesh_ray_accelerator(p.get('stl_mesh_data'))
    return trace_led_in_room(
        led, p['led_idx'], p['lumens'], p['ray_uniformity'], p['settings'],
        p['wall_specs'], absorbers=p['absorbers'], accel=accel,
    )


def _mesh_absorbed_room_grids(active, settings, wall_specs, emission, accel, verbose):
    """Lux the STL mesh removes from the (mesh-free) GPU grids, per wall cell."""
    absorbed = empty_wall_grids(wall_specs)
    cell_areas = wall_cell_areas_m2(wall_specs)
    n_rays = settings.rays_per_led
    total_hits = 0
    for i, (led, led_idx) in enumerate(active):
        t_led = time.perf_counter()
        dirs, lumens_per_ray = generate_led_rays(
            led, n_rays, led_lumens(led, emission.default_lumens),
            emission.ray_uniformity, led_rng(led_idx),
        )
        origins = np.tile(led.position, (n_rays, 1)).astype(np.float64)
        hits = batch_ray_mesh_intersection(origins, dirs, accel, verbose=verbose)
        n_hits = int(np.sum(hits))
        total_hits += n_hits
        if n_hits:
            wall_id, c1, c2, _ = nearest_wall_hits(
                origins[hits], dirs[hits], settings.front_dist, settings.side_dist,
                settings.top_bottom_dist, settings.back_dist,
            )
            flux = lumens_per_ray[hits]
            for wid in np.unique(wall_id[wall_id >= 0]):
                name = _WALL_BY_ID[int(wid)]
                sel = wall_id == wid
                _deposit(absorbed, cell_areas, name, wall_specs[name], c1[sel], c2[sel], flux[sel])
        dt = time.perf_counter() - t_led
        if verbose and dt > 0.3:
            print(f"    LED {i+1}/{len(active)}: {n_rays:,} rays, {n_hits:,} hits, {dt:.2f}s")
    return absorbed, len(active) * n_rays, total_hits


def compute_room_intensity(leds, settings: RoomSettings, emission: EmissionSettings | None = None,
                           absorbers=None, stl_mesh_data=None, use_gpu=None, verbose=True):
    """Per-wall lux grids for the room.

    Returns ``(grids, wall_specs)`` where ``grids`` maps wall name → 2-D lux
    array (see ``room_geometry`` for index conventions).
    """
    emission = emission or EmissionSettings()
    absorbers = list(absorbers or [])
    wall_specs = build_wall_specs(
        settings.front_dist, settings.side_dist, settings.top_bottom_dist,
        int(settings.grid_size), settings.led_x_center, settings.back_dist,
    )
    grids = empty_wall_grids(wall_specs)
    ray_hits = {name: 0 for name in wall_specs}
    total_rays = 0

    active = active_leds_with_index(leds)
    if not active:
        return grids, wall_specs

    if verbose:
        print(f"\n=== ROOM MODE === front x={settings.front_dist}, sides y=±{settings.side_dist}, "
              f"top/bottom z=±{settings.top_bottom_dist}"
              + (f", back x=-{settings.back_dist}" if settings.back_dist is not None else "")
              + f" | grid {settings.grid_size}²/wall, {len(active)} LEDs, {settings.rays_per_led} rays/LED")
        if settings.max_bounces > 0 and settings.wall_reflectance > 0:
            print(f"Reflections: ON (ρ={settings.wall_reflectance:.2f}, max {settings.max_bounces} bounces)")

    if use_gpu is None:
        use_gpu = gpu_backend.gpu_available()

    t0 = time.perf_counter()
    if use_gpu:
        leds_data, per_led_lumens = [], []
        for led, _ in active:
            leds_data.append({
                'position': np.array(led.position, dtype=np.float32),
                'direction': np.array(led.direction, dtype=np.float32),
                'viewing_angle': float(led.viewing_angle),
                'ext_lens_angle': getattr(led, 'ext_lens_angle', None),
            })
            per_led_lumens.append(led_lumens(led, emission.default_lumens))
        grid_shapes = {name: g.shape for name, g in grids.items()}
        gpu_params = {
            'front_dist': settings.front_dist,
            'side_dist': settings.side_dist,
            'top_bottom_dist': settings.top_bottom_dist,
            'back_dist': settings.back_dist,
            'led_x_center': settings.led_x_center,
            'num_rays_per_led': settings.rays_per_pixel,
            'grid_size': int(settings.grid_size),
            'lumens_per_led': emission.default_lumens,
            'per_led_lumens': np.array(per_led_lumens, dtype=np.float32),
            'absorbers': absorbers,
            'stl_mesh_data': None,
            'ray_uniformity': emission.ray_uniformity,
            'grid_shapes': grid_shapes,
            'wall_specs': wall_specs,
            'max_bounces': settings.max_bounces,
            'wall_reflectance': settings.wall_reflectance,
        }
        if verbose:
            print(f"[GPU] {gpu_backend.gpu_backend_label()} acceleration for room mode ({len(active)} LEDs)...")
        gpu_grids, gpu_hits, total_rays = gpu_backend.gpu_process_room_batch(leds_data, gpu_params)
        for name in grids:
            grids[name] = gpu_grids[name]
            ray_hits[name] = gpu_hits.get(name, 0)
        if stl_mesh_data is not None:
            t_gpu = time.perf_counter()
            accel = prepare_mesh_ray_accelerator(stl_mesh_data)
            absorbed, n_cand, n_hits = _mesh_absorbed_room_grids(
                active, settings, wall_specs, emission, accel, verbose)
            for name in grids:
                grids[name] = np.maximum(0, grids[name] - absorbed[name])
            if verbose:
                print(f"  GPU grids {t_gpu - t0:.2f}s, mesh subtraction "
                      f"{time.perf_counter() - t_gpu:.2f}s ({n_hits:,}/{n_cand:,} rays)")
    else:
        common = {
            'ray_uniformity': emission.ray_uniformity,
            'settings': settings,
            'wall_specs': wall_specs,
            'absorbers': absorbers,
            'stl_mesh_data': None,
        }
        if stl_mesh_data is not None:
            accel = prepare_mesh_ray_accelerator(stl_mesh_data)
            if verbose:
                print(f"[CPU] STL mesh active → sequential LEDs, threaded BVH")
            results = [
                trace_led_in_room(led, led_idx, led_lumens(led, emission.default_lumens),
                                  emission.ray_uniformity, settings, wall_specs,
                                  absorbers=absorbers, accel=accel)
                for led, led_idx in active
            ]
        else:
            worker_args = [
                (copy.copy(led), dict(common, led_idx=led_idx,
                                      lumens=led_lumens(led, emission.default_lumens)))
                for led, led_idx in active
            ]
            num_processes = min(multiprocessing.cpu_count(), len(active))
            if verbose:
                print(f"[CPU] {num_processes} processes for {len(active)} LEDs...")
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_room_worker, worker_args)
        for led_grids, led_hits, led_rays in results:
            for name in grids:
                grids[name] += led_grids[name]
                ray_hits[name] += led_hits[name]
            total_rays += led_rays

    if verbose:
        cell_areas = wall_cell_areas_m2(wall_specs)
        total_emitted = sum(led_lumens(led, emission.default_lumens) for led, _ in active)
        total_on_walls = sum(float(g.sum()) * cell_areas[name] for name, g in grids.items())
        pct = (total_on_walls / total_emitted * 100) if total_emitted > 0 else 0
        print(f"Done in {time.perf_counter() - t0:.2f}s | rays {total_rays} | emitted {total_emitted:.1f} lm, "
              f"on walls {total_on_walls:.1f} lm ({pct:.1f}%)")
        for name, grid in grids.items():
            hits = ray_hits[name]
            hp = (hits / total_rays * 100) if total_rays > 0 else 0
            print(f"  {name.capitalize()}: {grid.sum() * cell_areas[name]:.1f} lm ({hits} rays, {hp:.1f}%)")
    return grids, wall_specs
