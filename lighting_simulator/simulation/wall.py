"""Single-wall illuminance simulation.

Traces Monte Carlo rays from every enabled LED onto the plane ``x = wall_dist``
and accumulates illuminance (lux) on a square grid. Dispatches to the GPU
backend when available, otherwise to CPU workers (multiprocessing when no STL
mesh is involved, sequential + threaded BVH queries otherwise).
"""

import multiprocessing
import time

import numpy as np

from lighting_simulator.raytracing.boxes import ray_box_intersection_batch
from lighting_simulator.raytracing.mesh import (
    batch_ray_mesh_intersection,
    prepare_mesh_ray_accelerator,
)

from . import gpu_backend
from .emission import emission_cone_deg, emission_weight_lut, generate_led_rays, led_lumens, led_rng
from .settings import EmissionSettings, WallSettings


def active_leds_with_index(leds):
    """[(led, led_index)] for enabled LEDs; index defaults to list position."""
    out = []
    for pos, led in enumerate(leds):
        if hasattr(led, 'enabled') and not led.enabled:
            continue
        out.append((led, int(getattr(led, 'led_index', pos))))
    return out


def rays_per_led_for_budget(grid_size, rays_per_pixel, num_active_leds):
    total_pixels = int(grid_size) ** 2
    return max(1, int((total_pixels * rays_per_pixel) / max(1, num_active_leds)))


def scatter_wall_hits(grid, origin, dirs, lumens_per_ray, wall_dist, wall_size):
    """Accumulate lux of rays (from ``origin``) hitting the wall plane into ``grid``."""
    grid_size = grid.shape[0]
    cell_size = wall_size / grid_size
    cell_area_m2 = (cell_size * cell_size) / 10000.0
    half_size = wall_size / 2

    towards = np.where(dirs[:, 0] > 0)[0]
    if len(towards) == 0:
        return
    t = (wall_dist - origin[0]) / dirs[towards, 0]
    pos_t = t > 0
    vi = towards[pos_t]
    t = t[pos_t]
    hit_y = origin[1] + dirs[vi, 1] * t
    hit_z = origin[2] + dirs[vi, 2] * t
    gy = ((hit_y + half_size) / cell_size).astype(int)
    gz = ((hit_z + half_size) / cell_size).astype(int)
    in_bounds = (gy >= 0) & (gy < grid_size) & (gz >= 0) & (gz < grid_size)
    np.add.at(grid, (gz[in_bounds], gy[in_bounds]), lumens_per_ray[vi[in_bounds]] / cell_area_m2)


def trace_led_to_wall(led, led_idx, rays_per_led, lumens, ray_uniformity,
                      wall_dist, grid_size, wall_size, absorbers=(), accel=None):
    """Lux grid contributed by one LED (pure function; safe for threads/processes)."""
    grid = np.zeros((grid_size, grid_size))
    rng = led_rng(led_idx)
    dirs, lumens_per_ray = generate_led_rays(led, rays_per_led, lumens, ray_uniformity, rng)

    absorbed = np.zeros(rays_per_led, dtype=bool)
    if absorbers:
        origins = np.broadcast_to(led.position, (rays_per_led, 3)).astype(np.float32)
        absorbed = ray_box_intersection_batch(origins, dirs.astype(np.float32), list(absorbers))
    if accel is not None:
        alive = np.where(~absorbed)[0]
        if len(alive) > 0:
            origins = np.tile(led.position, (len(alive), 1)).astype(np.float64)
            absorbed[alive[batch_ray_mesh_intersection(origins, dirs[alive], accel)]] = True

    alive = ~absorbed
    scatter_wall_hits(grid, led.position, dirs[alive], lumens_per_ray[alive], wall_dist, wall_size)
    return grid


def _wall_worker(args):
    """Multiprocessing entry point: ``args = (led, params)``."""
    led, p = args
    accel = prepare_mesh_ray_accelerator(p.get('stl_mesh_data'))
    return trace_led_to_wall(
        led, p['led_idx'], p['rays_per_led'], p['lumens'], p['ray_uniformity'],
        p['wall_dist'], p['grid_size'], p['wall_size'],
        absorbers=p['absorbers'], accel=accel,
    )


def _gpu_led_payload(active, default_lumens, ray_uniformity=0.0):
    leds_data, lumens = [], []
    for led, led_idx in active:
        leds_data.append({
            'position': np.array(led.position, dtype=np.float32),
            'direction': np.array(led.direction, dtype=np.float32),
            'viewing_angle': emission_cone_deg(led),
            'weight_lut': emission_weight_lut(led, ray_uniformity),
            'led_idx': int(led_idx),
        })
        lumens.append(led_lumens(led, default_lumens))
    return leds_data, np.array(lumens, dtype=np.float32)


def _mesh_absorbed_wall_grid(active, settings, rays_per_led, emission, absorbers, accel, verbose):
    """Lux that the STL mesh removes from the (mesh-free) GPU grid, per cell."""
    absorbed_grid = np.zeros((settings.grid_size, settings.grid_size))
    total_candidates = total_hits = 0
    for i, (led, led_idx) in enumerate(active):
        t_led = time.perf_counter()
        dirs, lumens_per_ray = generate_led_rays(
            led, rays_per_led, led_lumens(led, emission.default_lumens),
            emission.ray_uniformity, led_rng(led_idx),
        )
        candidate = dirs[:, 0] > 0
        if absorbers:
            idx = np.where(candidate)[0]
            if len(idx) > 0:
                origins = np.broadcast_to(led.position, (len(idx), 3)).astype(np.float32)
                blocked = ray_box_intersection_batch(origins, dirs[idx].astype(np.float32), absorbers)
                candidate[idx[blocked]] = False
        idx = np.where(candidate)[0]
        if len(idx) == 0:
            continue
        total_candidates += len(idx)
        origins = np.tile(led.position, (len(idx), 1)).astype(np.float64)
        hits = batch_ray_mesh_intersection(origins, dirs[idx], accel, verbose=verbose)
        n_hits = int(np.sum(hits))
        total_hits += n_hits
        if n_hits:
            scatter_wall_hits(absorbed_grid, led.position, dirs[idx][hits],
                              lumens_per_ray[idx][hits], settings.wall_dist, settings.wall_size)
        dt = time.perf_counter() - t_led
        if verbose and dt > 0.3:
            print(f"    LED {i+1}/{len(active)}: {len(idx):,} rays, {n_hits:,} hits, {dt:.2f}s")
    return absorbed_grid, total_candidates, total_hits


def compute_wall_intensity(leds, settings: WallSettings, emission: EmissionSettings | None = None,
                           absorbers=None, stl_mesh_data=None, use_gpu=None, verbose=True,
                           parallel=True):
    """Illuminance grid (lux, shape ``grid_size × grid_size``) on the wall.

    Rows index Z (bottom → top), columns index Y. ``use_gpu=None`` auto-detects.
    ``parallel=False`` traces LEDs sequentially in-process (no worker pool), which
    is faster for the small, repeated evaluations of an optimisation loop.
    """
    emission = emission or EmissionSettings()
    absorbers = list(absorbers or [])
    grid_size = int(settings.grid_size)
    wall_size = float(settings.wall_size)
    wall_dist = float(settings.wall_dist)
    grid = np.zeros((grid_size, grid_size))

    active = active_leds_with_index(leds)
    if not active:
        return grid
    rays_per_led = rays_per_led_for_budget(grid_size, settings.rays_per_pixel, len(active))

    if verbose:
        print(f"\n=== WALL MODE === wall x={wall_dist:.1f} cm, grid {grid_size}², "
              f"{len(active)} active LEDs, {rays_per_led} rays/LED, "
              f"CPU cores: {multiprocessing.cpu_count()}")

    if use_gpu is None:
        use_gpu = gpu_backend.gpu_available()

    if use_gpu:
        leds_data, per_led_lumens = _gpu_led_payload(active, emission.default_lumens, emission.ray_uniformity)
        gpu_params = {
            'wall_dist': wall_dist,
            'rays_per_led': rays_per_led,
            'grid_size': grid_size,
            'wall_size': wall_size,
            'lumens_per_led': emission.default_lumens,
            'per_led_lumens': per_led_lumens,
            'absorbers': absorbers,
            'stl_mesh_data': None,  # GPU always traces without mesh
            'ray_uniformity': emission.ray_uniformity,
            'verbose': verbose,
        }
        if verbose:
            print(f"[GPU] {gpu_backend.gpu_backend_label()} acceleration for {len(active)} LEDs...")
        t0 = time.perf_counter()
        grid_full = gpu_backend.gpu_process_led_wall_batch(leds_data, gpu_params)
        if stl_mesh_data is None:
            if verbose:
                print(f"GPU ray tracing complete in {time.perf_counter() - t0:.2f}s\n")
            return grid_full
        # Hybrid: subtract lux carried by rays the mesh would have blocked.
        t_gpu = time.perf_counter()
        accel = prepare_mesh_ray_accelerator(stl_mesh_data)
        absorbed_grid, n_cand, n_hits = _mesh_absorbed_wall_grid(
            active, settings, rays_per_led, emission, absorbers, accel, verbose)
        grid = np.maximum(0, grid_full - absorbed_grid)
        if verbose:
            pct = np.sum(absorbed_grid) / max(np.sum(grid_full), 1e-10) * 100
            print(f"  GPU grid {t_gpu - t0:.2f}s, mesh subtraction {time.perf_counter() - t_gpu:.2f}s "
                  f"({n_hits:,}/{n_cand:,} rays, {pct:.1f}% absorbed)\n")
        return grid

    # CPU paths
    t0 = time.perf_counter()
    if stl_mesh_data is not None or not parallel:
        accel = prepare_mesh_ray_accelerator(stl_mesh_data)
        if verbose and accel is not None:
            print(f"[CPU] STL mesh active → sequential LEDs, threaded BVH (built in {time.perf_counter() - t0:.2f}s)")
        for led, led_idx in active:
            grid += trace_led_to_wall(
                led, led_idx, rays_per_led, led_lumens(led, emission.default_lumens),
                emission.ray_uniformity, wall_dist, grid_size, wall_size,
                absorbers=absorbers, accel=accel,
            )
    else:
        worker_args = [
            (led, {
                'led_idx': led_idx,
                'rays_per_led': rays_per_led,
                'lumens': led_lumens(led, emission.default_lumens),
                'ray_uniformity': emission.ray_uniformity,
                'wall_dist': wall_dist,
                'grid_size': grid_size,
                'wall_size': wall_size,
                'absorbers': absorbers,
                'stl_mesh_data': None,
            })
            for led, led_idx in active
        ]
        num_processes = min(multiprocessing.cpu_count(), len(active))
        if verbose:
            print(f"[CPU] {num_processes} processes for {len(active)} LEDs...")
        with multiprocessing.Pool(processes=num_processes) as pool:
            for local_grid in pool.map(_wall_worker, worker_args):
                grid += local_grid
    if verbose:
        print(f"Ray tracing complete in {time.perf_counter() - t0:.2f}s\n")
    return grid
