"""
GPU-accelerated ray tracing using CuPy (NVIDIA CUDA).
Provides drop-in replacements for the CPU multiprocessing workers
in interactive_lighting.py, processing ALL rays in parallel on GPU.

Falls back to CPU (NumPy) automatically if CuPy/CUDA is not available.

IMPORTANT: CuPy is lazily initialized on first use to avoid loading CUDA
in multiprocessing worker subprocesses (which causes memory errors and spam).
"""

import numpy as np
import time
import warnings

# --- Lazy GPU initialization ---
# We do NOT import CuPy at module level to prevent multiprocessing workers
# (which import this module) from each loading CUDA DLLs (causes MemoryError).
GPU_AVAILABLE = None  # None = not yet checked, True/False after _ensure_gpu_init()
_cp = None  # Will hold cupy module after lazy init


def _preload_pip_cuda_libs():
    """Preload CUDA libraries shipped as pip wheels (nvidia-*-cu12 packages).

    The cupy-cuda12x wheel does not bundle NVRTC; CuPy dlopen()s it by bare
    soname (e.g. libnvrtc.so.12), which fails unless a system CUDA toolkit is
    installed. Loading the wheel's copies with RTLD_GLOBAL first makes those
    sonames resolvable without a system-wide CUDA install.
    """
    import ctypes
    import glob
    import os

    try:
        import nvidia
    except ImportError:
        return

    for base in nvidia.__path__:
        for lib in sorted(glob.glob(os.path.join(base, "*", "lib", "lib*.so*"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        # CuPy also needs a CUDA root for toolkit headers (cuda_fp16.h etc.).
        # The nvidia-cuda-runtime-cu12 wheel ships them under include/.
        runtime_root = os.path.join(base, "cuda_runtime")
        if "CUDA_PATH" not in os.environ and os.path.isdir(
                os.path.join(runtime_root, "include")):
            os.environ["CUDA_PATH"] = runtime_root


def _ensure_gpu_init():
    """Lazily initialize CuPy/CUDA on first use. Thread-safe via GIL."""
    global GPU_AVAILABLE, _cp
    if GPU_AVAILABLE is not None:
        return  # Already initialized
    
    try:
        _preload_pip_cuda_libs()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA path could not be detected")
            import cupy as cp_module
        # Smoke test that forces NVRTC kernel compilation, so a missing
        # CUDA toolkit is detected here instead of mid-raytrace.
        _test = cp_module.arange(3, dtype=cp_module.float64)
        _ = float(cp_module.sum(_test))
        del _test
        _cp = cp_module
        GPU_AVAILABLE = True
        sms = cp_module.cuda.Device(0).attributes.get('MultiProcessorCount', '?')
        gpu_name = cp_module.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"[GPU] CUDA acceleration enabled - CuPy {cp_module.__version__}, "
              f"GPU: {gpu_name}, SMs: {sms}")
    except Exception as e:
        GPU_AVAILABLE = False
        _cp = None
        print(f"[GPU] CUDA not available ({e}), using CPU fallback")


def _xp():
    """Return cupy if GPU is available, else numpy."""
    _ensure_gpu_init()
    return _cp if GPU_AVAILABLE else np


def _calculate_lambertian_exponent(viewing_angle, ray_uniformity):
    """Calculate Lambertian exponent for LED (CPU, used for setup)."""
    theta_half = np.radians(viewing_angle / 2.0)
    cos_half = np.cos(theta_half)
    if cos_half > 0.01:
        n_base = np.log(0.5) / np.log(cos_half)
        n_base = np.clip(n_base, 0.1, 10.0)
    else:
        n_base = 1.0
    n = n_base * (1.0 + ray_uniformity * 2.0)
    n = np.clip(n, 0.1, 30.0)
    return float(n)


def _lens_efficiency(viewing_angle):
    """Optical efficiency of an LED collimating lens."""
    MAX_LOSS = 0.20
    sin2_half = np.sin(np.radians(viewing_angle / 2.0)) ** 2
    SIN2_REF = 0.75
    frac = max(0.0, 1.0 - sin2_half / SIN2_REF)
    return 1.0 - MAX_LOSS * frac


def _gpu_ray_box_intersection_batch(origins, directions, absorbers):
    """
    Vectorized ray-box intersection for ALL rays against ALL absorbers.
    
    Args:
        origins: (N, 3) array of ray origins (on GPU if available)
        directions: (N, 3) array of ray directions
        absorbers: list of dicts with 'center', 'half_sizes', optional 'rotation'
    
    Returns:
        (N,) boolean mask - True where ray is absorbed
    """
    xp = _xp()
    N = origins.shape[0]
    absorbed = xp.zeros(N, dtype=bool)

    for a in absorbers:
        center = xp.array(a['center'], dtype=xp.float32)
        half = xp.array(a['half_sizes'], dtype=xp.float32)
        rotation = a.get('rotation', None)

        if rotation is not None:
            qw, qx, qy, qz = rotation
            R = xp.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ], dtype=xp.float32)
            R_inv = R.T
            # Transform rays to local space: (N,3) @ (3,3).T
            local_origins = (origins - center[None, :]) @ R_inv.T
            local_dirs = directions @ R_inv.T
            local_center = xp.zeros(3, dtype=xp.float32)
        else:
            local_origins = origins - center[None, :]
            local_dirs = directions
            local_center = xp.zeros(3, dtype=xp.float32)

        # Vectorized slab intersection test for axis-aligned box
        tmin = xp.full(N, -1e30, dtype=xp.float32)
        tmax = xp.full(N, 1e30, dtype=xp.float32)
        valid = xp.ones(N, dtype=bool)

        for k in range(3):
            d_k = local_dirs[:, k]
            o_k = local_origins[:, k]
            lo = local_center[k] - half[k]
            hi = local_center[k] + half[k]

            # Rays parallel to slab
            parallel = xp.abs(d_k) < 1e-12
            outside = parallel & ((o_k < lo) | (o_k > hi))
            valid &= ~outside

            # Non-parallel rays
            inv_d = xp.where(parallel, xp.float32(1.0), xp.float32(1.0) / xp.where(parallel, xp.float32(1.0), d_k))
            t1 = (lo - o_k) * inv_d
            t2 = (hi - o_k) * inv_d
            t_near = xp.minimum(t1, t2)
            t_far = xp.maximum(t1, t2)

            # Only update for non-parallel rays
            mask_np = ~parallel
            tmin = xp.where(mask_np & (t_near > tmin), t_near, tmin)
            tmax = xp.where(mask_np & (t_far < tmax), t_far, tmax)

        # Box hit: valid, tmin <= tmax, tmax > 0
        box_hit = valid & (tmin <= tmax) & (tmax > 0)
        absorbed |= box_hit

    return absorbed


def _stl_check_absorption(stl_mesh_data, ray_positions, world_dirs, absorbed, xp):
    """
    Check STL mesh intersection for non-absorbed rays (front-face only).
    Runs on CPU via trimesh (transfers only non-absorbed rays from GPU).
    Only the outer surface (front face) absorbs — back/internal faces are ignored.
    Processes rays in chunks to avoid OOM with large batches.
    Modifies `absorbed` in-place.
    """
    import trimesh
    
    not_absorbed_mask = ~absorbed
    n_to_check = int(xp.sum(not_absorbed_mask))
    if n_to_check == 0:
        return
    
    # Get indices of non-absorbed rays
    indices = xp.where(not_absorbed_mask)[0]
    
    # Transfer to CPU for trimesh
    try:
        if GPU_AVAILABLE:
            origins_cpu = _cp.asnumpy(ray_positions[indices]).astype(np.float64)
            dirs_cpu = _cp.asnumpy(world_dirs[indices]).astype(np.float64)
        else:
            origins_cpu = np.asarray(ray_positions[indices], dtype=np.float64)
            dirs_cpu = np.asarray(world_dirs[indices], dtype=np.float64)
    except Exception as e:
        print(f"[STL] Error transferring rays to CPU: {e}")
        return
    
    # Build trimesh if needed
    if 'trimesh_obj' not in stl_mesh_data:
        stl_mesh_data['trimesh_obj'] = trimesh.Trimesh(
            vertices=stl_mesh_data['vertices'],
            faces=stl_mesh_data['faces'],
            process=False
        )
    
    mesh = stl_mesh_data['trimesh_obj']
    transform = stl_mesh_data.get('transform', np.eye(4))
    
    try:
        inv_transform = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        inv_transform = np.eye(4)
    
    inv_rot = inv_transform[:3, :3]
    inv_trans = inv_transform[:3, 3]
    
    # Force BVH construction once
    _ = mesh.ray
    
    # Process in chunks to avoid OOM (trimesh allocates internal buffers)
    CHUNK = 500_000
    all_hits = np.zeros(n_to_check, dtype=bool)
    
    # Pre-compute mesh AABB for fast pre-filtering
    bbox = mesh.bounds  # (2, 3)
    pad = np.maximum((bbox[1] - bbox[0]) * 0.001, 1e-6)
    bmin = bbox[0] - pad
    bmax = bbox[1] + pad
    
    for start in range(0, n_to_check, CHUNK):
        end = min(start + CHUNK, n_to_check)
        
        # Transform rays to mesh local coordinates
        o_chunk = origins_cpu[start:end]
        d_chunk = dirs_cpu[start:end]
        
        origins_local = (inv_rot @ o_chunk.T).T + inv_trans
        dirs_local = (inv_rot @ d_chunk.T).T
        norms = np.linalg.norm(dirs_local, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        dirs_local = dirs_local / norms
        
        # AABB pre-filter: skip rays that can't hit the mesh bounding box
        eps = 1e-10
        safe_d = np.where(np.abs(dirs_local) < eps,
                          np.copysign(eps, dirs_local + 1e-30), dirs_local)
        inv_d = 1.0 / safe_d
        t1 = (bmin - origins_local) * inv_d
        t2 = (bmax - origins_local) * inv_d
        t_enter = np.max(np.minimum(t1, t2), axis=1)
        t_exit = np.min(np.maximum(t1, t2), axis=1)
        can_hit = (t_enter <= t_exit) & (t_exit > 0)
        cand_idx = np.where(can_hit)[0]
        
        if len(cand_idx) == 0:
            continue
        
        try:
            index_tri, index_ray = mesh.ray.intersects_id(
                ray_origins=origins_local[cand_idx],
                ray_directions=dirs_local[cand_idx],
                multiple_hits=False,
            )
            
            if len(index_ray) > 0:
                # Filter to front-face hits only (ray opposes face normal)
                face_normals = mesh.face_normals[index_tri]
                hit_ray_dirs = dirs_local[cand_idx[index_ray]]
                dot_products = np.sum(hit_ray_dirs * face_normals, axis=1)
                front_face_mask = dot_products < 0
                # Map back to chunk-level indices
                front_face_rays = cand_idx[index_ray[front_face_mask]]
                all_hits[start + front_face_rays] = True
        except Exception as e:
            print(f"[STL] Mesh intersection error on chunk {start}-{end}: {e}")
            continue
    
    # Mark absorbed rays
    if GPU_AVAILABLE:
        hits_gpu = _cp.array(all_hits)
        absorbed[indices[hits_gpu]] = True
    else:
        absorbed[indices[all_hits]] = True


def gpu_process_led_wall_batch(leds_data, params):
    """
    GPU-accelerated single-wall ray tracing for ALL LEDs at once.
    
    Args:
        leds_data: list of dicts with keys:
            'position': (3,) array, 'direction': (3,) array,
            'viewing_angle': float, 'led_idx': int
        params: dict with wall_dist, rays_per_led, grid_size, wall_size,
                lumens_per_led, absorbers, ray_uniformity
    
    Returns:
        grid: (grid_size, grid_size) numpy array with lux values
    """
    _ensure_gpu_init()
    xp = _xp()
    
    wall_dist = params['wall_dist']
    rays_per_led = params['rays_per_led']
    grid_size = params['grid_size']
    wall_size = params['wall_size']
    lumens_per_led = params['lumens_per_led']
    absorbers = params.get('absorbers', [])
    ray_uniformity = params.get('ray_uniformity', 0.0)
    stl_mesh_data = params.get('stl_mesh_data', None)

    cell_size = wall_size / grid_size
    cell_area_cm2 = cell_size * cell_size
    cell_area_m2 = cell_area_cm2 / 10000.0
    half_size = wall_size / 2

    num_leds = len(leds_data)
    total_rays = num_leds * rays_per_led

    t0 = time.perf_counter()

    # ---- Generate ALL ray origins and directions on GPU ----
    # Batch size management: process in chunks if too many rays for GPU memory
    MAX_BATCH = 20_000_000  # ~480 MB at float32
    grid_accum = xp.zeros((grid_size, grid_size), dtype=xp.float64)

    # Prepare per-LED data as GPU arrays ONCE (outside batch loop)
    positions_np = np.array([ld['position'] for ld in leds_data], dtype=np.float32)
    directions_np = np.array([ld['direction'] for ld in leds_data], dtype=np.float32)
    viewing_np = np.array([ld['viewing_angle'] for ld in leds_data], dtype=np.float32)
    positions = xp.asarray(positions_np)
    directions_led = xp.asarray(directions_np)
    viewing_angles = xp.asarray(viewing_np)

    rays_processed = 0
    for batch_start in range(0, total_rays, MAX_BATCH):
        batch_end = min(batch_start + MAX_BATCH, total_rays)
        batch_size = batch_end - batch_start

        # Determine which LED each ray belongs to
        ray_indices = xp.arange(batch_start, batch_end, dtype=xp.int64)
        led_indices = (ray_indices // rays_per_led).astype(xp.int32)

        # Gather per-ray LED data
        ray_positions = positions[led_indices]  # (batch, 3)
        ray_led_dirs = directions_led[led_indices]  # (batch, 3)
        ray_viewing_angles = viewing_angles[led_indices]  # (batch,)

        # Build per-ray local coordinate systems (vectorized)
        z_axes = ray_led_dirs  # (batch, 3)
        abs_z2 = xp.abs(z_axes[:, 2])

        up_z = xp.zeros((batch_size, 3), dtype=xp.float32)
        mask_z = abs_z2 < 0.9
        up_z[mask_z] = xp.array([0, 0, 1], dtype=xp.float32)
        up_z[~mask_z] = xp.array([0, 1, 0], dtype=xp.float32)

        x_axes = xp.cross(z_axes, up_z)
        x_norms = xp.linalg.norm(x_axes, axis=1, keepdims=True)
        x_norms = xp.maximum(x_norms, xp.float32(1e-10))
        x_axes = x_axes / x_norms
        y_axes = xp.cross(z_axes, x_axes)

        # ---- Sample random directions in viewing cone ----
        # Use reproducible-ish random with unique seed per batch
        batch_seed = (42 + batch_start) % (2**32)
        if GPU_AVAILABLE:
            rng = _cp.random.RandomState(seed=batch_seed)
            u1 = rng.uniform(0, 1, batch_size).astype(xp.float32)
            u2 = rng.uniform(0, 1, batch_size).astype(xp.float32)
        else:
            rng = np.random.RandomState(seed=batch_seed)
            u1 = rng.uniform(0, 1, batch_size).astype(np.float32)
            u2 = rng.uniform(0, 1, batch_size).astype(np.float32)

        _ext_la = leds_data[0].get('ext_lens_angle', None)
        n_val = _calculate_lambertian_exponent(float(_ext_la) if _ext_la is not None else float(leds_data[0]['viewing_angle']), ray_uniformity)
        max_theta = xp.radians(ray_viewing_angles / 2.0)
        cos_max = xp.cos(max_theta)
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        cos_theta = xp.clip(cos_theta, -1.0, 1.0)
        theta = xp.arccos(cos_theta)
        phi = 2.0 * xp.float32(np.pi) * u2

        sin_theta = xp.sin(theta)
        cos_phi = xp.cos(phi)
        sin_phi = xp.sin(phi)

        # Local directions -> world directions (vectorized)
        lx = sin_theta * cos_phi
        ly = sin_theta * sin_phi
        lz = cos_theta

        world_dirs = (lx[:, None] * x_axes +
                      ly[:, None] * y_axes +
                      lz[:, None] * z_axes)
        world_dir_norms = xp.linalg.norm(world_dirs, axis=1, keepdims=True)
        world_dir_norms = xp.maximum(world_dir_norms, xp.float32(1e-10))
        world_dirs = world_dirs / world_dir_norms

        # ---- Compute per-ray lumens with cone normalization ----
        cos_max_f = float(np.cos(np.radians(max(float(leds_data[0]['viewing_angle']), 120.0) / 2.0)))
        cos_max_n1 = cos_max_f ** (n_val + 1.0)
        denom = 1.0 - cos_max_n1
        norm_factor = (n_val + 1.0) * (1.0 - cos_max_f) / denom if denom > 1e-12 else 1.0
        cos_theta_clamped = xp.clip(cos_theta, 0.0, 1.0)
        intensity = xp.power(cos_theta_clamped, xp.float32(n_val))
        # Support per-LED lumens override
        per_led_lumens = params.get('per_led_lumens', None)
        if per_led_lumens is not None:
            per_led_lumens_gpu = xp.asarray(per_led_lumens, dtype=xp.float32)
            ray_lumens = per_led_lumens_gpu[led_indices]
            lumens_per_ray = (ray_lumens / xp.float32(rays_per_led)) * intensity * xp.float32(norm_factor)
        else:
            lumens_per_ray = (xp.float32(lumens_per_led) / xp.float32(rays_per_led)) * intensity * xp.float32(norm_factor)

        # ---- Check absorber intersections (vectorized) ----
        absorbed = xp.zeros(batch_size, dtype=bool)
        if absorbers:
            absorbed = _gpu_ray_box_intersection_batch(ray_positions, world_dirs, absorbers)

        # ---- STL mesh intersection (hybrid GPU/CPU) ----
        if stl_mesh_data is not None and not xp.all(absorbed):
            _stl_check_absorption(stl_mesh_data, ray_positions, world_dirs, absorbed, xp)

        # ---- Wall intersection (vectorized) ----
        dir_x = world_dirs[:, 0]
        pointing_forward = dir_x > 0
        not_absorbed = ~absorbed

        valid = pointing_forward & not_absorbed
        # Avoid division by zero
        safe_dir_x = xp.where(valid, dir_x, xp.float32(1.0))
        t = (xp.float32(wall_dist) - ray_positions[:, 0]) / safe_dir_x
        t_valid = valid & (t > 0)

        hit_y = ray_positions[:, 1] + world_dirs[:, 1] * t
        hit_z = ray_positions[:, 2] + world_dirs[:, 2] * t

        grid_y = ((hit_y + xp.float32(half_size)) / xp.float32(cell_size)).astype(xp.int32)
        grid_z = ((hit_z + xp.float32(half_size)) / xp.float32(cell_size)).astype(xp.int32)

        in_bounds = t_valid & (grid_y >= 0) & (grid_y < grid_size) & (grid_z >= 0) & (grid_z < grid_size)

        # ---- Accumulate lux on grid ----
        lux = lumens_per_ray / xp.float32(cell_area_m2)

        # Filter valid hits
        valid_gy = grid_y[in_bounds]
        valid_gz = grid_z[in_bounds]
        valid_lux = lux[in_bounds].astype(xp.float64)

        # Atomic scatter-add to grid
        # grid[gz, gy] += lux for each valid hit
        if GPU_AVAILABLE:
            flat_idx = valid_gz * grid_size + valid_gy
            _cp.add.at(grid_accum.ravel(), flat_idx, valid_lux)
        else:
            flat_idx = valid_gz * grid_size + valid_gy
            np.add.at(grid_accum.ravel(), flat_idx.astype(np.intp), valid_lux)

        rays_processed += batch_size

    # Synchronize GPU to ensure all work is complete before timing
    if GPU_AVAILABLE:
        _cp.cuda.Stream.null.synchronize()

    t1 = time.perf_counter()
    
    device = "GPU" if GPU_AVAILABLE else "CPU"
    print(f"[{device}] Wall ray tracing: {total_rays:,} rays in {t1-t0:.2f}s "
          f"({total_rays/(t1-t0)/1e6:.1f}M rays/s)")

    # Transfer back to CPU
    if GPU_AVAILABLE:
        result = _cp.asnumpy(grid_accum)
    else:
        result = grid_accum

    return result


def gpu_process_room_batch(leds_data, params):
    """
    GPU-accelerated room (multi-wall) ray tracing for ALL LEDs at once.
    
    Args:
        leds_data: list of dicts with keys:
            'position': (3,) array, 'direction': (3,) array,
            'viewing_angle': float
        params: dict with front_dist, side_dist, top_bottom_dist, back_dist,
                num_rays_per_led, grid_size, lumens_per_led, absorbers,
                ray_uniformity, grid_shapes, wall_specs
    
    Returns:
        grids: dict of wall_name -> (grid_shape) numpy arrays
        ray_hits: dict of wall_name -> int
        total_rays: int
    """
    _ensure_gpu_init()
    xp = _xp()

    front_dist = params['front_dist']
    side_dist = params['side_dist']
    top_bottom_dist = params['top_bottom_dist']
    back_dist = params.get('back_dist')
    num_rays_per_led = params['num_rays_per_led']
    grid_size = params['grid_size']
    lumens_per_led = params['lumens_per_led']
    absorbers = params.get('absorbers', [])
    ray_uniformity = params.get('ray_uniformity', 0.0)
    grid_shapes = params['grid_shapes']
    wall_specs = params['wall_specs']

    num_leds = len(leds_data)
    rays_per_led = num_rays_per_led * grid_size * grid_size
    total_rays = num_leds * rays_per_led

    # Pre-compute cell areas
    cell_areas_m2 = {}
    for wall_name, spec in wall_specs.items():
        if wall_name in ('front', 'back'):
            cw = spec['size_y'] / spec['grid_y']
            ch = spec['size_z'] / spec['grid_z']
        elif wall_name in ['left', 'right']:
            cw = spec['size_x'] / spec['grid_x']
            ch = spec['size_z'] / spec['grid_z']
        else:
            cw = spec['size_x'] / spec['grid_x']
            ch = spec['size_y'] / spec['grid_y']
        cell_areas_m2[wall_name] = (cw * ch) / 10000.0

    # Initialize grids on GPU
    grids = {}
    for wall_name, shape in grid_shapes.items():
        grids[wall_name] = xp.zeros(shape, dtype=xp.float64)
    ray_hits = {wn: 0 for wn in grid_shapes}

    t0 = time.perf_counter()

    MAX_BATCH = 10_000_000  # Process in chunks for memory

    # Prepare per-LED data as GPU arrays ONCE (outside batch loop)
    positions_np = np.array([ld['position'] for ld in leds_data], dtype=np.float32)
    dir_leds_np = np.array([ld['direction'] for ld in leds_data], dtype=np.float32)
    view_angles_np = np.array([ld['viewing_angle'] for ld in leds_data], dtype=np.float32)
    positions = xp.asarray(positions_np)
    dir_leds = xp.asarray(dir_leds_np)
    view_angles = xp.asarray(view_angles_np)

    for batch_start in range(0, total_rays, MAX_BATCH):
        batch_end = min(batch_start + MAX_BATCH, total_rays)
        batch_size = batch_end - batch_start

        ray_indices = xp.arange(batch_start, batch_end, dtype=xp.int32)
        led_indices = ray_indices // rays_per_led

        ray_pos = positions[led_indices]
        ray_led_dir = dir_leds[led_indices]
        ray_va = view_angles[led_indices]

        # Build local coordinate systems
        z_axes = ray_led_dir
        abs_z2 = xp.abs(z_axes[:, 2])
        up = xp.zeros((batch_size, 3), dtype=xp.float32)
        mz = abs_z2 < 0.9
        up[mz] = xp.array([0, 0, 1], dtype=xp.float32)
        up[~mz] = xp.array([0, 1, 0], dtype=xp.float32)

        x_axes = xp.cross(z_axes, up)
        xn = xp.linalg.norm(x_axes, axis=1, keepdims=True)
        xn = xp.maximum(xn, xp.float32(1e-10))
        x_axes = x_axes / xn
        y_axes = xp.cross(z_axes, x_axes)

        # Random sampling
        if GPU_AVAILABLE:
            rng = _cp.random.RandomState(seed=42 + batch_start)
            u1 = rng.uniform(0, 1, batch_size).astype(xp.float32)
            u2 = rng.uniform(0, 1, batch_size).astype(xp.float32)
        else:
            rng = np.random.RandomState(seed=42 + batch_start)
            u1 = rng.uniform(0, 1, batch_size).astype(np.float32)
            u2 = rng.uniform(0, 1, batch_size).astype(np.float32)

        _ext_la = leds_data[0].get('ext_lens_angle', None)
        n_val = _calculate_lambertian_exponent(float(_ext_la) if _ext_la is not None else float(leds_data[0]['viewing_angle']), ray_uniformity)
        max_theta = xp.radians(ray_va / 2.0)
        cos_max = xp.cos(max_theta)
        cos_theta = 1.0 - u1 * (1.0 - cos_max)
        cos_theta = xp.clip(cos_theta, -1.0, 1.0)
        theta = xp.arccos(cos_theta)
        phi = 2.0 * xp.float32(np.pi) * u2

        sin_theta = xp.sin(theta)
        world_dirs = (sin_theta * xp.cos(phi))[:, None] * x_axes + \
                     (sin_theta * xp.sin(phi))[:, None] * y_axes + \
                     cos_theta[:, None] * z_axes
        wdn = xp.linalg.norm(world_dirs, axis=1, keepdims=True)
        wdn = xp.maximum(wdn, xp.float32(1e-10))
        world_dirs = world_dirs / wdn

        cos_max_f = float(np.cos(np.radians(float(leds_data[0]['viewing_angle']) / 2.0)))
        cos_max_n1 = cos_max_f ** (n_val + 1.0)
        denom = 1.0 - cos_max_n1
        norm_factor = (n_val + 1.0) * (1.0 - cos_max_f) / denom if denom > 1e-12 else 1.0
        cos_tc = xp.clip(cos_theta, 0.0, 1.0)
        intensity = xp.power(cos_tc, xp.float32(n_val))
        # Support per-LED lumens override
        per_led_lumens = params.get('per_led_lumens', None)
        if per_led_lumens is not None:
            per_led_lumens_gpu = xp.asarray(per_led_lumens, dtype=xp.float32)
            ray_lumens = per_led_lumens_gpu[led_indices]
            lpr = (ray_lumens / xp.float32(rays_per_led)) * intensity * xp.float32(norm_factor)
        else:
            lpr = (xp.float32(lumens_per_led) / xp.float32(rays_per_led)) * intensity * xp.float32(norm_factor)

        # Absorber check
        absorbed = xp.zeros(batch_size, dtype=bool)
        if absorbers:
            absorbed = _gpu_ray_box_intersection_batch(ray_pos, world_dirs, absorbers)

        # STL mesh check (hybrid GPU/CPU)
        stl_mesh_data = params.get('stl_mesh_data', None)
        if stl_mesh_data is not None and not xp.all(absorbed):
            _stl_check_absorption(stl_mesh_data, ray_pos, world_dirs, absorbed, xp)

        not_absorbed = ~absorbed
        dx = world_dirs[:, 0]
        dy = world_dirs[:, 1]
        dz = world_dirs[:, 2]
        ox = ray_pos[:, 0]
        oy = ray_pos[:, 1]
        oz = ray_pos[:, 2]

        # ---- Find closest wall intersection for each ray ----
        INF = xp.float32(1e30)
        best_t = xp.full(batch_size, INF, dtype=xp.float32)
        best_wall = xp.full(batch_size, -1, dtype=xp.int32)
        # Wall coding: 0=front, 1=left, 2=right, 3=top, 4=bottom, 5=back
        best_c1 = xp.zeros(batch_size, dtype=xp.float32)
        best_c2 = xp.zeros(batch_size, dtype=xp.float32)

        def _check_wall(condition, t_val, wall_id, coord1, coord2):
            valid = not_absorbed & condition & (t_val > 0) & (t_val < best_t)
            best_t_out = xp.where(valid, t_val, best_t)
            best_wall_out = xp.where(valid, xp.int32(wall_id), best_wall)
            best_c1_out = xp.where(valid, coord1, best_c1)
            best_c2_out = xp.where(valid, coord2, best_c2)
            return best_t_out, best_wall_out, best_c1_out, best_c2_out

        # Front wall (x = front_dist)
        safe_dx = xp.where(dx > 0, dx, xp.float32(1.0))
        t_front = (xp.float32(front_dist) - ox) / safe_dx
        y_front = oy + dy * t_front
        z_front = oz + dz * t_front
        best_t, best_wall, best_c1, best_c2 = _check_wall(dx > 0, t_front, 0, y_front, z_front)

        # Left wall (y = -side_dist)
        safe_dy_neg = xp.where(dy < 0, dy, xp.float32(-1.0))
        t_left = (xp.float32(-side_dist) - oy) / safe_dy_neg
        x_left = ox + dx * t_left
        z_left = oz + dz * t_left
        best_t, best_wall, best_c1, best_c2 = _check_wall(dy < 0, t_left, 1, x_left, z_left)

        # Right wall (y = +side_dist)
        safe_dy_pos = xp.where(dy > 0, dy, xp.float32(1.0))
        t_right = (xp.float32(side_dist) - oy) / safe_dy_pos
        x_right = ox + dx * t_right
        z_right = oz + dz * t_right
        best_t, best_wall, best_c1, best_c2 = _check_wall(dy > 0, t_right, 2, x_right, z_right)

        # Top wall (z = +top_bottom_dist)
        safe_dz_pos = xp.where(dz > 0, dz, xp.float32(1.0))
        t_top = (xp.float32(top_bottom_dist) - oz) / safe_dz_pos
        x_top = ox + dx * t_top
        y_top = oy + dy * t_top
        best_t, best_wall, best_c1, best_c2 = _check_wall(dz > 0, t_top, 3, x_top, y_top)

        # Bottom wall (z = -top_bottom_dist)
        safe_dz_neg = xp.where(dz < 0, dz, xp.float32(-1.0))
        t_bottom = (xp.float32(-top_bottom_dist) - oz) / safe_dz_neg
        x_bottom = ox + dx * t_bottom
        y_bottom = oy + dy * t_bottom
        best_t, best_wall, best_c1, best_c2 = _check_wall(dz < 0, t_bottom, 4, x_bottom, y_bottom)

        # Back wall (x = -back_dist)
        if back_dist is not None:
            back_x = xp.float32(-back_dist)
            safe_dx_neg = xp.where(dx < 0, dx, xp.float32(-1.0))
            t_back = (back_x - ox) / safe_dx_neg
            y_back = oy + dy * t_back
            z_back = oz + dz * t_back
            best_t, best_wall, best_c1, best_c2 = _check_wall(dx < 0, t_back, 5, y_back, z_back)

        # ---- Scatter-add to per-wall grids ----
        has_hit = best_wall >= 0

        # Process each wall type
        wall_configs = [
            (0, 'front'), (1, 'left'), (2, 'right'),
            (3, 'top'), (4, 'bottom')
        ]
        if back_dist is not None:
            wall_configs.append((5, 'back'))

        for wall_id, wall_name in wall_configs:
            mask = has_hit & (best_wall == wall_id)
            if not xp.any(mask):
                continue

            c1 = best_c1[mask]
            c2 = best_c2[mask]
            ray_lux = lpr[mask] / xp.float32(cell_areas_m2[wall_name])

            spec = wall_specs[wall_name]
            shape = grid_shapes[wall_name]

            if wall_name in ('front', 'back'):
                sz_y = xp.float32(spec['size_y'])
                sz_z = xp.float32(spec['size_z'])
                gy = spec['grid_y']
                gz = spec['grid_z']
                gi = ((c1 + sz_y / 2) / (sz_y / gy)).astype(xp.int32)
                gj = ((c2 + sz_z / 2) / (sz_z / gz)).astype(xp.int32)
                # grid[z, y] -> grid[gj, gi]
                row = xp.clip(gj, 0, shape[0] - 1)
                col = xp.clip(gi, 0, shape[1] - 1)
            elif wall_name in ('left', 'right'):
                sz_x = xp.float32(spec['size_x'])
                sz_z = xp.float32(spec['size_z'])
                gx = spec['grid_x']
                gz = spec['grid_z']
                x_min = xp.float32(spec['x_min'])
                gi = ((c1 - x_min) / (sz_x / gx)).astype(xp.int32)
                gj = ((c2 + sz_z / 2) / (sz_z / gz)).astype(xp.int32)
                row = xp.clip(gj, 0, shape[0] - 1)
                col = xp.clip(gi, 0, shape[1] - 1)
            else:  # top, bottom
                sz_x = xp.float32(spec['size_x'])
                sz_y = xp.float32(spec['size_y'])
                gx = spec['grid_x']
                gy = spec['grid_y']
                x_min = xp.float32(spec['x_min'])
                gi = ((c1 - x_min) / (sz_x / gx)).astype(xp.int32)
                gj = ((c2 + sz_y / 2) / (sz_y / gy)).astype(xp.int32)
                row = xp.clip(gj, 0, shape[0] - 1)
                col = xp.clip(gi, 0, shape[1] - 1)

            flat_idx = row * shape[1] + col
            if GPU_AVAILABLE:
                _cp.add.at(grids[wall_name].ravel(), flat_idx, ray_lux.astype(xp.float64))
            else:
                np.add.at(grids[wall_name].ravel(), flat_idx.astype(np.intp), ray_lux.astype(np.float64))

            ray_hits[wall_name] += int(xp.sum(mask))

        # ---- REFLECTION BOUNCES ----
        max_bounces = params.get('max_bounces', 0)
        wall_reflectance = params.get('wall_reflectance', 0.0)
        
        if max_bounces > 0 and wall_reflectance > 0:
            # Pre-compute wall normals, tangents, bitangents (axis-aligned walls)
            # wall_id: 0=front, 1=left, 2=right, 3=top, 4=bottom, 5=back
            _bnormals = xp.array([[-1,0,0],[0,1,0],[0,-1,0],[0,0,-1],[0,0,1],[1,0,0]], dtype=xp.float32)
            _btangents = xp.array([[0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0]], dtype=xp.float32)
            _bbitangents = xp.array([[0,0,-1],[0,0,-1],[0,0,1],[0,-1,0],[0,1,0],[0,0,1]], dtype=xp.float32)
            
            bounce_active = has_hit.copy()
            # Hit points
            bounce_ox = ox + dx * best_t
            bounce_oy = oy + dy * best_t
            bounce_oz = oz + dz * best_t
            bounce_lpr = lpr * xp.float32(wall_reflectance)
            bounce_wall = best_wall.copy()
            
            for bounce_i in range(max_bounces):
                n_active = int(xp.sum(bounce_active))
                if n_active == 0:
                    break
                
                # Get wall normals/tangents/bitangents for each ray's hit wall
                wall_ids = xp.clip(bounce_wall, 0, 5)
                n_xyz = _bnormals[wall_ids]
                t_xyz = _btangents[wall_ids]
                b_xyz = _bbitangents[wall_ids]
                
                # Offset position along wall normal
                bounce_ox = bounce_ox + n_xyz[:, 0] * 0.01
                bounce_oy = bounce_oy + n_xyz[:, 1] * 0.01
                bounce_oz = bounce_oz + n_xyz[:, 2] * 0.01
                
                # Generate cosine-weighted reflected directions (Malley's method)
                if GPU_AVAILABLE:
                    rng_b = _cp.random.RandomState(seed=(42 + batch_start + (bounce_i + 1) * 31337) % (2**31))
                    bu1 = rng_b.uniform(0, 1, batch_size).astype(xp.float32)
                    bu2 = rng_b.uniform(0, 1, batch_size).astype(xp.float32)
                else:
                    rng_b = np.random.RandomState(seed=(42 + batch_start + (bounce_i + 1) * 31337) % (2**31))
                    bu1 = rng_b.uniform(0, 1, batch_size).astype(np.float32)
                    bu2 = rng_b.uniform(0, 1, batch_size).astype(np.float32)
                
                br = xp.sqrt(bu1)
                bphi = 2.0 * xp.float32(np.pi) * bu2
                bx_local = br * xp.cos(bphi)
                by_local = br * xp.sin(bphi)
                bz_local = xp.sqrt(xp.maximum(xp.float32(0.0), 1.0 - bu1))
                
                # Transform local to world using per-ray tangent frames
                bounce_dx = bx_local * t_xyz[:, 0] + by_local * b_xyz[:, 0] + bz_local * n_xyz[:, 0]
                bounce_dy = bx_local * t_xyz[:, 1] + by_local * b_xyz[:, 1] + bz_local * n_xyz[:, 1]
                bounce_dz = bx_local * t_xyz[:, 2] + by_local * b_xyz[:, 2] + bz_local * n_xyz[:, 2]
                
                bnorm = xp.sqrt(bounce_dx**2 + bounce_dy**2 + bounce_dz**2)
                bnorm = xp.maximum(bnorm, xp.float32(1e-10))
                bounce_dx /= bnorm
                bounce_dy /= bnorm
                bounce_dz /= bnorm
                
                # Find closest wall for bounced rays
                INF_b = xp.float32(1e30)
                best_t_b = xp.full(batch_size, INF_b, dtype=xp.float32)
                best_wall_b = xp.full(batch_size, -1, dtype=xp.int32)
                best_c1_b = xp.zeros(batch_size, dtype=xp.float32)
                best_c2_b = xp.zeros(batch_size, dtype=xp.float32)
                
                def _check_bounce_wall(cond, t_val, wid, c1, c2):
                    v = bounce_active & cond & (t_val > 0) & (t_val < best_t_b)
                    return (xp.where(v, t_val, best_t_b),
                            xp.where(v, xp.int32(wid), best_wall_b),
                            xp.where(v, c1, best_c1_b),
                            xp.where(v, c2, best_c2_b))
                
                # Front
                s_dx_p = xp.where(bounce_dx > 0, bounce_dx, xp.float32(1.0))
                t_bf = (xp.float32(front_dist) - bounce_ox) / s_dx_p
                best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                    bounce_dx > 0, t_bf, 0, bounce_oy + bounce_dy * t_bf, bounce_oz + bounce_dz * t_bf)
                # Left
                s_dy_n = xp.where(bounce_dy < 0, bounce_dy, xp.float32(-1.0))
                t_bl = (xp.float32(-side_dist) - bounce_oy) / s_dy_n
                best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                    bounce_dy < 0, t_bl, 1, bounce_ox + bounce_dx * t_bl, bounce_oz + bounce_dz * t_bl)
                # Right
                s_dy_p = xp.where(bounce_dy > 0, bounce_dy, xp.float32(1.0))
                t_br = (xp.float32(side_dist) - bounce_oy) / s_dy_p
                best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                    bounce_dy > 0, t_br, 2, bounce_ox + bounce_dx * t_br, bounce_oz + bounce_dz * t_br)
                # Top
                s_dz_p = xp.where(bounce_dz > 0, bounce_dz, xp.float32(1.0))
                t_bt = (xp.float32(top_bottom_dist) - bounce_oz) / s_dz_p
                best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                    bounce_dz > 0, t_bt, 3, bounce_ox + bounce_dx * t_bt, bounce_oy + bounce_dy * t_bt)
                # Bottom
                s_dz_n = xp.where(bounce_dz < 0, bounce_dz, xp.float32(-1.0))
                t_bb = (xp.float32(-top_bottom_dist) - bounce_oz) / s_dz_n
                best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                    bounce_dz < 0, t_bb, 4, bounce_ox + bounce_dx * t_bb, bounce_oy + bounce_dy * t_bb)
                # Back
                if back_dist is not None:
                    s_dx_n = xp.where(bounce_dx < 0, bounce_dx, xp.float32(-1.0))
                    t_bk = (xp.float32(-back_dist) - bounce_ox) / s_dx_n
                    best_t_b, best_wall_b, best_c1_b, best_c2_b = _check_bounce_wall(
                        bounce_dx < 0, t_bk, 5, bounce_oy + bounce_dy * t_bk, bounce_oz + bounce_dz * t_bk)
                
                # Deposit bounced lux on grids
                has_hit_b = best_wall_b >= 0
                for wall_id, wall_name in wall_configs:
                    mask_b = has_hit_b & (best_wall_b == wall_id) & bounce_active
                    if not xp.any(mask_b):
                        continue
                    c1_b = best_c1_b[mask_b]
                    c2_b = best_c2_b[mask_b]
                    ray_lux_b = bounce_lpr[mask_b] / xp.float32(cell_areas_m2[wall_name])
                    
                    spec = wall_specs[wall_name]
                    shape = grid_shapes[wall_name]
                    if wall_name in ('front', 'back'):
                        gi_b = ((c1_b + xp.float32(spec['size_y'])/2) / (xp.float32(spec['size_y']) / spec['grid_y'])).astype(xp.int32)
                        gj_b = ((c2_b + xp.float32(spec['size_z'])/2) / (xp.float32(spec['size_z']) / spec['grid_z'])).astype(xp.int32)
                    elif wall_name in ('left', 'right'):
                        gi_b = ((c1_b - xp.float32(spec['x_min'])) / (xp.float32(spec['size_x']) / spec['grid_x'])).astype(xp.int32)
                        gj_b = ((c2_b + xp.float32(spec['size_z'])/2) / (xp.float32(spec['size_z']) / spec['grid_z'])).astype(xp.int32)
                    else:
                        gi_b = ((c1_b - xp.float32(spec['x_min'])) / (xp.float32(spec['size_x']) / spec['grid_x'])).astype(xp.int32)
                        gj_b = ((c2_b + xp.float32(spec['size_y'])/2) / (xp.float32(spec['size_y']) / spec['grid_y'])).astype(xp.int32)
                    row_b = xp.clip(gj_b, 0, shape[0] - 1)
                    col_b = xp.clip(gi_b, 0, shape[1] - 1)
                    flat_b = row_b * shape[1] + col_b
                    if GPU_AVAILABLE:
                        _cp.add.at(grids[wall_name].ravel(), flat_b, ray_lux_b.astype(xp.float64))
                    else:
                        np.add.at(grids[wall_name].ravel(), flat_b.astype(np.intp), ray_lux_b.astype(np.float64))
                    ray_hits[wall_name] += int(xp.sum(mask_b))
                
                # Update for next bounce
                bounce_active = has_hit_b & bounce_active
                bounce_ox = bounce_ox + bounce_dx * best_t_b
                bounce_oy = bounce_oy + bounce_dy * best_t_b
                bounce_oz = bounce_oz + bounce_dz * best_t_b
                bounce_lpr = bounce_lpr * xp.float32(wall_reflectance)
                bounce_wall = best_wall_b
                # Kill negligible flux rays
                bounce_active = bounce_active & (bounce_lpr > 1e-8)

    # Synchronize GPU to ensure all work is complete before timing
    if GPU_AVAILABLE:
        _cp.cuda.Stream.null.synchronize()

    t1 = time.perf_counter()
    device = "GPU" if GPU_AVAILABLE else "CPU"
    total_counted = sum(ray_hits.values())
    print(f"[{device}] Room ray tracing: {total_rays:,} rays in {t1-t0:.2f}s "
          f"({total_rays/(t1-t0)/1e6:.1f}M rays/s), {total_counted:,} wall hits")

    # Transfer to CPU
    result_grids = {}
    for wn, g in grids.items():
        result_grids[wn] = _cp.asnumpy(g) if GPU_AVAILABLE else g.copy()

    return result_grids, ray_hits, total_rays
