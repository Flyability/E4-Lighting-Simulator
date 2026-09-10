"""Optional GPU backend adapter (CUDA via CuPy, or Vulkan via Taichi).

The heavy module is imported lazily so multiprocessing workers never load GPU
drivers. All engine code should go through this adapter instead of importing
``lighting_simulator.raytracing.gpu`` directly.
"""

try:
    from lighting_simulator.raytracing import gpu as _gpu
    HAS_GPU_MODULE = True
except ImportError:  # pragma: no cover - only when optional deps are missing
    _gpu = None
    HAS_GPU_MODULE = False

# Relative flux mismatch (GPU vs CPU) above which the GPU backend is rejected.
_SELF_TEST_TOLERANCE = 0.10
_self_test_passed = None


def _self_test():
    """Trace one LED on GPU and CPU; reject the GPU if the results disagree.

    Some Taichi/Vulkan driver combinations have been observed to return stale
    or mismatched kernel outputs, so a cheap numerical check is done once
    before trusting the backend.
    """
    import numpy as np

    from lighting_simulator.domain.led import LED
    from .wall import trace_led_to_wall

    led = LED(position=(-0.2, 8.9, -0.6), direction=(0.84, 0.21, -0.5), viewing_angle=120.0, lumens=100.0)
    n_rays, grid_size, wall_size, wall_dist = 40000, 40, 80.0, 100.0
    cpu = trace_led_to_wall(led, 0, n_rays, 100.0, 0.0, wall_dist, grid_size, wall_size)
    leds_data = [{
        'position': np.asarray(led.position, dtype=np.float32),
        'direction': np.asarray(led.direction, dtype=np.float32),
        'viewing_angle': 120.0, 'ext_lens_angle': None, 'led_idx': 0,
    }]
    params = {
        'wall_dist': wall_dist, 'rays_per_led': n_rays, 'grid_size': grid_size, 'wall_size': wall_size,
        'lumens_per_led': 100.0, 'per_led_lumens': np.array([100.0], dtype=np.float32),
        'absorbers': [], 'stl_mesh_data': None, 'ray_uniformity': 0.0,
    }
    try:
        gpu = _gpu.gpu_process_led_wall_batch(leds_data, params)
    except Exception as exc:  # pragma: no cover - driver failures
        print(f"[GPU] Self-test raised {exc!r}; falling back to CPU.")
        return False
    cpu_sum, gpu_sum = float(cpu.sum()), float(np.asarray(gpu).sum())
    if cpu_sum <= 0 or not np.isfinite(gpu_sum):
        return False
    rel = abs(gpu_sum - cpu_sum) / cpu_sum
    if rel > _SELF_TEST_TOLERANCE:
        print(f"[GPU] Self-test FAILED: GPU/CPU wall flux mismatch {rel*100:.1f}% "
              f"(GPU {gpu_sum:.3g} vs CPU {cpu_sum:.3g}). Using CPU instead.")
        return False
    return True


def gpu_available():
    """Initialise the GPU backend on first call; True if a GPU backend is usable.

    The backend must also pass a one-time numerical self-test against the CPU
    tracer; otherwise it is disabled for the rest of the process.
    """
    global _self_test_passed
    if not HAS_GPU_MODULE:
        return False
    _gpu._ensure_gpu_init()
    if not _gpu.GPU_AVAILABLE:
        return False
    if _self_test_passed is None:
        _self_test_passed = _self_test()
        if not _self_test_passed:
            _gpu.GPU_AVAILABLE = False
            _gpu.GPU_BACKEND = 'cpu'
    return bool(_self_test_passed)


def gpu_backend():
    """'cuda' | 'taichi' | 'cpu' | None."""
    if not HAS_GPU_MODULE:
        return None
    _gpu._ensure_gpu_init()
    return _gpu.GPU_BACKEND


def gpu_backend_label():
    backend = gpu_backend()
    if backend == 'cuda':
        return "NVIDIA CUDA"
    if backend == 'taichi':
        return "Vulkan (Intel Arc/AMD/Apple)"
    return "CPU"


def gpu_process_led_wall_batch(leds_data, params):
    return _gpu.gpu_process_led_wall_batch(leds_data, params)


def gpu_process_room_batch(leds_data, params):
    return _gpu.gpu_process_room_batch(leds_data, params)
