"""Benchmark: GPU vs CPU ray tracing performance."""
import sys, time, numpy as np
sys.path.insert(0, '.')
from gpu_raytrace import gpu_process_led_wall_batch, GPU_AVAILABLE
import multiprocessing

# Import CPU worker
from interactive_lighting import _process_led_wall_worker, _calculate_lambertian_exponent, LED

print(f'GPU Available: {GPU_AVAILABLE}')
print(f'CPU cores: {multiprocessing.cpu_count()}')
print()

# Create test LEDs
num_test_leds = 24
rays_per = 50000

leds = []
leds_data = []
for i in range(num_test_leds):
    angle = np.radians(12 + (i % 4) * 10)
    pos = np.array([35*np.cos(angle)-35, 35*np.sin(angle) + (i%2)*6.5, 0], dtype=np.float64)
    dir_vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float64)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    led = LED(width=0.5, viewing_angle=120, position=tuple(pos), direction=tuple(dir_vec), color=(1,0,0))
    led.enabled = True
    led.led_index = i
    leds.append(led)
    
    leds_data.append({
        'position': pos.astype(np.float32),
        'direction': dir_vec.astype(np.float32),
        'viewing_angle': 120.0,
        'led_idx': i,
    })

total_rays = num_test_leds * rays_per
print(f'Test: {num_test_leds} LEDs x {rays_per:,} rays = {total_rays:,} total rays')
print()

# === CPU (multiprocessing) ===
print('--- CPU (multiprocessing) ---')
worker_args = []
for i, led in enumerate(leds):
    params = {
        'wall_dist': 100.0,
        'rays_per_led': rays_per,
        'grid_size': 50,
        'wall_size': 80.0,
        'lumens_per_led': 300.0,
        'absorbers': [],
        'stl_mesh_data': None,
        'ray_uniformity': 0.0,
        'led_idx': i,
    }
    worker_args.append((led, params))

t0 = time.perf_counter()
num_processes = min(multiprocessing.cpu_count(), num_test_leds)
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.map(_process_led_wall_worker, worker_args)
cpu_grid = np.zeros((50, 50))
for r in results:
    cpu_grid += r
t1 = time.perf_counter()
cpu_time = t1 - t0
print(f'CPU time: {cpu_time:.3f}s ({total_rays/cpu_time/1e6:.2f}M rays/s)')
print(f'CPU grid sum: {cpu_grid.sum():.1f}')
print()

# === GPU ===
print('--- GPU (CUDA) ---')
gpu_params = {
    'wall_dist': 100.0,
    'rays_per_led': rays_per,
    'grid_size': 50,
    'wall_size': 80.0,
    'lumens_per_led': 300.0,
    'absorbers': [],
    'ray_uniformity': 0.0,
}

# Warm-up
_ = gpu_process_led_wall_batch(leds_data, gpu_params)

t0 = time.perf_counter()
gpu_grid = gpu_process_led_wall_batch(leds_data, gpu_params)
t1 = time.perf_counter()
gpu_time = t1 - t0
print(f'GPU time: {gpu_time:.3f}s ({total_rays/gpu_time/1e6:.2f}M rays/s)')
print(f'GPU grid sum: {gpu_grid.sum():.1f}')
print()

# === Speedup ===
speedup = cpu_time / gpu_time
print(f'=== SPEEDUP: {speedup:.1f}x faster with GPU ===')
