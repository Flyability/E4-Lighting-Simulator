"""Quick test of GPU ray tracing performance.  Usage: PYTHONPATH=. python scripts/test_gpu.py"""
import time, numpy as np
from lighting_simulator.raytracing import gpu as _gpu
from lighting_simulator.raytracing.gpu import gpu_process_led_wall_batch

_gpu._ensure_gpu_init()
print(f'GPU Available: {_gpu.GPU_AVAILABLE} ({_gpu.GPU_BACKEND})')

# Create test LEDs simulating typical scenario
leds_data = []
for i in range(24):  # 24 active LEDs
    angle = np.radians(12 + (i % 4) * 10)
    pos = np.array([35*np.cos(angle)-35, 35*np.sin(angle) + (i%2)*6.5, 0], dtype=np.float32)
    dir_vec = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float32)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    leds_data.append({
        'position': pos,
        'direction': dir_vec,
        'viewing_angle': 120.0,
        'led_idx': i,
    })

params = {
    'wall_dist': 100.0,
    'rays_per_led': 100000,
    'grid_size': 50,
    'wall_size': 80.0,
    'lumens_per_led': 300.0,
    'absorbers': [],
    'ray_uniformity': 0.0,
}

total = len(leds_data) * params['rays_per_led']
print(f'Testing with {len(leds_data)} LEDs x {params["rays_per_led"]:,} rays = {total:,} total rays')

t0 = time.perf_counter()
grid = gpu_process_led_wall_batch(leds_data, params)
t1 = time.perf_counter()

print(f'Result grid shape: {grid.shape}, sum: {grid.sum():.1f}')
print(f'Total time: {t1-t0:.3f}s')
print(f'Throughput: {total/(t1-t0)/1e6:.1f}M rays/s')
print('SUCCESS!')
