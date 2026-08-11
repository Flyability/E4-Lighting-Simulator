"""Verify which GPU is actually doing computation."""
import warnings, time
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import cupy as cp

print('Before computation:')
mem = cp.cuda.Device(0).mem_info
print(f'  GPU free: {mem[0]/1e9:.2f} GB / total: {mem[1]/1e9:.2f} GB')

n_rays = 500000
origins = cp.random.rand(n_rays, 3).astype(cp.float32)
dirs = cp.random.rand(n_rays, 3).astype(cp.float32)

print(f'\nComputing {n_rays:,} rays on GPU...')
t0 = time.perf_counter()
for _ in range(10):
    result = cp.sum(origins * dirs, axis=1)
    cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
print(f'Done in {t1-t0:.3f}s')

print(f'\nAfter computation:')
mem = cp.cuda.Device(0).mem_info
print(f'  GPU free: {mem[0]/1e9:.2f} GB / total: {mem[1]/1e9:.2f} GB')
print(f'  Used: {(mem[1]-mem[0])/1e6:.0f} MB')

props = cp.cuda.runtime.getDeviceProperties(0)
name = props['name'].decode()
print(f'\nConfirm device: {name}')
print(f'Compute capability: {cp.cuda.Device(0).compute_capability}')
