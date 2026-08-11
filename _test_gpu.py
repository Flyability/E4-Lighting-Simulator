"""Quick test: verify CuPy uses the NVIDIA GPU (not Intel iGPU)."""
import os, time, warnings

print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import cupy as cp

print(f"CuPy version: {cp.__version__}")
n_dev = cp.cuda.runtime.getDeviceCount()
print(f"CUDA devices found: {n_dev}")
for i in range(n_dev):
    props = cp.cuda.runtime.getDeviceProperties(i)
    name = props["name"].decode()
    mem_gb = props["totalGlobalMem"] / (1024**3)
    print(f"  Device {i}: {name} ({mem_gb:.1f} GB)")

dev = cp.cuda.Device(0)
dev.use()
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"\nActive CUDA device: {props['name'].decode()}")

# Run sustained GPU workload so user can see Task Manager spike on NVIDIA
print("\nRunning GPU stress test for 5 sec — watch Task Manager for NVIDIA activity...")
a = cp.random.rand(8000, 8000, dtype=cp.float32)
t0 = time.perf_counter()
iters = 0
while time.perf_counter() - t0 < 5.0:
    b = cp.matmul(a, a)
    cp.cuda.Stream.null.synchronize()
    iters += 1
elapsed = time.perf_counter() - t0
print(f"Done: {iters} matmuls in {elapsed:.1f}s on {props['name'].decode()}")
