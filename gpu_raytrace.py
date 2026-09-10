"""Compatibility shim: the GPU ray tracer now lives in
``lighting_simulator.raytracing.gpu``. Kept so the root-level benchmark and
test scripts keep working unchanged.
"""

from lighting_simulator.raytracing import gpu as _gpu
from lighting_simulator.raytracing.gpu import *  # noqa: F401,F403
from lighting_simulator.raytracing.gpu import (  # noqa: F401
    _ensure_gpu_init,
    _calculate_lambertian_exponent,
    _lens_efficiency,
)


def __getattr__(name):
    # GPU_AVAILABLE / GPU_BACKEND are mutated lazily inside the real module,
    # so resolve them dynamically instead of copying stale values.
    return getattr(_gpu, name)
