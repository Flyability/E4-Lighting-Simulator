"""Monte Carlo LED ray emission shared by all CPU tracing paths."""

import numpy as np

from lighting_simulator.domain.geometry import emission_frame
from lighting_simulator.domain.optics import effective_lambertian_exponent

RAY_SEED_BASE = 42


def led_rng(led_idx):
    """Deterministic, thread-safe RNG for one LED (reproducible Monte Carlo)."""
    return np.random.RandomState((RAY_SEED_BASE + int(led_idx)) % (2**32))


def led_lumens(led, default_lumens):
    """Per-LED flux, falling back to ``default_lumens`` when unset/zero."""
    return float(getattr(led, 'lumens', None) or default_lumens)


EMISSION_LUT_SIZE = 256


def emission_cone_deg(led):
    """Full cone angle (deg) inside which the LED emits: the profile's extent, else ``viewing_angle``."""
    profile = getattr(led, 'beam_profile', None)
    return 2.0 * profile.max_angle_deg if profile is not None else float(led.viewing_angle)


def emission_weight_lut(led, ray_uniformity=0.0, size=EMISSION_LUT_SIZE):
    """Per-ray weight w(θ) tabulated on θ ∈ [0, cone/2] for rays drawn uniformly in solid angle in the cone.

    ``lumens_per_ray = Φ / N · w(θ)`` with w = I(θ)/Φ · 2π(1 − cos θmax), so the weights average to 1.
    Shared by the CPU and GPU tracers; the GPU kernels index it with θ/θmax.
    """
    theta_max = np.radians(emission_cone_deg(led) / 2.0)
    theta = np.linspace(0.0, theta_max, int(size))
    cos_max = np.cos(theta_max)
    profile = getattr(led, 'beam_profile', None)
    if profile is not None:
        w = profile.intensity_cd(np.degrees(theta), 1.0) * 2.0 * np.pi * (1.0 - cos_max)
    else:
        n = effective_lambertian_exponent(led, ray_uniformity)
        denom = 1.0 - cos_max ** (n + 1.0)
        norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
        w = norm_factor * np.power(np.clip(np.cos(theta), 0.0, 1.0), n)
    return w.astype(np.float32)


def generate_led_rays(led, n_rays, lumens, ray_uniformity, rng):
    """Sample ``n_rays`` directions from the LED's emission cone.

    Directions are drawn uniformly (in solid angle) inside the cone of
    half-angle ``viewing_angle / 2``; each ray is weighted by the Lambertian
    intensity cosⁿ(θ) and normalised so the weights sum to ``lumens``.

    Returns (world_dirs (N, 3) float64, lumens_per_ray (N,)).
    """
    x_axis, y_axis, z_axis = emission_frame(led.direction)
    profile = getattr(led, 'beam_profile', None)
    if profile is not None:
        n, cos_max = 0.0, profile.cos_max
    else:
        n = effective_lambertian_exponent(led, ray_uniformity)
        cos_max = np.cos(np.radians(led.viewing_angle / 2.0))

    u = rng.uniform(0, 1, (n_rays, 2))
    cos_theta = np.clip(1.0 - u[:, 0] * (1.0 - cos_max), -1.0, 1.0)
    sin_theta = np.sin(np.arccos(cos_theta))
    phi = 2 * np.pi * u[:, 1]
    local_dirs = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

    world_dirs = (
        local_dirs[:, 0:1] * x_axis
        + local_dirs[:, 1:2] * y_axis
        + local_dirs[:, 2:3] * z_axis
    )
    world_dirs /= np.linalg.norm(world_dirs, axis=1, keepdims=True)

    if profile is not None:
        # uniform solid-angle sampling in the cone; weight = I(θ) / pdf, pdf = 1 / (2π (1 - cos_max))
        theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        lumens_per_ray = profile.intensity_cd(theta_deg, lumens) * (2.0 * np.pi * (1.0 - cos_max)) / n_rays
        return world_dirs, lumens_per_ray

    denom = 1.0 - cos_max ** (n + 1.0)
    norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
    lumens_per_ray = (lumens / n_rays) * np.power(np.clip(cos_theta, 0.0, 1.0), n) * norm_factor
    return world_dirs, lumens_per_ray
