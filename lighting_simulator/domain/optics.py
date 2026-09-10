"""LED emission optics: Lambertian exponent, lens losses, diffuse sampling."""

import numpy as np


def lambertian_exponent(viewing_angle, ray_uniformity=0.0):
    """Exponent ``n`` of the I(θ) = I₀·cosⁿ(θ) model.

    ``viewing_angle`` is the full cone angle (deg) at which intensity is 50%.
    ``ray_uniformity`` in [0, 1] sharpens the beam (n scaled by 1 + 2·u).
    """
    theta_half = np.radians(viewing_angle / 2.0)
    cos_half = np.cos(theta_half)
    if cos_half > 0.01:
        n_base = np.log(0.5) / np.log(cos_half)
        n_base = np.clip(n_base, 0.1, 10.0)
    else:
        n_base = 1.0
    n = n_base * (1.0 + ray_uniformity * 2.0)
    return float(np.clip(n, 0.1, 30.0))


def effective_lambertian_exponent(led, ray_uniformity=0.0):
    """Lambertian exponent for an LED, honouring an external lens if attached."""
    ext_lens_angle = getattr(led, "ext_lens_angle", None)
    if ext_lens_angle is not None:
        return lambertian_exponent(ext_lens_angle, ray_uniformity)
    return lambertian_exponent(led.viewing_angle, ray_uniformity)


def lens_efficiency(viewing_angle):
    """Optical efficiency of a collimating lens producing ``viewing_angle``.

    A bare LED (~120° beam) needs no lens (eta = 1.0). Narrower beams
    incur Fresnel/absorption losses proportional to the fraction of
    Lambertian flux that must be redirected.
    Typical results: 120->1.00  90->0.93  60->0.87  30->0.82  10->0.80
    """
    max_loss = 0.20
    sin2_half = np.sin(np.radians(viewing_angle / 2.0)) ** 2
    sin2_ref = 0.75  # sin²(60°), Lambertian reference
    frac = max(0.0, 1.0 - sin2_half / sin2_ref)
    return 1.0 - max_loss * frac


def sample_cosine_hemisphere(normal, rng=None):
    """Cosine-weighted random direction in the hemisphere around ``normal``.

    Importance-samples a Lambertian BRDF (Malley's method).
    """
    rand = rng if rng is not None else np.random
    u1 = rand.random()
    u2 = rand.random()
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    x_local = r * np.cos(phi)
    y_local = r * np.sin(phi)
    z_local = np.sqrt(max(0.0, 1.0 - u1))
    if abs(normal[2]) < 0.999:
        tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    world_dir = x_local * tangent + y_local * bitangent + z_local * normal
    norm = np.linalg.norm(world_dir)
    if norm > 1e-10:
        world_dir /= norm
    return world_dir


# Backwards-compatible aliases (old private names used by benchmark scripts).
_calculate_lambertian_exponent = lambertian_exponent
_get_effective_n = effective_lambertian_exponent
_lens_efficiency = lens_efficiency
_sample_cosine_hemisphere = sample_cosine_hemisphere
