"""Measured LED beam profiles: relative luminous intensity vs. angle off the optical axis.

A profile replaces the ``cosⁿ`` / hard-cone model of ``optics.lambertian_exponent`` with the
datasheet curve, so a flat-topped emitter such as the XFL12K HD (80 % at 60°, 18 % at 90°) is
simulated as it really radiates. Intensity is normalised so the curve integrates to the LED
flux:  I(θ) = Φ · p(θ) / K,  K = 2π ∫₀^θmax p(θ) sinθ dθ.

Profiles live in ``beam_profiles/<name>.json`` (``{"name", "angles_deg", "relative_intensity"}``,
angles ascending from 0) plus the built-ins below; ``LED.beam_profile`` holds the resolved object,
``LedSpec.profile`` its name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os

import numpy as np

LAMBERTIAN = "cos^n (beam angle)"
"""Pseudo-profile name meaning "use the cosⁿ / beam-angle model"."""

_BUILTIN = {
    # Digitised from the XFL12K HD datasheet polar diagram (relative intensity, % → 0..1).
    "XFL12K HD": ([0, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90],
                  [1.00, 0.99, 0.96, 0.93, 0.90, 0.86, 0.80, 0.76, 0.65, 0.55, 0.42, 0.30, 0.18]),
}


@dataclass
class BeamProfile:
    name: str
    angles_deg: np.ndarray
    rel_intensity: np.ndarray
    _norm: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.angles_deg = np.asarray(self.angles_deg, dtype=float)
        self.rel_intensity = np.asarray(self.rel_intensity, dtype=float)
        if self.angles_deg.ndim != 1 or self.angles_deg.shape != self.rel_intensity.shape or len(self.angles_deg) < 2:
            raise ValueError("BeamProfile needs matching 1-D angle / intensity arrays (≥ 2 points)")
        if np.any(np.diff(self.angles_deg) <= 0) or self.angles_deg[0] < 0 or self.angles_deg[-1] > 180:
            raise ValueError("BeamProfile angles must be ascending within [0, 180]")
        peak = float(self.rel_intensity.max())
        if peak <= 0:
            raise ValueError("BeamProfile intensity must be positive somewhere")
        self.rel_intensity = np.clip(self.rel_intensity / peak, 0.0, None)
        th = np.radians(np.linspace(0.0, self.angles_deg[-1], 4096))
        self._norm = float(2.0 * np.pi * np.trapz(self.relative(np.degrees(th)) * np.sin(th), th))

    @property
    def max_angle_deg(self) -> float:
        """Half-angle beyond which the profile emits nothing."""
        return float(self.angles_deg[-1])

    @property
    def cos_max(self) -> float:
        return float(np.cos(np.radians(self.max_angle_deg)))

    def relative(self, theta_deg):
        """p(θ) in [0, 1]; 0 beyond the last tabulated angle."""
        t = np.asarray(theta_deg, dtype=float)
        return np.where(t <= self.angles_deg[-1], np.interp(t, self.angles_deg, self.rel_intensity), 0.0)

    def intensity_cd(self, theta_deg, flux_lm):
        """Luminous intensity (cd) at θ for an emitter of total flux ``flux_lm``."""
        return float(flux_lm) * self.relative(theta_deg) / self._norm

    def half_intensity_angle_deg(self) -> float:
        """Full angle at which the intensity drops to 50 % (datasheet 'viewing angle')."""
        t = np.linspace(0.0, self.max_angle_deg, 2048)
        p = self.relative(t)
        idx = np.argmax(p < 0.5)
        return 2.0 * float(t[idx]) if p[idx] < 0.5 else 2.0 * self.max_angle_deg

    def to_dict(self):
        return {"name": self.name, "angles_deg": self.angles_deg.tolist(), "relative_intensity": self.rel_intensity.tolist()}


def builtin_profiles() -> dict[str, BeamProfile]:
    return {n: BeamProfile(n, a, i) for n, (a, i) in _BUILTIN.items()}


def load_profiles(directory="beam_profiles") -> dict[str, BeamProfile]:
    """Built-ins plus every ``*.json`` in ``directory`` (file entries override built-ins of the same name)."""
    out = builtin_profiles()
    if directory and os.path.isdir(directory):
        for fn in sorted(os.listdir(directory)):
            if not fn.lower().endswith(".json"):
                continue
            try:
                with open(os.path.join(directory, fn), encoding="utf-8") as f:
                    d = json.load(f)
                name = d.get("name") or fn[:-5]
                out[name] = BeamProfile(name, d["angles_deg"], d["relative_intensity"])
            except Exception as exc:  # a bad file must not take the app down
                print(f"⚠ beam profile {fn}: {exc}")
    return out


_registry: dict[str, BeamProfile] = {}


def get_profile(name, directory="beam_profiles") -> BeamProfile | None:
    """Resolve a profile by name (None / '' / LAMBERTIAN → None). Loaded once, cached."""
    if not name or name == LAMBERTIAN:
        return None
    if not _registry:
        _registry.update(load_profiles(directory))
    return _registry.get(name)


def profile_names(directory="beam_profiles"):
    if not _registry:
        _registry.update(load_profiles(directory))
    return [LAMBERTIAN] + sorted(_registry)


def refresh_profiles(directory="beam_profiles"):
    _registry.clear()
    _registry.update(load_profiles(directory))
    return profile_names(directory)
