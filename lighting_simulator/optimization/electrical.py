"""LED drive electronics: current ↔ luminous flux and driver-channel count.

Flux is assumed linear in current (``lm = I · V · efficacy``), which is what
"lumen per LED is strictly tied to current" means at the design stage. All
per-LED currents are in amperes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DriverModel:
    voltage_v: float = 6.0
    """Forward voltage of one LED."""
    efficacy_lm_per_w: float = 180.0
    max_current_a: float = 13.0
    """Absolute per-LED ceiling (flash pulse); bounds any current variable."""
    leds_per_driver: int = 1
    """Channels per driver IC — ``n_drivers = ceil(n_leds / leds_per_driver)``."""

    def lumens(self, current_a):
        return float(current_a) * self.voltage_v * self.efficacy_lm_per_w

    def current(self, lumens):
        return float(lumens) / max(1e-9, self.voltage_v * self.efficacy_lm_per_w)

    def n_drivers(self, n_leds):
        return int(math.ceil(n_leds / max(1, int(self.leds_per_driver)))) if n_leds else 0

    def led_currents(self, leds):
        """Per-LED current (A) implied by each LED's flux."""
        return np.array([self.current(led.lumens) for led in leds], dtype=float)
