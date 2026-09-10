"""Domain model: LEDs, their placement, optics and geometry helpers."""

from .led import LED
from .led_factory import create_leds
from .placement import LEDPlacement

__all__ = ["LED", "LEDPlacement", "create_leds"]
