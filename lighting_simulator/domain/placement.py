from dataclasses import dataclass

import numpy as np

from .led import LED


@dataclass
class LEDPlacement:
    """A physical LED mounted at a position and orientation in a scene.

    Attribute access falls through to the wrapped :class:`LED`, so a
    placement can be used anywhere an LED is expected. Extra per-placement
    optical data (e.g. ``ext_lens_angle``) is stored on the LED as well.
    """

    led: LED
    index: int
    row_direction: np.ndarray | None = None
    square_normal: np.ndarray | None = None
    owner: tuple[str, int] | None = None
    is_custom: bool = False
    is_dynamic_group: bool = False
    is_individual: bool = False

    _OWN_FIELDS = frozenset({
        "led", "index", "row_direction", "square_normal", "owner",
        "is_custom", "is_dynamic_group", "is_individual",
    })

    def __getattr__(self, name):
        # Only called when normal lookup fails. Guard against recursion while
        # unpickling (``led`` not yet set) or for dunder probes.
        if name in LEDPlacement._OWN_FIELDS or name.startswith("__"):
            raise AttributeError(name)
        led = self.__dict__.get("led")
        if led is None:
            raise AttributeError(name)
        return getattr(led, name)

    def __setattr__(self, name, value):
        if name in LEDPlacement._OWN_FIELDS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.led, name, value)

    @property
    def position(self) -> np.ndarray:
        return self.led.position

    @property
    def mesh_normal(self) -> np.ndarray:
        """Normal of the physical LED square (beam direction when no tilt is applied)."""
        return self.square_normal if self.square_normal is not None else self.led.direction

    @property
    def led_index(self) -> int:
        return self.index

    @led_index.setter
    def led_index(self, value: int) -> None:
        self.index = int(value)

    @property
    def direction(self) -> np.ndarray:
        return self.led.direction

    @property
    def enabled(self) -> bool:
        return self.led.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.led.enabled = value

    @property
    def width(self) -> float:
        return self.led.width

    @property
    def viewing_angle(self) -> float:
        return self.led.viewing_angle

    @property
    def lumens(self) -> float:
        return self.led.lumens
