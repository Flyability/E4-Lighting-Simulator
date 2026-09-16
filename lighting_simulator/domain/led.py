from dataclasses import dataclass, field
import numpy as np

# Operating roles: 'vio' = continuous only (VIO / video), 'flash' = photogrammetry pulse only,
# 'both' = continuous and pulsed. 'both' reproduces the pre-role behaviour and is the default.
ROLES = ('vio', 'flash', 'both')
DEFAULT_ROLE = 'both'
# Integer encoding for optimiser variables (0 = LED off).
ROLE_CODES = {'off': 0, 'vio': 1, 'flash': 2, 'both': 3}
CODE_ROLES = {v: k for k, v in ROLE_CODES.items()}


def normalize_role(role):
    role = role or DEFAULT_ROLE
    return role if role in ROLES else DEFAULT_ROLE


def normalize_roles(roles, n):
    """Length-``n`` list of valid role strings (missing / unknown entries -> 'both')."""
    roles = list(roles or [])
    return [normalize_role(roles[i] if i < len(roles) else None) for i in range(int(n))]


def led_role(led):
    return normalize_role(getattr(led, 'role', None))


def split_by_role(leds):
    """``{'vio': [...], 'flash': [...], 'both': [...]}`` over the enabled LEDs."""
    out = {r: [] for r in ROLES}
    for led in leds:
        if getattr(led, 'enabled', True):
            out[led_role(led)].append(led)
    return out


def apply_operating_mode(leds, flash_on, flash_lumens=None):
    """Set ``enabled`` / ``lumens`` of each LED for one operating point (in place).

    flight (``flash_on=False``): 'flash'-only LEDs are switched off (flagged ``mode_idle``
    so the viewer can tell them from LEDs the user turned off).
    flash  (``flash_on=True``): 'flash' + 'both' LEDs run at ``flash_lumens`` (when given).
    """
    for led in leds:
        led.mode_idle = False
        if not getattr(led, 'enabled', True):
            continue
        role = led_role(led)
        if not flash_on:
            if role == 'flash':
                led.enabled = False
                led.mode_idle = True
        elif role != 'vio' and flash_lumens is not None:
            led.lumens = float(flash_lumens)
    return leds


@dataclass
class LED:
    """A single square LED in the lighting simulator.

    Attributes:
        position (np.array): The position of the LED in 3D space.
        direction (np.array): The direction the LED is facing.
        lumens (float): The intensity of the LED.
        color (tuple): The color of the LED as an RGB tuple.
        width (float): The width of the LED in centimeters. Default is 1.0.
        view_angle (float): The view angle of the LED in degrees. Default is 60.0.
        enabled (bool): Whether the LED is enabled or not. Default is True.
        role (str): 'vio' | 'flash' | 'both' operating role (see module header).
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))  # Default position at origin
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))  # Default direction along +Z axis
    lumens: float = 100.0  # Default intensity
    color: tuple = (1.0, 1.0, 1.0)  # Default to white light
    width: float = 1.0 # in centimeters
    viewing_angle: float = 60.0 # in degrees
    enabled: bool = True
    role: str = DEFAULT_ROLE

    def __post_init__(self):
        # Convert position and direction to numpy arrays and ensure they are of type float
        self.position = np.array(self.position,dtype=float,copy=True)
        self.direction = np.array(self.direction,dtype=float,copy=True)
                
        # Check values and types
        if self.position.shape != (3,):
            raise ValueError("LED position must contain exactly 3 values")

        if self.direction.shape != (3,):
            raise ValueError("LED direction must contain exactly 3 values")

        if self.width <= 0:
            raise ValueError("LED width must be positive")

        if not 0 < self.viewing_angle <= 180:
            raise ValueError("LED viewing angle must be between 0 and 180 degrees")

        if self.lumens < 0:
            raise ValueError("LED lumens cannot be negative")

        if len(self.color) != 3 or not all(0 <= value <= 1 for value in self.color):
            raise ValueError("LED color must be an RGB tuple with values in [0, 1]")

        self.role = normalize_role(self.role)

        # Normalize the direction vector
        norm = np.linalg.norm(self.direction)
        if norm < 1e-10:
            raise ValueError("LED direction cannot be the zero vector")

        self.direction = self.direction / norm

    def get_visualization_rays(self, ray_length=60.0):
        """Chief ray plus four marginal rays at the viewing-angle edge.

        Returns a list of (origin, unit_direction) tuples.
        """
        z_axis = self.direction
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        rays = [(self.position.copy(), self.direction.copy())]
        theta = np.radians(self.viewing_angle / 2.0)
        s, c = np.sin(theta), np.cos(theta)
        for local_dir in [(s, 0, c), (-s, 0, c), (0, s, c), (0, -s, c)]:
            world_dir = local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis
            world_dir = world_dir / np.linalg.norm(world_dir)
            rays.append((self.position.copy(), world_dir))
        return rays