"""Monte Carlo illuminance engine (wall and room modes)."""

from .direct import direct_illuminance
from .emission import generate_led_rays
from .room import compute_room_intensity, trace_led_in_room
from .room_geometry import build_wall_specs, room_wall_cell_centers, wall_grid_cell_centers_cm
from .settings import EmissionSettings, RoomSettings, WallSettings
from .wall import compute_wall_intensity, trace_led_to_wall

__all__ = [
    "EmissionSettings",
    "RoomSettings",
    "WallSettings",
    "build_wall_specs",
    "compute_room_intensity",
    "compute_wall_intensity",
    "direct_illuminance",
    "generate_led_rays",
    "room_wall_cell_centers",
    "trace_led_in_room",
    "trace_led_to_wall",
    "wall_grid_cell_centers_cm",
]
