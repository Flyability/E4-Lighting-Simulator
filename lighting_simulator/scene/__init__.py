"""Scene construction (LED rig + occluders) from saved configuration files."""

from .absorbers import build_elios_absorbers, rotate_absorbers_z
from .builder import (
    Scene,
    apply_diffuser,
    apply_global_transform,
    build_leds_from_config,
    build_scene_from_config,
    group_config_to_factory,
    individual_led_config_to_factory,
    load_config,
)
from .stl import global_z_rotation_4x4, stl_mesh_data, stl_transform

__all__ = [
    "Scene",
    "apply_diffuser",
    "apply_global_transform",
    "build_elios_absorbers",
    "build_leds_from_config",
    "build_scene_from_config",
    "global_z_rotation_4x4",
    "group_config_to_factory",
    "individual_led_config_to_factory",
    "load_config",
    "rotate_absorbers_z",
    "stl_mesh_data",
    "stl_transform",
]
