"""STL transform from GUI controls and per-vertex mesh shading by the LED set."""
import numpy as np
from lighting_simulator.scene.stl import stl_transform


def _build_stl_transform(stl_scale_ctrl, stl_rot_x_ctrl, stl_rot_y_ctrl, stl_rot_z_ctrl,
                          stl_pos_x_ctrl, stl_pos_y_ctrl, stl_pos_z_ctrl):
    """Build 4x4 transform matrix from STL GUI controls. Shared by all ray tracing paths."""
    return stl_transform(
        stl_scale_ctrl.value,
        (stl_rot_x_ctrl.value, stl_rot_y_ctrl.value, stl_rot_z_ctrl.value),
        (stl_pos_x_ctrl.value, stl_pos_y_ctrl.value, stl_pos_z_ctrl.value),
    )



def calculate_mesh_lighting(mesh_vertices, mesh_normals, leds, base_color=(0.7, 0.7, 0.9), led_lumens=100):
    """
    Calculate per-vertex lighting for STL mesh using physical lux calculation.
    Vectorized: processes all LEDs in a single batched computation.
    
    Args:
        mesh_vertices: Nx3 array of vertex positions in meters (Viser units)
        mesh_normals: Nx3 array of vertex normals
        leds: List of LED objects with position, direction, enabled
        base_color: RGB tuple for base mesh color (0-1 range)
        led_lumens: Luminous flux per LED in lumens (default 100)
    
    Returns:
        Nx3 array of RGB colors for each vertex
    """
    num_vertices = len(mesh_vertices)
    
    # Get only enabled LEDs
    active_leds = [led for led in leds if getattr(led, 'enabled', True)]
    
    if len(active_leds) == 0:
        # Just return ambient
        ambient = 0.15
        base = np.array(base_color, dtype=np.float32) * ambient
        return np.tile(base, (num_vertices, 1))
    
    num_leds = len(active_leds)
    
    # Pre-extract LED data into contiguous arrays for vectorized computation
    led_positions = np.array([led.position for led in active_leds], dtype=np.float32) / 100.0  # cm -> m, shape (L, 3)
    led_directions = np.array([led.direction for led in active_leds], dtype=np.float32)  # (L, 3)
    led_half_angles = np.array([np.radians(led.viewing_angle / 2.0) for led in active_leds], dtype=np.float32)  # (L,)
    
    # Per-LED lumens: use override if available, else global
    per_led_lumens = np.array([
        float(getattr(led, 'lumens', None) or led_lumens)
        for led in active_leds
    ], dtype=np.float32)
    
    # Normalize LED directions
    led_dir_norms = np.linalg.norm(led_directions, axis=1, keepdims=True) + 1e-10
    led_directions = led_directions / led_dir_norms  # (L, 3)
    
    # Calculate luminous intensity per LED (candelas)
    solid_angles = 2 * np.pi * (1 - np.cos(led_half_angles))  # (L,)
    solid_angles = np.maximum(solid_angles, 0.001)
    luminous_intensities = per_led_lumens / solid_angles  # (L,)
    cos_half_angles = np.cos(led_half_angles)  # (L,)
    
    # Filter out LEDs with invalid intensity
    valid = np.isfinite(luminous_intensities) & (luminous_intensities > 0)
    if not np.any(valid):
        ambient = 0.15
        base = np.array(base_color, dtype=np.float32) * ambient
        return np.tile(base, (num_vertices, 1))
    
    led_positions = led_positions[valid]
    led_directions = led_directions[valid]
    luminous_intensities = luminous_intensities[valid]
    cos_half_angles = cos_half_angles[valid]
    num_leds = len(led_positions)
    
    # --- Batched computation: all LEDs x all vertices ---
    # Process in chunks to limit memory (L * N * 3 can be large)
    chunk_size = max(1, min(num_leds, 50_000_000 // max(num_vertices, 1)))  # ~200MB limit
    
    total_illuminance = np.zeros(num_vertices, dtype=np.float32)
    
    for led_start in range(0, num_leds, chunk_size):
        led_end = min(led_start + chunk_size, num_leds)
        L = led_end - led_start
        
        # to_vertex[l, v, :] = mesh_vertices[v] - led_positions[l]
        # Shape: (L, N, 3) 
        to_vertex = mesh_vertices[np.newaxis, :, :] - led_positions[led_start:led_end, np.newaxis, :]  # (L, N, 3)
        distances = np.linalg.norm(to_vertex, axis=2) + 1e-6  # (L, N)
        to_vertex_norm = to_vertex / distances[:, :, np.newaxis]  # (L, N, 3)
        
        # Cone check: cos(angle) between LED direction and to_vertex
        cos_angle = np.einsum('ld,lnd->ln', led_directions[led_start:led_end], to_vertex_norm)  # (L, N)
        in_cone = cos_angle > cos_half_angles[led_start:led_end, np.newaxis]  # (L, N)
        
        # Lambert's cosine: angle between surface normal and incoming light
        cos_incident = np.einsum('nd,lnd->ln', mesh_normals, -to_vertex_norm)  # (L, N)
        cos_incident = np.maximum(0, cos_incident)
        
        # Illuminance = I * cos_incident / d^2 (only in cone)
        illuminance = luminous_intensities[led_start:led_end, np.newaxis] * cos_incident / (distances ** 2)  # (L, N)
        
        # Cone edge falloff
        denom = 1.0 - cos_half_angles[led_start:led_end, np.newaxis]
        denom = np.maximum(denom, 1e-10)
        angle_factor = np.maximum(0, (cos_angle - cos_half_angles[led_start:led_end, np.newaxis]) / denom) ** 2
        illuminance *= angle_factor
        
        # Zero out contributions outside cone
        illuminance *= in_cone.astype(np.float32)
        
        # Sum across LEDs in this chunk
        total_illuminance += np.nansum(illuminance, axis=0)
    
    # Convert lux to color intensity
    lux_to_color_scale = 0.002
    intensity_normalized = np.clip(total_illuminance * lux_to_color_scale, 0.0, 1.0)
    
    # Map to blue-to-white gradient
    blue_base = np.array([0.0, 0.0, 0.2], dtype=np.float32)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    t = intensity_normalized[:, np.newaxis]  # (N, 1)
    final_colors = blue_base * (1.0 - t) + white * t
    final_colors = np.clip(final_colors, 0.0, 1.0)
    
    return final_colors



