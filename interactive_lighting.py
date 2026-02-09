"""
Interactive lighting design tool using Viser.
Allows real-time adjustment of LED parameters with sliders.
"""

import numpy as np
import viser
import viser.transforms as tf
import time


class LED:
    """Square planar LED with hemispherical (Lambertian) emission pattern."""

    def __init__(
        self,
        width=1.0,
        viewing_angle=60,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        color=(1.0, 0.0, 0.0),
    ):
        self.width = width
        self.viewing_angle = viewing_angle  # half-angle in degrees
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.color = color

    def get_visualization_rays(self, ray_length=60.0):
        """Get chief ray + marginal rays for visualization."""
        rays = []

        # Build local coordinate system
        z_axis = self.direction
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Chief ray
        rays.append((self.position.copy(), self.direction.copy()))

        # Marginal rays at viewing angle edges
        theta = np.radians(self.viewing_angle)
        s, c = np.sin(theta), np.cos(theta)

        for local_dir in [(s, 0, c), (-s, 0, c), (0, s, c), (0, -s, c)]:
            world_dir = (
                local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis
            )
            world_dir = world_dir / np.linalg.norm(world_dir)
            rays.append((self.position.copy(), world_dir))

        return rays


def create_leds(
    front_angle_deg, side_angle_deg, viewing_angle, radius, circle_center_x
):
    """Create 4 LEDs based on parameters."""
    angles_deg = [front_angle_deg, -front_angle_deg, side_angle_deg, -side_angle_deg]
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]

    leds = []
    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)
        x = circle_center_x + radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        z = 0.0

        # Direction: radially outward from circle center
        dir_x = x - circle_center_x
        dir_y = y
        dir_z = 0

        led = LED(
            width=1.0,
            viewing_angle=viewing_angle,
            position=(x, y, z),
            direction=(dir_x, dir_y, dir_z),
            color=colors[i],
        )
        leds.append(led)

    return leds


def main():
    # Create Viser server
    server = viser.ViserServer()
    print(f"Viser server running at: http://localhost:8080")

    # --- GUI Controls ---
    with server.gui.add_folder("LED Configuration"):
        front_angle_slider = server.gui.add_slider(
            "Front LED angle (°)", min=0, max=45, step=1, initial_value=12
        )
        side_angle_slider = server.gui.add_slider(
            "Side LED angle (°)", min=20, max=80, step=1, initial_value=47
        )
        viewing_angle_slider = server.gui.add_slider(
            "Viewing angle (°)", min=10, max=90, step=5, initial_value=60
        )

    with server.gui.add_folder("Geometry"):
        radius_slider = server.gui.add_slider(
            "Circle radius (cm)", min=10, max=60, step=1, initial_value=35
        )
        wall_dist_slider = server.gui.add_slider(
            "Wall distance (cm)", min=20, max=100, step=5, initial_value=50
        )
        circle_center_slider = server.gui.add_slider(
            "Circle center X (cm)", min=-60, max=0, step=5, initial_value=-35
        )

    with server.gui.add_folder("Display"):
        ray_length_slider = server.gui.add_slider(
            "Ray length (cm)", min=20, max=100, step=5, initial_value=60
        )
        show_random_rays = server.gui.add_checkbox(
            "Show random rays (20/LED)", initial_value=True
        )
        show_led_markers = server.gui.add_checkbox(
            "Show LED markers", initial_value=False
        )
        show_intensity_map = server.gui.add_checkbox(
            "Show intensity on wall", initial_value=True
        )
        intensity_rays_slider = server.gui.add_slider(
            "Rays for intensity", min=100, max=5000, step=100, initial_value=1000
        )
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=10, max=100, step=10, initial_value=50
        )
        wall_view_size = server.gui.add_slider(
            "Wall view size (cm)", min=20, max=200, step=10, initial_value=80
        )

    # Store handles for dynamic objects
    led_handles = []
    ray_handles = []
    intensity_handles = []

    def compute_wall_intensity(
        leds, wall_dist, num_rays_per_led, grid_size=50, wall_size=80
    ):
        """Trace rays and compute intensity grid on wall."""
        # Grid covers -wall_size/2 to +wall_size/2 cm in Y and Z
        grid = np.zeros((grid_size, grid_size))
        cell_size = wall_size / grid_size  # cm per cell
        half_size = wall_size / 2

        for led in leds:
            # Generate random rays
            z_axis = led.direction
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            for _ in range(num_rays_per_led):
                # Lambertian distribution
                u1, u2 = np.random.uniform(0, 1, 2)
                theta = np.arcsin(np.sqrt(u1)) * (
                    np.radians(led.viewing_angle) / (np.pi / 2)
                )
                phi = 2 * np.pi * u2

                local_dir = np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta),
                    ]
                )
                world_dir = (
                    local_dir[0] * x_axis
                    + local_dir[1] * y_axis
                    + local_dir[2] * z_axis
                )
                world_dir = world_dir / np.linalg.norm(world_dir)

                # Check if ray hits wall
                if world_dir[0] > 0:  # Ray going towards wall
                    t = (wall_dist - led.position[0]) / world_dir[0]
                    if t > 0:
                        hit_y = led.position[1] + world_dir[1] * t
                        hit_z = led.position[2] + world_dir[2] * t

                        # Convert to grid indices (centered at 0,0)
                        grid_y = int((hit_y + half_size) / cell_size)
                        grid_z = int((hit_z + half_size) / cell_size)

                        if 0 <= grid_y < grid_size and 0 <= grid_z < grid_size:
                            grid[grid_z, grid_y] += 1

        return grid, wall_size

    def intensity_to_color(value, max_val):
        """Convert intensity to inferno-like colormap."""
        if max_val == 0:
            return (0.0, 0.0, 0.0)
        t = min(value / max_val, 1.0)
        # Simple inferno-like gradient: black -> purple -> red -> orange -> yellow
        if t < 0.25:
            r, g, b = t * 4 * 0.5, 0, t * 4 * 0.5
        elif t < 0.5:
            r, g, b = 0.5 + (t - 0.25) * 4 * 0.5, 0, 0.5 - (t - 0.25) * 4 * 0.5
        elif t < 0.75:
            r, g, b = 1.0, (t - 0.5) * 4 * 0.5, 0
        else:
            r, g, b = 1.0, 0.5 + (t - 0.75) * 4 * 0.5, (t - 0.75) * 4
        return (r, g, b)

    def update_scene():
        """Redraw the scene based on current slider values."""
        nonlocal led_handles, ray_handles, intensity_handles

        # Clear previous objects
        for handle in led_handles + ray_handles + intensity_handles:
            handle.remove()
        led_handles = []
        ray_handles = []
        intensity_handles = []

        # Get current values
        front_angle = front_angle_slider.value
        side_angle = side_angle_slider.value
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        wall_dist = wall_dist_slider.value
        circle_center_x = circle_center_slider.value
        ray_length = ray_length_slider.value

        # Create LEDs
        leds = create_leds(
            front_angle, side_angle, viewing_angle, radius, circle_center_x
        )

        # Draw LEDs as spheres (if enabled)
        if show_led_markers.value:
            for i, led in enumerate(leds):
                handle = server.scene.add_icosphere(
                    f"/leds/led_{i}",
                    radius=0.5,
                    color=led.color,
                    position=tuple(led.position / 100.0),  # Convert cm to m for viser
                )
                led_handles.append(handle)

        # Draw rays
        for i, led in enumerate(leds):
            vis_rays = led.get_visualization_rays(ray_length)

            for j, (pos, direction) in enumerate(vis_rays):
                # Calculate end point, clipping at wall
                end = pos + direction * ray_length

                # Clip at wall
                if pos[0] < wall_dist and end[0] > wall_dist:
                    t = (wall_dist - pos[0]) / (end[0] - pos[0])
                    end = pos + direction * ray_length * t

                # Draw line (positions in meters)
                points = np.array([pos / 100.0, end / 100.0])
                handle = server.scene.add_line_segments(
                    f"/rays/led_{i}/ray_{j}",
                    points=points.reshape(1, 2, 3),
                    colors=led.color,  # Single color tuple
                    line_width=2.0,
                )
                ray_handles.append(handle)

            # Add random rays if enabled
            if show_random_rays.value:
                np.random.seed(42 + i)  # Consistent random rays
                for k in range(20):  # 20 random rays per LED
                    # Random direction within viewing cone
                    u1, u2 = np.random.uniform(0, 1, 2)
                    theta = np.arcsin(np.sqrt(u1)) * (
                        np.radians(viewing_angle) / (np.pi / 2)
                    )
                    phi = 2 * np.pi * u2

                    z_axis = led.direction
                    if abs(z_axis[2]) < 0.9:
                        x_axis = np.cross(z_axis, [0, 0, 1])
                    else:
                        x_axis = np.cross(z_axis, [0, 1, 0])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)

                    local_dir = np.array(
                        [
                            np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta),
                        ]
                    )
                    world_dir = (
                        local_dir[0] * x_axis
                        + local_dir[1] * y_axis
                        + local_dir[2] * z_axis
                    )
                    world_dir = world_dir / np.linalg.norm(world_dir)

                    end = led.position + world_dir * ray_length
                    if led.position[0] < wall_dist and end[0] > wall_dist:
                        t = (wall_dist - led.position[0]) / (end[0] - led.position[0])
                        end = led.position + world_dir * ray_length * t

                    points = np.array([led.position / 100.0, end / 100.0])
                    # Dimmer color for random rays
                    dim_color = (
                        led.color[0] * 0.5,
                        led.color[1] * 0.5,
                        led.color[2] * 0.5,
                    )
                    handle = server.scene.add_line_segments(
                        f"/rays/led_{i}/random_{k}",
                        points=points.reshape(1, 2, 3),
                        colors=dim_color,
                        line_width=1.0,
                    )
                    ray_handles.append(handle)

        # Compute and display intensity map on wall
        if show_intensity_map.value:
            grid_size = int(intensity_grid_size.value)
            wall_size = int(wall_view_size.value)
            num_rays = int(intensity_rays_slider.value)
            intensity_grid, actual_wall_size = compute_wall_intensity(
                leds, wall_dist, num_rays // 4, grid_size, wall_size
            )
            max_intensity = intensity_grid.max()

            cell_size_cm = actual_wall_size / grid_size
            cell_size_m = cell_size_cm / 100.0
            half_size = actual_wall_size / 2

            for gz in range(grid_size):
                for gy in range(grid_size):
                    intensity = intensity_grid[gz, gy]
                    if intensity > 0:
                        color = intensity_to_color(intensity, max_intensity)
                        # Position in meters (centered at 0,0)
                        y_pos = (
                            -half_size + gy * cell_size_cm + cell_size_cm / 2
                        ) / 100.0
                        z_pos = (
                            -half_size + gz * cell_size_cm + cell_size_cm / 2
                        ) / 100.0
                        x_pos = wall_dist / 100.0 - 0.005  # Slightly in front of wall

                        handle = server.scene.add_box(
                            f"/intensity/cell_{gz}_{gy}",
                            dimensions=(0.001, cell_size_m * 0.95, cell_size_m * 0.95),
                            color=color,
                            position=(x_pos, y_pos, z_pos),
                        )
                        intensity_handles.append(handle)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, 2.0, 2.0),  # 200cm x 200cm wall, thin
        color=(0.5, 0.5, 0.5),
        position=(0.5, 0.0, 0.0),  # Will be updated
    )

    # Grid on XY plane
    grid_points = []
    for i in range(-10, 11):
        grid_points.append([[-1.0, i * 0.1, 0], [1.0, i * 0.1, 0]])
        grid_points.append([[i * 0.1, -1.0, 0], [i * 0.1, 1.0, 0]])

    server.scene.add_line_segments(
        "/grid",
        points=np.array(grid_points),
        colors=(0.3, 0.3, 0.3),  # Single color for all segments
        line_width=1.0,
    )

    # Origin axes
    server.scene.add_line_segments(
        "/axes/x",
        points=np.array([[[0, 0, 0], [0.5, 0, 0]]]),
        colors=(1.0, 0.0, 0.0),
        line_width=3.0,
    )
    server.scene.add_line_segments(
        "/axes/y",
        points=np.array([[[0, 0, 0], [0, 0.5, 0]]]),
        colors=(0.0, 1.0, 0.0),
        line_width=3.0,
    )
    server.scene.add_line_segments(
        "/axes/z",
        points=np.array([[[0, 0, 0], [0, 0, 0.5]]]),
        colors=(0.0, 0.0, 1.0),
        line_width=3.0,
    )

    # Callback to update wall position
    def update_wall():
        wall_dist = wall_dist_slider.value
        wall_handle.position = (wall_dist / 100.0, 0.0, 0.0)

    # Register callbacks
    front_angle_slider.on_update(lambda _: update_scene())
    side_angle_slider.on_update(lambda _: update_scene())
    viewing_angle_slider.on_update(lambda _: update_scene())
    radius_slider.on_update(lambda _: update_scene())
    circle_center_slider.on_update(lambda _: update_scene())
    ray_length_slider.on_update(lambda _: update_scene())
    show_random_rays.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: update_scene())
    intensity_grid_size.on_update(lambda _: update_scene())
    wall_view_size.on_update(lambda _: update_scene())
    wall_dist_slider.on_update(lambda _: (update_wall(), update_scene()))

    # Initial draw
    update_scene()

    print("\n" + "=" * 60)
    print("INTERACTIVE LIGHTING DESIGN")
    print("Open http://localhost:8080 in your browser")
    print("Use the sliders on the left to adjust LED parameters")
    print("=" * 60 + "\n")

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
