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
    front_angle_deg,
    side_angle_deg,
    viewing_angle,
    radius,
    circle_center_x,
    group_rotations=None,
    row_enabled=None,
):
    """Create 4 LED groups; each group contains 3 LEDs spaced by 3 mm."""
    angles_deg = [front_angle_deg, -front_angle_deg, side_angle_deg, -side_angle_deg]
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]

    if group_rotations is None:
        group_rotations = [0.0, 0.0, 0.0, 0.0]

    leds = []
    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)
        x = circle_center_x + radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        z = 0.0

        # Direction: radially outward from circle center (before offset)
        dir_x = x - circle_center_x
        dir_y = y
        dir_z = 0

        # Apply Y offset (green axis) based on group type to position only
        # Front groups (i=0, i=1): ±6.5 cm (13 cm apart)
        # Side groups (i=2, i=3): ±11.75 cm (23.5 cm apart)
        if i in (0, 1):  # Front
            y_offset = 6.5 if i == 0 else -6.5
        else:  # Side (i in (2, 3))
            y_offset = 11.75 if i == 2 else -11.75
        y = y + y_offset

        # Build local in-plane axes for arranging rows and LEDs
        z_axis = np.array((dir_x, dir_y, dir_z), dtype=float)
        if np.linalg.norm(z_axis) == 0:
            z_axis = np.array((1.0, 0.0, 0.0))
        else:
            z_axis = z_axis / np.linalg.norm(z_axis)

        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Parameters: 3 mm between LEDs in a row, 5 mm between rows
        led_spacing_cm = 0.3  # 3 mm
        row_spacing_cm = 0.5  # 5 mm

        # Four rows with specified in-plane inclinations (degrees)
        inclinations = [90,30, -30,-90]
        # Row centers offsets along Z-axis (blue axis) so rows are spaced by 5mm vertically
        row_offsets = [(-1.5 * row_spacing_cm), (-0.5 * row_spacing_cm), (0.5 * row_spacing_cm), (1.5 * row_spacing_cm)]

        for row_idx, alpha_deg in enumerate(inclinations):
            # Skip row if disabled
            if row_enabled is not None and not row_enabled[row_idx]:
                continue
            alpha = np.radians(alpha_deg)
            # Row direction: along X-axis (which is Y-green in the LED plane coordinate system)
            row_dir = x_axis

            # Center point for this row: distributed along Z axis (blue axis, vertical)
            center_off = row_offsets[row_idx]
            row_center = np.array((x, y, z)) + np.array([0, 0, 1]) * center_off

            # Compute rotated LED direction for this row
            # Base azimuth: radial direction in XY plane (points toward group's angle)
            radial = np.array((dir_x, dir_y, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)
            # Apply group azimuth rotation (rotate radial_unit around Z)
            rot_deg = float(group_rotations[i])
            rot_rad = np.radians(rot_deg)
            ca_r, sa_r = np.cos(rot_rad), np.sin(rot_rad)
            rotated_radial = np.array([
                ca_r * radial_unit[0] - sa_r * radial_unit[1],
                sa_r * radial_unit[0] + ca_r * radial_unit[1],
                0.0,
            ])
            # Tilt around axis perpendicular to radial in the plane: tilt toward Z by alpha
            z_unit = np.array((0.0, 0.0, 1.0))
            rotated_dir = np.cos(alpha) * rotated_radial + (-np.sin(alpha)) * z_unit
            rotated_dir = rotated_dir / np.linalg.norm(rotated_dir)

            # If row 1 or 4 (indices 0 or 3), move row center 1 cm back toward circle center
            if row_idx in (0, 3):
                row_center = row_center - radial_unit * 1.0

            # Three LEDs along the row (spaced along green/Y axis direction: row_dir)
            for off in [-led_spacing_cm, 0.0, led_spacing_cm]:
                pos = tuple(row_center + row_dir * off)
                led = LED(
                    width=1.0,
                    viewing_angle=viewing_angle,
                    position=pos,
                    direction=tuple(rotated_dir),
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
            "Viewing angle (°)", min=10, max=90, step=5, initial_value=10
        )
        # Per-group rotation sliders (rotate beam and visual together)
        rot_front_pos = server.gui.add_slider("Rotate front+ (°)", min=-180, max=180, step=1, initial_value=0)
        rot_front_neg = server.gui.add_slider("Rotate front- (°)", min=-180, max=180, step=1, initial_value=0)
        rot_side_pos = server.gui.add_slider("Rotate side+ (°)", min=-180, max=180, step=1, initial_value=0)
        rot_side_neg = server.gui.add_slider("Rotate side- (°)", min=-180, max=180, step=1, initial_value=0)

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
        show_rays_output = server.gui.add_checkbox(
            "Show rays in output", initial_value=True
        )
        show_led_markers = server.gui.add_checkbox(
            "Show LED markers", initial_value=False
        )
        show_intensity_map = server.gui.add_checkbox(
            "Show intensity on wall", initial_value=True
        )
        intensity_rays_slider = server.gui.add_slider(
            "Rays for intensity", min=100, max=5000, step=100, initial_value=500
        )
        led_lumens_slider = server.gui.add_slider(
            "LED lumens (lm)", min=1, max=2000, step=1, initial_value=100
        )
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=10, max=100, step=10, initial_value=30
        )
        # Intensity legend shown under the sliders as HTML with color swatches
        legend_html = server.gui.add_html("")
        wall_view_size = server.gui.add_slider(
            "Wall view size (cm)", min=20, max=200, step=10, initial_value=80
        )
        # Reset button: some Viser button handles don't support on_update;
        # we'll detect clicks by polling `reset_button.value` in the main loop.
        reset_button = server.gui.add_button("Reset to original positions")
        # Per-row enable toggles
        row1_chk = server.gui.add_checkbox("Row 1 on", initial_value=False)
        row2_chk = server.gui.add_checkbox("Row 2 on", initial_value=True)
        row3_chk = server.gui.add_checkbox("Row 3 on", initial_value=True)
        row4_chk = server.gui.add_checkbox("Row 4 on", initial_value=False)
        # Absorber controls moved to dedicated folder for clarity

    # Store handles for dynamic objects
    led_handles = []
    ray_handles = []
    intensity_handles = []
    absorber_handles = []

    # Absorbers folder (separate group for easier access)
    with server.gui.add_folder("Absorbers"):
        absorbers_enable = server.gui.add_checkbox("Enable absorbers", initial_value=True)
        abs0_off_x = server.gui.add_slider("Abs0 offset X (cm)", min=-50, max=200, step=0.1, initial_value=0.0)
        abs0_off_y = server.gui.add_slider("Abs0 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs0_off_z = server.gui.add_slider("Abs0 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs1_off_x = server.gui.add_slider("Abs1 offset X (cm)", min=-50, max=200, step=0.1, initial_value=0.0)
        abs1_off_y = server.gui.add_slider("Abs1 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs1_off_z = server.gui.add_slider("Abs1 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)

    def compute_wall_intensity(
        leds, wall_dist, num_rays_per_led, grid_size=50, wall_size=80, absorbers=None
    ):
        """Trace rays and compute intensity grid on wall."""
        # Grid covers -wall_size/2 to +wall_size/2 cm in Y and Z
        grid = np.zeros((grid_size, grid_size))
        cell_size = wall_size / grid_size  # cm per cell
        half_size = wall_size / 2

        # Assume uniform luminous flux per LED provided by GUI
        lumens_per_led = float(led_lumens_slider.value) if 'led_lumens_slider' in globals() or True else 100.0
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
                # Ray-box intersection helper (positions in cm)
                def ray_box_intersection(pos, direction, box):
                    center = np.array(box['center'], dtype=float)
                    half = np.array(box['half_sizes'], dtype=float)
                    tmin = -np.inf
                    tmax = np.inf
                    for k in range(3):
                        if abs(direction[k]) < 1e-12:
                            if pos[k] < center[k] - half[k] or pos[k] > center[k] + half[k]:
                                return None
                        else:
                            t1 = (center[k] - half[k] - pos[k]) / direction[k]
                            t2 = (center[k] + half[k] - pos[k]) / direction[k]
                            t_near = min(t1, t2)
                            t_far = max(t1, t2)
                            tmin = max(tmin, t_near)
                            tmax = min(tmax, t_far)
                            if tmin > tmax:
                                return None
                    if tmax < 0:
                        return None
                    return tmin if tmin > 0 else (tmax if tmax > 0 else None)

                # Check absorbers intersection
                hit_absorbed = False
                if absorbers is not None:
                    for a in absorbers:
                        t_hit = ray_box_intersection(led.position, world_dir, a)
                        if t_hit is not None and t_hit > 0:
                            hit_absorbed = True
                            break

                if hit_absorbed:
                    continue

                if world_dir[0] > 0:  # Ray going towards wall
                    t = (wall_dist - led.position[0]) / world_dir[0]
                    if t > 0:
                        hit_y = led.position[1] + world_dir[1] * t
                        hit_z = led.position[2] + world_dir[2] * t

                        # Convert to grid indices (centered at 0,0)
                        grid_y = int((hit_y + half_size) / cell_size)
                        grid_z = int((hit_z + half_size) / cell_size)

                        if 0 <= grid_y < grid_size and 0 <= grid_z < grid_size:
                            # Convert ray count to lumens: each sampled ray represents equal fraction of LED lumens
                            lumens_per_ray = lumens_per_led / max(1, num_rays_per_led)
                            grid[grid_z, grid_y] += lumens_per_ray

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
        nonlocal led_handles, ray_handles, intensity_handles, absorber_handles

        # Ray-box intersection helper for update_scene (positions in cm)
        def ray_box_intersection(pos, direction, box):
            center = np.array(box['center'], dtype=float)
            half = np.array(box['half_sizes'], dtype=float)
            tmin = -np.inf
            tmax = np.inf
            for k in range(3):
                if abs(direction[k]) < 1e-12:
                    if pos[k] < center[k] - half[k] or pos[k] > center[k] + half[k]:
                        return None
                else:
                    t1 = (center[k] - half[k] - pos[k]) / direction[k]
                    t2 = (center[k] + half[k] - pos[k]) / direction[k]
                    t_near = min(t1, t2)
                    t_far = max(t1, t2)
                    tmin = max(tmin, t_near)
                    tmax = min(tmax, t_far)
                    if tmin > tmax:
                        return None
            if tmax < 0:
                return None
            return tmin if tmin > 0 else (tmax if tmax > 0 else None)

        # Clear previous objects (safely ignore already-removed handles)
        for handle in led_handles + ray_handles + intensity_handles + absorber_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles = []
        ray_handles = []
        intensity_handles = []
        absorber_handles = []

        # Get current values
        front_angle = front_angle_slider.value
        side_angle = side_angle_slider.value
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        wall_dist = wall_dist_slider.value
        circle_center_x = circle_center_slider.value
        ray_length = ray_length_slider.value

        # Create LEDs
        # Read per-group rotation slider values
        rotations = [
            rot_front_pos.value,
            rot_front_neg.value,
            rot_side_pos.value,
            rot_side_neg.value,
        ]

        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            group_rotations=rotations,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
        )

        # Build absorbers for front groups (one per front group)
        absorbers = []
        # angles matching create_leds
        angles_deg = [front_angle, -front_angle, side_angle, -side_angle]
        for i, angle_deg in enumerate(angles_deg):
            if i not in (0, 1):
                continue
            angle_rad = np.radians(angle_deg)
            gx = circle_center_x + radius * np.cos(angle_rad)
            gy = radius * np.sin(angle_rad)
            # Y offset same logic used earlier (front groups ±6.5 cm)
            y_offset = 6.5 if i == 0 else -6.5
            gy = gy + y_offset

            # Radial unit (from circle center to LED position), used to shift toward center
            radial = np.array((gx - circle_center_x, gy, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)

            # Center of absorber base position: 5.0 cm toward the wall from LED position
            base_abs_cx = gx + radial_unit[0] * 5.0 - 5.5  # -5 cm offset along X
            # Y offset: +4.2 cm for front+ (i==0), -4.2 cm for front- (i==1)
            y_base_offset =-4.2 if i == 0 else 4.2
            base_abs_cy = gy + radial_unit[1] * 5.0 + y_base_offset
            base_abs_cz = 0.0

            # Apply user offsets (sliders) if available
            if 'absorbers_enable' in globals() and not absorbers_enable.value:
                continue
            if i == 0:
                abs_cx = base_abs_cx + abs0_off_x.value
                abs_cy = base_abs_cy + abs0_off_y.value
                abs_cz = base_abs_cz + abs0_off_z.value
            else:
                abs_cx = base_abs_cx + abs1_off_x.value
                abs_cy = base_abs_cy + abs1_off_y.value
                abs_cz = base_abs_cz + abs1_off_z.value

            # Absorber half-sizes in cm: length along X = 5.0 cm, width along Y = 1.5 cm,
            # small thickness along Z (we use 0.05 cm)
            half_length_x =5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0

            # Here we set box half_sizes as (hx, hy, hz) corresponding to (X, Y, Z)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
            })

        # Draw absorber boxes (red) in the scene
        for idx, a in enumerate(absorbers):
            cx, cy, cz = a['center']
            hx, hy, hz = a['half_sizes']
            # Viser add_box dimensions are in meters (x,y,z)
            dims = ((hx * 2) / 100.0, (hy * 2) / 100.0, (hz * 2) / 100.0)
            pos_m = (cx / 100.0, cy / 100.0, cz / 100.0)
            handle = server.scene.add_box(
                f"/absorbers/abs_{idx}",
                dimensions=dims,
                color=(1.0, 0.0, 0.0),
                position=pos_m,
            )
            absorber_handles.append(handle)

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

        # Draw rays (toggleable)
        if show_rays_output.value:
            for i, led in enumerate(leds):
                vis_rays = led.get_visualization_rays(ray_length)

                for j, (pos, direction) in enumerate(vis_rays):
                    # Calculate end point, clipping at wall
                    # Calculate end point, clipping at absorbers and wall
                    end = pos + direction * ray_length

                    # Check absorbers first via box intersection
                    t_abs_min = None
                    if absorbers is not None:
                        for a in absorbers:
                            t_hit = ray_box_intersection(pos, direction, a)
                            if t_hit is not None and t_hit > 0:
                                if t_abs_min is None or t_hit < t_abs_min:
                                    t_abs_min = t_hit

                    # Clip at wall (compute t_wall as distance along direction, not fraction)
                    t_wall = None
                    if direction[0] != 0:
                        t_wall = (wall_dist - pos[0]) / direction[0]

                    # Choose nearest positive intersection (absorber before wall)
                    t_clip = None
                    if t_abs_min is not None and t_abs_min > 0:
                        t_clip = t_abs_min
                    if t_wall is not None and t_wall > 0:
                        if t_clip is None or t_wall < t_clip:
                            t_clip = t_wall

                    if t_clip is not None:
                        end = pos + direction * min(t_clip, ray_length)

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
                    for k in range(10):  # 10 random rays per LED (reduced for performance)
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

                        # Compute nearest intersection with absorbers or wall and clip
                        end = led.position + world_dir * ray_length
                        t_abs_min = None
                        if absorbers is not None:
                            for a in absorbers:
                                t_hit = ray_box_intersection(led.position, world_dir, a)
                                if t_hit is not None and t_hit > 0:
                                    if t_abs_min is None or t_hit < t_abs_min:
                                        t_abs_min = t_hit

                        t_wall = None
                        if world_dir[0] != 0:
                            t_wall = (wall_dist - led.position[0]) / world_dir[0]

                        t_clip = None
                        if t_abs_min is not None and t_abs_min > 0:
                            t_clip = t_abs_min
                        if t_wall is not None and t_wall > 0:
                            if t_clip is None or t_wall < t_clip:
                                t_clip = t_wall

                        if t_clip is not None:
                            end = led.position + world_dir * min(t_clip, ray_length)

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
                leds, wall_dist, num_rays // 4, grid_size, wall_size, absorbers=absorbers
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

            # Update GUI legend HTML (show numeric lumen levels with color swatches)
            legend_steps = 6
            legend_vals = np.linspace(0, max_intensity, legend_steps)
            html_lines = ["<div style='font-family: sans-serif;'>",
                          "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend (lm)</div>"]
            for val in reversed(legend_vals):
                color = intensity_to_color(val, max_intensity)
                hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
                html_lines.append(
                    f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                    f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                    f"<div>{val:.1f} lm</div></div>"
                )
            html_lines.append("</div>")
            legend_html.content = "".join(html_lines)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, 2.0, 2.0),  # 200cm x 200cm wall, thin
        color=(0.5, 0.5, 0.5),
        position=(0.5, 0.0, 0.0),  # Will be updated
    )

    # Grid on XY plane (1mm x 1mm squares)
    grid_points = []
    for i in range(-100, 101):
        grid_points.append([[-1.0, i * 0.001, 0], [1.0, i * 0.001, 0]])  # 1mm spacing
        grid_points.append([[i * 0.001, -1.0, 0], [i * 0.001, 1.0, 0]])  # 1mm spacing

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
    rot_front_pos.on_update(lambda _: update_scene())
    rot_front_neg.on_update(lambda _: update_scene())
    rot_side_pos.on_update(lambda _: update_scene())
    rot_side_neg.on_update(lambda _: update_scene())
    radius_slider.on_update(lambda _: update_scene())
    circle_center_slider.on_update(lambda _: update_scene())
    ray_length_slider.on_update(lambda _: update_scene())
    led_lumens_slider.on_update(lambda _: update_scene())
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: update_scene())
    row1_chk.on_update(lambda _: update_scene())
    row2_chk.on_update(lambda _: update_scene())
    row3_chk.on_update(lambda _: update_scene())
    row4_chk.on_update(lambda _: update_scene())
    absorbers_enable.on_update(lambda _: update_scene())
    abs0_off_x.on_update(lambda _: update_scene())
    abs0_off_y.on_update(lambda _: update_scene())
    abs0_off_z.on_update(lambda _: update_scene())
    abs1_off_x.on_update(lambda _: update_scene())
    abs1_off_y.on_update(lambda _: update_scene())
    abs1_off_z.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: update_scene())
    intensity_grid_size.on_update(lambda _: update_scene())
    wall_view_size.on_update(lambda _: update_scene())
    wall_dist_slider.on_update(lambda _: (update_wall(), update_scene()))

    # Capture default values so reset restores them
    defaults = {
        "front_angle": front_angle_slider.value,
        "side_angle": side_angle_slider.value,
        "viewing_angle": viewing_angle_slider.value,
        "rot_front_pos": rot_front_pos.value,
        "rot_front_neg": rot_front_neg.value,
        "rot_side_pos": rot_side_pos.value,
        "rot_side_neg": rot_side_neg.value,
        "radius": radius_slider.value,
        "circle_center_x": circle_center_slider.value,
        "wall_dist": wall_dist_slider.value,
        "row1": row1_chk.value,
        "row2": row2_chk.value,
        "row3": row3_chk.value,
        "row4": row4_chk.value,
        "absorbers_enable": absorbers_enable.value,
        "abs0_off_x": abs0_off_x.value,
        "abs0_off_y": abs0_off_y.value,
        "abs0_off_z": abs0_off_z.value,
        "abs1_off_x": abs1_off_x.value,
        "abs1_off_y": abs1_off_y.value,
        "abs1_off_z": abs1_off_z.value,
    }

    # Initial draw
    update_scene()

    print("\n" + "=" * 60)
    print("INTERACTIVE LIGHTING DESIGN")
    print("Open http://localhost:8080 in your browser")
    print("Use the sliders on the left to adjust LED parameters")
    print("=" * 60 + "\n")

    # Keep server running; poll the reset button (some Viser button handles
    # don't expose event callbacks). When pressed, restore defaults and redraw.
    try:
        while True:
            time.sleep(0.2)
            try:
                if getattr(reset_button, "value", False):
                    # Restore default slider values
                    front_angle_slider.value = defaults["front_angle"]
                    side_angle_slider.value = defaults["side_angle"]
                    viewing_angle_slider.value = defaults["viewing_angle"]
                    rot_front_pos.value = defaults["rot_front_pos"]
                    rot_front_neg.value = defaults["rot_front_neg"]
                    rot_side_pos.value = defaults["rot_side_pos"]
                    rot_side_neg.value = defaults["rot_side_neg"]
                    radius_slider.value = defaults["radius"]
                    circle_center_slider.value = defaults["circle_center_x"]
                    wall_dist_slider.value = defaults["wall_dist"]
                    # Restore row checkbox states
                    try:
                        row1_chk.value = defaults["row1"]
                        row2_chk.value = defaults["row2"]
                        row3_chk.value = defaults["row3"]
                        row4_chk.value = defaults["row4"]
                    except Exception:
                        pass
                    # Restore absorber controls
                    try:
                        absorbers_enable.value = defaults["absorbers_enable"]
                        abs0_off_x.value = defaults["abs0_off_x"]
                        abs0_off_y.value = defaults["abs0_off_y"]
                        abs0_off_z.value = defaults["abs0_off_z"]
                        abs1_off_x.value = defaults["abs1_off_x"]
                        abs1_off_y.value = defaults["abs1_off_y"]
                        abs1_off_z.value = defaults["abs1_off_z"]
                    except Exception:
                        pass

                    # Force wall update and scene redraw
                    update_wall()
                    update_scene()

                    # Clear the button press (Viser button may keep value True)
                    try:
                        reset_button.value = False
                    except Exception:
                        pass
            except Exception:
                # Be defensive: ignore polling errors to keep server alive
                pass
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
