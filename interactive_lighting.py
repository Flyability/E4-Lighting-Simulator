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
    led_states=None,
):
    """Create 4 LED groups; each group contains 3 LEDs spaced by 3 mm."""
    angles_deg = [front_angle_deg, -front_angle_deg, side_angle_deg, -side_angle_deg]
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0), (1.0, 1.0, 0.2)]

    if group_rotations is None:
        group_rotations = [0.0, 0.0, 0.0, 0.0]

    leds = []
    led_index = 0  # Track global LED index
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
            # Skip row if disabled, but still increment led_index for proper tracking
            if row_enabled is not None and not row_enabled[row_idx]:
                led_index += 3  # Skip 3 LEDs for this row
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
            for led_in_row, off in enumerate([-led_spacing_cm, 0.0, led_spacing_cm]):
                pos = tuple(row_center + row_dir * off)
                
                # Check if this LED should be enabled based on row_enabled and led_states
                is_row_enabled = row_enabled is None or row_enabled[row_idx]
                is_led_enabled = led_states is None or led_states[led_index]
                
                # Create LED object (always create it to maintain fixed indices)
                led = LED(
                    width=1.0,
                    viewing_angle=viewing_angle,
                    position=pos,
                    direction=tuple(rotated_dir),
                    color=colors[i],
                )
                # Add enabled flag to LED object
                led.enabled = is_row_enabled and is_led_enabled
                led.led_index = led_index  # Store the global index
                leds.append(led)
                led_index += 1

    return leds


def main():
    # Create Viser server
    server = viser.ViserServer()
    print(f"Viser server running at: http://localhost:8080")

    # LED state array: 4 groups × 4 rows × 3 LEDs = 48 LEDs total
    led_states = [True] * 48  # All LEDs initially on
    
    # Store button handles for LED control
    led_buttons = {}
    row_buttons = {}
    group_buttons = {}

    # --- GUI Controls ---
    with server.gui.add_folder("LED Configuration"):
        front_angle_slider = server.gui.add_slider(
            "Front LED angle (°)", min=0, max=45, step=1, initial_value=12
        )
        side_angle_slider = server.gui.add_slider(
            "Side LED angle (°)", min=20, max=80, step=1, initial_value=47
        )
        viewing_angle_slider = server.gui.add_slider(
            "Viewing angle (°)", min=10, max=130, step=5, initial_value=120
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
            "Show random rays (scales with intensity)", initial_value=False
        )
        show_rays_output = server.gui.add_checkbox(
            "Show rays in output", initial_value=False
        )
        show_led_markers = server.gui.add_checkbox(
            "Show LED markers", initial_value=False
        )
        show_intensity_map = server.gui.add_checkbox(
            "Show intensity on wall", initial_value=False
        )
        intensity_rays_slider = server.gui.add_slider(
            "Rays for intensity", min=100, max=5000, step=100, initial_value=500
        )
        ray_uniformity_slider = server.gui.add_slider(
            "Ray uniformity (0=Lambertian, 1=Focused)", min=0.0, max=1.0, step=0.2, initial_value=0.0
        )
        led_lumens_slider = server.gui.add_slider(
            "LED lumens (lm/LED)", min=100, max=1000, step=10, initial_value=100
        )
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=5, max=100, step=5, initial_value=30
        )
        # Intensity legend shown under the sliders as HTML with color swatches
        legend_html = server.gui.add_html("")
        wall_view_size = server.gui.add_slider(
            "Wall view size (cm)", min=100, max=200, step=10, initial_value=100
        )
        # Manual update button for intensity map (computationally expensive)
        update_intensity_button = server.gui.add_button("Update Intensity Map")
        # Reset button: some Viser button handles don't support on_update;
        # we'll detect clicks by polling `reset_button.value` in the main loop.
        reset_button = server.gui.add_button("Reset to original positions")
        # Per-row enable toggles
        row1_chk = server.gui.add_checkbox("Row 1 on", initial_value=False)
        row2_chk = server.gui.add_checkbox("Row 2 on", initial_value=True)
        row3_chk = server.gui.add_checkbox("Row 3 on", initial_value=True)
        row4_chk = server.gui.add_checkbox("Row 4 on", initial_value=False)
        # Absorber controls moved to dedicated folder for clarity

    # Camera FOV visualization
    with server.gui.add_folder("Camera FOV"):
        show_camera_fov = server.gui.add_checkbox("Show Camera FOV", initial_value=True)
        camera_fov_h = server.gui.add_slider(
            "Horizontal FOV (°)", min=10, max=120, step=1, initial_value=75
        )
        camera_fov_v = server.gui.add_slider(
            "Vertical FOV (°)", min=10, max=90, step=1, initial_value=60
        )
        camera_pos_x = server.gui.add_slider(
            "Camera X pos (cm)", min=-100, max=100, step=1, initial_value=0
        )
        capture_fov_btn = server.gui.add_button("Capture FOV Image", color="green")

    # Store handles for dynamic objects
    camera_fov_handles = []
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

    # LED Control Matrix (individual LED and row control)
    group_names = ["Front+", "Front-", "Side+", "Side-"]
    group_colors_hex = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33"]
    
    # Create control folders for each group with HTML buttons
    for group_idx, (group_name, color_hex) in enumerate(zip(group_names, group_colors_hex)):
        with server.gui.add_folder(f"{group_name}"):
            # Group control button (toggle entire group)
            group_btn = server.gui.add_button(f"ALL", color=color_hex)
            group_buttons[group_idx] = group_btn
            
            server.gui.add_html("<hr style='margin:4px 0;'>")
            
            # Create controls for each row in the group
            for row_idx in range(4):
                # Row header with row toggle button
                html_content = f"""
                <div style='margin:6px 0 2px 0;'>
                    <span style='font-weight:600;font-size:11px;'>Row {row_idx + 1}:</span>
                </div>
                """
                server.gui.add_html(html_content)
                
                # Row toggle button
                row_btn = server.gui.add_button(f"Row {row_idx + 1}", color="#666666")
                row_buttons[(group_idx, row_idx)] = row_btn
                
                # LED buttons for this row (3 LEDs) - small square buttons
                for led_in_row_idx in range(3):
                    led_global_idx = group_idx * 12 + row_idx * 3 + led_in_row_idx
                    led_btn = server.gui.add_button(f"L{led_in_row_idx+1}", color=color_hex)
                    led_buttons[led_global_idx] = led_btn
                    led_buttons[led_global_idx] = led_btn


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
        for led_idx, led in enumerate(leds):
            # Skip disabled LEDs (check if attribute exists)
            if hasattr(led, 'enabled') and not led.enabled:
                continue
            
            # Set deterministic seed for this LED to get reproducible results
            # Use led.led_index if available, otherwise use enumerate index
            idx = getattr(led, 'led_index', led_idx)
            np.random.seed((42 + idx) % (2**32))  # Keep seed within valid range
            
            # Generate random rays
            z_axis = led.direction
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            for _ in range(num_rays_per_led):
                # Cosine power distribution: intensity proportional to cos^n(theta)
                # uniformity: 0=Lambertian (n=1), 1=Focused (n=4)
                u1, u2 = np.random.uniform(0, 1, 2)
                max_theta = np.radians(led.viewing_angle)
                
                uniformity = float(ray_uniformity_slider.value)
                n = 1.0 + uniformity * 3.0  # Exponent from 1 to 4
                
                # Cosine power distribution sampling with clamping to avoid numerical issues
                cos_max = np.cos(max_theta)
                base = 1 - u1 * (1 - np.power(cos_max, n + 1))
                base = np.clip(base, 0.0, 1.0)  # Clamp to valid range
                cos_theta_sampled = np.power(base, 1.0 / (n + 1))
                cos_theta_sampled = np.clip(cos_theta_sampled, -1.0, 1.0)  # Clamp for arccos
                theta = np.arccos(cos_theta_sampled)
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
                            # Calculate intensity coefficient based on cosine power distribution
                            # For uniformity > 0, intensity follows cos^n(theta)
                            # theta is the angle between ray direction and LED central axis
                            cos_theta = np.cos(theta)  # theta was calculated earlier in sampling
                            intensity_coefficient = np.power(cos_theta, n)
                            
                            # Convert ray count to lumens with intensity modulation
                            # Each ray represents a fraction of LED lumens, modulated by cos^n(theta)
                            lumens_per_ray = (lumens_per_led / max(1, num_rays_per_led)) * intensity_coefficient
                            grid[grid_z, grid_y] += lumens_per_ray

        return grid, wall_size

    def capture_camera_fov_image():
        """Capture intensity image within camera FOV at 1cm resolution."""
        from datetime import datetime
        from PIL import Image
        
        # Get current camera and wall settings
        wall_dist = wall_dist_slider.value
        cam_x = camera_pos_x.value
        fov_h_deg = camera_fov_h.value
        fov_v_deg = camera_fov_v.value
        
        # Calculate FOV dimensions on wall
        fov_h_rad = np.radians(fov_h_deg)
        fov_v_rad = np.radians(fov_v_deg)
        fov_width_cm = 2 * (wall_dist - cam_x) * np.tan(fov_h_rad / 2)
        fov_height_cm = 2 * (wall_dist - cam_x) * np.tan(fov_v_rad / 2)
        
        # Cell resolution: 1cm × 1cm (10mm²)
        cell_size_cm = 1.0
        grid_width = int(np.ceil(fov_width_cm / cell_size_cm))
        grid_height = int(np.ceil(fov_height_cm / cell_size_cm))
        
        # Create grid for FOV region
        fov_grid = np.zeros((grid_height, grid_width))
        
        # Get LEDs configuration
        front_angle = front_angle_slider.value
        side_angle = side_angle_slider.value
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value
        num_rays = int(intensity_rays_slider.value)
        
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
            led_states=led_states,
        )
        
        # Build absorbers
        absorbers = []
        angles_deg = [front_angle, -front_angle, side_angle, -side_angle]
        for i, angle_deg in enumerate(angles_deg):
            if i not in (0, 1):
                continue
            angle_rad = np.radians(angle_deg)
            gx = circle_center_x + radius * np.cos(angle_rad)
            gy = radius * np.sin(angle_rad)
            y_offset = 6.5 if i == 0 else -6.5
            gy = gy + y_offset
            
            radial = np.array((gx - circle_center_x, gy, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)
            
            abs_cx = gx - radial_unit[0] * 1.0
            abs_cy = gy - radial_unit[1] * 1.0
            abs_cz = 0.0
            half_length_x = 0.35
            half_width_y = 1.0
            half_thickness_z = 1.5
            
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
            })
        
        # Ray tracing for FOV region
        lumens_per_led = float(led_lumens_slider.value)
        num_rays_per_led = num_rays // 4
        
        print(f"Capturing FOV image: {grid_width}x{grid_height} pixels...")
        
        for led_idx, led in enumerate(leds):
            if hasattr(led, 'enabled') and not led.enabled:
                continue
            
            idx = getattr(led, 'led_index', led_idx)
            np.random.seed((42 + idx) % (2**32))
            
            z_axis = led.direction
            if abs(z_axis[2]) < 0.9:
                x_axis = np.cross(z_axis, [0, 0, 1])
            else:
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            for _ in range(num_rays_per_led):
                u1, u2 = np.random.uniform(0, 1, 2)
                max_theta = np.radians(led.viewing_angle)
                
                uniformity = float(ray_uniformity_slider.value)
                n = 1.0 + uniformity * 3.0
                
                # Cosine power distribution sampling with clamping
                cos_max = np.cos(max_theta)
                base = 1 - u1 * (1 - np.power(cos_max, n + 1))
                base = np.clip(base, 0.0, 1.0)
                cos_theta_sampled = np.power(base, 1.0 / (n + 1))
                cos_theta_sampled = np.clip(cos_theta_sampled, -1.0, 1.0)
                theta = np.arccos(cos_theta_sampled)
                phi = 2 * np.pi * u2
                
                local_dir = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
                world_dir = (
                    local_dir[0] * x_axis
                    + local_dir[1] * y_axis
                    + local_dir[2] * z_axis
                )
                world_dir = world_dir / np.linalg.norm(world_dir)
                
                # Check absorber intersection
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
                
                hit_absorbed = False
                for a in absorbers:
                    t_hit = ray_box_intersection(led.position, world_dir, a)
                    if t_hit is not None and t_hit > 0:
                        hit_absorbed = True
                        break
                
                if hit_absorbed:
                    continue
                
                if world_dir[0] > 0:
                    t = (wall_dist - led.position[0]) / world_dir[0]
                    if t > 0:
                        hit_y = led.position[1] + world_dir[1] * t
                        hit_z = led.position[2] + world_dir[2] * t
                        
                        # Check if hit is within FOV bounds (centered at 0,0)
                        half_w = fov_width_cm / 2
                        half_h = fov_height_cm / 2
                        
                        if -half_w <= hit_y <= half_w and -half_h <= hit_z <= half_h:
                            # Convert to FOV grid indices
                            grid_x = int((hit_y + half_w) / cell_size_cm)
                            grid_y = int((hit_z + half_h) / cell_size_cm)
                            
                            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                                cos_theta = np.cos(theta)
                                intensity_coefficient = np.power(cos_theta, n)
                                lumens_per_ray = (lumens_per_led / max(1, num_rays_per_led)) * intensity_coefficient
                                fov_grid[grid_y, grid_x] += lumens_per_ray
        
        # Convert to lux and create image
        cell_area_m2 = (cell_size_cm / 100.0) ** 2
        lux_grid = fov_grid / cell_area_m2
        
        # Get max intensity for color mapping
        max_lumens = fov_grid.max()
        max_lux = lux_grid.max()
        
        # Create image using same colormap as render (intensity_to_color)
        img_rgb = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                lumen_val = fov_grid[i, j]
                color = intensity_to_color(lumen_val, max_lumens)
                img_rgb[i, j] = [int(c * 255) for c in color]
        
        # Add legend to the right (50 pixels wide)
        legend_width = 50
        legend_steps = 100
        full_width = grid_width + legend_width + 10  # 10px padding
        img_with_legend = np.ones((grid_height, full_width, 3), dtype=np.uint8) * 255  # White background
        
        # Copy main image
        img_with_legend[:, :grid_width, :] = img_rgb
        
        # Draw legend bar
        legend_x_start = grid_width + 5
        legend_x_end = legend_x_start + 30
        
        for i in range(legend_steps):
            # Map i to grid_height
            y_start = int(i * grid_height / legend_steps)
            y_end = int((i + 1) * grid_height / legend_steps)
            
            # Intensity from top (max) to bottom (min)
            intensity_fraction = 1.0 - (i / legend_steps)
            lumen_val = intensity_fraction * max_lumens
            color = intensity_to_color(lumen_val, max_lumens)
            rgb = [int(c * 255) for c in color]
            
            img_with_legend[y_start:y_end, legend_x_start:legend_x_end, :] = rgb
        
        # Use PIL to add text labels
        from PIL import ImageDraw, ImageFont
        img_pil = Image.fromarray(img_with_legend, 'RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use a default font, fallback to PIL default
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # Add text labels at key points
        num_labels = 6
        for i in range(num_labels):
            fraction = i / (num_labels - 1)
            y_pos = int((1.0 - fraction) * grid_height)
            lux_val = fraction * max_lux
            
            # Draw tick mark
            draw.line([(legend_x_end, y_pos), (legend_x_end + 3, y_pos)], fill=(0, 0, 0), width=1)
            
            # Draw text
            text = f"{lux_val:.0f}"
            draw.text((legend_x_end + 5, y_pos - 5), text, fill=(0, 0, 0), font=font)
        
        # Add "lux" label
        draw.text((legend_x_start, 5), "lux", fill=(0, 0, 0), font=font)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fov_intensity_{timestamp}.png"
        img_pil.save(filename)
        
        print(f"FOV image saved to {filename}")
        print(f"Image size: {grid_width} x {grid_height} pixels (1 pixel = 1cm²)")
        print(f"FOV dimensions: {fov_width_cm:.2f} x {fov_height_cm:.2f} cm")
        print(f"Total lumens in FOV: {fov_grid.sum():.2f} lm")
        print(f"Max illuminance: {max_lux:.2f} lux")

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

    def update_intensity_map():
        """Update only the intensity map on the wall (expensive operation)."""
        nonlocal intensity_handles
        
        # Clear previous intensity handles
        for handle in intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        intensity_handles = []
        
        if not show_intensity_map.value:
            return
        
        # Get current values
        wall_dist = wall_dist_slider.value
        grid_size = int(intensity_grid_size.value)
        wall_size = int(wall_view_size.value)
        num_rays = int(intensity_rays_slider.value)
        
        # Get current LEDs configuration
        front_angle = front_angle_slider.value
        side_angle = side_angle_slider.value
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value
        
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
            led_states=led_states,
        )
        
        # Build absorbers
        absorbers = []
        angles_deg = [front_angle, -front_angle, side_angle, -side_angle]
        for i, angle_deg in enumerate(angles_deg):
            if i not in (0, 1):
                continue
            angle_rad = np.radians(angle_deg)
            gx = circle_center_x + radius * np.cos(angle_rad)
            gy = radius * np.sin(angle_rad)
            y_offset = 6.5 if i == 0 else -6.5
            gy = gy + y_offset
            
            radial = np.array((gx - circle_center_x, gy, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)
            
            base_abs_cx = gx + radial_unit[0] * 5.0 - 5.0
            y_base_offset = -4.2 if i == 0 else 4.2
            base_abs_cy = gy + radial_unit[1] * 5.0 + y_base_offset
            base_abs_cz = 0.0
            
            if not absorbers_enable.value:
                continue
            if i == 0:
                abs_cx = base_abs_cx + abs0_off_x.value
                abs_cy = base_abs_cy + abs0_off_y.value
                abs_cz = base_abs_cz + abs0_off_z.value
            else:
                abs_cx = base_abs_cx + abs1_off_x.value
                abs_cy = base_abs_cy + abs1_off_y.value
                abs_cz = base_abs_cz + abs1_off_z.value
            
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
            })
        
        # Compute intensity
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
                    y_pos = (
                        -half_size + gy * cell_size_cm + cell_size_cm / 2
                    ) / 100.0
                    z_pos = (
                        -half_size + gz * cell_size_cm + cell_size_cm / 2
                    ) / 100.0
                    x_pos = wall_dist / 100.0 - 0.005
                    
                    handle = server.scene.add_box(
                        f"/intensity/cell_{gz}_{gy}",
                        dimensions=(0.001, cell_size_m * 0.95, cell_size_m * 0.95),
                        color=color,
                        position=(x_pos, y_pos, z_pos),
                    )
                    intensity_handles.append(handle)
        
        # Update legend
        legend_steps = 6
        legend_vals = np.linspace(0, max_intensity, legend_steps)
        # Conversion: Illuminance (lux) = Luminous flux (lumen) / Area (m²)
        # Each grid cell area = (cell_size_m)²
        cell_area_m2 = cell_size_m * cell_size_m
        html_lines = ["<div style='font-family: sans-serif;'>",
                      "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"]
        for val in reversed(legend_vals):
            color = intensity_to_color(val, max_intensity)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            # lux = total lumen hitting the cell / cell area in m²
            lux_val = val / cell_area_m2 if cell_area_m2 > 0 else 0
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{val:.1f} lm</div>"
                f"<div style='color:#888;font-size:11px;'>({lux_val:.0f} lx)</div></div>"
            )
        html_lines.append("</div>")
        legend_html.content = "".join(html_lines)
    
    def update_scene():
        """Redraw the scene based on current slider values (without intensity map)."""
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles

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
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles = []
        ray_handles = []
        absorber_handles = []
        camera_fov_handles = []

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
            led_states=led_states,
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
            base_abs_cx = gx + radial_unit[0] * 5.0 - 5.0  # -5 cm offset along X
            # Y offset: -4.2 cm for left/front+ (i==0), +4.2 cm for right/front- (i==1)
            y_base_offset = -4.2 if i == 0 else 4.2
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
                if hasattr(led, 'enabled') and not led.enabled:
                    continue
                led_idx = getattr(led, 'led_index', i)
                handle = server.scene.add_icosphere(
                    f"/leds/led_{led_idx}",
                    radius=0.5,
                    color=led.color,
                    position=tuple(led.position / 100.0),  # Convert cm to m for viser
                )
                led_handles.append(handle)

        # Draw rays (toggleable)
        if show_rays_output.value:
            for i, led in enumerate(leds):
                if hasattr(led, 'enabled') and not led.enabled:
                    continue
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
                    led_idx = getattr(led, 'led_index', i)
                    handle = server.scene.add_line_segments(
                        f"/rays/led_{led_idx}/ray_{j}",
                        points=points.reshape(1, 2, 3),
                        colors=led.color,  # Single color tuple
                        line_width=2.0,
                    )
                    ray_handles.append(handle)

                # Add random rays if enabled
                if show_random_rays.value:
                    led_idx = getattr(led, 'led_index', i)
                    np.random.seed(42 + led_idx)  # Consistent random rays
                    num_random_rays = 50  # Fixed number of visualization rays
                    for k in range(num_random_rays):
                        # Random direction within viewing cone using cosine power distribution
                        u1, u2 = np.random.uniform(0, 1, 2)
                        max_theta = np.radians(viewing_angle)
                        
                        uniformity = float(ray_uniformity_slider.value)
                        n = 1.0 + uniformity * 3.0  # Exponent from 1 to 4
                        
                        # Cosine power distribution sampling with clamping
                        cos_max = np.cos(max_theta)
                        base = 1 - u1 * (1 - np.power(cos_max, n + 1))
                        base = np.clip(base, 0.0, 1.0)
                        cos_theta_sampled = np.power(base, 1.0 / (n + 1))
                        cos_theta_sampled = np.clip(cos_theta_sampled, -1.0, 1.0)
                        theta = np.arccos(cos_theta_sampled)
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
                            f"/rays/led_{led_idx}/random_{k}",
                            points=points.reshape(1, 2, 3),
                            colors=dim_color,
                            line_width=1.0,
                        )
                        ray_handles.append(handle)

        # Draw camera FOV rectangle on wall
        if show_camera_fov.value:
            wall_dist = wall_dist_slider.value
            cam_x = camera_pos_x.value
            fov_h_deg = camera_fov_h.value
            fov_v_deg = camera_fov_v.value
            
            # Calculate FOV dimensions on wall based on viewing angles
            # Using simple trigonometry: width = 2 * distance * tan(angle/2)
            fov_h_rad = np.radians(fov_h_deg)
            fov_v_rad = np.radians(fov_v_deg)
            
            # FOV dimensions at wall distance (in cm)
            fov_width_cm = 2 * (wall_dist - cam_x) * np.tan(fov_h_rad / 2)
            fov_height_cm = 2 * (wall_dist - cam_x) * np.tan(fov_v_rad / 2)
            
            # Draw FOV border lines only (no fill)
            half_w = fov_width_cm / 200.0  # Half width in meters
            half_h = fov_height_cm / 200.0  # Half height in meters
            wall_x = wall_dist / 100.0 - 0.008
            
            # Four corner lines
            corners = [
                [[wall_x, -half_w, -half_h], [wall_x, half_w, -half_h]],  # Bottom
                [[wall_x, half_w, -half_h], [wall_x, half_w, half_h]],    # Right
                [[wall_x, half_w, half_h], [wall_x, -half_w, half_h]],    # Top
                [[wall_x, -half_w, half_h], [wall_x, -half_w, -half_h]],  # Left
            ]
            
            handle = server.scene.add_line_segments(
                "/camera/fov_border",
                points=np.array(corners),
                colors=(0.0, 1.0, 0.0),  # Green
                line_width=6.0,  # Linee più spesse
            )
            camera_fov_handles.append(handle)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, 2.0, 2.0),  # 200cm x 200cm wall, thin
        color=(0.5, 0.5, 0.5),
        position=(0.5, 0.0, 0.0),  # Will be updated
    )

    # Grid on XY plane (millimeter resolution)
    grid_points = []
    for i in range(-10, 11):
        grid_points.append([[-1.0, i * 0.01, 0], [1.0, i * 0.01, 0]])  # 1mm spacing
        grid_points.append([[i * 0.01, -1.0, 0], [i * 0.01, 1.0, 0]])  # 1mm spacing

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
    led_lumens_slider.on_update(lambda _: None)  # No auto-update, use manual button
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: update_intensity_map())
    row1_chk.on_update(lambda _: update_scene())
    row2_chk.on_update(lambda _: update_scene())
    row3_chk.on_update(lambda _: update_scene())
    row4_chk.on_update(lambda _: update_scene())
    absorbers_enable.on_update(lambda _: update_scene())
    show_camera_fov.on_update(lambda _: update_scene())
    camera_fov_h.on_update(lambda _: update_scene())
    camera_fov_v.on_update(lambda _: update_scene())
    camera_pos_x.on_update(lambda _: update_scene())
    abs0_off_x.on_update(lambda _: update_scene())
    abs0_off_y.on_update(lambda _: update_scene())
    abs0_off_z.on_update(lambda _: update_scene())
    abs1_off_x.on_update(lambda _: update_scene())
    abs1_off_y.on_update(lambda _: update_scene())
    abs1_off_z.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: None)  # No auto-update for expensive params
    ray_uniformity_slider.on_update(lambda _: None)  # No auto-update for expensive params
    intensity_grid_size.on_update(lambda _: None)  # No auto-update for expensive params
    wall_view_size.on_update(lambda _: None)  # No auto-update for expensive params
    wall_dist_slider.on_update(lambda _: (update_wall(), update_scene()))
    
    # Register LED control button callbacks
    # Group buttons
    for group_idx, btn in group_buttons.items():
        def make_group_handler(g_idx):
            def handler(_):
                start_idx = g_idx * 12
                end_idx = start_idx + 12
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
            return handler
        btn.on_click(make_group_handler(group_idx))
    
    # Row buttons
    for (group_idx, row_idx), btn in row_buttons.items():
        def make_row_handler(g_idx, r_idx):
            def handler(_):
                start_idx = g_idx * 12 + r_idx * 3
                end_idx = start_idx + 3
                any_on = any(led_states[start_idx:end_idx])
                new_state = not any_on
                for i in range(start_idx, end_idx):
                    led_states[i] = new_state
                print(f"Group {g_idx} Row {r_idx} toggled: LEDs {start_idx}-{end_idx-1} set to {new_state}")
                update_scene()
            return handler
        btn.on_click(make_row_handler(group_idx, row_idx))
    
    # Individual LED buttons
    for led_idx, btn in led_buttons.items():
        def make_led_handler(l_idx):
            def handler(_):
                led_states[l_idx] = not led_states[l_idx]
                print(f"LED {l_idx} toggled to {led_states[l_idx]}")
                update_scene()
            return handler
        btn.on_click(make_led_handler(led_idx))
    
    # Button for manual intensity map update
    update_intensity_button.on_click(lambda _: update_intensity_map())
    
    # Button for capturing FOV intensity image
    capture_fov_btn.on_click(lambda _: capture_camera_fov_image())

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
