
import argparse
import time
import numpy as np
# Monkeypatch np.float to float to fix pvtrace compatibility with newer numpy
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

import matplotlib.pyplot as plt
import functools
import sys
import os

import logging
# Configure logging to reduce noise from libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

try:
    from pvtrace import Node, Scene, Sphere, Box, Material, Light, MeshcatRenderer, Ray
    from pvtrace.algorithm.photon_tracer import follow
    from pvtrace.light.event import Event
    from pvtrace.material.utils import lambertian
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


class LED:
    """
    Square planar LED with hemispherical (Lambertian) emission pattern.
    
    Attributes:
        width: Width and height of the square LED (cm)
        viewing_angle: Half-angle of emission cone (degrees), 90 = hemisphere
        power: Electrical power (Watts)
        efficiency: Luminous efficacy (lumens/Watt)
        position: (x, y, z) position in world coordinates
        direction: (dx, dy, dz) unit vector for chief ray direction (normal to LED surface)
    """
    def __init__(self, width=1.0, viewing_angle=90, power=1.0, efficiency=100.0, 
                 position=(0,0,0), direction=(1,0,0), color=0xFFFFFF):
        self.width = width  # cm
        self.viewing_angle = viewing_angle  # degrees (half-angle)
        self.power = power  # Watts
        self.efficiency = efficiency  # lumens/Watt
        self.lumens = power * efficiency
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(direction)  # Normalize
        self.color = color
    
    def emit_rays(self, num_rays):
        """
        Generate rays with Lambertian (hemispherical) distribution.
        Returns list of (position, direction) tuples in world coordinates.
        """
        rays = []
        
        # Build local coordinate system where Z is the emission direction
        z_axis = self.direction
        # Find a perpendicular vector for x_axis
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        for _ in range(num_rays):
            # Lambertian distribution: cos(theta) weighted
            # theta = arcsin(sqrt(random)), phi = 2*pi*random
            u1, u2 = np.random.uniform(0, 1, 2)
            theta = np.arcsin(np.sqrt(u1))  # Lambertian
            phi = 2 * np.pi * u2
            
            # Limit to viewing angle
            max_theta = np.radians(self.viewing_angle)
            theta = theta * (max_theta / (np.pi/2))  # Scale to viewing angle
            
            # Local direction
            local_dir = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            
            # Transform to world coordinates
            world_dir = local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis
            world_dir = world_dir / np.linalg.norm(world_dir)
            
            rays.append((self.position.copy(), world_dir))
        
        return rays
    
    def get_visualization_rays(self):
        """
        Get specific rays for visualization: chief ray + marginal rays.
        Returns list of (position, direction) tuples.
        """
        rays = []
        
        # Build local coordinate system
        z_axis = self.direction
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [0, 1, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Chief ray (center)
        rays.append((self.position.copy(), self.direction.copy()))
        
        # Marginal rays at viewing angle edges
        theta = np.radians(self.viewing_angle)
        s = np.sin(theta)
        c = np.cos(theta)
        
        # 4 marginal rays at the edge of the viewing cone
        for local_dir in [(s, 0, c), (-s, 0, c), (0, s, c), (0, -s, c)]:
            world_dir = local_dir[0] * x_axis + local_dir[1] * y_axis + local_dir[2] * z_axis
            world_dir = world_dir / np.linalg.norm(world_dir)
            rays.append((self.position.copy(), world_dir))
        
        return rays


def main():
    parser = argparse.ArgumentParser(description="Lighting Simulation")
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization")
    parser.add_argument("--rays", type=int, default=1000, help="Rays per light source for heatmap")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show-rays", dest="show_rays", action="store_true", help="Show rays in 3D visualization")
    group.add_argument("--hide-rays", dest="show_rays", action="store_false", help="Hide rays in 3D visualization")
    parser.set_defaults(show_rays=True)
    args = parser.parse_args()

    # Units: cm
    
    # 1. Setup World
    # World is a large sphere of air
    world_radius = 500.0
    world = Node(
        name="world",
        geometry=Sphere(
            radius=world_radius,
            material=Material(refractive_index=1.0)
        )
    )

    # 2. Setup Wall
    # Flat wall at distance 50 cm in front (X=50)
    # Wall should be large enough to catch light. Let's make it 200x200 cm.
    # It is perpendicular to X axis.
    wall_dist = 50.0
    wall_size = 200.0
    
    # Material for the wall:
    # We want it to be opaque/absorptive so rays stop at it (or reflect), but NOT transmit.
    # refractive_index=1.5 is dielectric (glass).
    # To make it opaque in pvtrace, we can use an Absorber component or just rely on "hits" being recorded.
    # However, for visualization, rays passing through look confusing.
    # Let's use a LambertianSurface which reflects/scatters, or make it fully absorbing.
    # BUT, pvtrace Material takes a list of components.
    # A simple way to stop transmission is to use a high absorption coefficient or a SurfaceDelegate.
    # Let's stick to dielectric but handle visualization carefully, OR switch to a standard "opaque" material concept if available.
    # Actually, if we just want to Visualize it as solid, we change the renderer.
    # If we want the Physics to be solid, we need a surface that doesn't transmit.
    # Using LambertianSurface will reflect rays diffusely.
    # For now, let's keep physics as is (counting hits) but make the visualization clearly solid.
    
    wall = Node(
        name="wall",
        geometry=Box(
            size=(1.0, wall_size, wall_size), # Thickness 1cm
            material=Material(refractive_index=1.5) # Absorbing or standard material to register hits
        ),
        parent=world
    )
    # Move wall to x=50. The box center is at (0,0,0) locally.
    # If thickness is 1cm, center at 50.5 puts front face at 50.
    wall.translate((wall_dist + 0.5, 0, 0))

    # 3. Setup Lights using LED objects
    # Circle radius 35 cm, centered at X=-35cm (so circle passes through origin)
    # LEDs at 12° and 47° angles from forward (+X direction from circle center)
    # Chief ray points NORMAL to the circumference (radially outward from circle center)
    
    radius = 35.0
    circle_center_x = -35.0  # Circle passes through origin
    angles_deg = [12, -12, 47, -47]
    
    # LED parameters
    led_width = 1.0  # cm (square LED)
    led_viewing_angle = 10  # degrees half-angle (typical LED)
    led_power = 1.0  # Watts
    led_efficiency = 100.0  # lumens/Watt
    
    # Colors for each light: Red, Green, Blue, Yellow
    light_colors = [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00]
    
    leds = []
    
    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)
        
        # Position on the circle
        # Circle centered at (circle_center_x, 0, 0), forward is +X.
        x = circle_center_x + radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        z = 0.0
        
        # Direction: Normal to circumference = radially outward from circle center (before offset)
        # Vector from circle center to LED position
        dir_x = x - circle_center_x  # = radius * cos(angle)
        dir_y = y - 0  # = radius * sin(angle)
        dir_z = 0
        
        # Apply Y offset (green axis) based on group type to position only
        # Front groups (i=0, i=1): ±6.5 cm (13 cm apart)
        # Side groups (i=2, i=3): ±11.75 cm (23.5 cm apart)
        if i in (0, 1):  # Front
            y_offset = 6.5 if i == 0 else -6.5
        else:  # Side (i in (2, 3))
            y_offset = 11.75 if i == 2 else -11.75
        y = y + y_offset
        
        position = (x, y, z)
        direction = (dir_x, dir_y, dir_z)
        
        # Create 4 rows, each with 3 sub-LEDs. Rows spaced by 5mm, LEDs in row spaced by 3mm.
        z_axis = np.array(direction, dtype=float)
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

        led_spacing_cm = 0.3  # 3 mm
        row_spacing_cm = 1 # 5 mm
        inclinations = [-90, 30, -30, 90]
        row_offsets = [(-1.5 * row_spacing_cm), (-0.5 * row_spacing_cm), (0.5 * row_spacing_cm), (1.5 * row_spacing_cm)]

        for row_idx, alpha_deg in enumerate(inclinations):
            alpha = np.radians(alpha_deg)
            # Row direction: along X-axis (which is Y-green in the LED plane coordinate system)
            row_dir = x_axis

            # Center point for this row: distributed along Z axis (blue axis, vertical)
            center_off = row_offsets[row_idx]
            row_center = np.array((x, y, z)) + np.array([0, 0, 1]) * center_off

            # If row 1 or 4 (indices 0 or 3), move row center 1 cm back toward circle center
            if row_idx in (0, 1, 2, 3):
                row_center = row_center - radial_unit * 1.0

            # Compute rotated LED direction for this row
            # Base azimuth: radial direction in XY plane (points toward group's angle)
            radial = np.array((dir_x, dir_y, 0.0), dtype=float)
            if np.linalg.norm(radial) == 0:
                radial_unit = np.array((1.0, 0.0, 0.0))
            else:
                radial_unit = radial / np.linalg.norm(radial)
            z_unit = np.array((0.0, 0.0, 1.0))
            rotated_dir = np.cos(alpha) * radial_unit + (-np.sin(alpha)) * z_unit
            rotated_dir = rotated_dir / np.linalg.norm(rotated_dir)

            for off in [-led_spacing_cm, 0.0, led_spacing_cm]:
                pos_off = tuple(row_center + row_dir * off)
                led = LED(
                    width=led_width,
                    viewing_angle=led_viewing_angle,
                    power=led_power,
                    efficiency=led_efficiency,
                    position=pos_off,
                    direction=tuple(rotated_dir),
                    color=light_colors[i % len(light_colors)]
                )
                leds.append(led)
        print(f"LED group {i}: angle={angle_deg}°, center_pos=({x:.1f}, {y:.1f}, {z:.1f}), dir=({dir_x:.2f}, {dir_y:.2f}, {dir_z:.2f}) -> 4 rows x 3 LEDs")

    # 4. Run Simulation
    scene = Scene(world)
    
    renderer = None
    if args.visualize:
        print("Starting Meshcat renderer...")
        # open_browser=False because we are likely in a remote env
        try:
            # wireframe=False to make wall visible (but transparent)
            renderer = MeshcatRenderer(open_browser=False, wireframe=False, transparency=True, opacity=0.3)
            
            # --- CUSTOM GRID AND AXES ---
            # Remove default grid/axes if possible or override them
            vis = renderer.vis
            
            # Disable fog or increase far plane if possible, but meshcat simple viewer handles this well.
            # Maybe the user refers to the black background or clipping.
            # We can set background color to something lighter if "fog of war" means dark.
            vis["/Background"].set_property("visible", False) # Hide default background if any
            # Actually, to change background color in meshcat:
            vis.set_property("background", [0.1, 0.1, 0.1]) # Dark gray instead of black might help?
            
            # Fix disappearing objects when zooming out (Camera clipping planes)
            # Default Three.js perspective camera has far plane.
            # We can try to set it via properties if exposed, but meshcat often handles this automatically.
            # A common trick is to ensure the scene bounding box is well defined.
            # Our scene is large (500cm world).
            # We can force the camera to have a larger far clipping plane.
            vis['/Cameras/default/rotated/<object>'].set_property('far', 10000.0)
            vis['/Cameras/default/rotated/<object>'].set_property('near', 0.1)
            
            # Create a custom grid
            # 1cm spacing. Scene is ~200cm wide.
            # Draw lines every 10cm (major) and 1cm (minor) or just every 10cm for clarity?
            # User wants "1 grid cell for each centimeter". That's very dense for a 200cm scene.
            # Let's do 10cm grid lines with maybe a different color.
            
            # Using LineSegments to draw a grid
            import meshcat.geometry as g
            import meshcat.transformations as tf

            # Grid parameters
            size = 200 # cm total size
            step = 10   # cm spacing (10cm cells to avoid clutter, user asked for 1cm but that's 40,000 lines)
                        # Let's try 5cm or stick to 10cm and mention scale.
                        # Or maybe user means "I want to see the scale".
            
            # Let's make a grid with 10cm main lines and maybe 1cm markers?
            # Actually, standard meshcat grid is likely unit-less or meters.
            # If our units are cm, then 1 unit = 1 cm.
            # Default meshcat grid is probably 10x10 units.
            
            # We can disable the default grid in the browser UI, but harder from python.
            # We can draw our own axes.
            
            # Draw Axes (X=Red, Y=Green, Z=Blue) length 50cm
            vis["/Axes"].set_object(g.LineSegments(
                g.PointsGeometry(np.array([
                    [0, 0, 0], [50, 0, 0], # X
                    [0, 0, 0], [0, 50, 0], # Y
                    [0, 0, 0], [0, 0, 50]  # Z
                ]).astype(np.float32).T),
                g.LineBasicMaterial(vertexColors=True)
            ))
            # Set colors for axes manually? LineBasicMaterial can take vertex colors but needs color array.
            
            # Simpler: Draw 3 separate lines
            vis["/Axes/X"].set_object(g.Line(g.PointsGeometry(np.array([[0,0,0],[50,0,0]], dtype=np.float32).T), g.LineBasicMaterial(color=0xff0000)))
            vis["/Axes/Y"].set_object(g.Line(g.PointsGeometry(np.array([[0,0,0],[0,50,0]], dtype=np.float32).T), g.LineBasicMaterial(color=0x00ff00)))
            vis["/Axes/Z"].set_object(g.Line(g.PointsGeometry(np.array([[0,0,0],[0,0,50]], dtype=np.float32).T), g.LineBasicMaterial(color=0x0000ff)))
            
            # Draw Grid
            # Grid on XY plane (Z=0)
            # Lines parallel to X and Y
            # Range -100 to 100
            grid_lines = []
            grid_color = 0x444444
            
            for i in range(-100, 101, 10): # Every 10cm
                # Line parallel to X
                grid_lines.append([[ -100.0, float(i), 0.0], [ 100.0, float(i), 0.0]])
                # Line parallel to Y
                grid_lines.append([[ float(i), -100.0, 0.0], [ float(i), 100.0, 0.0]])
            
            # Flatten
            grid_points = []
            for start, end in grid_lines:
                grid_points.append(start)
                grid_points.append(end)
            
            vis["/Grid"].set_object(g.LineSegments(
                g.PointsGeometry(np.array(grid_points, dtype=np.float32).T),
                g.LineBasicMaterial(color=grid_color)
            ))

            renderer.render(scene)
            
            # Make wall more visible - draw a filled quad
            # Draw a solid plane for the wall
            vis["/WallPlane"].set_object(
                g.Box([0.5, wall_size, wall_size]),  # Thin box as wall
                g.MeshBasicMaterial(color=0x808080, opacity=0.5, transparent=True)
            )
            vis["/WallPlane"].set_transform(tf.translation_matrix([50.25, 0, 0]))
            
            # We can add a wireframe box around it to define edges clearly.
            vis["/WallOutline"].set_object(g.LineSegments(
                g.PointsGeometry(np.array([
                    [50, -100, -100], [50, 100, -100],
                    [50, 100, -100], [50, 100, 100],
                    [50, 100, 100], [50, -100, 100],
                    [50, -100, 100], [50, -100, -100]
                ], dtype=np.float32).T),
                g.LineBasicMaterial(color=0xFFFFFF, linewidth=2)
            ))
            
            # Also add a solid but very transparent plane for the wall to catch "fog" or just be visible
            # Wall itself should be rendered by renderer.render(scene) but maybe it's too subtle.
            # Let's override the wall material in the visualizer to be sure.
            # Node name is "wall", so path is "world | wall" (spaces around | are standard in pvtrace renderer)
            # Make it SOLID (opacity 1.0) and WHITE/GRAY to be clearly visible as a wall.
            vis["world | wall"].set_property("opacity", 0.5) 
            vis["world | wall"].set_property("color", 0x808080) # Gray
            vis["world | wall"].set_property("transparent", True) # Keep semi-transparent to see rays inside/behind?
            # User said "I don't see the wall at all now" and "this wall need to be solid".
            # So let's make it more opaque.
            vis["world | wall"].set_property("opacity", 0.8)
            
            # Explicitly create a solid box for the wall if the renderer didn't pick it up well
            # But renderer.render(scene) should have done it.
            # If "world | wall" path is correct. Let's verify paths? 
            # In renderer.py: pathname = " | ".join([x.name for x in node.path])
            # root name is "world". child name is "wall". Path is "world | wall". Correct.

            print("Visualization server started. Open the following URL in your browser:")
            print(renderer.vis.url())
        except Exception as e:
            print(f"Failed to start renderer: {e}")
            renderer = None
    
    # If visualizing, draw 100 rays per LED + chief/marginal rays for clarity
    num_rays_per_light = args.rays  # Default 5000, used for heatmap calculation
    
    # Always calculate heatmap rays for the PNG output
    # Visualization rays are separate (drawn below)

    hit_points = []
    hit_weights = []
    
    print(f"Simulating {len(leds)} LEDs with {num_rays_per_light} rays each for heatmap...")
    
    total_rays = 0
    wall_hits = 0
    all_hit_points_3d = []  # Store (x,y,z, color)
    
    # Pre-draw LED sources as spheres so we can see them
    if renderer:
        import meshcat.geometry as g
        import meshcat.transformations as tf
        
        for led_idx, led in enumerate(leds):
            renderer.vis[f"/Lights/Source_{led_idx}"].set_object(
                g.Sphere(led.width / 2),  # Radius = half width
                g.MeshBasicMaterial(color=led.color)
            )
            renderer.vis[f"/Lights/Source_{led_idx}"].set_transform(
                tf.translation_matrix(led.position)
            )

    # Ray tracing for heatmap AND visualization
    for led_idx, led in enumerate(leds):
        
        # If visualizing, draw rays from this LED (optional)
        if renderer:
            if args.show_rays:
                # Get visualization rays (chief + 4 marginals)
                vis_rays = led.get_visualization_rays()

                # Also get 100 random rays for better coverage visualization
                random_vis_rays = led.emit_rays(100)

                all_rays_to_draw = vis_rays + random_vis_rays

                for pos, direction in all_rays_to_draw:
                    # Create ray
                    r = Ray(
                        position=tuple(pos),
                        direction=tuple(direction),
                        wavelength=555.0
                    )

                    # Trace this ray
                    history = follow(scene, r)

                    # Draw all segments but STOP at wall (x=50) to show solid wall
                    path_points = [ray_step.position for ray_step, evt in history]

                    for j in range(len(path_points) - 1):
                        start = path_points[j]
                        end = path_points[j+1]

                        # If ray crosses wall plane (x=50), clip it there
                        if start[0] < 50.0 and end[0] >= 50.0:
                            if abs(end[0] - start[0]) > 1e-6:
                                t = (50.0 - start[0]) / (end[0] - start[0])
                                end = (
                                    50.0,
                                    start[1] + t * (end[1] - start[1]),
                                    start[2] + t * (end[2] - start[2])
                                )
                            renderer.add_line_segment(start, end, colour=led.color)
                            break  # Stop drawing - wall is solid
                        else:
                            renderer.add_line_segment(start, end, colour=led.color)

        # Heatmap rays (always run for PNG generation)
        rays_data = led.emit_rays(num_rays_per_light)
        
        for pos, direction in rays_data:
            total_rays += 1
            
            r = Ray(
                position=tuple(pos),
                direction=tuple(direction),
                wavelength=555.0
            )
            
            history = follow(scene, r)
            
            for ray_step, event in history:
                p = ray_step.position
                if 49.9 < p[0] < 50.6:
                    # Convert hit count to lumens using LED lumens and total rays per light
                    lumens_per_ray = led.lumens / max(1, num_rays_per_light)
                    hit_points.append((p[1], p[2]))
                    hit_weights.append(lumens_per_ray)
                    all_hit_points_3d.append((p[0], p[1], p[2], led.color))
                    wall_hits += 1
                    break


    # 5. Analyze and Plot
    print(f"Simulation complete. {wall_hits} hits on wall out of {total_rays} rays.")
    
    if renderer and all_hit_points_3d:
        print(f"Adding {len(all_hit_points_3d)} hit points to visualization...")
        # Add point cloud for wall hits - BATCH all points at once
        points_xyz = []
        points_colors = []
        
        for x, y, z, c in all_hit_points_3d:
            # Shift x slightly so it sits on top of the wall surface (x=50) for visibility
            points_xyz.append([x - 0.1, y, z]) 
            # Convert hex color to RGB floats 0-1
            r = ((c >> 16) & 0xFF) / 255.0
            g_Val = ((c >> 8) & 0xFF) / 255.0
            b = (c & 0xFF) / 255.0
            points_colors.append([r, g_Val, b])
        
        # Add all points at once (not inside the loop!)
        vis["/WallHits"].set_object(g.Points(
            g.PointsGeometry(
                np.array(points_xyz, dtype=np.float32).T,
                color=np.array(points_colors, dtype=np.float32).T
            ),
            g.PointsMaterial(size=0.05)
        ))
        vis["/WallHits"].set_property("vertexColors", 2)

    if not hit_points:
        print("No hits recorded. Check geometry and directions.")
        if not args.visualize:
            return
        # In visualization mode, we still want to show the 3D view even without heatmap data

    if hit_points:
        hits_y = [p[0] for p in hit_points]
        hits_z = [p[1] for p in hit_points]
        weights = hit_weights if hit_weights else None
        
        plt.figure(figsize=(10, 8))
        # 2D Histogram
        # Bin range should cover the wall size (-100 to 100)
        bins = 100
        plt.hist2d(hits_y, hits_z, bins=bins, range=[[-100, 100], [-100, 100]], cmap='inferno', weights=weights)
        plt.colorbar(label='Intensity (Lumens)')
        plt.title(f'Lighting Intensity on Wall (Distance {wall_dist}cm)\n'
                  f'4 LEDs at ±{angles_deg[0]}°, ±{angles_deg[2]}° | '
                  f'Viewing angle: {led_viewing_angle}° | Circle radius: {radius}cm')
        plt.xlabel('Y (cm)')
        plt.ylabel('Z (cm)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        output_file = 'lighting_uniformity.png'
        plt.savefig(output_file, dpi=150)
        print(f"Result saved to {output_file}")

    if renderer:
        print("\n" + "="*60)
        print("VISUALIZATION RUNNING")
        print("To view the 3D simulation, open the following URL:")
        print(f"{renderer.vis.url()}")
        print("="*60 + "\n")
        print("Press Ctrl+C to exit and stop the server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main()
