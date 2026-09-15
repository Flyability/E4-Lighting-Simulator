"""
Interactive lighting design tool (Viser front-end).

This module hosts the GUI application. All physics, geometry and analysis live
in the sibling packages (``domain``, ``camera``, ``raytracing``, ``simulation``,
``scene``, ``analysis``) and are imported here; nothing in this file should be
needed to run a headless simulation (see ``lighting_simulator.pipeline``).
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
else:
    # Force line-buffered output so print() messages appear immediately in terminals
    if not sys.stdout.line_buffering:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors='replace', line_buffering=True)
    if not sys.stderr.line_buffering:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors='replace', line_buffering=True)

import numpy as np
import warnings
import viser
import time
import json
import os
import copy
import hashlib
import trimesh
import webbrowser as _wb
import socket as _socket

from lighting_simulator.domain.led_factory import create_leds
from lighting_simulator.domain.geometry import as_vec3 as _as_vec3
from lighting_simulator.domain.optics import effective_lambertian_exponent as _get_effective_n
from lighting_simulator.domain.guides import (
    bake_and_disable_guide,
    circle_line_segments_m as _circle_line_segments_m,
    dynamic_group_world_geometry as _dynamic_group_world_geometry,
    enable_circular_guide,
    guide_is_enabled as _guide_is_enabled,
    restore_group_guide,
    serialize_guide,
)
from lighting_simulator.domain.mirroring import expand_mirror_configs
from lighting_simulator.camera.fov import (
    camera_fov_wall_trapezoid as _camera_fov_wall_trapezoid,
    fov_plane_mask_to_quads_and_contour,
    points_in_fisheye_fov,
    points_in_pinhole_fov,
    rasterize_fisheye_fov_on_plane,
    vio_hfov_vfov_deg,
    vio_optical_axis,
)
from lighting_simulator.raytracing.boxes import ray_box_intersection_batch as _ray_box_intersection_batch_np
from lighting_simulator.raytracing.mesh import (
    batch_ray_mesh_intersection as _batch_ray_mesh_intersection,
    prepare_mesh_ray_accelerator as _prepare_mesh_ray_accelerator,
    ray_mesh_intersection as _ray_mesh_intersection,
)
from lighting_simulator.simulation import (
    EmissionSettings,
    RoomSettings,
    WallSettings,
    room_wall_cell_centers,
    wall_grid_cell_centers_cm,
)
from lighting_simulator.simulation import wall as _wall_engine
from lighting_simulator.simulation import room as _room_engine
from lighting_simulator.simulation import gpu_backend as _gpu_backend
from lighting_simulator.ui import csv_overlay as _csv_overlay
from lighting_simulator.ui import optimize_tab as _optimize_tab
from lighting_simulator.ui import room_mode as _room_mode
from lighting_simulator.ui.mesh_lighting import _build_stl_transform, calculate_mesh_lighting
from types import SimpleNamespace as _SimpleNamespace
from lighting_simulator.ui import config_io as _config_io
from lighting_simulator.ui import panels as _panels
from lighting_simulator.analysis.uniformity import compute_uniformity_html as _compute_uniformity_html
from lighting_simulator.scene.absorbers import build_elios_absorbers, rotate_absorbers_z
from lighting_simulator.scene.builder import apply_diffuser, apply_global_transform
from lighting_simulator.scene.stl import (
    _rot4_x, _rot4_y, _rot4_z,
    global_z_rotation_4x4,
    stl_mesh_data as stl_mesh_data_payload,
    stl_transform,
)

# Suppress viser warnings about removing already-removed nodes
warnings.filterwarnings("ignore", message="Attempted to remove already removed node")


def _json_default(o):
    """json.dump fallback: configs built from GUI state may carry numpy scalars/arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main():
    # Create Viser server — bind to 0.0.0.0 so other computers on the LAN can connect
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    
    # Dark theme; panel placement is handled via main_panel.dock_right() below
    server.gui.configure_theme(dark_mode=True)
    
    # Get local IP for LAN access
    _local_ip = "127.0.0.1"
    try:
        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _local_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        pass
    
    # Try to add Windows Firewall rule for LAN access (requires admin, fails silently)
    try:
        import subprocess as _sp
        _sp.run(
            ['netsh', 'advfirewall', 'firewall', 'add', 'rule',
             'name=LightingSim', 'dir=in', 'action=allow',
             'protocol=TCP', 'localport=8080'],
            capture_output=True, timeout=5
        )
    except Exception:
        pass
    
    # Auto-open browser
    _wb.open("http://localhost:8080")
    
    print(f"\n  🖥️  Local:   http://localhost:8080")
    print(f"  🌐  Network: http://{_local_ip}:8080")
    print(f"\n  ⚠️  If network access does not work, run as Administrator")
    print(f"      or manually open port 8080 in Windows Firewall.")
    print("\n" + "="*60)
    print("  LED Lighting Simulation - Interactive Tool")
    print("="*60)
    print("\n📋 To get started:")
    print("  1. Create a 🆕 New Project (empty, add custom groups)")
    print("  2. Or 📂 Load an existing configuration (e.g., Elios 3)")
    print("\n💡 All LEDs are disabled until you load or create a project.")
    print("="*60 + "\n")

    # --- Configuration Management ---
    # Prefer the launch directory (so a bundled exe finds its data next to it);
    # otherwise fall back to the project root so it works from any cwd.
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = "configs" if os.path.isdir("configs") else os.path.join(_project_root, "configs")
    custom_groups_templates_dir = (
        "custom_groups_templates" if os.path.isdir("custom_groups_templates")
        else os.path.join(_project_root, "custom_groups_templates")
    )
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(custom_groups_templates_dir):
        os.makedirs(custom_groups_templates_dir)
    print(f"  📁  Configs: {os.path.abspath(config_dir)}")
    
    # Save the Elios 3 configuration if it doesn't exist
    elios3_config = {
        "name": "Elios 3",
        "description": "Standard configuration: front +/-, side +/-",
        "viewing_angle": 120,
        "radius": 35,
        "circle_center_x": -35,
        "group_rotations": [0.7, -0.7, 18, -18],
        "group_rotations_y": [0, 0, 0, 0],
        "group_offsets": [
            [0.0, 1.6, 0.0],
            [0.0, -1.6, 0.0],
            [-1.3, -33.1, 0.0],
            [-1.3, 33.1, 0.0]
        ],
        "row_enabled": [False, True, True, False],
        "led_states": [True] * 48,
        "custom_groups": []
    }
    elios3_path = os.path.join(config_dir, "elios3.json")
    if not os.path.exists(elios3_path):
        with open(elios3_path, "w") as f:
            json.dump(elios3_config, f, indent=4)

    # LED state array: 4 groups × 4 rows × 3 LEDs = 48 LEDs total
    # Start with all LEDs disabled (user must load a project or create new)
    led_states = [False] * 48
    
    # Flag to track if a project is loaded
    project_loaded = [False]  # Use list for mutability in nested functions
    current_config_name = [""]  # Track which configuration is loaded
    loading_in_progress = [False]  # Flag to prevent callbacks during config loading
    
    # Custom groups - list of dictionaries, each containing group configuration
    custom_groups = []  # Each group: {id, enable, pos_x, pos_y, pos_z, rot, led_states, buttons, folder}
    next_custom_group_id = [0]  # Counter for unique IDs (use list to allow modification in nested functions)
    
    # Individual LEDs - list of single LED configurations
    individual_leds = []  # Each LED: {id, enable, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, size, folder}
    next_individual_led_id = [0]  # Counter for unique IDs
    
    # Template folders - track loaded templates with master controls
    template_folders = []  # List of template folder handles to remove on new project/load
    
    # Panel Configurator state (populated later when UI is created)
    _panel_slot_data = [None, None, None, None]
    _panel_dropdowns = []
    _panel_mode_dropdowns = []

    # Live panel mirroring: the selected "primary" panel is reflected across
    # the XZ plane at scene-build time; its counterpart panel is disabled.
    _mirror_primary = [None]           # owner tuple ('slot', idx) / ('custom_group', id)
    _mirror_disabled_handles = []      # enable checkboxes we turned off (to restore)
    _mirror_counterpart_name = [None]  # display name of the disabled panel
    
    # Store button handles for LED control
    led_buttons = {}
    row_buttons = {}
    group_buttons = {}
    
    # Group colors (defined early for use in config functions)
    group_colors_hex = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33"]

    # --- UI layout: docked/floating panels + main-panel tabs (viser 1.1+) ---
    server.gui.set_panel_label("Controls")
    server.gui.main_panel.dock_right()
    server.gui.main_panel.set_width(400)

    main_tabs = server.gui.add_tab_group()
    tab_project = main_tabs.add_tab("Project")
    tab_panels = main_tabs.add_tab("Panels & LEDs")
    tab_fov = main_tabs.add_tab("FOV")
    tab_advanced = main_tabs.add_tab("Advanced")
    tab_optim = main_tabs.add_tab("Optimize")

    # Display / Global: full-height left dock. Intensity Map is *not* stacked
    # above this (Viser's dock_below split defaults to 50/50 and cannot be
    # weighted from Python), so these tabs get the whole column.
    left_panel = server.gui.add_panel()
    toolbar_tab = left_panel.add_tab("Display")
    global_tab = left_panel.add_tab("Global")
    left_panel.dock_left()
    left_panel.set_width(320)

    # Intensity Map + legend, floating at the top-left of the 3D canvas
    # (to the right of the docked Display/Global column). Auto-height wraps
    # to the controls and legend so there is no empty splitter pane.
    quick_panel = server.gui.add_panel()
    quick_tab = quick_panel.add_tab("Intensity Map")
    quick_panel.float(x=12, y=8, width=300)

    inspector_panel = server.gui.add_panel()
    inspector_tab = inspector_panel.add_tab("Selected")
    inspector_panel.float(x=-12, y=48)
    inspector_panel.set_width(340)

    selected_owner = [None]
    _just_clicked_mesh = [False]
    _inspector_handles = []
    _inspector_syncing = [False]
    _select_panel_impl = [lambda owner: None]

    def select_panel(owner):
        _select_panel_impl[0](owner)

    def save_custom_group_template(name, groups_list, individual_leds_list):
        """Save all custom groups and individual LEDs as a reusable template."""
        path = os.path.join(custom_groups_templates_dir, f"{name.lower().replace(' ', '_')}.json")
        template = {
            "name": name,
            "groups": groups_list,
            "individual_leds": individual_leds_list
        }
        with open(path, "w") as f:
            json.dump(template, f, indent=4, default=_json_default)
        print(f"✓ Template saved with {len(groups_list)} custom group(s) and {len(individual_leds_list)} individual LED(s): {name}")
    
    def get_available_templates():
        """Get list of available custom group templates."""
        if not os.path.exists(custom_groups_templates_dir):
            return []
        files = [f for f in os.listdir(custom_groups_templates_dir) if f.lower().endswith(".json")]
        return sorted((f[:-5] for f in files), key=str.lower)
    
    # --- GUI Controls ---
    with tab_project:
        _project_folder = server.gui.add_folder("Project Management")
    with _project_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Start a new project or load existing</div>")
        
        new_project_btn = server.gui.add_button("🆕 New Project (Empty)", color="#4CAF50")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        
        def get_available_configs():
            files = [f for f in os.listdir(config_dir) if f.lower().endswith(".json")]
            return sorted((f[:-5] for f in files), key=str.lower)

        config_dropdown = server.gui.add_dropdown(
            "Select Configuration",
            options=get_available_configs(),
            initial_value=None
        )

        load_config_btn = server.gui.add_button("📂 Load Configuration")
        refresh_configs_btn = server.gui.add_button("🔄 Refresh List")

        @refresh_configs_btn.on_click
        def _(_):
            current = config_dropdown.value
            config_dropdown.options = get_available_configs()
            if current in config_dropdown.options:
                config_dropdown.value = current
            print(f"✓ {len(config_dropdown.options)} configuration(s) in {os.path.abspath(config_dir)}")
        
        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;margin-bottom:6px;'>Save Current Project</div>")
        
        save_name_input = server.gui.add_text("Project Name", initial_value="")
        save_type_dropdown = server.gui.add_dropdown(
            "Save As",
            options=["Full Configuration", "Custom Group Template"],
            initial_value="Full Configuration"
        )
        save_project_btn = server.gui.add_button("💾 Save Project")

        @new_project_btn.on_click
        def _(_):
            new_project()

        @load_config_btn.on_click
        def _(_):
            name = config_dropdown.value
            if not name:
                print("Error: Please select a configuration to load.")
                return
            path = os.path.join(config_dir, f"{name}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    cfg = json.load(f)
                    num_custom = len(cfg.get("custom_groups", []))
                    print(f"Loading configuration: {name} ({num_custom} custom groups)")
                    current_config_name[0] = name
                    project_loaded[0] = True
                    apply_config(cfg)
                # Set the loaded config name in the save field for easy re-saving
                save_name_input.value = name
                print(f"✓ Configuration loaded: {name}")
                print(f"  💡 Modify and click 'Save Project' to update the configuration")

        @save_project_btn.on_click
        def _(_):
            nonlocal template_dropdown
            name = save_name_input.value.strip()
            if not name:
                print("Error: Please enter a project name.")
                return
            
            save_type = save_type_dropdown.value
            
            if save_type == "Full Configuration":
                # Save complete configuration
                cfg = get_current_config()
                cfg["name"] = name
                
                path = os.path.join(config_dir, f"{name.lower().replace(' ', '_')}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4, default=_json_default)
                
                print(f"✓ Configuration saved: {name}")
                # Refresh dropdown options
                config_dropdown.options = get_available_configs()
            else:
                # Save as custom group template - save groups separately
                if len(custom_groups) > 0 or len(individual_leds) > 0:
                    # Save custom groups
                    custom_groups_data = []
                    for group in custom_groups:
                        group_cfg = {
                            'enabled': group['enable'].value,
                            'position': [group['pos_x'].value, group['pos_y'].value, group['pos_z'].value],
                            'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                            'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                            'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                            'led_states': group['led_states'][:]
                        }
                        # Save dynamic group properties if present
                        if group.get('is_dynamic', False):
                            group_cfg['is_dynamic'] = True
                            group_cfg['num_leds'] = group.get('num_leds', 12)
                            # CRITICAL: Save ORIGINAL positions (not the current rotated ones!)
                            # This ensures template always contains undeformed geometry
                            group_cfg['led_positions'] = group.get('original_led_positions', group.get('led_positions', []))
                            group_cfg['led_rotations'] = group.get('original_led_rotations', group.get('led_rotations', []))
                            group_cfg['led_row_directions'] = group.get('original_led_row_directions', group.get('led_row_directions', []))
                            group_cfg['led_sizes'] = group.get('led_sizes', [])
                            group_cfg['led_viewing_angles'] = group.get('led_viewing_angles', [])
                            group_cfg['led_rows'] = group.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
                            group_cfg['led_euler_angles'] = group.get('led_euler_angles', [])
                            group_cfg['led_beam_tilts'] = group.get('led_beam_tilts', [])
                            group_cfg['led_lumens'] = group.get('led_lumens', [])
                        # Save lumens override for template
                        group_cfg['lumens_override_enabled'] = group.get('lumens_override') and group['lumens_override'].value
                        group_cfg['lumens_value'] = group['lumens_value'].value if group.get('lumens_value') else 100
                        if _guide_is_enabled(group):
                            group_cfg['guide'] = serialize_guide(group['guide'])
                        custom_groups_data.append(group_cfg)
                    
                    # Save individual LEDs
                    individual_leds_data = []
                    for led in individual_leds:
                        individual_leds_data.append({
                            'enabled': led['enable'].value,
                            'led_on': led['led_on'],
                            'pos_x': led['pos_x'].value,
                            'pos_y': led['pos_y'].value,
                            'pos_z': led['pos_z'].value,
                            'rot_x': led['rot_x'].value,
                            'rot_y': led['rot_y'].value,
                            'rot_z': led['rot_z'].value,
                            'size': led['size'].value,
                            'viewing_angle': led['viewing_angle'].value,
                            'square_roll': led['square_roll'].value,
                            'beam_tilt': led['beam_tilt'].value,
                            'lumens_override_enabled': led.get('lumens_override') and led['lumens_override'].value,
                            'lumens_value': led['lumens_value'].value if led.get('lumens_value') else 100,
                        })
                    
                    # Save as template with separate groups
                    save_custom_group_template(name, custom_groups_data, individual_leds_data)
                    # Refresh template dropdown list
                    _fresh = get_available_templates()
                    template_dropdown.options = ["Empty"] + _fresh
                    for _pdd in _panel_dropdowns:
                        _cur = _pdd.value
                        _pdd.options = ["-- Nessuno --"] + _fresh
                        _pdd.value = _cur if _cur in _pdd.options else "-- Nessuno --"
                    
                    print(f"✓ Template saved: {len(custom_groups_data)} group(s) + {len(individual_leds_data)} individual LED(s)")
                else:
                    print("Error: No custom groups or individual LEDs to save as template")

    # Store reference to LED Configuration folder and visibility state
    base_groups_active = [False]  # Track if base LED groups are active in current project
    
    with tab_panels:
        led_config_folder = server.gui.add_folder("LED Configuration (Base Groups)")
    led_config_folder.visible = False  # Hidden by default (empty project)
    
    with led_config_folder:
        viewing_angle_slider = server.gui.add_slider(
            "Viewing angle (°) [GWP9LR35: 120°]", min=10, max=130, step=5, initial_value=120
        )
        # Per-group rotation sliders Z axis (rotate beam and visual together)
        rot_front_pos = server.gui.add_slider("Rotate front+ Z (°)", min=-180, max=180, step=1, initial_value=0.7)
        rot_front_neg = server.gui.add_slider("Rotate front- Z (°)", min=-180, max=180, step=1, initial_value=-0.7)
        rot_side_pos = server.gui.add_slider("Rotate side+ Z (°)", min=-180, max=180, step=1, initial_value=18)
        rot_side_neg = server.gui.add_slider("Rotate side- Z (°)", min=-180, max=180, step=1, initial_value=-18)
        
        # Per-group rotation sliders Y axis (local tangent axis - tilts forward/backward)
        rot_y_front_pos = server.gui.add_slider("Rotate front+ local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_front_neg = server.gui.add_slider("Rotate front- local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_side_pos = server.gui.add_slider("Rotate side+ local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        rot_y_side_neg = server.gui.add_slider("Rotate side- local Y (tilt °)", min=-180, max=180, step=1, initial_value=0)
        
        # Per-group translation sliders (move entire group along X, Y, Z axes)
        with server.gui.add_folder("Group Positions"):
            # Front+ (Red group)
            with server.gui.add_folder("Front+ (Red)"):
                offset_front_pos_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
                offset_front_pos_y = server.gui.add_slider("Offset Y (cm)", min=-30, max=30, step=0.1, initial_value=1.6)
                offset_front_pos_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Front- (Green group)
            with server.gui.add_folder("Front- (Green)"):
                offset_front_neg_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
                offset_front_neg_y = server.gui.add_slider("Offset Y (cm)", min=-30, max=30, step=0.1, initial_value=-1.6)
                offset_front_neg_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Side+ (Blue group)
            with server.gui.add_folder("Side+ (Blue)"):
                offset_side_pos_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=-1.3)
                offset_side_pos_y = server.gui.add_slider("Offset Y (cm)", min=-40, max=40, step=0.1, initial_value=-33.1)
                offset_side_pos_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)
            # Side- (Yellow group)
            with server.gui.add_folder("Side- (Yellow)"):
                offset_side_neg_x = server.gui.add_slider("Offset X (cm)", min=-30, max=30, step=0.1, initial_value=-1.3)
                offset_side_neg_y = server.gui.add_slider("Offset Y (cm)", min=-40, max=50, step=0.1, initial_value=33.1)
                offset_side_neg_z = server.gui.add_slider("Offset Z (cm)", min=-30, max=30, step=0.1, initial_value=0.0)

    with global_tab:
        server.gui.add_markdown("**Geometry**")
        radius_slider = server.gui.add_slider(
            "Circle radius (cm)", min=10, max=60, step=1, initial_value=35
        )
        wall_dist_slider = server.gui.add_slider(
            "Wall distance (cm)", min=10, max=1500, step=5, initial_value=50
        )
        circle_center_slider = server.gui.add_slider(
            "Circle center X (cm)", min=-60, max=0, step=5, initial_value=-35
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Global Rotation:</b>")
        global_rotation_z_slider = server.gui.add_slider(
            "Rotate configuration (°)", min=-180, max=180, step=1, initial_value=0
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Global Position:</b>")
        global_pos_x_slider = server.gui.add_slider(
            "Global offset X (cm)", min=-100, max=100, step=0.5, initial_value=0
        )
        global_pos_y_slider = server.gui.add_slider(
            "Global offset Y (cm)", min=-100, max=100, step=0.5, initial_value=0
        )
        global_pos_z_slider = server.gui.add_slider(
            "Global offset Z (cm)", min=-100, max=100, step=0.5, initial_value=0
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Wall Settings:</b>")
        intensity_grid_size = server.gui.add_slider(
            "Wall grid resolution", min=5, max=1000 , step=5, initial_value=30
        )
        wall_view_size = server.gui.add_slider("Wall view size (cm)", min=100, max=2000, step=10, initial_value=250)
        bw_scale_chk = server.gui.add_checkbox("B/W Scale", initial_value=False)

    with quick_tab:
        update_intensity_button = server.gui.add_button("Update Intensity Map")
        show_intensity_map = server.gui.add_checkbox(
            "Show intensity on wall", initial_value=False
        )
        legend_max_input = server.gui.add_number("Legend max (lux)", initial_value=3500, min=1, max=100000, step=50)
        server.gui.add_html("<div style='color:#888;font-size:10px;margin-top:-4px;'>Fixed cap: colors scale 0–this value. If peak exceeds it, switches to AUTO.</div>")
        intensity_threshold_slider = server.gui.add_slider(
            "Black threshold (lux)", min=0, max=10000, step=10, initial_value=0
        )
        server.gui.add_html("<div style='color:#888;font-size:10px;margin-top:-4px;'>Cells below this lux are drawn pitch black (0 = off). Applies to wall and room maps after the next update.</div>")
        uniformity_percentile_slider = server.gui.add_slider(
            "Robust Eₘᵢₙ percentile (%)", min=0.0, max=10.0, step=0.5, initial_value=2.0,
            hint="Adds U0 = P(E)/Eavg to the legend (the optimiser's metric); 0 hides it. Legend updates instantly.",
        )
        cell_area_html = server.gui.add_html(
            "<div style='font-family: sans-serif; font-size: 11px; color: #666; margin-top: -8px; margin-bottom: 8px;'>"
            "Cell area: calculating..."
            "</div>"
        )
        legend_html = server.gui.add_html(
            "<div style='font-family: sans-serif;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
            "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click<br>'Update Intensity Map' to populate legend</div>"
            "</div>"
        )

    with toolbar_tab:
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
            "Show LED markers", initial_value=True
        ) 
        intensity_rays_slider = server.gui.add_slider(
            "Rays per pixel (↑quality, ↓speed)", min=10, max=500000, step=10, initial_value=10500
        )
        ray_uniformity_slider = server.gui.add_slider(
            "Focus factor (0=Standard, 1=3x focused)", min=0.0, max=1.0, step=0.05, initial_value=0.0
        )
        led_lumens_slider = server.gui.add_slider(
            "LED lumens (lm/LED)", min=10, max=1000, step=10, initial_value=168
        )
        
        server.gui.add_html("<hr style='margin:8px 0;'><b>Diffuser Lens:</b>")
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-bottom:4px;'>"
            "Simulates a diffuser lens in front of the LEDs: widens the beam and makes it more uniform. "
            "Typical transmission 85-95%.</div>"
        )
        diffuser_enable_chk = server.gui.add_checkbox(
            "Enable diffuser lens", initial_value=False
        )
        diffuser_angle_slider = server.gui.add_slider(
            "Diffuser output angle (°)", min=60, max=180, step=5, initial_value=170
        )
        diffuser_transmission_slider = server.gui.add_slider(
            "Diffuser transmission (%)", min=50, max=100, step=1, initial_value=90
        )
        calibration_factor_slider = server.gui.add_slider(
            "Calibration factor", min=0.5, max=1.5, step=0.001, initial_value=1.0
        )
        run_benchmark_button = server.gui.add_button("Run Benchmark (multi-distance)")
        export_lux_matrix_button = server.gui.add_button("Export Lux Matrix (±40cm)")

    # --- CSV Pattern Import (initially collapsed) ---
    with tab_advanced:
        _gpu_folder = server.gui.add_folder("🎮 GPU", expand_by_default=False)
    with _gpu_folder:
        gpu_status_html = server.gui.add_html(
            "<div style='font-size:11px;color:#888;'>Backend is initialised on the first ray trace.</div>"
        )
        gpu_purge_btn = server.gui.add_button("🧹 Purge GPU memory",
                                              hint="Release cached device buffers (Vulkan: resets the runtime)")

        @gpu_purge_btn.on_click
        def _(_):
            try:
                t0 = time.perf_counter()
                msg = _gpu_backend.purge_memory()
                gpu_status_html.content = (f"<div style='font-size:11px;color:#4CAF50;'>{msg} "
                                           f"({time.perf_counter() - t0:.1f}s)</div>")
            except Exception as exc:
                gpu_status_html.content = f"<div style='font-size:11px;color:#ff6666;'>Purge failed: {exc}</div>"
            print(f"[GPU] purge: {gpu_status_html.content}")

    with tab_advanced:
        _csv_folder = server.gui.add_folder("📊 Import CSV Pattern", expand_by_default=False)
    with _csv_folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-bottom:6px;'>"
            "Import a benchmark or FOV intensity CSV and overlay the measured pattern on the wall."
            "</div>"
        )
        csv_import_path = server.gui.add_text("CSV File Path", initial_value="")
        csv_import_btn = server.gui.add_button("📂 Import CSV", color="#4CAF50")
        csv_clear_btn = server.gui.add_button("🗑️ Clear Imported Pattern", color="#FF5555")
        csv_import_status = server.gui.add_html("<div style='font-size:11px;color:#888;'>No file imported</div>")
        csv_legend_max_input = server.gui.add_number("CSV Legend max (lux)", initial_value=523, min=1, max=100000, step=10)
        server.gui.add_html("<div style='color:#888;font-size:10px;margin-top:-4px;'>Fixed cap for CSV pattern legend. AUTO if peak exceeds it.</div>")
        csv_legend_html = server.gui.add_html("")
        csv_diff_html = server.gui.add_html("")

        # Reset button: some Viser button handles don't support on_update;
        # we'll detect clicks by polling `reset_button.value` in the main loop.
        reset_button = server.gui.add_button("Reset to original positions")
        # Per-row enable toggles
        row1_chk = server.gui.add_checkbox("Row 1 on", initial_value=False)
        row2_chk = server.gui.add_checkbox("Row 2 on", initial_value=True)
        row3_chk = server.gui.add_checkbox("Row 3 on", initial_value=True)
        row4_chk = server.gui.add_checkbox("Row 4 on", initial_value=False)
        # Absorber controls moved to dedicated folder for clarity

    # 3D Model Import (STL files)
    stl_mesh_handle = [None]  # Store mesh handle for removal/update
    stl_mesh_data = [None]  # Store loaded trimesh object

    with tab_advanced:
        _stl_folder = server.gui.add_folder("3D Models (STL)")
    with _stl_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Import 3D CAD models</div>")
        stl_file_path = server.gui.add_text("STL File Path", initial_value=r"C:\Users\gianmatteo.marietti_\Downloads\109045 E3 CAGE ASSEMBLY_Coarse.STL")
        stl_load_button = server.gui.add_button("📂 Load STL", color="#4CAF50")
        stl_clear_button = server.gui.add_button("🗑️ Clear Model", color="#FF5555")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        stl_absorber_enable = server.gui.add_checkbox("Enable as Light Absorber", initial_value=True)
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>When enabled, the 3D model blocks light rays</div>")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        stl_visible = server.gui.add_checkbox("Show Model", initial_value=True)
        stl_scale = server.gui.add_slider("Scale", min=0.01, max=10.0, step=0.01, initial_value=0.1)
        stl_pos_x = server.gui.add_slider("Position X (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_pos_y = server.gui.add_slider("Position Y (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_pos_z = server.gui.add_slider("Position Z (cm)", min=-200, max=200, step=1, initial_value=0)
        stl_rot_x = server.gui.add_slider("Rotation X (°)", min=-180, max=180, step=1, initial_value=0)
        stl_rot_y = server.gui.add_slider("Rotation Y (°)", min=-180, max=180, step=1, initial_value=0)
        stl_rot_z = server.gui.add_slider("Rotation Z (°)", min=-180, max=180, step=1, initial_value=0)
        stl_opacity = server.gui.add_slider("Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.8)
        stl_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        update_mesh_lighting_btn = server.gui.add_button("🔆 Update Mesh Lighting", color="#FFA500")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Recalculate lighting based on current LED configuration</div>")
        
# Mesh lighting legend (gradient from dark blue to white)
        server.gui.add_html(
            "<div style='font-family: sans-serif; margin-top: 12px;'>" 
            "<div style='font-weight:600; margin-bottom:6px; font-size:12px;'>Mesh Lighting Intensity:</div>"
            "<div style='display:flex; align-items:center; gap:8px;'>"
            "<span style='font-size:10px; color:#888;'>Dark</span>"
            "<div style='flex:1; height:20px; background:linear-gradient(to right, "
            "#000033 0%, #000055 10%, #0000AA 20%, #1133CC 30%, #2255DD 40%, "
            "#3366EE 50%, #5588FF 60%, #77AAFF 65%, #99CCFF 75%, #BBDDFF 85%, #DDEEFF 92%, #FFFFFF 100%); "
            "border:1px solid #444; border-radius:3px;'></div>"
            "<span style='font-size:10px; color:#888;'>Bright</span>"
            "</div>"
            "<div style='font-size:10px; color:#666; margin-top:4px;'>"
            "• Color map: Dark Blue → White<br>"
            "• Physics: Lux = (Lumens × cos θ) / d² | Responds to LED lumens"
            "</div>"
            "</div>"
        )
        
        stl_info_html = server.gui.add_html(
            "<div style='font-family: sans-serif; font-size: 11px; color: #666;'>"
            "No model loaded"
            "</div>"
        )
        
        def _get_stl_cache_path(file_path):
            """Get cache file path for an STL file based on its content hash."""
            # Use file path + modification time as cache key for speed
            stat = os.stat(file_path)
            cache_key = f"{file_path}|{stat.st_size}|{stat.st_mtime_ns}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(os.path.dirname(file_path), '.stl_cache')
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, f"{cache_hash}.npz")

        def load_stl_file():
            """Load STL file and display in scene. Uses numpy binary cache for fast reloads."""
            file_path = stl_file_path.value.strip()
            if not file_path:
                print("⚠️ Please enter a file path")
                return
            
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return
            
            try:
                t_start = time.perf_counter()
                cache_path = _get_stl_cache_path(file_path)
                
                # Try loading from numpy binary cache first (10-50x faster)
                if os.path.exists(cache_path):
                    print(f"Loading STL from cache: {os.path.basename(file_path)}")
                    cached = np.load(cache_path)
                    mesh = trimesh.Trimesh(
                        vertices=cached['vertices'],
                        faces=cached['faces'],
                        vertex_normals=cached['vertex_normals'],
                        process=False  # Skip expensive validation since we know data is good
                    )
                    t_load = time.perf_counter()
                    print(f"  Cache loaded in {t_load - t_start:.2f}s")
                else:
                    print(f"Loading STL file: {file_path} (first load, will cache)")
                    mesh = trimesh.load(file_path, force='mesh', process=False)
                    
                    # Handle multiple meshes (Scene object)
                    if isinstance(mesh, trimesh.Scene):
                        mesh = trimesh.util.concatenate(
                            [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                        )
                    
                    # Center mesh at origin (move centroid to 0,0,0)
                    mesh.vertices -= mesh.centroid
                    
                    t_load = time.perf_counter()
                    print(f"  STL parsed in {t_load - t_start:.2f}s")
                    
                    # Save to numpy binary cache for fast future loads
                    # np.savez (uncompressed) is much faster than np.savez_compressed
                    try:
                        np.savez(
                            cache_path,
                            vertices=mesh.vertices.astype(np.float32),
                            faces=mesh.faces,
                            vertex_normals=mesh.vertex_normals.astype(np.float32)
                        )
                        print(f"  Cache saved: {os.path.basename(cache_path)}")
                    except Exception as ce:
                        print(f"  Warning: could not save cache: {ce}")
                
                stl_mesh_data[0] = mesh
                
                # Calculate mesh dimensions
                num_vertices = len(mesh.vertices)
                num_faces = len(mesh.faces)
                bounds = mesh.bounds
                size = bounds[1] - bounds[0]
                
                # Auto-calculate ideal scale
                target_size_cm = 70.0
                max_dimension = np.max(size)
                ideal_scale = 1.0
                
                if max_dimension > 0:
                    ideal_scale = target_size_cm / max_dimension
                    stl_scale.value = ideal_scale
                
                # Update info
                info_text = (
                    f"<div style='font-family: sans-serif; font-size: 11px; color: #4CAF50;'>"
                    f"✓ Model loaded<br>"
                    f"Vertices: {num_vertices:,}<br>"
                    f"Faces: {num_faces:,}<br>"
                    f"Original size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}<br>"
                    f"Scaled size: {size[0]*ideal_scale:.1f} × {size[1]*ideal_scale:.1f} × {size[2]*ideal_scale:.1f} cm"
                    f"</div>"
                )
                stl_info_html.content = info_text
                
                t_total = time.perf_counter() - t_start
                print(f"✓ STL loaded: {num_vertices:,} vertices, {num_faces:,} faces ({t_total:.2f}s total)")
                update_stl_mesh(skip_lighting=True)
                
            except Exception as e:
                print(f"❌ Error loading STL: {e}")
                import traceback; traceback.print_exc()
                stl_info_html.content = f"<div style='color:#FF5555;'>Error: {str(e)}</div>"
        
        def update_stl_mesh(skip_lighting=False):
            """Update STL mesh visualization in scene.
            
            Args:
                skip_lighting: If True, use uniform base color instead of computing
                    per-vertex lighting. Much faster for initial load.
            """
            nonlocal stl_mesh_handle
            
            try:
                t_start = time.perf_counter()
                
                # Remove existing mesh
                if stl_mesh_handle[0] is not None:
                    try:
                        stl_mesh_handle[0].remove()
                    except:
                        pass
                    stl_mesh_handle[0] = None
                
                # Add mesh if loaded and visible
                if stl_mesh_data[0] is not None and stl_visible.value:
                    orig_mesh = stl_mesh_data[0]
                    
                    # Validate values (protect against NaN)
                    scale = float(stl_scale.value)
                    if not np.isfinite(scale) or scale <= 0:
                        scale = 1.0
                    
                    pos_x = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
                    pos_y = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
                    pos_z = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
                    rot_x = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
                    rot_y = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0
                    rot_z = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
                    opacity = float(stl_opacity.value) if np.isfinite(float(stl_opacity.value)) else 0.8
                    
                    # Update info display
                    bounds = orig_mesh.bounds
                    orig_size = bounds[1] - bounds[0]
                    scaled_size = orig_size * scale
                    num_vertices = len(orig_mesh.vertices)
                    num_faces = len(orig_mesh.faces)
                    info_text = (
                        f"<div style='font-family: sans-serif; font-size: 11px; color: #4CAF50;'>"
                        f"✓ Model loaded<br>"
                        f"Vertices: {num_vertices:,}<br>"
                        f"Faces: {num_faces:,}<br>"
                        f"Original size: {orig_size[0]:.1f} × {orig_size[1]:.1f} × {orig_size[2]:.1f}<br>"
                        f"Scaled size: {scaled_size[0]:.1f} × {scaled_size[1]:.1f} × {scaled_size[2]:.1f} cm"
                        f"</div>"
                    )
                    stl_info_html.content = info_text
                    
                    # --- Build combined 4x4 transform (no mesh.copy() needed) ---
                    orig_vertices = orig_mesh.vertices  # Direct reference, no copy
                    orig_normals = orig_mesh.vertex_normals
                    orig_faces = orig_mesh.faces
                    
                    # Build rotation matrix using pure numpy helpers (much faster than trimesh)
                    T_rot = np.eye(4)
                    if rot_x != 0:
                        T_rot = T_rot @ _rot4_x(np.radians(rot_x))
                    if rot_y != 0:
                        T_rot = T_rot @ _rot4_y(np.radians(rot_y))
                    if rot_z != 0:
                        T_rot = T_rot @ _rot4_z(np.radians(rot_z))
                    
                    # Apply global Z rotation on top of STL-local rotation
                    g_rot_deg = global_rotation_z_slider.value
                    if abs(g_rot_deg) > 0.01:
                        T_rot = _rot4_z(np.radians(g_rot_deg)) @ T_rot

                    # Combined transform: Scale -> Rotate -> Translate -> cm-to-meters
                    # Instead of building 4 separate matrices and multiplying, build directly
                    R = T_rot[:3, :3]
                    scale_m = scale * 0.01  # scale * cm_to_meters
                    
                    # Transform vertices: v' = (R * scale_m) @ v + translate_m
                    RS = R * scale_m  # 3x3 scaled rotation
                    # Also rotate the translation vector by global rotation
                    translate_cm = np.array([pos_x, pos_y, pos_z])
                    if abs(g_rot_deg) > 0.01:
                        cg2, sg2 = np.cos(np.radians(g_rot_deg)), np.sin(np.radians(g_rot_deg))
                        translate_cm = np.array([cg2*pos_x - sg2*pos_y, sg2*pos_x + cg2*pos_y, pos_z])
                    translate_m = translate_cm * 0.01  # cm to meters
                    
                    vertices_transformed = (orig_vertices @ RS.T) + translate_m
                    vertices_transformed = vertices_transformed.astype(np.float32)
                    
                    # --- Compute vertex colors & render ---
                    if skip_lighting:
                        # FAST PATH: use add_mesh_simple (no Trimesh/GLB overhead, no normal transform)
                        stl_mesh_handle[0] = server.scene.add_mesh_simple(
                            name="/stl_model",
                            vertices=vertices_transformed,
                            faces=orig_faces.astype(np.uint32),
                            color=(0, 0, 51),  # dark blue (0.0, 0.0, 0.2) * 255
                            opacity=opacity if opacity < 1.0 else None,
                            flat_shading=False,
                            side="double",
                            wireframe=stl_wireframe.value,
                            visible=True,
                        )
                    else:
                        # Transform normals (rotation only, then normalize) — only needed for lighting
                        normals_transformed = orig_normals @ R.T
                        norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
                        norms[norms < 1e-10] = 1.0
                        normals_transformed = (normals_transformed / norms).astype(np.float32)

                        # Full lighting calculation — need per-vertex colors → Trimesh path
                        vertex_colors = calculate_mesh_lighting(
                            vertices_transformed, 
                            normals_transformed, 
                            current_leds,
                            base_color=(0.7, 0.7, 0.9),
                            led_lumens=led_lumens_slider.value
                        )
                        vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)
                    
                        # Add alpha channel
                        alpha_value = int(opacity * 255)
                        vertex_colors_rgba = np.concatenate([
                            vertex_colors_uint8,
                            np.full((num_vertices, 1), alpha_value, dtype=np.uint8)
                        ], axis=1)
                    
                        # Build trimesh with transformed data (process=False skips expensive validation)
                        mesh_transformed = trimesh.Trimesh(
                            vertices=vertices_transformed,
                            faces=orig_faces,
                            vertex_normals=normals_transformed,
                            process=False
                        )
                        from trimesh.visual import ColorVisuals
                        mesh_transformed.visual = ColorVisuals(mesh=mesh_transformed, vertex_colors=vertex_colors_rgba)
                    
                        # Add to scene
                        stl_mesh_handle[0] = server.scene.add_mesh_trimesh(
                            name="/stl_model",
                            mesh=mesh_transformed,
                            visible=True,
                        )
                    
                    t_total = time.perf_counter() - t_start
                    if t_total > 0.5:
                        print(f"  Mesh update: {t_total:.2f}s {'(no lighting)' if skip_lighting else '(with lighting)'}")
            except Exception as e:
                print(f"Error updating STL mesh: {e}")
                import traceback; traceback.print_exc()
        
        def clear_stl_model():
            """Clear loaded STL model."""
            nonlocal stl_mesh_handle
            if stl_mesh_handle[0] is not None:
                try:
                    stl_mesh_handle[0].remove()
                except:
                    pass
                stl_mesh_handle[0] = None
            stl_mesh_data[0] = None
            stl_info_html.content = "<div style='color:#666;'>No model loaded</div>"
            print("STL model cleared")
        
        # Button callbacks
        @stl_load_button.on_click
        def _(_):
            load_stl_file()
        
        @stl_clear_button.on_click
        def _(_):
            clear_stl_model()
        
        @update_mesh_lighting_btn.on_click
        def _(_):
            try:
                if stl_mesh_data[0] is not None:
                    print("Calculating mesh lighting...")
                    update_stl_mesh(skip_lighting=False)
                    print("✓ Mesh lighting updated")
                else:
                    print("⚠️ No mesh loaded")
            except Exception as e:
                print(f"Error updating mesh lighting: {e}")
        
        # Update mesh when parameters change (geometry only, no lighting)
        def safe_update_stl(_):
            try:
                update_stl_mesh(skip_lighting=True)
                # Update ray visualization with new STL mesh
                update_scene()
                # Note: Intensity calculations remain manual (use update buttons)
            except Exception as e:
                print(f"Error in STL update callback: {e}")
        
        def safe_update_stl_absorber(_):
            """Update when STL absorber enable/disable changes - affects ray blocking."""
            try:
                # Update ray visualization with new STL absorber state
                update_scene()
                # Note: Intensity calculations remain manual (use update buttons)
            except Exception as e:
                print(f"Error in STL absorber update callback: {e}")
        
        stl_visible.on_update(safe_update_stl)
        stl_scale.on_update(safe_update_stl)
        stl_pos_x.on_update(safe_update_stl)
        stl_pos_y.on_update(safe_update_stl)
        stl_pos_z.on_update(safe_update_stl)
        stl_rot_x.on_update(safe_update_stl)
        stl_rot_y.on_update(safe_update_stl)
        stl_rot_z.on_update(safe_update_stl)
        stl_opacity.on_update(safe_update_stl)
        stl_wireframe.on_update(safe_update_stl)
        stl_absorber_enable.on_update(safe_update_stl_absorber)

    # Room Mode - Cubic room with 5 or 6 walls (back wall optional)
    with tab_advanced:
        _room_folder = server.gui.add_folder("Room Mode")
    with _room_folder:
        room_mode_enable = server.gui.add_checkbox("Enable Room Mode", initial_value=False)
        show_room_walls = server.gui.add_checkbox("Show Room Walls", initial_value=True)
        show_room_intensity = server.gui.add_checkbox("Show Room Intensity", initial_value=False)
        room_front_dist = server.gui.add_slider(
            "Front wall distance (cm)", min=20, max=200, step=10, initial_value=200
        )
        room_side_dist = server.gui.add_slider(
            "Side walls distance (cm)", min=20, max=300, step=10, initial_value=200
        )
        room_top_bottom_dist = server.gui.add_slider(
            "Top/Bottom walls distance (cm)", min=20, max=300, step=10, initial_value=200
        )
        show_back_wall = server.gui.add_checkbox("Show Back Wall", initial_value=False)
        room_back_dist = server.gui.add_slider(
            "Back wall distance (cm)", min=10, max=100, step=5, initial_value=50
        )
        room_grid_size = server.gui.add_slider(
            "Room walls grid resolution", min=10, max=50, step=5, initial_value=20
        )
        update_room_button = server.gui.add_button("Update Room Intensity")
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Wall Reflections</div>")
        reflections_enable = server.gui.add_checkbox("Enable Reflections", initial_value=False)
        wall_material_dropdown = server.gui.add_dropdown(
            "Wall Material",
            options=["White Paint (\u03c1=0.85)", "Light Gray (\u03c1=0.65)", "Concrete (\u03c1=0.30)",
                     "Wood (\u03c1=0.45)", "Brick (\u03c1=0.25)", "Dark Paint (\u03c1=0.15)", "Custom"],
            initial_value="White Paint (\u03c1=0.85)"
        )
        custom_reflectance_slider = server.gui.add_slider(
            "Reflectance (\u03c1)", min=0.0, max=0.99, step=0.01, initial_value=0.85
        )
        max_bounces_slider_room = server.gui.add_slider(
            "Max Bounces", min=1, max=10, step=1, initial_value=3
        )
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-top:-4px;'>More bounces = more accurate but slower.<br>"
            "Reflected flux per bounce: \u03a6 \u00d7 \u03c1<sup>n</sup></div>"
        )
        _MATERIAL_REFLECTANCE = {
            "White Paint (\u03c1=0.85)": 0.85,
            "Light Gray (\u03c1=0.65)": 0.65,
            "Concrete (\u03c1=0.30)": 0.30,
            "Wood (\u03c1=0.45)": 0.45,
            "Brick (\u03c1=0.25)": 0.25,
            "Dark Paint (\u03c1=0.15)": 0.15,
        }
        @wall_material_dropdown.on_update
        def _on_material_change(_):
            mat = wall_material_dropdown.value
            if mat in _MATERIAL_REFLECTANCE:
                custom_reflectance_slider.value = _MATERIAL_REFLECTANCE[mat]

    # Camera FOV visualization
    with tab_fov:
        _cam_folder = server.gui.add_folder("Camera FOV")
    with _cam_folder:
        show_camera_fov = server.gui.add_checkbox("Show Camera FOV", initial_value=True)
        camera_fov_h = server.gui.add_slider(
            "Horizontal FOV (°)", min=10, max=120, step=1, initial_value=75
        )
        camera_fov_v = server.gui.add_slider(
            "Vertical FOV (°)", min=10, max=120, step=1, initial_value=60
        )
        camera_pos_x = server.gui.add_slider(
            "Camera X pos (cm)", min=-100, max=100, step=1, initial_value=10
        )
        camera_pos_y = server.gui.add_slider(
            "Camera Y pos (cm)", min=-100, max=100, step=1, initial_value=-4
        )
        camera_pitch = server.gui.add_slider(
            "Camera pitch (°)", min=-60, max=60, step=1, initial_value=0
        )
        capture_fov_btn = server.gui.add_button("Capture FOV Image", color="green")

    # VIO cameras (2× VD66GY equidistant fisheye)
    with tab_fov:
        _vio_folder = server.gui.add_folder("VIO Cameras")
    with _vio_folder:
        show_vio_fov = server.gui.add_checkbox("Show VIO FOV", initial_value=True)
        vio_fill_fov = server.gui.add_checkbox("Fill VIO FOV", initial_value=False)
        vio_pos_x = server.gui.add_slider(
            "VIO X pos (cm)", min=-100, max=100, step=0.5, initial_value=10.0
        )
        vio_pos_y = server.gui.add_slider(
            "VIO Y pos (cm)", min=-50, max=50, step=0.5, initial_value=0.0
        )
        vio_pos_z = server.gui.add_slider(
            "VIO Z pos (cm)", min=-50, max=50, step=0.5, initial_value=0.0
        )
        vio_cam1_pitch = server.gui.add_slider(
            "Cam 1 pitch (°)", min=-90, max=90, step=1, initial_value=45
        )
        vio_cam1_yaw = server.gui.add_slider(
            "Cam 1 yaw (°)", min=-180, max=180, step=1, initial_value=0
        )
        vio_cam2_pitch = server.gui.add_slider(
            "Cam 2 pitch (°)", min=-90, max=90, step=1, initial_value=-45
        )
        vio_cam2_yaw = server.gui.add_slider(
            "Cam 2 yaw (°)", min=-180, max=180, step=1, initial_value=0
        )
        vio_long_fov = server.gui.add_slider(
            "Long-side FOV (°)", min=60, max=180, step=1, initial_value=170
        )
        vio_landscape = server.gui.add_checkbox(
            "Landscape mount (long side horizontal)", initial_value=True
        )
        _hfov0, _vfov0 = vio_hfov_vfov_deg(170, True)
        vio_derived_fov_html = server.gui.add_html(
            f"<div style='color:#ccc;font-size:12px;'>HFOV = {_hfov0:.1f}° &nbsp; VFOV = {_vfov0:.1f}°"
            f"<br>(VD66GY 1124×1364, equidistant fisheye)</div>"
        )
        server.gui.add_html(
            "<div style='font-size:12px;margin-top:4px;'>"
            "<span style='color:#ff00ff;'>● Cam 1 (up)</span> &nbsp; "
            "<span style='color:#00ffff;'>● Cam 2 (down)</span>"
            "<br><span style='color:#888;'>Outline always shown. Enable Fill VIO FOV for semi-transparent coverage (hides intensity under the fill).</span>"
            "</div>"
        )

        def _refresh_vio_fov_label(_=None):
            hfov, vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
            vio_derived_fov_html.content = (
                f"<div style='color:#ccc;font-size:12px;'>HFOV = {hfov:.1f}° &nbsp; VFOV = {vfov:.1f}°"
                f"<br>(VD66GY 1124×1364, equidistant fisheye)</div>"
            )

    # Store handles for dynamic objects
    camera_fov_handles = []
    vio_fov_handles = []
    led_handles = []
    ray_handles = []
    guide_handles = []
    intensity_handles = []
    room_intensity_handles = []
    room_wall_handles = []
    absorber_handles = []
    imported_csv_handles = []
    
    # Store current LED objects (for reuse in room intensity calculation)
    current_leds = []

    # Absorbers folder (separate group for easier access)
    with tab_advanced:
        absorbers_folder = server.gui.add_folder("Absorbers")
    absorbers_folder.visible = False  # Hidden by default
    
    with absorbers_folder:
        absorbers_enable = server.gui.add_checkbox("Enable absorbers", initial_value=False)
        abs0_off_x = server.gui.add_slider("Abs0 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1)
        abs0_off_y = server.gui.add_slider("Abs0 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=2.5)
        abs0_off_z = server.gui.add_slider("Abs0 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs1_off_x = server.gui.add_slider("Abs1 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1)
        abs1_off_y = server.gui.add_slider("Abs1 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=-2.5)
        abs1_off_z = server.gui.add_slider("Abs1 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs2_off_x = server.gui.add_slider("Abs2 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1.8)
        abs2_off_y = server.gui.add_slider("Abs2 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=-10.5)
        abs2_off_z = server.gui.add_slider("Abs2 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs2_rot_z = server.gui.add_slider("Abs2 rotation Z (deg)", min=-180, max=180, step=1, initial_value=-14)
        abs3_off_x = server.gui.add_slider("Abs3 offset X (cm)", min=-50, max=200, step=0.1, initial_value=-1.8)
        abs3_off_y = server.gui.add_slider("Abs3 offset Y (cm)", min=-50, max=50, step=0.1, initial_value=10.5)
        abs3_off_z = server.gui.add_slider("Abs3 offset Z (cm)", min=-50, max=50, step=0.1, initial_value=0.0)
        abs3_rot_z = server.gui.add_slider("Abs3 rotation Z (deg)", min=-180, max=180, step=1, initial_value=14)

    # Custom LED Groups folder (dynamic groups)
    with tab_panels:
        custom_groups_folder = server.gui.add_folder("Custom LED Groups")
        server.gui.add_markdown("Click a panel or LED group in the 3D view to edit its parameters.")
        open_panel_designer_btn = server.gui.add_button("Create New Panel", color="green")
    template_dropdown = None  # Will be initialized later

    # Panel designer state is deliberately separate from scene groups.  This
    # lets Cancel leave the currently placed panel completely untouched.
    designer_mode = [False]
    designer_state = [{
        'name': 'Untitled Panel',
        'editing_owner': None,
        'leds': [],
        'selected_led': None,
    }]
    designer_ui_handles = []
    designer_scene_handles = []
    designer_led_nodes = []  # [{box, frame}, ...] persistent LED meshes
    designer_gizmo = [None]
    designer_widget_refs = [{}]  # field -> (slider, number) for gizmo→UI sync
    designer_syncing = [False]
    static_scene_handles = []  # /grid + /axes (filled when static scene is built)
    
    with custom_groups_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Add New Custom Group</div>")
        
        template_dropdown = server.gui.add_dropdown(
            "From Template",
            options=["Empty"] + get_available_templates(),
            initial_value="Empty"
        )
        
        load_mode_dropdown = server.gui.add_dropdown(
            "Load Mode",
            options=["As Group (Solid)", "As Individual LEDs (Editable)"],
            initial_value="As Group (Solid)"
        )
        
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>"
                           "• Group: Fast, moves as one unit<br>"
                           "• Individual LEDs: Edit each LED position/rotation/size separately</div>")
        
        add_custom_group_btn = server.gui.add_button("➕ Add Custom Group", color="green")
        
        @add_custom_group_btn.on_click
        def _(_):
            selected_template = template_dropdown.value
            load_mode = load_mode_dropdown.value
            
            if selected_template == "Empty":
                # Create empty custom group
                create_custom_group()
                print("✓ Empty custom group added")
            else:
                # Load from template
                if load_mode == "As Individual LEDs (Editable)":
                    # Load template as individual LEDs
                    load_template_as_individual_leds(selected_template)
                else:
                    # Load as solid group (default behavior)
                    load_custom_group_from_template(selected_template)
                # Refresh template list in case new templates were added
                _fresh = get_available_templates()
                template_dropdown.options = ["Empty"] + _fresh
                for _pdd in _panel_dropdowns:
                    _cur = _pdd.value
                    _pdd.options = ["-- Nessuno --"] + _fresh
                    _pdd.value = _cur if _cur in _pdd.options else "-- Nessuno --"

    # =====================================================================
    #  PANEL CONFIGURATOR  – 4 Elios 3 slots with per-slot template choice
    # =====================================================================
    # Pre-computed Elios3 slot data (centroid + outward Z-rotation angle)
    _ELIOS3_SLOTS = [
        {"name": "Front +",  "centroid": [18.06, -7.92, -1.59], "angle_deg": -23.7, "config_rot": 0.7,  "config_roll":  0,  "config_lr":  10},
        {"name": "Front -",  "centroid": [18.03,  7.97, -1.59], "angle_deg":  23.8, "config_rot": -0.7, "config_roll": -3,  "config_lr":  -6},
        {"name": "Side -",   "centroid": [16.07, 12.57, -0.83], "angle_deg":  38.0, "config_rot": -18,  "config_roll":  0,  "config_lr":  21},
        {"name": "Side +",   "centroid": [16.07, -12.63, -0.82], "angle_deg": -38.2, "config_rot": 18,  "config_roll":  0,  "config_lr": -21},
    ]

    def _Rz_matrix(deg):
        """Build a 3×3 rotation matrix around Z axis."""
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # ── Panel mirroring (XZ plane) ──────────────────────────────────────
    # The selected "primary" panel gets a live mirrored shadow generated at
    # scene-build time (_expand_mirror_configs); the panel that used to sit
    # on the opposite side is simply disabled.

    def _owner_groups(owner):
        """Custom-group dicts belonging to a panel owner tuple."""
        if owner is None:
            return []
        kind, key = owner
        if kind == 'slot':
            return [g for g in custom_groups if g.get('panel_slot') == key]
        if kind == 'custom_group':
            return [g for g in custom_groups
                    if g.get('id') == key and g.get('panel_slot') is None]
        return []

    def _owner_individual_leds(owner):
        """Individual-LED dicts belonging to a slot owner (individual mode)."""
        if owner is None or owner[0] != 'slot':
            return []
        key = owner[1]
        data = _panel_slot_data[key] if 0 <= key < len(_panel_slot_data) else None
        if data and data.get('individual_leds'):
            return list(data['individual_leds'])
        return [led for led in individual_leds if led.get('panel_slot') == key]

    def _owner_display_name(owner):
        kind, key = owner
        if kind == 'slot':
            if 0 <= key < len(_ELIOS3_SLOTS):
                return _ELIOS3_SLOTS[key]['name']
            return f"Slot {key + 1}"
        for g in custom_groups:
            if g.get('id') == key:
                return g.get('name') or f"Group {key}"
        return f"Group {key}"

    def _owner_mean_y(owner):
        ys = []
        for g in _owner_groups(owner):
            try:
                ys.append(float(g['pos_y'].value))
            except Exception:
                pass
        for led in _owner_individual_leds(owner):
            try:
                ys.append(float(led['pos_y'].value))
            except Exception:
                pass
        return float(np.mean(ys)) if ys else None

    def _mirror_candidate_owners():
        owners = []
        for si in range(len(_ELIOS3_SLOTS)):
            o = ('slot', si)
            if _owner_groups(o) or _owner_individual_leds(o):
                owners.append(o)
        for g in custom_groups:
            if g.get('panel_slot') is None:
                owners.append(('custom_group', g.get('id')))
        return owners

    def _find_mirror_counterpart(owner):
        """Panel on the opposite side of the XZ plane (closest Y match)."""
        yp = _owner_mean_y(owner)
        if yp is None or abs(yp) < 0.1:
            return None
        best, best_err = None, None
        for cand in _mirror_candidate_owners():
            if cand == owner:
                continue
            yc = _owner_mean_y(cand)
            if yc is None or yc * yp >= 0:
                continue
            err = abs(yc + yp)
            if best_err is None or err < best_err:
                best, best_err = cand, err
        return best

    def _clear_mirror_state():
        """Turn mirroring off and re-enable whatever panel it had disabled."""
        prev_loading = loading_in_progress[0]
        loading_in_progress[0] = True
        try:
            for h in _mirror_disabled_handles:
                try:
                    h.value = True
                except Exception:
                    pass
        finally:
            loading_in_progress[0] = prev_loading
        _mirror_disabled_handles.clear()
        _mirror_primary[0] = None
        _mirror_counterpart_name[0] = None

    def _enable_mirror_for(owner):
        """Make `owner` the mirror primary and disable its counterpart panel.

        Returns the counterpart owner, or None if no opposite-side panel was
        found (mirroring is still enabled in that case)."""
        _clear_mirror_state()
        counterpart = _find_mirror_counterpart(owner)
        if counterpart is not None:
            handles = [g.get('enable') for g in _owner_groups(counterpart)]
            handles += [l.get('enable') for l in _owner_individual_leds(counterpart)]
            if counterpart[0] == 'slot':
                data = _panel_slot_data[counterpart[1]]
                if data:
                    ctl = (data.get('controls') or {}).get('enable')
                    if ctl is not None:
                        handles.append(ctl)
            prev_loading = loading_in_progress[0]
            loading_in_progress[0] = True
            try:
                for h in handles:
                    if h is None:
                        continue
                    try:
                        if h.value:
                            h.value = False
                            _mirror_disabled_handles.append(h)
                    except Exception:
                        pass
            finally:
                loading_in_progress[0] = prev_loading
            _mirror_counterpart_name[0] = _owner_display_name(counterpart)
        _mirror_primary[0] = owner
        return counterpart

    def _expand_mirror_configs(custom_groups_configs, individual_leds_configs):
        """Append XZ-mirrored copies of the mirror-primary panel's configs."""
        expand_mirror_configs(custom_groups_configs, individual_leds_configs, _mirror_primary[0])

    # --- Panel Configurator UI ---
    with tab_panels:
        panel_config_folder = server.gui.add_folder("Panel Configurator (Elios 3 Slots)", expand_by_default=False)

    # _panel_dropdowns / _panel_mode_dropdowns are created (empty) near the top of main()
    _panel_load_btns = []
    _panel_clear_btns = []

    with panel_config_folder:
        server.gui.add_html(
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>"
            "Assign a template to each of the 4 Elios 3 panel positions.<br>"
            "Positions and rotations are pre-set to match the drone geometry.<br>"
            "Tip: click a panel in 3D and use 'Mirror to other side' in the Selected tab.</div>"
        )

        for _si, _slot_info in enumerate(_ELIOS3_SLOTS):
            with server.gui.add_folder(f"📍 {_slot_info['name']}"):
                _tpl_dd = server.gui.add_dropdown(
                    "Template",
                    options=["-- Nessuno --"] + get_available_templates(),
                    initial_value="-- Nessuno --",
                )
                _mode_dd = server.gui.add_dropdown(
                    "Mode",
                    options=["Solid (Group)", "Individual LEDs"],
                    initial_value="Solid (Group)",
                )
                _load_btn = server.gui.add_button("✅ Load Panel", color="green")
                _clear_btn = server.gui.add_button("🗑️ Remove Panel", color="red")

                _panel_dropdowns.append(_tpl_dd)
                _panel_mode_dropdowns.append(_mode_dd)
                _panel_load_btns.append(_load_btn)
                _panel_clear_btns.append(_clear_btn)

                def _make_load_handler(si, dd, mdd):
                    def handler(_):
                        tpl = dd.value
                        if tpl == "-- Nessuno --":
                            _clear_panel_slot(si)
                            update_scene()
                            print(f"Panel Configurator: Slot {_ELIOS3_SLOTS[si]['name']} cleared.")
                            return
                        as_individual = mdd.value == "Individual LEDs"
                        _load_template_into_slot(si, tpl, as_individual)
                    return handler

                def _make_clear_handler(si, dd):
                    def handler(_):
                        _clear_panel_slot(si)
                        dd.value = "-- Nessuno --"
                        update_scene()
                        print(f"Panel Configurator: Slot {_ELIOS3_SLOTS[si]['name']} cleared.")
                    return handler

                _load_btn.on_click(_make_load_handler(_si, _tpl_dd, _mode_dd))
                _clear_btn.on_click(_make_clear_handler(_si, _tpl_dd))

    with tab_panels:
        individual_leds_folder = server.gui.add_folder("Individual LEDs")
    
    def export_individual_leds_simple():
        """Export all individual LEDs to a simple JSON format (preserves exact coordinates)."""
        if len(individual_leds) == 0:
            print("⚠️ No individual LEDs to export")
            return
        
        # Export directory
        export_dir = "exports"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Collect current LED data (exact coordinates, no transformations)
        leds_export = []
        for led_data in individual_leds:
            led_export = {
                "id": led_data['id'],
                "enabled": led_data['enable'].value,
                "led_on": led_data['led_on'],
                "position": {
                    "x": float(led_data['pos_x'].value),
                    "y": float(led_data['pos_y'].value),
                    "z": float(led_data['pos_z'].value)
                },
                "rotation": {
                    "x": float(led_data['rot_x'].value),
                    "y": float(led_data['rot_y'].value),
                    "z": float(led_data['rot_z'].value)
                },
                "size": float(led_data['size'].value),
                "viewing_angle": float(led_data['viewing_angle'].value),
                "square_roll": float(led_data['square_roll'].value),
                "beam_tilt": float(led_data['beam_tilt'].value)
            }
            
            # Add metadata if present (template source info)
            if 'template_source' in led_data:
                led_export['template_source'] = led_data['template_source']
            if 'group_index' in led_data:
                led_export['group_index'] = led_data['group_index']
            
            leds_export.append(led_export)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"individual_leds_{timestamp}.json"
        filepath = os.path.join(export_dir, filename)
        
        # Save to file
        export_data = {
            "format_version": "1.0",
            "description": "Individual LEDs export - exact coordinates (no transformations)",
            "export_date": timestamp,
            "num_leds": len(leds_export),
            "leds": leds_export
        }
        
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Exported {len(leds_export)} individual LED(s) to: {filename}")
        return filepath
    
    def export_leds_to_stl():
        """Export each LED as an editable planar surface in STEP format.
        
        One face per LED (rectangle with filleted corners). The output is a
        true B-Rep STEP file with planar faces, fully editable in SolidWorks
        (selectable as reference plane, extrudable, etc.).
        """
        if len(current_leds) == 0:
            print("⚠️ No LEDs in the scene. Update the scene first.")
            return None
        
        try:
            import cadquery as cq
        except ImportError:
            print("⚠️ 'cadquery' library required for STEP export.  pip install cadquery")
            return None
        try:
            from shapely.geometry import Polygon
        except ImportError:
            print("⚠️ 'shapely' library required.  pip install shapely")
            return None
        
        active_leds = [led for led in current_leds
                       if not (hasattr(led, 'enabled') and not led.enabled)]
        if not active_leds:
            print("⚠️ No active LEDs to export.")
            return None
        
        # ── Parameters (cm) ──
        margin   = 0.05   # 0.5 mm border around each LED
        fillet_r = 0.04   # 0.4 mm fillet on outer corners
        faces    = []
        
        def _normal(led):
            n = np.array(getattr(led, 'mesh_normal', led.direction), dtype=float)
            nm = np.linalg.norm(n)
            return n / nm if nm > 1e-10 else np.array([1., 0., 0.])
        
        for led in active_leds:
            pos = np.array(led.position, dtype=float)
            nrm = _normal(led)
            hw  = led.width / 2.0
            
            # Local 2-D frame on the LED's plane
            if abs(nrm[2]) < 0.9:
                lx = np.cross(nrm, [0, 0, 1])
            else:
                lx = np.cross(nrm, [0, 1, 0])
            lx /= np.linalg.norm(lx)
            ly = np.cross(nrm, lx)
            ly /= np.linalg.norm(ly)
            
            # Use row_direction for consistent orientation
            row_d = getattr(led, 'row_direction', None)
            if row_d is not None:
                row_d = np.array(row_d, dtype=float)
                r2x = np.dot(row_d, lx)
                r2y = np.dot(row_d, ly)
                n2  = np.hypot(r2x, r2y)
                if n2 > 0.01:
                    r_hat = np.array([r2x, r2y]) / n2
                else:
                    r_hat = np.array([1., 0.])
            else:
                r_hat = np.array([1., 0.])
            p_hat = np.array([-r_hat[1], r_hat[0]])
            
            # ── Outer panel outline (LED square + margin) with filleted corners ──
            m = hw + margin
            outer_corners = [(r_hat[0]*sx*m + p_hat[0]*sy*m,
                              r_hat[1]*sx*m + p_hat[1]*sy*m)
                             for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
            outer = Polygon(outer_corners)
            try:
                sm = outer.buffer(-fillet_r, resolution=8).buffer(fillet_r, resolution=8)
                if sm.is_valid and not sm.is_empty and sm.area > outer.area * 0.5:
                    outer = sm
            except Exception:
                pass
            
            if outer.is_empty:
                continue
            
            polys = (list(outer.geoms)
                     if outer.geom_type == 'MultiPolygon'
                     else [outer])
            
            # Build a CadQuery Workplane on the LED's local plane.
            # Units: cadquery uses mm; our scene is in cm → multiply by 10.
            plane = cq.Plane(
                origin=cq.Vector(float(pos[0])*10, float(pos[1])*10, float(pos[2])*10),
                xDir=cq.Vector(float(lx[0]), float(lx[1]), float(lx[2])),
                normal=cq.Vector(float(nrm[0]), float(nrm[1]), float(nrm[2])),
            )
            
            for poly in polys:
                try:
                    coords = list(poly.exterior.coords)
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords = coords[:-1]
                    pts_mm = [(float(x)*10, float(y)*10) for (x, y) in coords]
                    wire = (
                        cq.Workplane(plane)
                        .polyline(pts_mm)
                        .close()
                        .val()
                    )
                    face = cq.Face.makeFromWires(wire)
                    faces.append(face)
                except Exception as e:
                    print(f"   [skip face] {e}")
        
        if not faces:
            print("⚠️ No faces generated.")
            return None
        
        compound = cq.Compound.makeCompound(faces)
        
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"led_panel_{ts}.step"
        filepath = os.path.join(export_dir, filename)
        cq.exporters.export(compound, filepath, exportType='STEP')
        
        print(f"✓ Exported STEP: {filename}")
        print(f"  Planar faces: {len(faces)} (editable in SolidWorks)")
        print(f"  Margin: {margin*10:.1f} mm  Fillet: {fillet_r*10:.1f} mm  Units: mm")
        print(f"  Path: {os.path.abspath(filepath)}")
        return filepath
    
    def export_custom_group_dxf():
        """Export a 2D DXF file for CNC cutting of the custom group LEDs.
        
        Projects all custom-group LEDs onto their unfolded flat plane,
        groups them into rows by Y-coordinate clustering, then places
        horizontal living-hinge slot patterns between rows whose normals
        differ so the flat panel can be bent into the 3-D shape.
        
        Units in the DXF are millimetres.
        
        Layers:
          PANEL_OUTLINE  – outer contour (white)
          LED_HOLES      – square LED apertures (red)
          FLEX_CUTS      – living-hinge slots between rows (green)
        """
        try:
            import ezdxf
        except ImportError:
            print("⚠️ 'ezdxf' library required.  pip install ezdxf")
            return None
        
        # Gather custom-group LEDs that are active
        custom_leds = [
            led for led in current_leds
            if getattr(led, 'is_custom', False)
            and not (hasattr(led, 'enabled') and not led.enabled)
        ]
        if not custom_leds:
            print("⚠️ No active custom-group LEDs in the scene.")
            return None
        
        # --- Helper: normalised normal vector ---
        def _led_normal(led):
            n = np.array(getattr(led, 'mesh_normal', led.direction), dtype=float)
            nm = np.linalg.norm(n)
            return n / nm if nm > 1e-10 else np.array([1., 0., 0.])
        
        # --- Compute local 2-D frame from the average LED normal ---
        positions_3d = np.array([led.position for led in custom_leds])
        normals_3d = np.array([_led_normal(led) for led in custom_leds])
        
        avg_normal = normals_3d.mean(axis=0)
        n_len = np.linalg.norm(avg_normal)
        avg_normal = avg_normal / n_len if n_len > 1e-10 else np.array([1., 0., 0.])
        
        centroid = positions_3d.mean(axis=0)
        
        # Orthonormal frame on the projection plane
        if abs(avg_normal[2]) < 0.9:
            x_local = np.cross(avg_normal, [0, 0, 1])
        else:
            x_local = np.cross(avg_normal, [0, 1, 0])
        x_local /= np.linalg.norm(x_local)
        y_local = np.cross(avg_normal, x_local)
        y_local /= np.linalg.norm(y_local)
        
        # --- Project each LED onto the 2-D plane (cm → mm) ---
        margin_mm      = 1.5   # margin around each LED hole
        panel_border_mm = 3.0  # extra border around the panel edges
        
        led_data = []  # [(cx_mm, cy_mm, hw_mm, normal_3d), ...]
        for idx, led in enumerate(custom_leds):
            delta = np.array(led.position) - centroid
            cx = np.dot(delta, x_local) * 10.0  # cm → mm
            cy = np.dot(delta, y_local) * 10.0
            hw = (led.width / 2.0) * 10.0
            led_data.append((cx, cy, hw, normals_3d[idx]))
        
        # --- Outer panel bounding rectangle ---
        all_x  = [d[0] for d in led_data]
        all_y  = [d[1] for d in led_data]
        max_hw = max(d[2] for d in led_data)
        border = max_hw + margin_mm + panel_border_mm
        
        x_min = min(all_x) - border
        x_max = max(all_x) + border
        y_min = min(all_y) - border
        y_max = max(all_y) + border
        
        # ================================================================
        #  Cluster LEDs into rows by Y coordinate, then add flex cuts
        #  between adjacent rows whose average normals differ
        # ================================================================
        # Sort LEDs by Y coordinate
        sorted_indices = sorted(range(len(led_data)), key=lambda i: led_data[i][1])
        
        # Cluster into rows: LEDs within cluster_tol mm of each other in Y
        cluster_tol = max_hw * 1.5  # LEDs in same row are close in Y
        rows = []  # list of lists of led_data indices
        current_row = [sorted_indices[0]]
        for k in range(1, len(sorted_indices)):
            prev_y = led_data[sorted_indices[k - 1]][1]
            curr_y = led_data[sorted_indices[k]][1]
            if abs(curr_y - prev_y) < cluster_tol:
                current_row.append(sorted_indices[k])
            else:
                rows.append(current_row)
                current_row = [sorted_indices[k]]
        rows.append(current_row)
        
        # Compute per-row average Y and average normal
        row_info = []  # (avg_y, avg_normal_3d, min_x, max_x)
        for row in rows:
            avg_y = np.mean([led_data[i][1] for i in row])
            avg_n = np.mean([led_data[i][3] for i in row], axis=0)
            nm = np.linalg.norm(avg_n)
            avg_n = avg_n / nm if nm > 1e-10 else np.array([1., 0., 0.])
            r_min_x = min(led_data[i][0] - led_data[i][2] for i in row)
            r_max_x = max(led_data[i][0] + led_data[i][2] for i in row)
            row_info.append((avg_y, avg_n, r_min_x, r_max_x))
        
        # --- Generate flex cuts between adjacent rows ---
        flex_angle_threshold_deg = 2.0
        slot_length_mm  = 4.0   # length of each slot segment
        slot_gap_mm     = 1.5   # gap between consecutive slots in a line
        n_slot_lines    = 3     # parallel lines of slots
        slot_line_gap   = 1.0   # spacing between parallel lines
        
        flex_cuts = []  # ((x1,y1),(x2,y2))
        
        for r in range(len(rows) - 1):
            # Angle between adjacent row normals
            dot = np.clip(np.dot(row_info[r][1], row_info[r + 1][1]), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(abs(dot)))
            if angle_deg < flex_angle_threshold_deg:
                continue
            
            # Y zone: between the bottom of upper row and top of lower row
            # (rows sorted bottom to top, i.e. ascending Y)
            row_top_leds    = rows[r]
            row_bottom_leds = rows[r + 1]
            
            y_top_of_lower = max(led_data[i][1] + led_data[i][2] + margin_mm for i in row_top_leds)
            y_bot_of_upper = min(led_data[i][1] - led_data[i][2] - margin_mm for i in row_bottom_leds)
            
            zone_y_center = (y_top_of_lower + y_bot_of_upper) / 2.0
            zone_y_height = y_bot_of_upper - y_top_of_lower
            
            if zone_y_height < 1.5:
                # Not enough vertical space for flex cuts; place them anyway at midpoint
                zone_y_center = (row_info[r][0] + row_info[r + 1][0]) / 2.0
                zone_y_height = abs(row_info[r + 1][0] - row_info[r][0]) * 0.3
                if zone_y_height < 1.0:
                    continue
            
            # X extent of the flex zone = full panel width minus a small inset
            inset = panel_border_mm * 0.5
            zone_x_min = x_min + inset
            zone_x_max = x_max - inset
            zone_width = zone_x_max - zone_x_min
            if zone_width < slot_length_mm:
                continue
            
            # Place n_slot_lines parallel horizontal lines of staggered slots
            total_lines_span = (n_slot_lines - 1) * slot_line_gap
            
            for line_k in range(n_slot_lines):
                line_y = zone_y_center - total_lines_span / 2.0 + line_k * slot_line_gap
                
                # Stagger odd lines by half a stride
                stride = slot_length_mm + slot_gap_mm
                stagger = (stride / 2.0) if (line_k % 2 == 1) else 0.0
                
                x_pos = zone_x_min + stagger
                while x_pos + slot_length_mm <= zone_x_max:
                    x1 = x_pos
                    x2 = x_pos + slot_length_mm
                    flex_cuts.append(((x1, line_y), (x2, line_y)))
                    x_pos += stride
        
        # --- Build DXF ---
        doc = ezdxf.new(dxfversion='R2010')
        doc.units = ezdxf.units.MM
        msp = doc.modelspace()
        
        # Outer panel contour
        msp.add_lwpolyline(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)],
            close=True,
            dxfattribs={'layer': 'PANEL_OUTLINE', 'color': 7}
        )
        
        # LED holes (square)
        for cx, cy, hw, _ in led_data:
            msp.add_lwpolyline(
                [(cx - hw, cy - hw), (cx + hw, cy - hw),
                 (cx + hw, cy + hw), (cx - hw, cy + hw)],
                close=True,
                dxfattribs={'layer': 'LED_HOLES', 'color': 1}
            )
        
        # Flex cuts (horizontal living-hinge slots)
        for (x1, y1), (x2, y2) in flex_cuts:
            msp.add_line(
                (x1, y1), (x2, y2),
                dxfattribs={'layer': 'FLEX_CUTS', 'color': 3}
            )
        
        # --- Save ---
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cnc_panel_{ts}.dxf"
        filepath = os.path.join(export_dir, filename)
        doc.saveas(filepath)
        
        panel_w = x_max - x_min
        panel_h = y_max - y_min
        print(f"✓ Exported CNC DXF: {filename}")
        print(f"  Custom LEDs: {len(custom_leds)}  Rows detected: {len(rows)}")
        print(f"  Panel size: {panel_w:.1f} x {panel_h:.1f} mm")
        print(f"  Flex zones: {max(0, len(rows)-1)}  Slots: {len(flex_cuts)}")
        print(f"  Layers: PANEL_OUTLINE, LED_HOLES, FLEX_CUTS")
        print(f"  Path: {os.path.abspath(filepath)}")
        return filepath
    
    with individual_leds_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Add New Individual LED</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Add single LEDs with custom position, rotation, and size</div>")
        
        add_individual_led_btn = server.gui.add_button("➕ Add LED", color="cyan")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export Individual LEDs</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Save exact coordinates without transformations</div>")
        
        export_individual_leds_btn = server.gui.add_button("💾 Export to JSON", color="#4CAF50")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export Cover Panel (STEP)</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>Editable planar surfaces in SolidWorks (selectable as a plane, extrudable). One face per LED.</div>")
        
        export_stl_btn = server.gui.add_button("📦 Export Panel STEP", color="#FF9800")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Export CNC Cutting (DXF)</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>2D DXF file for CNC cutting of the custom panel. Can be opened in AutoCAD, LibreCAD, etc.</div>")
        
        export_cnc_btn = server.gui.add_button("🔩 Export CNC DXF", color="#2196F3")
        
        @add_individual_led_btn.on_click
        def _(_):
            create_individual_led()
            print("✓ Individual LED added")
        
        @export_individual_leds_btn.on_click
        def _(_):
            export_individual_leds_simple()
        
        @export_stl_btn.on_click
        def _(_):
            export_leds_to_stl()
        
        @export_cnc_btn.on_click
        def _(_):
            export_custom_group_dxf()


    # LED Control Matrix (individual LED and row control for base groups)
    # These controls are placed inside led_config_folder so they are hidden when no base groups are active
    group_names = ["Front+", "Front-", "Side+", "Side-"]
    
    # Create control folders for each group with HTML buttons inside led_config_folder
    with led_config_folder:
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


    def _emission_settings():
        """Engine emission parameters from the GUI.

        Per-LED flux already carries the diffuser transmission (applied in
        _build_current_leds_and_absorbers), so only the fallback flux is scaled here.
        """
        default_lumens = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        if diffuser_enable_chk.value:
            default_lumens *= float(diffuser_transmission_slider.value) / 100.0
        return EmissionSettings(
            default_lumens=default_lumens,
            ray_uniformity=float(ray_uniformity_slider.value),
        )

    def compute_wall_intensity(
        leds, wall_dist, num_rays_per_led, grid_size=50, wall_size=80, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays onto the wall. Returns (lux grid, wall_size)."""
        settings = WallSettings(
            wall_dist=float(wall_dist), grid_size=int(grid_size),
            wall_size=float(wall_size), rays_per_pixel=int(num_rays_per_led),
        )
        grid = _wall_engine.compute_wall_intensity(
            leds, settings, _emission_settings(),
            absorbers=absorbers, stl_mesh_data=stl_mesh_data,
        )
        return grid, wall_size

    def compute_room_intensity(
        leds, front_dist, side_dist, top_bottom_dist, num_rays_per_led, grid_size=20, back_dist=None, absorbers=None, stl_mesh_data=None
    ):
        """Trace rays inside the room. Returns (per-wall lux grids, wall_specs)."""
        reflections_on = bool(reflections_enable.value)
        settings = RoomSettings(
            front_dist=float(front_dist), side_dist=float(side_dist),
            top_bottom_dist=float(top_bottom_dist), back_dist=back_dist,
            grid_size=int(grid_size), rays_per_pixel=int(num_rays_per_led),
            led_x_center=float(circle_center_slider.value),
            max_bounces=int(max_bounces_slider_room.value) if reflections_on else 0,
            wall_reflectance=float(custom_reflectance_slider.value) if reflections_on else 0.0,
        )
        return _room_engine.compute_room_intensity(
            leds, settings, _emission_settings(),
            absorbers=absorbers, stl_mesh_data=stl_mesh_data,
        )

    def capture_camera_fov_image():
        """Capture intensity image within camera FOV at 1cm resolution."""
        from datetime import datetime
        from PIL import Image
        
        # Get current camera and wall settings
        # Use correct wall distance based on mode
        if room_mode_enable.value:
            wall_dist = room_front_dist.value
        else:
            wall_dist = wall_dist_slider.value
        
        cam_x = camera_pos_x.value
        cam_y = camera_pos_y.value
        fov_h_deg = camera_fov_h.value
        fov_v_deg = camera_fov_v.value
        pitch_deg = camera_pitch.value

        # FOV footprint on the wall (trapezoid when the camera is pitched)
        fov_z_bot, fov_z_top, fov_w_bot, fov_w_top = _camera_fov_wall_trapezoid(
            wall_dist - cam_x, pitch_deg, fov_h_deg, fov_v_deg
        )
        fov_half_w_max = max(fov_w_bot, fov_w_top)
        fov_width_cm = 2.0 * fov_half_w_max   # bounding-box width on wall
        fov_height_cm = fov_z_top - fov_z_bot  # bounding-box height on wall
        
        # Cell resolution: 1cm × 1cm (10mm²)
        cell_size_cm = 1.0
        grid_width = int(np.ceil(fov_width_cm / cell_size_cm))
        grid_height = int(np.ceil(fov_height_cm / cell_size_cm))
        
        # Create grid for FOV region
        fov_grid = np.zeros((grid_height, grid_width))
        
        # Get LEDs configuration (fixed angles: front=0°, side=90°)
        front_angle = 0.0  # Fixed front angle
        side_angle = 90.0  # Fixed side angle
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value
        
        rotations = [
            rot_front_pos.value,
            rot_front_neg.value,
            rot_side_pos.value,
            rot_side_neg.value,
        ]
        
        rotations_y = [
            rot_y_front_pos.value,
            rot_y_front_neg.value,
            rot_y_side_pos.value,
            rot_y_side_neg.value,
        ]
        
        offsets = [
            (offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value),
            (offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value),
            (offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value),
            (offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value),
        ]
        
        # Build custom groups configs list
        custom_groups_configs = []
        for group in custom_groups:
            config = {
                'enabled': group['enable'].value,
                'position': (group['pos_x'].value, group['pos_y'].value, group['pos_z'].value),
                'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                'led_states': group['led_states'],
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                translated_positions, rotated_directions, rotated_row_dirs = _dynamic_group_world_geometry(group)
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_sizes'] = group.get('led_sizes', [])
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                if rotated_row_dirs:
                    config['led_row_directions'] = rotated_row_dirs
            # Pass lumens override for custom group
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
            if group.get('panel_slot') is not None:
                config['owner'] = ('slot', group['panel_slot'])
            else:
                config['owner'] = ('custom_group', group['id'])
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
                'beam_tilt': led['beam_tilt'].value,
            }
            # Pass lumens override for individual LED
            if led.get('lumens_override') and led['lumens_override'].value:
                config['lumens_override'] = float(led['lumens_value'].value)
            else:
                config['lumens_override'] = None
            # Pass external lens settings
            if led.get('ext_lens_enable') and led['ext_lens_enable'].value:
                config['ext_lens_angle'] = float(led['ext_lens_angle'].value)
                config['ext_lens_efficiency'] = float(led['ext_lens_efficiency'].value) / 100.0
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and led in _pdata.get('individual_leds', []):
                    config['owner'] = ('slot', _si)
                    break
            if 'owner' not in config and led.get('panel_slot') is not None:
                config['owner'] = ('slot', led['panel_slot'])
            individual_leds_configs.append(config)
        
        # Inject the live XZ-mirrored copy of the mirror-primary panel (if any)
        _expand_mirror_configs(custom_groups_configs, individual_leds_configs)

        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            default_lumens=float(led_lumens_slider.value),
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
        # ── Apply diffuser lens effect (FOV camera) ──
        if diffuser_enable_chk.value:
            _diff_angle = float(diffuser_angle_slider.value)
            _diff_trans = float(diffuser_transmission_slider.value) / 100.0
            for led in leds:
                led.viewing_angle = max(led.viewing_angle, _diff_angle)
                if led.lumens is not None:
                    led.lumens = led.lumens * _diff_trans

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
                'rotation': None,
            })
        
        # Add abs2 and abs3 at origin with offsets (if absorbers enabled)
        if absorbers_enable.value:
            # Abs2 with rotation
            abs_cx = 0.0 + abs2_off_x.value
            abs_cy = 0.0 + abs2_off_y.value
            abs_cz = 0.0 + abs2_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs2_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
            
            # Abs3 with rotation
            abs_cx = 0.0 + abs3_off_x.value
            abs_cy = 0.0 + abs3_off_y.value
            abs_cz = 0.0 + abs3_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs3_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
        
        # Ray tracing for FOV region
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
        rays_per_pixel = int(intensity_rays_slider.value)
        
        # Count active LEDs
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        if num_active_leds == 0:
            print("No active LEDs")
            return
        
        # Calculate rays per LED to achieve target rays per pixel
        total_pixels = grid_width * grid_height
        num_rays_per_led = max(1, int((total_pixels * rays_per_pixel) / num_active_leds))
        
        print(f"Capturing FOV image: {grid_width}x{grid_height} pixels ({total_pixels} total)...")
        print(f"Active LEDs: {num_active_leds}, Rays per LED: {num_rays_per_led}, Total rays: {num_active_leds * num_rays_per_led}")
        print(f"Target: {rays_per_pixel} rays/pixel, Actual: {(num_active_leds * num_rays_per_led) / total_pixels:.2f} rays/pixel")
        
        # Pre-build STL mesh accelerator ONCE (outside LED loop)
        fov_stl_accel = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_obj = stl_mesh_data[0]
            transform = np.eye(4)
            scale = float(stl_scale.value)
            if np.isfinite(scale) and scale > 0:
                transform[:3, :3] *= scale
            rot_x_v = float(stl_rot_x.value) if np.isfinite(float(stl_rot_x.value)) else 0.0
            rot_y_v = float(stl_rot_y.value) if np.isfinite(float(stl_rot_y.value)) else 0.0
            rot_z_v = float(stl_rot_z.value) if np.isfinite(float(stl_rot_z.value)) else 0.0
            if rot_x_v != 0:
                transform = _rot4_x(np.radians(rot_x_v)) @ transform
            if rot_y_v != 0:
                transform = _rot4_y(np.radians(rot_y_v)) @ transform
            if rot_z_v != 0:
                transform = _rot4_z(np.radians(rot_z_v)) @ transform
            pos_xv = float(stl_pos_x.value) if np.isfinite(float(stl_pos_x.value)) else 0.0
            pos_yv = float(stl_pos_y.value) if np.isfinite(float(stl_pos_y.value)) else 0.0
            pos_zv = float(stl_pos_z.value) if np.isfinite(float(stl_pos_z.value)) else 0.0
            transform[:3, 3] = [pos_xv, pos_yv, pos_zv]
            fov_mesh_data = {
                'vertices': mesh_obj.vertices,
                'faces': mesh_obj.faces,
                'transform': transform,
            }
            fov_stl_accel = _prepare_mesh_ray_accelerator(fov_mesh_data)
            print(f"STL mesh absorber active ({len(mesh_obj.faces)} triangles)")
        
        led_total_lumens_emitted = 0.0
        
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
            
            uniformity = float(ray_uniformity_slider.value)
            n = _get_effective_n(led, uniformity)
            max_theta = np.radians(led.viewing_angle / 2.0)
            cos_max = np.cos(max_theta)
            
            # --- Generate ALL rays for this LED at once ---
            u = np.random.uniform(0, 1, (num_rays_per_led, 2))
            cos_theta = 1.0 - u[:, 0] * (1.0 - cos_max)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            phi = 2 * np.pi * u[:, 1]
            
            sin_theta = np.sin(theta)
            local_dirs = np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta,
            ])
            
            world_dirs = (local_dirs[:, 0:1] * x_axis +
                          local_dirs[:, 1:2] * y_axis +
                          local_dirs[:, 2:3] * z_axis)
            norms_wd = np.linalg.norm(world_dirs, axis=1, keepdims=True)
            world_dirs = world_dirs / norms_wd
            
            # Calculate lumens per ray with cone normalization and lens efficiency
            cos_max_n1 = cos_max ** (n + 1.0)
            denom = 1.0 - cos_max_n1
            norm_factor = (n + 1.0) * (1.0 - cos_max) / denom if denom > 1e-12 else 1.0
            cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)
            intensity_coefficients = np.power(cos_theta_clamped, n)
            fov_led_lumens = float(getattr(led, 'lumens', None) or lumens_per_led)
            lumens_per_ray_arr = (fov_led_lumens / max(1, num_rays_per_led)) * intensity_coefficients * norm_factor
            led_total_lumens_emitted += np.sum(lumens_per_ray_arr)
            
            # --- Check box absorber intersection (vectorized) ---
            hit_absorbed = np.zeros(num_rays_per_led, dtype=bool)
            if absorbers:
                ray_origins = np.broadcast_to(led.position, (num_rays_per_led, 3)).copy().astype(np.float32)
                hit_absorbed = _ray_box_intersection_batch_np(ray_origins, world_dirs.astype(np.float32), absorbers)
            
            # --- Batch STL mesh intersection ---
            if fov_stl_accel is not None:
                not_abs = np.where(~hit_absorbed)[0]
                if len(not_abs) > 0:
                    origins = np.tile(led.position, (len(not_abs), 1)).astype(np.float64)
                    mesh_hits = _batch_ray_mesh_intersection(origins, world_dirs[not_abs], fov_stl_accel)
                    hit_absorbed[not_abs[mesh_hits]] = True
            
            # --- Wall hits ---
            alive = ~hit_absorbed
            towards_wall = world_dirs[:, 0] > 0
            valid = alive & towards_wall
            vi = np.where(valid)[0]
            
            if len(vi) > 0:
                t = (wall_dist - led.position[0]) / world_dirs[vi, 0]
                pos_t = t > 0
                vi2 = vi[pos_t]
                t2 = t[pos_t]
                
                hit_y = led.position[1] + world_dirs[vi2, 1] * t2
                hit_z = led.position[2] + world_dirs[vi2, 2] * t2
                
                # Keep only hits inside the (possibly pitched) FOV trapezoid,
                # centred at the camera's Y position
                z_span = max(fov_z_top - fov_z_bot, 1e-9)
                frac_z = np.clip((hit_z - fov_z_bot) / z_span, 0.0, 1.0)
                half_w_at_z = fov_w_bot + (fov_w_top - fov_w_bot) * frac_z
                in_fov = (
                    (hit_z >= fov_z_bot) & (hit_z <= fov_z_top)
                    & (np.abs(hit_y - cam_y) <= half_w_at_z)
                )
                fi = np.where(in_fov)[0]
                
                grid_x = ((hit_y[fi] - cam_y + fov_half_w_max) / cell_size_cm).astype(int)
                grid_y = ((hit_z[fi] - fov_z_bot) / cell_size_cm).astype(int)
                
                in_bounds = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
                bi = np.where(in_bounds)[0]
                
                lux_values = lumens_per_ray_arr[vi2[fi[bi]]]
                np.add.at(fov_grid, (grid_y[bi], grid_x[bi]), lux_values)
        
        # Diagnostic: print first LED's flux conservation
        print(f"FOV Capture: First LED emitted {led_total_lumens_emitted:.2f} lm total (target: {lumens_per_led:.2f} lm)")
        
        # Convert to lux: Lux = Lumen / Area_m²
        cell_area_m2 = (cell_size_cm / 100.0) ** 2
        lux_grid = fov_grid / cell_area_m2
        
        # Clean up any NaN or Inf values
        fov_grid = np.nan_to_num(fov_grid, nan=0.0, posinf=0.0, neginf=0.0)
        lux_grid = np.nan_to_num(lux_grid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get max lux for color mapping
        max_lux = lux_grid.max()
        
        # Create image using same colormap as render (intensity_to_color)
        img_rgb = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i in range(grid_height):
            for j in range(grid_width):
                lux_val = lux_grid[i, j]
                color = intensity_to_color(lux_val, max_lux)
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
            lux_val = intensity_fraction * max_lux
            color = intensity_to_color(lux_val, max_lux)
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

    FIXED_LEGEND_MAX = 3500.0  # Default fixed absolute legend cap (overridden by GUI)

    def intensity_to_color(value, max_val):
        """Convert intensity to colormap (inferno-like or black-to-white)."""
        # Handle invalid values
        if max_val == 0 or not np.isfinite(value) or not np.isfinite(max_val):
            return (0.0, 0.0, 0.0)
        
        # Below-threshold cells are rendered pitch black (threshold in lux, 0 = off)
        if value < float(intensity_threshold_slider.value):
            return (0.0, 0.0, 0.0)
        
        t = np.clip(value / max_val, 0.0, 1.0)
        
        # Black-to-white grayscale mode
        if bw_scale_chk.value:
            return (t, t, t)
        
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

    # Cache for last computed intensity grid so uniformity can be recalculated
    # when FOV changes without re-running ray tracing
    _last_intensity_cache = {'grid': None, 'wall_size_cm': None, 'wall_dist': None,
                             'cell_area_m2': None, 'max_lux': None, 'color_scale_max': None}
    _last_room_cache = {
        'grids': None, 'wall_specs': None,
        'front_dist': None, 'side_dist': None, 'top_bottom_dist': None, 'back_dist': None,
        'max_lux': None, 'color_scale_max': None, 'avg_cell_area_m2': None,
    }
    # Guard programmatic checkbox writes in on_room_mode_toggle so we don't
    # retrigger update_intensity_map / update_room_intensity_map.
    _mode_toggle_syncing = [False]

    def _build_lux_legend_html(max_lux, color_scale_max, cell_area_m2, cell_caption="lm/cell"):
        """Intensity-scale HTML used by both wall and room legend panels."""
        _legend_cap = float(legend_max_input.value)
        if color_scale_max <= _legend_cap:
            _step = max(1, _legend_cap / 8)
            legend_vals_lux = np.arange(0, _legend_cap + 1, _step)
        else:
            legend_vals_lux = np.linspace(0, color_scale_max, 9)
        scale_label = f"(scale 0\u2013{int(color_scale_max)} lx" + (
            ", FIXED)" if color_scale_max <= _legend_cap else ", AUTO)")
        html_lines = [
            "<div style='font-family: sans-serif;'>",
            "<div style='font-weight:600;margin-bottom:2px;'>Intensity legend (lux)</div>",
            f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} \u2014 peak {max_lux:.0f} lx</div>",
        ]
        for lux_val in reversed(legend_vals_lux):
            color = intensity_to_color(lux_val, color_scale_max)
            hex_color = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
            lumen_val = lux_val * cell_area_m2
            html_lines.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                f"<div style='width:18px;height:12px;background:{hex_color};margin-right:8px;border:1px solid #222;'></div>"
                f"<div style='min-width:70px;'>{lux_val:.0f} lx</div>"
                f"<div style='color:#888;font-size:11px;'>({lumen_val:.4f} {cell_caption})</div></div>"
            )
        html_lines.append("</div>")
        return "".join(html_lines)

    def _empty_fov_html():
        return (
            "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>Pattern Uniformity</div>"
            "<div style='color:#888;font-size:12px;'>No intensity data inside camera FOV</div></div>"
        )

    def _wall_grid_cell_centers_cm(grid, wall_size_cm, wall_dist):
        return wall_grid_cell_centers_cm(grid.shape, wall_size_cm, wall_dist)

    def _collect_room_fov_lux(cache):
        """Lux values of room-wall cells that fall inside the main-camera FOV."""
        grids = cache.get('grids')
        specs = cache.get('wall_specs')
        if not grids or not specs:
            return np.array([])
        cam_pos = np.array([camera_pos_x.value, camera_pos_y.value, 0.0], dtype=float)
        parts = []
        for name, grid in grids.items():
            spec = specs.get(name)
            if spec is None:
                continue
            pts = room_wall_cell_centers(
                name, spec,
                cache['front_dist'], cache['side_dist'],
                cache['top_bottom_dist'], cache['back_dist'],
            )
            mask = points_in_pinhole_fov(
                cam_pos, camera_pitch.value,
                camera_fov_h.value, camera_fov_v.value, pts,
            )
            if np.any(mask):
                parts.append(grid[mask])
        if not parts:
            return np.array([])
        return np.concatenate(parts)

    def _collect_active_cell_samples():
        """(points_cm Nx3, lux N) for the active intensity mode, or (None, None)."""
        if room_mode_enable.value:
            cache = _last_room_cache
            if cache['grids'] is None:
                return None, None
            pts_list, lux_list = [], []
            for name, grid in cache['grids'].items():
                spec = cache['wall_specs'].get(name)
                if spec is None:
                    continue
                pts = room_wall_cell_centers(
                    name, spec,
                    cache['front_dist'], cache['side_dist'],
                    cache['top_bottom_dist'], cache['back_dist'],
                )
                pts_list.append(pts.reshape(-1, 3))
                lux_list.append(np.asarray(grid).reshape(-1))
            if not pts_list:
                return None, None
            return np.concatenate(pts_list, axis=0), np.concatenate(lux_list)
        cache = _last_intensity_cache
        if cache['grid'] is None:
            return None, None
        pts = _wall_grid_cell_centers_cm(cache['grid'], cache['wall_size_cm'], cache['wall_dist'])
        return pts.reshape(-1, 3), np.asarray(cache['grid']).reshape(-1)

    def _compute_vio_occupancy_html():
        """% of VIO-FOV wall cells whose lux is above the black threshold."""
        pts, lux = _collect_active_cell_samples()
        if pts is None or pts.size == 0:
            return ""
        threshold = float(intensity_threshold_slider.value)
        cam_pos = np.array([vio_pos_x.value, vio_pos_y.value, vio_pos_z.value], dtype=float)
        hfov, vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
        mask1 = points_in_fisheye_fov(
            cam_pos, vio_cam1_pitch.value, vio_cam1_yaw.value, hfov, vfov, pts)
        mask2 = points_in_fisheye_fov(
            cam_pos, vio_cam2_pitch.value, vio_cam2_yaw.value, hfov, vfov, pts)
        good = lux > threshold

        def _row(mask):
            n = int(np.count_nonzero(mask))
            if n == 0:
                return None, "\u2014"
            pct = 100.0 * float(np.count_nonzero(mask & good)) / n
            return pct, f"{pct:.1f}%"

        _, s1 = _row(mask1)
        _, s2 = _row(mask2)
        union_pct, s_u = _row(mask1 | mask2)
        union_color = "#4CAF50" if (union_pct is not None and union_pct >= 50.0) else "#FF9800"
        if union_pct is None:
            union_color = "#888"
        return (
            "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>VIO FOV Lighting Occupancy</div>"
            f"<div style='color:#888;font-size:10px;margin-bottom:6px;'>"
            f"Share of FOV wall cells brighter than {threshold:.0f} lx"
            f"{' (any light)' if threshold <= 0 else ''}</div>"
            "<table style='font-size:11px;color:#ccc;border-collapse:collapse;width:100%;'>"
            f"<tr><td style='padding:1px 6px 1px 0;color:#ff00ff;'>Cam 1 (up)</td><td>{s1}</td></tr>"
            f"<tr><td style='padding:1px 6px 1px 0;color:#00ffff;'>Cam 2 (down)</td><td>{s2}</td></tr>"
            f"<tr><td style='padding:1px 6px 1px 0;'>Union (Cam 1 \u222a Cam 2)</td>"
            f"<td style='color:{union_color};font-weight:700;'>{s_u}</td></tr>"
            "</table>"
            "<div style='font-size:10px;color:#888;margin-top:6px;'>"
            "Denominator is FOV-covered wall cells, not empty space.</div></div>"
        )

    def _wall_metrics_html(grid, wall_size_cm, wall_dist):
        _trap = _camera_fov_wall_trapezoid(
            wall_dist - camera_pos_x.value, camera_pitch.value,
            camera_fov_h.value, camera_fov_v.value,
        )
        html = _compute_uniformity_html(
            grid,
            fov_trapezoid=(*_trap, camera_pos_y.value),
            wall_size_cm=wall_size_cm,
            min_percentile=float(uniformity_percentile_slider.value),
        )
        if not html:
            html = _empty_fov_html()
        return html + _compute_vio_occupancy_html()

    def _room_metrics_html():
        fov_lux = _collect_room_fov_lux(_last_room_cache)
        if fov_lux.size == 0:
            html = _empty_fov_html()
        else:
            html = (_compute_uniformity_html(fov_lux.reshape(1, -1),
                                             min_percentile=float(uniformity_percentile_slider.value))
                    or _empty_fov_html())
        return html + _compute_vio_occupancy_html()

    def _refresh_uniformity():
        """Recalculate FOV-only uniformity (and VIO occupancy) from cache (cheap)."""
        if room_mode_enable.value:
            cache = _last_room_cache
            if cache['grids'] is None:
                return
            legend = _build_lux_legend_html(
                cache['max_lux'], cache['color_scale_max'],
                cache['avg_cell_area_m2'], cell_caption="lm/cell avg",
            )
            legend_html.content = legend + _room_metrics_html()
            return
        cache = _last_intensity_cache
        if cache['grid'] is None:
            return
        legend = _build_lux_legend_html(
            cache['max_lux'], cache['color_scale_max'],
            cache['cell_area_m2'], cell_caption="lm/cell",
        )
        legend_html.content = legend + _wall_metrics_html(
            cache['grid'], cache['wall_size_cm'], cache['wall_dist'],
        )

    def _build_current_leds_and_absorbers():
        """Build LEDs and absorbers from current GUI state."""
        front_angle = 0.0
        side_angle = 90.0
        viewing_angle = viewing_angle_slider.value
        radius = radius_slider.value
        circle_center_x = circle_center_slider.value

        rotations = [
            rot_front_pos.value, rot_front_neg.value,
            rot_side_pos.value, rot_side_neg.value,
        ]
        rotations_y = [
            rot_y_front_pos.value, rot_y_front_neg.value,
            rot_y_side_pos.value, rot_y_side_neg.value,
        ]
        offsets = [
            (offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value),
            (offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value),
            (offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value),
            (offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value),
        ]

        custom_groups_configs = []
        for group in custom_groups:
            config = {
                'enabled': group['enable'].value,
                'position': (group['pos_x'].value, group['pos_y'].value, group['pos_z'].value),
                'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                'led_states': group['led_states'],
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                translated_positions, rotated_directions, rotated_row_dirs = _dynamic_group_world_geometry(group)
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                config['led_beam_tilts'] = group.get('led_beam_tilts', [])
                if rotated_row_dirs:
                    config['led_row_directions'] = rotated_row_dirs
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
            if group.get('panel_slot') is not None:
                config['owner'] = ('slot', group['panel_slot'])
            else:
                config['owner'] = ('custom_group', group['id'])
            custom_groups_configs.append(config)

        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),
                'pos_x': led['pos_x'].value, 'pos_y': led['pos_y'].value, 'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value, 'rot_y': led['rot_y'].value, 'rot_z': led['rot_z'].value,
                'size': led['size'].value, 'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value, 'beam_tilt': led['beam_tilt'].value,
            }
            if led.get('lumens_override') and led['lumens_override'].value:
                config['lumens_override'] = float(led['lumens_value'].value)
            else:
                config['lumens_override'] = None
            # Pass external lens settings
            if led.get('ext_lens_enable') and led['ext_lens_enable'].value:
                config['ext_lens_angle'] = float(led['ext_lens_angle'].value)
                config['ext_lens_efficiency'] = float(led['ext_lens_efficiency'].value) / 100.0
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and led in _pdata.get('individual_leds', []):
                    config['owner'] = ('slot', _si)
                    break
            if 'owner' not in config and led.get('panel_slot') is not None:
                config['owner'] = ('slot', led['panel_slot'])
            individual_leds_configs.append(config)

        # Inject the live XZ-mirrored copy of the mirror-primary panel (if any)
        _expand_mirror_configs(custom_groups_configs, individual_leds_configs)

        leds = create_leds(
            front_angle, side_angle, viewing_angle, radius, circle_center_x,
            default_lumens=float(led_lumens_slider.value),
            group_rotations=rotations, group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states, group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )

        _g_rot_z_deg = float(global_rotation_z_slider.value)
        apply_global_transform(
            leds, _g_rot_z_deg,
            (global_pos_x_slider.value, global_pos_y_slider.value, global_pos_z_slider.value),
        )

        if diffuser_enable_chk.value:
            apply_diffuser(leds, diffuser_angle_slider.value,
                           float(diffuser_transmission_slider.value) / 100.0)

        absorbers = build_elios_absorbers(
            _absorber_config(), radius, circle_center_x, front_angle,
        )
        rotate_absorbers_z(absorbers, _g_rot_z_deg)

        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            if abs(_g_rot_z_deg) > 0.01:
                transform = global_z_rotation_4x4(_g_rot_z_deg) @ transform
            stl_mesh_for_raytracing = stl_mesh_data_payload(stl_mesh_data[0], transform)

        return leds, absorbers, stl_mesh_for_raytracing

    def update_intensity_map():
        """Update only the intensity map on the wall (expensive operation)."""
        nonlocal intensity_handles, legend_html

        if room_mode_enable.value:
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Room Mode is active — use 'Update Room Intensity' in Advanced → Room Mode.</div>"
                "</div>"
            )
            return
        
        import time as _time
        t_total_start = _time.perf_counter()
        
        # Clear previous intensity handles
        for handle in intensity_handles:
            try:
                handle.remove()
            except KeyError:
                pass
        intensity_handles = []
        
        if not show_intensity_map.value:
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click 'Update Intensity Map' to see the legend</div>"
                "</div>"
            )
            return
        
        # Get current values
        wall_dist = wall_dist_slider.value
        grid_size = int(intensity_grid_size.value)
        wall_size = max(int(wall_view_size.value), 80)  # min 80cm to always cover +-40cm export range
        
        leds, absorbers, stl_mesh_for_raytracing = _build_current_leds_and_absorbers()
        
        # Compute intensity with rays_per_pixel from slider
        t_raytrace_start = _time.perf_counter()
        rays_per_pixel = int(intensity_rays_slider.value)
        intensity_grid, actual_wall_size = compute_wall_intensity(
            leds, wall_dist, rays_per_pixel, grid_size, wall_size, absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        t_raytrace_end = _time.perf_counter()
        print(f"  [TIMING] Ray tracing: {t_raytrace_end - t_raytrace_start:.2f}s")
        # Clean up any NaN or Inf values in the grid
        intensity_grid = np.nan_to_num(intensity_grid, nan=0.0, posinf=0.0, neginf=0.0)
        max_lux = intensity_grid.max()  # Grid now contains lux (lm/m²)
        # Use fixed scale; fall back to actual max if it exceeds the cap
        _legend_cap = float(legend_max_input.value)
        color_scale_max = _legend_cap if max_lux <= _legend_cap else max_lux
        
        # Calculate cell area for lux to lumen conversion
        cell_size_cm = actual_wall_size / grid_size
        cell_area_cm2 = cell_size_cm * cell_size_cm
        cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
        
        # === DIAGNOSTIC OUTPUT FOR FLUX CONSERVATION ===
        num_active_leds = sum(1 for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        lumens_per_led = float(led_lumens_slider.value) * float(calibration_factor_slider.value)
        # Apply diffuser transmission loss
        if diffuser_enable_chk.value:
            lumens_per_led *= float(diffuser_transmission_slider.value) / 100.0
        total_emitted_lumens = sum(float(getattr(led, 'lumens', None) or lumens_per_led) for led in leds if not (hasattr(led, 'enabled') and not led.enabled))
        # Convert lux to lumen: multiply each cell by its area and sum
        total_wall_lumens = np.sum(intensity_grid * cell_area_m2)
        conservation_ratio = (total_wall_lumens / total_emitted_lumens * 100) if total_emitted_lumens > 0 else 0
        
        # Calculate 7mm² sensor reading at center
        # Grid contains lux (lm/m²), convert to lumens for sensor area
        sensor_area_cm2 = 0.07  # 7mm² = 0.07cm²
        sensor_area_m2 = sensor_area_cm2 / 10000.0
        
        # Grid now stores lux (lm/m²), convert to lumens for sensor
        center_idx = grid_size // 2
        center_cell_lux = intensity_grid[center_idx, center_idx]
        
        # Convert sensor area to m²
        sensor_area_m2 = sensor_area_cm2 / 10000.0
        
        # Calculate lumens on sensor: Lumen = Lux × Area
        sensor_lumens_from_center_cell = center_cell_lux * sensor_area_m2
        
        print(f"\n=== FLUX CONSERVATION CHECK ===")
        print(f"Active LEDs: {num_active_leds}")
        print(f"Lumens per LED: {lumens_per_led:.1f} lm")
        print(f"Total emitted: {total_emitted_lumens:.1f} lm")
        print(f"Total on wall: {total_wall_lumens:.1f} lm")
        print(f"Conservation: {conservation_ratio:.1f}%")
        print(f"Wall distance: {wall_dist:.1f} cm")
        print(f"7mm² sensor at center: {sensor_lumens_from_center_cell:.4f} lm")
        print(f"================================\n")
        
        cell_size_cm = actual_wall_size / grid_size
        cell_size_m = cell_size_cm / 100.0
        half_size = actual_wall_size / 2
        
        t_viz_start = _time.perf_counter()
        
        # Build a single colored mesh for the entire intensity grid (much faster than per-cell boxes)
        # Each cell = 2 triangles (quad), with vertex colors for smooth rendering
        vertices_list = []
        faces_list = []
        colors_list = []
        
        x_pos = wall_dist / 100.0 - 0.008  # slightly in front of the wall
        vert_idx = 0
        gap = 0.025  # Small gap between cells (2.5% of cell)
        
        for gz in range(grid_size):
            for gy in range(grid_size):
                intensity = intensity_grid[gz, gy]
                if intensity > 0:
                    color = intensity_to_color(intensity, color_scale_max)
                    color_uint8 = [int(c * 255) for c in color] + [255]
                    
                    y_center = (-half_size + gy * cell_size_cm + cell_size_cm / 2) / 100.0
                    z_center = (-half_size + gz * cell_size_cm + cell_size_cm / 2) / 100.0
                    half_cell = cell_size_m * 0.5 * (1.0 - gap)
                    
                    # 4 corners of the quad
                    v0 = [x_pos, y_center - half_cell, z_center - half_cell]
                    v1 = [x_pos, y_center + half_cell, z_center - half_cell]
                    v2 = [x_pos, y_center + half_cell, z_center + half_cell]
                    v3 = [x_pos, y_center - half_cell, z_center + half_cell]
                    
                    vertices_list.extend([v0, v1, v2, v3])
                    # Double-sided: both winding orders so visible from any angle
                    faces_list.append([vert_idx, vert_idx + 1, vert_idx + 2])
                    faces_list.append([vert_idx, vert_idx + 2, vert_idx + 3])
                    faces_list.append([vert_idx, vert_idx + 2, vert_idx + 1])
                    faces_list.append([vert_idx, vert_idx + 3, vert_idx + 2])
                    colors_list.extend([color_uint8] * 4)
                    vert_idx += 4
        
        if len(vertices_list) > 0:
            vertices_np = np.array(vertices_list, dtype=np.float32)
            faces_np = np.array(faces_list, dtype=np.uint32)
            colors_np = np.array(colors_list, dtype=np.uint8)
            
            intensity_mesh = trimesh.Trimesh(
                vertices=vertices_np,
                faces=faces_np,
                process=False
            )
            from trimesh.visual import ColorVisuals
            intensity_mesh.visual = ColorVisuals(mesh=intensity_mesh, vertex_colors=colors_np)
            
            handle = server.scene.add_mesh_trimesh(
                name="/intensity_map",
                mesh=intensity_mesh,
                visible=True,
            )
            intensity_handles.append(handle)
        
        t_viz_end = _time.perf_counter()
        print(f"  [TIMING] Visualization: {t_viz_end - t_viz_start:.2f}s ({vert_idx // 4} cells)")
        print(f"  [TIMING] Total update_intensity_map: {t_viz_end - t_total_start:.2f}s")
        
        # Update legend (grid now stores lux = lm/m²)
        legend = _build_lux_legend_html(max_lux, color_scale_max, cell_area_m2, cell_caption="lm/cell")

        # Cache grid so FOV / VIO / threshold changes can recalculate metrics cheaply
        _last_intensity_cache['grid'] = intensity_grid
        _last_intensity_cache['wall_size_cm'] = actual_wall_size
        _last_intensity_cache['wall_dist'] = wall_dist
        _last_intensity_cache['cell_area_m2'] = cell_area_m2
        _last_intensity_cache['max_lux'] = max_lux
        _last_intensity_cache['color_scale_max'] = color_scale_max

        legend_html.content = legend + _wall_metrics_html(
            intensity_grid, actual_wall_size, wall_dist,
        )

    # ── CSV Pattern Import logic ──────────────────────────────────────────
    # --- CSV pattern import (benchmark / FOV captures), lux-matrix export and the multi-distance benchmark. (see ui/csv_overlay.py) ---
    _csv_overlay_ns = _csv_overlay.build(_SimpleNamespace(
        _build_current_leds_and_absorbers=_build_current_leds_and_absorbers,
        _last_intensity_cache=_last_intensity_cache,
        compute_wall_intensity=compute_wall_intensity,
        csv_diff_html=csv_diff_html,
        csv_import_path=csv_import_path,
        csv_import_status=csv_import_status,
        csv_legend_html=csv_legend_html,
        csv_legend_max_input=csv_legend_max_input,
        imported_csv_handles=imported_csv_handles,
        intensity_grid_size=intensity_grid_size,
        intensity_rays_slider=intensity_rays_slider,
        intensity_to_color=intensity_to_color,
        room_front_dist=room_front_dist,
        room_mode_enable=room_mode_enable,
        server=server,
        wall_dist_slider=wall_dist_slider,
        wall_view_size=wall_view_size,
    ))
    clear_csv_pattern = _csv_overlay_ns.clear_csv_pattern
    export_lux_matrix = _csv_overlay_ns.export_lux_matrix
    import_csv_pattern = _csv_overlay_ns.import_csv_pattern
    run_benchmark = _csv_overlay_ns.run_benchmark
    # --- Room mode (see ui/room_mode.py) ---
    _room_mode_ns = _room_mode.build(_SimpleNamespace(
        _build_lux_legend_html=_build_lux_legend_html,
        _build_stl_transform=_build_stl_transform,
        _last_room_cache=_last_room_cache,
        _room_metrics_html=_room_metrics_html,
        abs0_off_x=abs0_off_x,
        abs0_off_y=abs0_off_y,
        abs0_off_z=abs0_off_z,
        abs1_off_x=abs1_off_x,
        abs1_off_y=abs1_off_y,
        abs1_off_z=abs1_off_z,
        abs2_off_x=abs2_off_x,
        abs2_off_y=abs2_off_y,
        abs2_off_z=abs2_off_z,
        abs2_rot_z=abs2_rot_z,
        abs3_off_x=abs3_off_x,
        abs3_off_y=abs3_off_y,
        abs3_off_z=abs3_off_z,
        abs3_rot_z=abs3_rot_z,
        absorbers_enable=absorbers_enable,
        circle_center_slider=circle_center_slider,
        compute_room_intensity=compute_room_intensity,
        current_leds=current_leds,
        global_rotation_z_slider=global_rotation_z_slider,
        intensity_rays_slider=intensity_rays_slider,
        intensity_to_color=intensity_to_color,
        legend_html=legend_html,
        legend_max_input=legend_max_input,
        radius_slider=radius_slider,
        room_back_dist=room_back_dist,
        room_front_dist=room_front_dist,
        room_grid_size=room_grid_size,
        room_intensity_handles=room_intensity_handles,
        room_mode_enable=room_mode_enable,
        room_side_dist=room_side_dist,
        room_top_bottom_dist=room_top_bottom_dist,
        room_wall_handles=room_wall_handles,
        server=server,
        show_back_wall=show_back_wall,
        show_room_intensity=show_room_intensity,
        show_room_walls=show_room_walls,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
    ))
    draw_room_walls = _room_mode_ns.draw_room_walls
    update_room_intensity_map = _room_mode_ns.update_room_intensity_map
    def _clear_inspector():
        for h in _inspector_handles:
            try:
                h.remove()
            except Exception:
                pass
        _inspector_handles.clear()

    def _inspector_add(handle):
        _inspector_handles.append(handle)
        return handle

    def _mirror_slider(label, src, minv, maxv, step):
        sl = server.gui.add_slider(label, min=minv, max=maxv, step=step, initial_value=src.value)
        _inspector_add(sl)

        def _on(_):
            if _inspector_syncing[0]:
                return
            _inspector_syncing[0] = True
            try:
                src.value = sl.value
            finally:
                _inspector_syncing[0] = False

        sl.on_update(_on)
        return sl

    def _mirror_checkbox(label, src):
        chk = server.gui.add_checkbox(label, initial_value=bool(src.value))
        _inspector_add(chk)

        def _on(_):
            if _inspector_syncing[0]:
                return
            _inspector_syncing[0] = True
            try:
                src.value = chk.value
            finally:
                _inspector_syncing[0] = False

        chk.on_update(_on)
        return chk

    def _inspector_led_matrix(group):
        """ALL / Row / LED on-off buttons that write into group['led_states']."""
        led_states_g = group['led_states']
        led_rows = group.get('led_rows', [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        num_leds = group.get('num_leds', len(led_states_g))

        def _refresh_hidden_colors():
            if callable(group.get('update_button_colors')):
                group['update_button_colors']()
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and group in _pdata.get('groups', []):
                    uf = _pdata.get('update_btn_colors')
                    if callable(uf):
                        uf()

        _inspector_add(server.gui.add_html("<hr style='margin:6px 0;'><b>LED Controls:</b>"))
        all_btn = _inspector_add(server.gui.add_button(
            "ALL LEDs", color="#FF00FF" if any(led_states_g) else "#666666"
        ))

        def _on_all(_):
            new_state = not all(led_states_g)
            for i in range(len(led_states_g)):
                led_states_g[i] = new_state
            _refresh_hidden_colors()
            update_scene()
            populate_inspector(selected_owner[0])

        all_btn.on_click(_on_all)

        for row_idx, led_indices in enumerate(led_rows):
            any_on = any(led_states_g[i] for i in led_indices if i < len(led_states_g))
            row_btn = _inspector_add(server.gui.add_button(
                f"Row {row_idx + 1}", color="#FF00FF" if any_on else "#666666"
            ))

            def _make_row(indices):
                def _on(_):
                    new_state = not all(led_states_g[i] for i in indices if i < len(led_states_g))
                    for i in indices:
                        if i < len(led_states_g):
                            led_states_g[i] = new_state
                    _refresh_hidden_colors()
                    update_scene()
                    populate_inspector(selected_owner[0])
                return _on

            row_btn.on_click(_make_row(list(led_indices)))

        for led_idx in range(num_leds):
            if led_idx >= len(led_states_g):
                break
            color = "#FF00FF" if led_states_g[led_idx] else "#444444"
            led_btn = _inspector_add(server.gui.add_button(f"L{led_idx + 1}", color=color))

            def _make_led(idx):
                def _on(_):
                    led_states_g[idx] = not led_states_g[idx]
                    _refresh_hidden_colors()
                    update_scene()
                    populate_inspector(selected_owner[0])
                return _on

            led_btn.on_click(_make_led(led_idx))

    def _inspector_mirror_checkbox(owner):
        """Checkbox making `owner` the mirror primary (XZ reflection).

        Enabling it disables the panel on the opposite side and renders a live
        mirrored copy of this panel there instead."""
        is_primary = _mirror_primary[0] == owner
        chk = _inspector_add(server.gui.add_checkbox(
            "Mirror to other side (XZ)", initial_value=is_primary
        ))
        if is_primary:
            cp_name = _mirror_counterpart_name[0]
            note = (
                f"Mirroring active — **{cp_name}** is disabled."
                if cp_name else
                "Mirroring active — no opposite-side panel found to disable."
            )
            _inspector_add(server.gui.add_markdown(f"*{note}*"))

        @chk.on_update
        def _(_):
            if loading_in_progress[0]:
                return
            if chk.value:
                _enable_mirror_for(owner)
            elif _mirror_primary[0] == owner:
                _clear_mirror_state()
            update_scene()
            if selected_owner[0] == owner:
                populate_inspector(owner)

    def _inspector_group_pose_sliders(group):
        """Free XYZ/Euler sliders; hidden while the circular guide is anchoring the panel."""
        if _guide_is_enabled(group):
            return
        _mirror_slider("Position X (cm)", group['pos_x'], -100, 100, 0.1)
        _mirror_slider("Position Y (cm)", group['pos_y'], -50, 50, 0.1)
        _mirror_slider("Position Z (cm)", group['pos_z'], -50, 50, 0.1)
        if group.get('rot_tilt_lr') is not None:
            _mirror_slider("Tilt Left/Right (°)", group['rot_tilt_lr'], -180, 180, 1)
        if group.get('rot_tilt_ud') is not None:
            _mirror_slider("Tilt Up/Down (°)", group['rot_tilt_ud'], -180, 180, 1)
        if group.get('rot_roll') is not None:
            _mirror_slider("Rotate on axis (°)", group['rot_roll'], -180, 180, 1)

    def _inspector_guide_controls(group):
        """Checkbox to fit/anchor a circular guide and the slide slider while on."""
        enabled = _guide_is_enabled(group)
        chk = _inspector_add(server.gui.add_checkbox(
            "Anchor to circular guide", initial_value=enabled
        ))
        if group.get('guide_error'):
            _inspector_add(server.gui.add_markdown(f"**Guide:** {group['guide_error']}"))
        if enabled:
            guide = group.get('guide') or {}
            if guide.get('warning'):
                _inspector_add(server.gui.add_markdown(f"*{guide['warning']}*"))
            n_circ = len(guide.get('circles') or [])
            _inspector_add(server.gui.add_markdown(
                f"Pose locked — {n_circ} construction circle(s). "
                "Uncheck to restore free XYZ/Euler."
            ))
            sl = _inspector_add(server.gui.add_slider(
                "Slide along guide (°)",
                min=-180, max=180, step=0.5,
                initial_value=float(guide.get('theta_deg', 0.0)),
            ))

            @sl.on_update
            def _(_):
                if loading_in_progress[0] or _inspector_syncing[0]:
                    return
                if not isinstance(group.get('guide'), dict):
                    return
                group['guide']['theta_deg'] = float(sl.value)
                update_scene()

        @chk.on_update
        def _(_):
            if loading_in_progress[0] or _inspector_syncing[0]:
                return
            if chk.value:
                ok, err = enable_circular_guide(group)
                if not ok:
                    group['guide'] = None
                    group['guide_error'] = err or "Could not fit a circular guide."
                else:
                    group['guide_error'] = None
            else:
                bake_and_disable_guide(group)
            update_scene()
            if selected_owner[0] is not None:
                populate_inspector(selected_owner[0])

    def _designer_rotation_matrix(rx_deg, ry_deg, rz_deg):
        """LED-local Euler convention used by the designer: Rx @ Ry @ Rz."""
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def _designer_euler_from_axes(direction, row_direction=None):
        """Recover designer Euler values from an LED forward and row axis."""
        forward = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
        if row_direction is None:
            row = np.array([0.0, 1.0, 0.0])
        else:
            row = np.asarray(row_direction, dtype=float)
            row -= forward * np.dot(row, forward)
            if np.linalg.norm(row) < 1e-9:
                row = np.array([0.0, 1.0, 0.0])
        row /= np.linalg.norm(row)
        third = np.cross(forward, row)
        R = np.column_stack([forward, row, third])
        ry = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
        if abs(np.cos(ry)) > 1e-6:
            rx = np.arctan2(-R[1, 2], R[2, 2])
            rz = np.arctan2(-R[0, 1], R[0, 0])
        else:
            rx, rz = np.arctan2(R[2, 1], R[1, 1]), 0.0
        return tuple(float(v) for v in np.degrees([rx, ry, rz]))

    def _designer_quaternion(R):
        """Return a wxyz quaternion for an orthonormal 3x3 rotation matrix."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return (0.25 / s, (R[2, 1] - R[1, 2]) * s,
                    (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s)
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                    (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
        if i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                    0.25 * s, (R[1, 2] + R[2, 1]) / s)
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s, 0.25 * s)

    def _designer_matrix_from_wxyz(wxyz):
        """Convert a wxyz quaternion to a 3x3 rotation matrix."""
        w, x, y, z = [float(v) for v in wxyz]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=float)

    def _clear_designer_gizmo():
        if designer_gizmo[0] is not None:
            try:
                designer_gizmo[0].remove()
            except (KeyError, AttributeError):
                pass
            designer_gizmo[0] = None

    def _clear_designer_scene():
        _clear_designer_gizmo()
        for nodes in designer_led_nodes:
            for key in ('box', 'frame'):
                try:
                    nodes[key].remove()
                except (KeyError, AttributeError):
                    pass
        designer_led_nodes.clear()
        for handle in designer_scene_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_scene_handles.clear()

    def _designer_led_pose(led):
        """Return (position_m, wxyz) for an LED in designer state."""
        R = _designer_rotation_matrix(led['rx'], led['ry'], led['rz'])
        pos = (led['x'] / 100.0, led['y'] / 100.0, led['z'] / 100.0)
        return pos, _designer_quaternion(R)

    def _update_designer_led_pose(index):
        """In-place pose update for one LED box + frame (no recreate)."""
        if index is None or index < 0 or index >= len(designer_led_nodes):
            return
        led = designer_state[0]['leds'][index]
        pos, wxyz = _designer_led_pose(led)
        nodes = designer_led_nodes[index]
        nodes['box'].position = pos
        nodes['box'].wxyz = wxyz
        nodes['frame'].position = pos
        nodes['frame'].wxyz = wxyz

    def _update_designer_selection_colors():
        selected = designer_state[0]['selected_led']
        for index, nodes in enumerate(designer_led_nodes):
            try:
                # Viser expects 0-255 when assigning .color (floats stay near-black).
                nodes['box'].color = (38, 255, 255) if index == selected else (255, 255, 255)
            except Exception:
                pass

    def _sync_widgets_from_led(led):
        """Push LED state into slider/number widgets without feedback loops."""
        refs = designer_widget_refs[0]
        if not refs:
            return
        designer_syncing[0] = True
        try:
            for key, pair in refs.items():
                if key not in led:
                    continue
                slider, number = pair
                val = led[key]
                try:
                    slider.value = val
                except Exception:
                    pass
                try:
                    number.value = val
                except Exception:
                    pass
        finally:
            designer_syncing[0] = False

    def _place_gizmo():
        """Create/move the transform gizmo onto the selected LED (LED-local axes)."""
        index = designer_state[0]['selected_led']
        leds = designer_state[0]['leds']
        if index is None or index < 0 or index >= len(leds):
            _clear_designer_gizmo()
            return
        led = leds[index]
        pos, wxyz = _designer_led_pose(led)
        if designer_gizmo[0] is None:
            gizmo = server.scene.add_transform_controls(
                "/panel_designer/gizmo",
                scale=0.08,
                depth_test=False,
                position=pos,
                wxyz=wxyz,
            )

            @gizmo.on_update
            def _on_gizmo_update(_):
                if designer_syncing[0] or not designer_mode[0]:
                    return
                i = designer_state[0]['selected_led']
                if i is None or i >= len(designer_state[0]['leds']):
                    return
                cur = designer_state[0]['leds'][i]
                gx, gy, gz = designer_gizmo[0].position
                cur['x'], cur['y'], cur['z'] = gx * 100.0, gy * 100.0, gz * 100.0
                Rm = _designer_matrix_from_wxyz(designer_gizmo[0].wxyz)
                rx, ry, rz = _designer_euler_from_axes(Rm @ np.array([1.0, 0.0, 0.0]),
                                                      Rm @ np.array([0.0, 1.0, 0.0]))
                cur['rx'], cur['ry'], cur['rz'] = rx, ry, rz
                _update_designer_led_pose(i)
                _sync_widgets_from_led(cur)

            @gizmo.on_drag_end
            def _on_gizmo_drag_end(_):
                if not designer_mode[0]:
                    return
                i = designer_state[0]['selected_led']
                if i is None or i >= len(designer_state[0]['leds']):
                    return
                cur = designer_state[0]['leds'][i]
                # Snap to the same steps as the UI controls.
                cur['x'] = round(cur['x'] / 0.05) * 0.05
                cur['y'] = round(cur['y'] / 0.05) * 0.05
                cur['z'] = round(cur['z'] / 0.05) * 0.05
                cur['rx'] = round(cur['rx'] / 0.5) * 0.5
                cur['ry'] = round(cur['ry'] / 0.5) * 0.5
                cur['rz'] = round(cur['rz'] / 0.5) * 0.5
                _update_designer_led_pose(i)
                _sync_widgets_from_led(cur)
                designer_syncing[0] = True
                try:
                    pos2, wxyz2 = _designer_led_pose(cur)
                    designer_gizmo[0].position = pos2
                    designer_gizmo[0].wxyz = wxyz2
                finally:
                    designer_syncing[0] = False

            designer_gizmo[0] = gizmo
        else:
            designer_syncing[0] = True
            try:
                designer_gizmo[0].position = pos
                designer_gizmo[0].wxyz = wxyz
            finally:
                designer_syncing[0] = False

    def _select_designer_led_index(index, rebuild_ui=True):
        """Select an LED for editing; move gizmo and refresh inspector."""
        _just_clicked_mesh[0] = True
        designer_state[0]['selected_led'] = index
        _update_designer_selection_colors()
        _place_gizmo()
        if rebuild_ui:
            _build_designer_ui()

    def update_designer_scene(full_rebuild=True):
        """Render the isolated panel. full_rebuild recreates LED meshes + gizmo."""
        if not designer_mode[0]:
            _clear_designer_scene()
            return
        leds = designer_state[0]['leds']
        if full_rebuild:
            _clear_designer_scene()
            extent_cm = max(10.0, max(
                (max(abs(led[a]) for a in ('x', 'y', 'z')) + led['size']
                 for led in leds), default=5.0
            ))
            designer_scene_handles.append(server.scene.add_grid(
                "/panel_designer/grid", width=extent_cm * 2 / 100.0,
                height=extent_cm * 2 / 100.0, plane="yz", cell_size=0.01,
            ))
            designer_scene_handles.append(server.scene.add_frame(
                "/panel_designer/panel_reference_frame", axes_length=0.05,
                axes_radius=0.002, origin_radius=0.004,
            ))
            for index, led in enumerate(leds):
                pos, wxyz = _designer_led_pose(led)
                selected = index == designer_state[0]['selected_led']
                # Sibling paths (not child of the box) so poses are not double-applied.
                box = server.scene.add_box(
                    f"/panel_designer/led_{index}",
                    dimensions=(0.0005, led['size'] / 100.0, led['size'] / 100.0),
                    color=(38, 255, 255) if selected else (255, 255, 255),
                    position=pos,
                    wxyz=wxyz,
                )
                frame = server.scene.add_frame(
                    f"/panel_designer/frame_{index}",
                    axes_length=max(0.01, led['size'] / 150.0), axes_radius=0.001,
                    origin_radius=0.002,
                    position=pos,
                    wxyz=wxyz,
                )

                def _on_led_click(_event, selected_index=index):
                    _select_designer_led_index(selected_index)

                box.on_click(_on_led_click)
                designer_led_nodes.append({'box': box, 'frame': frame})
            _place_gizmo()
        else:
            for index in range(len(leds)):
                _update_designer_led_pose(index)
            _update_designer_selection_colors()
            _place_gizmo()

    def _designer_set_camera(view):
        """Place every connected client on an orthographic-style panel view."""
        leds = designer_state[0]['leds']
        extent = max(0.3, max(
            (max(abs(led[a]) for a in ('x', 'y', 'z')) / 100.0 + led['size'] / 100.0
             for led in leds), default=0.1
        ) * 3)
        views = {
            'XY': ((0, 0, extent), (0, 1, 0)),
            '-XY': ((0, 0, -extent), (0, 1, 0)),
            'XZ': ((0, extent, 0), (0, 0, 1)),
            '-XZ': ((0, -extent, 0), (0, 0, 1)),
            'YZ': ((extent, 0, 0), (0, 0, 1)),
            '-YZ': ((-extent, 0, 0), (0, 0, 1)),
        }
        position, up = views[view]
        for client in server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = (0, 0, 0)
            client.camera.up_direction = up

    def _designer_apply_to_group(group, leds):
        """Replace one dynamic group's local LED arrays from designer LEDs."""
        group['is_dynamic'] = True
        group['num_leds'] = len(leds)
        group['led_positions'] = [(led['x'], led['y'], led['z']) for led in leds]
        group['original_led_positions'] = list(group['led_positions'])
        rotations = []
        rows = []
        for led in leds:
            R = _designer_rotation_matrix(led['rx'], led['ry'], led['rz'])
            rotations.append(tuple(R @ np.array([1.0, 0.0, 0.0])))
            rows.append(tuple(R @ np.array([0.0, 1.0, 0.0])))
        group['led_rotations'] = rotations
        group['original_led_rotations'] = list(rotations)
        group['led_row_directions'] = rows
        group['original_led_row_directions'] = list(rows)
        group['led_euler_angles'] = [(led['rx'], led['ry'], led['rz']) for led in leds]
        group['led_sizes'] = [led['size'] for led in leds]
        group['led_viewing_angles'] = [led['view_angle'] for led in leds]
        group['led_lumens'] = [led['lumens'] if led['custom_lumens'] else None for led in leds]
        group['led_states'] = [True] * len(leds)
        group['led_rows'] = [list(range(len(leds)))] if leds else []

    def _designer_template_group():
        """Return a serializable dynamic group for Save As Template."""
        leds = designer_state[0]['leds']
        temp = {'is_dynamic': True, 'position': [0, 0, 0],
                'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0,
                'enabled': True}
        _designer_apply_to_group(temp, leds)
        return temp

    def _save_designer_template(_event=None):
        name = designer_state[0]['name'].strip() or 'Untitled Panel'
        save_custom_group_template(name, [_designer_template_group()], [])
        fresh = get_available_templates()
        template_dropdown.options = ["Empty"] + fresh
        for dropdown in _panel_dropdowns:
            current = dropdown.value
            dropdown.options = ["-- Nessuno --"] + fresh
            dropdown.value = current if current in dropdown.options else "-- Nessuno --"
        print(f"✓ Panel Designer template saved: {name}")

    def _save_designer(_event=None):
        state = designer_state[0]
        by_group = {}
        for led in state['leds']:
            group = led.get('_group')
            if group is not None:
                by_group.setdefault(id(group), [group, []])[1].append(led)
            elif led.get('_individual') is not None:
                item = led['_individual']
                item['pos_x'].value, item['pos_y'].value, item['pos_z'].value = led['x'], led['y'], led['z']
                item['rot_x'].value, item['rot_y'].value, item['rot_z'].value = led['rx'], led['ry'], led['rz']
                item['size'].value, item['viewing_angle'].value = led['size'], led['view_angle']
                item['lumens_override'].value = led['custom_lumens']
                item['lumens_value'].value = led['lumens']
        if state['editing_owner'] is None:
            group = create_custom_group(skip_update_scene=True, num_leds=max(1, len(state['leds'])),
                                        led_rows=[list(range(len(state['leds'])))] if state['leds'] else [[0]],
                                        group_name=state['name'])
            _designer_apply_to_group(group, state['leds'])
            group['template_name'] = state['name']
            selected_owner[0] = ('custom_group', group['id'])
        else:
            for group, group_leds in by_group.values():
                _designer_apply_to_group(group, group_leds)
        _exit_designer()

    def _collect_normal_scene_visibility_targets():
        """Handles that should be hidden while the panel designer is open."""
        targets = list(static_scene_handles)
        try:
            targets.append(wall_handle)
        except Exception:
            pass
        if stl_mesh_handle[0] is not None:
            targets.append(stl_mesh_handle[0])
        targets.extend(room_wall_handles)
        targets.extend(intensity_handles)
        targets.extend(room_intensity_handles)
        targets.extend(imported_csv_handles)
        return targets

    def _set_normal_scene_visible(visible):
        for handle in _collect_normal_scene_visibility_targets():
            try:
                handle.visible = visible
            except Exception:
                pass

    def _clear_normal_dynamic_scene():
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles, vio_fov_handles, guide_handles
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        led_handles = []
        ray_handles = []
        absorber_handles = []
        camera_fov_handles = []
        vio_fov_handles = []
        guide_handles = []

    def _exit_designer(_event=None):
        designer_mode[0] = False
        designer_widget_refs[0] = {}
        _clear_designer_scene()
        for handle in designer_ui_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_ui_handles.clear()
        _clear_inspector()
        _set_normal_scene_visible(True)
        update_scene()
        populate_inspector(selected_owner[0])

    def _build_designer_ui():
        """Rebuild designer controls inside the floating Selected inspector."""
        _clear_inspector()
        for handle in designer_ui_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        designer_ui_handles.clear()
        designer_widget_refs[0] = {}
        state = designer_state[0]
        with inspector_tab:
            _inspector_add(server.gui.add_markdown("**Panel Designer**"))
            exit_btn = _inspector_add(server.gui.add_button("Exit Designer", color="red"))
            exit_btn.on_click(_exit_designer)

            name_input = _inspector_add(server.gui.add_text("Panel name", initial_value=state['name']))
            name_input.on_update(lambda _: state.__setitem__('name', name_input.value))

            _inspector_add(server.gui.add_markdown("**View plane**"))
            for view in ('XY', 'XZ', 'YZ', '-XY', '-XZ', '-YZ'):
                button = _inspector_add(server.gui.add_button(view))
                button.on_click(lambda _, v=view: _designer_set_camera(v))

            _inspector_add(server.gui.add_markdown("**LEDs** — click a mesh or button; drag the gizmo arrows/rings"))
            for index, _led in enumerate(state['leds']):
                if index == state['selected_led']:
                    button = _inspector_add(server.gui.add_button(f"LED {index + 1}", color="#00CCCC"))
                else:
                    button = _inspector_add(server.gui.add_button(f"LED {index + 1}"))

                def _choose(_event, selected_index=index):
                    _select_designer_led_index(selected_index)

                button.on_click(_choose)

            add_btn = _inspector_add(server.gui.add_button("Add LED", color="green"))
            duplicate_btn = _inspector_add(server.gui.add_button("Duplicate LED"))
            remove_btn = _inspector_add(server.gui.add_button("Remove LED", color="red"))

            def _add(_):
                state['leds'].append({
                    'x': 0.0, 'y': 0.0, 'z': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0,
                    'size': 0.5, 'view_angle': 120.0, 'custom_lumens': False,
                    'lumens': 100.0, '_group': state.get('target_group'),
                })
                state['selected_led'] = len(state['leds']) - 1
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            add_btn.on_click(_add)

            def _duplicate(_):
                i = state['selected_led']
                if i is None:
                    return
                new_led = dict(state['leds'][i])
                new_led['y'] += 0.5
                state['leds'].append(new_led)
                state['selected_led'] = len(state['leds']) - 1
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            duplicate_btn.on_click(_duplicate)

            def _remove(_):
                i = state['selected_led']
                if i is None:
                    return
                state['leds'].pop(i)
                state['selected_led'] = min(i, len(state['leds']) - 1) if state['leds'] else None
                update_designer_scene(full_rebuild=True)
                _build_designer_ui()

            remove_btn.on_click(_remove)

            i = state['selected_led']
            if i is not None and i < len(state['leds']):
                led = state['leds'][i]
                _inspector_add(server.gui.add_markdown(
                    f"**LED {i + 1}** — drag gizmo (LED-local axes) or edit values"
                ))
                for label, key, low, high, step in (
                    ('Position X (cm)', 'x', -30, 30, 0.05),
                    ('Position Y (cm)', 'y', -30, 30, 0.05),
                    ('Position Z (cm)', 'z', -30, 30, 0.05),
                    ('Rotation X (°)', 'rx', -180, 180, 0.5),
                    ('Rotation Y (°)', 'ry', -180, 180, 0.5),
                    ('Rotation Z (°)', 'rz', -180, 180, 0.5),
                    ('Square side (cm)', 'size', 0.1, 10, 0.05),
                    ('Viewing angle (°)', 'view_angle', 1, 180, 1),
                ):
                    slider = _inspector_add(server.gui.add_slider(
                        label, min=low, max=high, step=step, initial_value=led[key]
                    ))
                    number = _inspector_add(server.gui.add_number(
                        f"{label} value", initial_value=led[key], step=step
                    ))
                    designer_widget_refs[0][key] = (slider, number)

                    def _bind(source, other, field, needs_rebuild):
                        def _update(_):
                            if designer_syncing[0]:
                                return
                            designer_syncing[0] = True
                            try:
                                led[field] = float(source.value)
                                other.value = source.value
                            finally:
                                designer_syncing[0] = False
                            if needs_rebuild:
                                update_designer_scene(full_rebuild=True)
                            else:
                                _update_designer_led_pose(state['selected_led'])
                                _place_gizmo()
                        source.on_update(_update)

                    needs_rebuild = key in ('size', 'view_angle')
                    _bind(slider, number, key, needs_rebuild)
                    _bind(number, slider, key, needs_rebuild)

                lumens_check = _inspector_add(server.gui.add_checkbox(
                    "Custom lumens", initial_value=led['custom_lumens']
                ))
                lumens_value = _inspector_add(server.gui.add_number(
                    "Lumens value", initial_value=led['lumens'], min=1, step=1
                ))
                lumens_check.on_update(lambda _: led.__setitem__('custom_lumens', lumens_check.value))
                lumens_value.on_update(lambda _: led.__setitem__('lumens', float(lumens_value.value)))

            _inspector_add(server.gui.add_html("<hr style='margin:8px 0;'>"))
            save_btn = _inspector_add(server.gui.add_button("Save", color="green"))
            template_btn = _inspector_add(server.gui.add_button("Save As Template"))
            cancel_btn = _inspector_add(server.gui.add_button("Cancel", color="red"))
            save_btn.on_click(_save_designer)
            template_btn.on_click(_save_designer_template)
            cancel_btn.on_click(_exit_designer)

    def _enter_panel_designer(owner=None):
        """Open an empty designer or extract a selected panel's local LEDs."""
        state = {'name': 'Untitled Panel', 'editing_owner': owner, 'leds': [],
                 'selected_led': None, 'target_group': None}
        groups = _owner_groups(owner) if owner is not None else []
        for group in groups:
            if not group.get('is_dynamic', False):
                continue
            state['name'] = group.get('template_name') or _owner_display_name(owner)
            positions = group.get('original_led_positions', group.get('led_positions', []))
            eulers = group.get('led_euler_angles', [])
            directions = group.get('original_led_rotations', group.get('led_rotations', []))
            rows = group.get('original_led_row_directions', group.get('led_row_directions', []))
            sizes = group.get('led_sizes', [])
            view_angles = group.get('led_viewing_angles', [])
            lumens_list = group.get('led_lumens', [])
            for i, pos in enumerate(positions):
                euler = eulers[i] if i < len(eulers) else _designer_euler_from_axes(
                    directions[i] if i < len(directions) else (1, 0, 0),
                    rows[i] if i < len(rows) else None,
                )
                state['leds'].append({
                    'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
                    'rx': float(euler[0]), 'ry': float(euler[1]), 'rz': float(euler[2]),
                    'size': float(sizes[i]) if i < len(sizes) else 0.5,
                    'view_angle': float(view_angles[i]) if i < len(view_angles) else 120.0,
                    'custom_lumens': i < len(lumens_list) and lumens_list[i] is not None,
                    'lumens': float(lumens_list[i]) if i < len(lumens_list) and lumens_list[i] is not None else 100.0,
                    '_group': group,
                })
            state['target_group'] = state['target_group'] or group
        for item in _owner_individual_leds(owner):
            state['name'] = _owner_display_name(owner)
            state['leds'].append({
                'x': item['pos_x'].value, 'y': item['pos_y'].value, 'z': item['pos_z'].value,
                'rx': item['rot_x'].value, 'ry': item['rot_y'].value, 'rz': item['rot_z'].value,
                'size': item['size'].value, 'view_angle': item['viewing_angle'].value,
                'custom_lumens': item['lumens_override'].value,
                'lumens': item['lumens_value'].value, '_individual': item,
            })
        if owner is not None and not state['leds']:
            print("Panel Designer requires a dynamic panel or individual LEDs.")
            return
        if state['leds']:
            state['selected_led'] = 0
        designer_state[0] = state
        designer_mode[0] = True
        # Isolate the view without destroying static nodes (wall/grid/axes/STL).
        _clear_normal_dynamic_scene()
        _set_normal_scene_visible(False)
        _build_designer_ui()
        update_designer_scene(full_rebuild=True)
        _designer_set_camera('YZ')

    open_panel_designer_btn.on_click(lambda _: _enter_panel_designer())

    def populate_inspector(owner):
        """Rebuild the floating inspector for the given owner (or empty state)."""
        if designer_mode[0]:
            # Designer owns the Selected tab while active.
            return
        _clear_inspector()
        with inspector_tab:
            if owner is None:
                _inspector_add(server.gui.add_markdown(
                    "Click a panel or LED group in the 3D view to edit it."
                ))
                return

            kind, key = owner
            deselect_btn = _inspector_add(server.gui.add_button("Deselect"))

            @deselect_btn.on_click
            def _(_):
                select_panel(None)

            if kind == 'slot':
                data = _panel_slot_data[key] if 0 <= key < len(_panel_slot_data) else None
                slot_groups = [g for g in custom_groups if g.get('panel_slot') == key]
                if data is None and not slot_groups:
                    _inspector_add(server.gui.add_markdown("This slot is empty."))
                    return
                if data is None:
                    # Panel restored from a saved configuration (no slot UI):
                    # expose its group controls directly so it stays editable.
                    slot_name = _ELIOS3_SLOTS[key]['name']
                    _inspector_add(server.gui.add_markdown(f"**Slot {slot_name}**"))
                    _inspector_mirror_checkbox(owner)
                    for g_idx, group in enumerate(slot_groups):
                        if len(slot_groups) > 1:
                            _inspector_add(server.gui.add_html(
                                f"<hr style='margin:8px 0;'><b>Group {g_idx + 1}</b>"
                            ))
                        _mirror_checkbox("Enable", group['enable'])
                        _inspector_guide_controls(group)
                        _inspector_group_pose_sliders(group)
                        if group.get('lumens_override') is not None:
                            _mirror_checkbox("Enable custom lumens", group['lumens_override'])
                            _mirror_slider("Lumens per LED (lm)", group['lumens_value'], 1, 900000, 1)
                        _inspector_led_matrix(group)
                    return
                slot_name = data.get('slot_label', _ELIOS3_SLOTS[key]['name'])
                tmpl = data.get('template_display', data.get('template_name', ''))
                _inspector_add(server.gui.add_markdown(
                    f"**Slot {slot_name}**  \n{tmpl}"
                    + ("  \n*Loaded as individual LEDs*" if data.get('as_individual') else "")
                ))
                edit_designer_btn = _inspector_add(server.gui.add_button("Edit in Panel Designer"))
                edit_designer_btn.on_click(lambda _, o=owner: _enter_panel_designer(o))
                _inspector_mirror_checkbox(owner)
                ctrl = data.get('controls') or {}
                slot_groups_live = data.get('groups') or slot_groups
                slot_anchored = any(_guide_is_enabled(g) for g in slot_groups_live)
                if ctrl.get('enable') is not None:
                    _mirror_checkbox("Enable Slot", ctrl['enable'])
                    if slot_anchored:
                        _inspector_add(server.gui.add_markdown(
                            "Slot offset/rotation locked while a group is anchored to a circular guide."
                        ))
                    else:
                        _mirror_slider("Offset X (cm)", ctrl['pos_x'], -50, 50, 0.1)
                        _mirror_slider("Offset Y (cm)", ctrl['pos_y'], -50, 50, 0.1)
                        _mirror_slider("Offset Z (cm)", ctrl['pos_z'], -50, 50, 0.1)
                        _mirror_slider("Rotate on axis (°)", ctrl['rot_x'], -180, 180, 1)
                        _mirror_slider("Tilt Up/Down (°)", ctrl['rot_y'], -180, 180, 1)
                        _mirror_slider("Tilt Left/Right (°)", ctrl['rot_z'], -180, 180, 1)
                    _inspector_add(server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>"))
                    _mirror_checkbox("Enable custom lumens", ctrl['lumens_chk'])
                    _mirror_slider("Lumens per LED (lm)", ctrl['lumens_slider'], 1, 900000, 1)
                if data.get('as_individual'):
                    _inspector_add(server.gui.add_markdown(
                        "Per-LED edits are in the **Individual LEDs** folder."
                    ))
                else:
                    for g_idx, group in enumerate(slot_groups_live):
                        _inspector_add(server.gui.add_html(
                            f"<hr style='margin:8px 0;'><b>Group {g_idx + 1}</b>"
                        ))
                        _inspector_guide_controls(group)
                        _inspector_led_matrix(group)
                remove_btn = _inspector_add(server.gui.add_button("Remove Slot", color="red"))

                @remove_btn.on_click
                def _(_):
                    _clear_panel_slot(key)
                    if key < len(_panel_dropdowns):
                        _panel_dropdowns[key].value = "-- Nessuno --"
                    update_scene()

            elif kind == 'custom_group':
                group = next((g for g in custom_groups if g.get('id') == key), None)
                if group is None:
                    _inspector_add(server.gui.add_markdown("This group no longer exists."))
                    return
                title = group.get('template_name') or f"Custom Group {key}"
                _inspector_add(server.gui.add_markdown(f"**{title}**"))
                edit_designer_btn = _inspector_add(server.gui.add_button("Edit in Panel Designer"))
                edit_designer_btn.on_click(lambda _, o=owner: _enter_panel_designer(o))
                _inspector_mirror_checkbox(owner)
                _mirror_checkbox("Enable", group['enable'])
                _inspector_guide_controls(group)
                _inspector_group_pose_sliders(group)
                if group.get('lumens_override') is not None:
                    _inspector_add(server.gui.add_html("<hr style='margin:4px 0;'><b>Lumens Override:</b>"))
                    _mirror_checkbox("Enable custom lumens", group['lumens_override'])
                    _mirror_slider("Lumens per LED (lm)", group['lumens_value'], 1, 900000, 1)
                _inspector_led_matrix(group)
                remove_btn = _inspector_add(server.gui.add_button("Remove Group", color="red"))

                @remove_btn.on_click
                def _(_):
                    if group in custom_groups:
                        custom_groups.remove(group)
                    try:
                        group['folder'].remove()
                    except Exception:
                        pass
                    select_panel(None)

            elif kind == 'base_group':
                names = ["Front+", "Front-", "Side+", "Side-"]
                rot_z = [rot_front_pos, rot_front_neg, rot_side_pos, rot_side_neg]
                rot_y = [rot_y_front_pos, rot_y_front_neg, rot_y_side_pos, rot_y_side_neg]
                off_x = [offset_front_pos_x, offset_front_neg_x, offset_side_pos_x, offset_side_neg_x]
                off_y = [offset_front_pos_y, offset_front_neg_y, offset_side_pos_y, offset_side_neg_y]
                off_z = [offset_front_pos_z, offset_front_neg_z, offset_side_pos_z, offset_side_neg_z]
                if key < 0 or key > 3:
                    _inspector_add(server.gui.add_markdown("Unknown base group."))
                    return
                _inspector_add(server.gui.add_markdown(f"**Base group: {names[key]}**"))
                _mirror_slider(f"Rotate {names[key]} Z (°)", rot_z[key], -180, 180, 1)
                _mirror_slider(f"Rotate {names[key]} local Y (tilt °)", rot_y[key], -180, 180, 1)
                ymin, ymax = ((-40, 40) if key == 2 else (-40, 50) if key == 3 else (-30, 30))
                _mirror_slider("Offset X (cm)", off_x[key], -30, 30, 0.1)
                _mirror_slider("Offset Y (cm)", off_y[key], ymin, ymax, 0.1)
                _mirror_slider("Offset Z (cm)", off_z[key], -30, 30, 0.1)
                _inspector_add(server.gui.add_html("<hr style='margin:6px 0;'><b>LED Controls:</b>"))
                color_hex = group_colors_hex[key]
                start = key * 12
                any_on = any(led_states[start:start + 12])
                all_btn = _inspector_add(server.gui.add_button("ALL", color=color_hex if any_on else "#444444"))

                def _on_all(_):
                    new_state = not all(led_states[start:start + 12])
                    for i in range(start, start + 12):
                        led_states[i] = new_state
                    update_all_led_buttons()
                    update_scene()
                    update_ui_visibility()
                    populate_inspector(selected_owner[0])

                all_btn.on_click(_on_all)
                for row_idx in range(4):
                    r0 = start + row_idx * 3
                    any_row = any(led_states[r0:r0 + 3])
                    row_btn = _inspector_add(server.gui.add_button(
                        f"Row {row_idx + 1}", color=color_hex if any_row else "#666666"
                    ))

                    def _make_row(s):
                        def _on(_):
                            new_state = not all(led_states[s:s + 3])
                            for i in range(s, s + 3):
                                led_states[i] = new_state
                            update_all_led_buttons()
                            update_scene()
                            update_ui_visibility()
                            populate_inspector(selected_owner[0])
                        return _on

                    row_btn.on_click(_make_row(r0))
                for li in range(12):
                    gi = start + li
                    on = led_states[gi]
                    led_btn = _inspector_add(server.gui.add_button(
                        f"L{li + 1}", color=color_hex if on else "#444444"
                    ))

                    def _make_led(idx):
                        def _on(_):
                            led_states[idx] = not led_states[idx]
                            update_all_led_buttons()
                            update_scene()
                            update_ui_visibility()
                            populate_inspector(selected_owner[0])
                        return _on

                    led_btn.on_click(_make_led(gi))

            else:
                _inspector_add(server.gui.add_markdown(f"Unknown selection: `{kind}`"))

    def _select_panel_real(owner):
        if designer_mode[0]:
            return
        if selected_owner[0] == owner:
            return
        selected_owner[0] = owner
        populate_inspector(owner)
        update_scene()

    _select_panel_impl[0] = _select_panel_real
    populate_inspector(None)
    
    def update_scene():
        """Redraw the scene based on current slider values (without intensity map)."""
        nonlocal led_handles, ray_handles, absorber_handles, camera_fov_handles, vio_fov_handles, guide_handles
        if designer_mode[0]:
            # Designer owns the 3D view; do not rebuild (would destroy the gizmo).
            return

        # Ray-box intersection helper for update_scene (positions in cm)
        def ray_box_intersection(pos, direction, box):
            center = np.array(box['center'], dtype=float)
            half = np.array(box['half_sizes'], dtype=float)
            rotation = box.get('rotation', None)
            
            # If box has rotation, transform ray to box's local space
            if rotation is not None:
                qw, qx, qy, qz = rotation
                # Convert quaternion to rotation matrix
                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                    [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                    [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
                ])
                # Transform ray to local space (inverse rotation)
                R_inv = R.T
                local_pos = R_inv @ (pos - center)
                local_dir = R_inv @ direction
                pos = local_pos
                direction = local_dir
                center = np.array([0.0, 0.0, 0.0])
            
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
        for handle in led_handles + ray_handles + absorber_handles + camera_fov_handles + vio_fov_handles + guide_handles:
            try:
                handle.remove()
            except KeyError:
                pass  # Handle already removed by server
        led_handles = []
        ray_handles = []
        absorber_handles = []
        camera_fov_handles = []
        vio_fov_handles = []
        guide_handles = []

        # Get current values (fixed angles: front=0°, side=90°)
        front_angle = 0.0  # Fixed front angle
        side_angle = 90.0  # Fixed side angle
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
        
        rotations_y = [
            rot_y_front_pos.value,
            rot_y_front_neg.value,
            rot_y_side_pos.value,
            rot_y_side_neg.value,
        ]
        
        offsets = [
            (offset_front_pos_x.value, offset_front_pos_y.value, offset_front_pos_z.value),
            (offset_front_neg_x.value, offset_front_neg_y.value, offset_front_neg_z.value),
            (offset_side_pos_x.value, offset_side_pos_y.value, offset_side_pos_z.value),
            (offset_side_neg_x.value, offset_side_neg_y.value, offset_side_neg_z.value),
        ]

        # Build custom groups configs list
        custom_groups_configs = []
        for group in custom_groups:
            config = {
                'enabled': group['enable'].value,
                'position': (group['pos_x'].value, group['pos_y'].value, group['pos_z'].value),
                'rotation_x': group['rot_roll'].value if 'rot_roll' in group else 0,
                'rotation_y': group['rot_tilt_ud'].value if 'rot_tilt_ud' in group else 0,
                'rotation_z': group['rot_tilt_lr'].value if 'rot_tilt_lr' in group else 0,
                'led_states': group['led_states'],
                'row_enabled': [row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            }
            # Add dynamic group info if present
            if group.get('is_dynamic', False):
                config['num_leds'] = group.get('num_leds', 0)
                translated_positions, rotated_directions, rotated_row_dirs = _dynamic_group_world_geometry(group)
                config['led_positions'] = translated_positions
                config['led_rotations'] = rotated_directions
                config['led_sizes'] = group.get('led_sizes', [])
                config['led_viewing_angles'] = group.get('led_viewing_angles', [])
                config['led_lumens'] = group.get('led_lumens', [])
                if rotated_row_dirs:
                    config['led_row_directions'] = rotated_row_dirs
            # Pass lumens override for custom group
            if group.get('lumens_override') and group['lumens_override'].value:
                config['lumens_override'] = float(group['lumens_value'].value)
            else:
                config['lumens_override'] = None
            if group.get('panel_slot') is not None:
                config['owner'] = ('slot', group['panel_slot'])
            else:
                config['owner'] = ('custom_group', group['id'])
            custom_groups_configs.append(config)
        
        # Build individual LEDs configs list
        individual_leds_configs = []
        for led in individual_leds:
            config = {
                'enabled': led['enable'].value,
                'led_on': led.get('led_on', True),  # Default to True if not set
                'pos_x': led['pos_x'].value,
                'pos_y': led['pos_y'].value,
                'pos_z': led['pos_z'].value,
                'rot_x': led['rot_x'].value,
                'rot_y': led['rot_y'].value,
                'rot_z': led['rot_z'].value,
                'size': led['size'].value,
                'viewing_angle': led['viewing_angle'].value,
                'square_roll': led['square_roll'].value,
                'beam_tilt': led['beam_tilt'].value,
            }
            # Pass lumens override for individual LED
            if led.get('lumens_override') and led['lumens_override'].value:
                config['lumens_override'] = float(led['lumens_value'].value)
            else:
                config['lumens_override'] = None
            # Pass external lens settings
            if led.get('ext_lens_enable') and led['ext_lens_enable'].value:
                config['ext_lens_angle'] = float(led['ext_lens_angle'].value)
                config['ext_lens_efficiency'] = float(led['ext_lens_efficiency'].value) / 100.0
            for _si, _pdata in enumerate(_panel_slot_data):
                if _pdata and led in _pdata.get('individual_leds', []):
                    config['owner'] = ('slot', _si)
                    break
            if 'owner' not in config and led.get('panel_slot') is not None:
                config['owner'] = ('slot', led['panel_slot'])
            individual_leds_configs.append(config)
        
        # Inject the live XZ-mirrored copy of the mirror-primary panel (if any)
        _expand_mirror_configs(custom_groups_configs, individual_leds_configs)

        leds = create_leds(
            front_angle,
            side_angle,
            viewing_angle,
            radius,
            circle_center_x,
            default_lumens=float(led_lumens_slider.value),
            group_rotations=rotations,
            group_rotations_y=rotations_y,
            row_enabled=[row1_chk.value, row2_chk.value, row3_chk.value, row4_chk.value],
            led_states=led_states,
            group_offsets=offsets,
            custom_groups_configs=custom_groups_configs,
            individual_leds_configs=individual_leds_configs,
            create_base_groups=any(led_states[:48]),
        )
        
        # ── Apply global Z rotation to all LEDs ──
        global_rot_z_deg = global_rotation_z_slider.value
        if abs(global_rot_z_deg) > 0.01:
            g_rad = np.radians(global_rot_z_deg)
            cg, sg = np.cos(g_rad), np.sin(g_rad)
            Rg = np.array([[cg, -sg, 0],
                           [sg,  cg, 0],
                           [0,   0,  1]], dtype=float)
            for led in leds:
                led.position = Rg @ led.position
                led.direction = Rg @ led.direction
                if hasattr(led, 'row_direction') and led.row_direction is not None:
                    led.row_direction = Rg @ np.asarray(led.row_direction)
                if hasattr(led, 'square_normal') and led.square_normal is not None:
                    led.square_normal = Rg @ np.asarray(led.square_normal)

        # ── Apply global position offset to all LEDs ──
        _gp_x = global_pos_x_slider.value
        _gp_y = global_pos_y_slider.value
        _gp_z = global_pos_z_slider.value
        if abs(_gp_x) > 0.001 or abs(_gp_y) > 0.001 or abs(_gp_z) > 0.001:
            _gp_offset = np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            for led in leds:
                led.position = led.position + _gp_offset

        # ── Apply diffuser lens effect ──
        # A diffuser lens scatters light, widening the viewing angle toward
        # a near-Lambertian distribution.  The transmission loss is applied
        # separately in the simulation functions (lumens_per_led *= transmission).
        if diffuser_enable_chk.value:
            diff_angle = float(diffuser_angle_slider.value)
            for led in leds:
                # Widen to diffuser output angle (only if wider than native)
                led.viewing_angle = max(led.viewing_angle, diff_angle)

        # Save LEDs for reuse in room intensity calculation (in place: ui.room_mode holds a reference)
        current_leds[:] = leds

        # Rest-pose construction circles + axis for the selected anchored panel
        def _globalize_cm(p):
            v = np.asarray(p, dtype=float).reshape(3)
            if abs(global_rot_z_deg) > 0.01:
                g_rad = np.radians(global_rot_z_deg)
                cg, sg = np.cos(g_rad), np.sin(g_rad)
                v = np.array([cg * v[0] - sg * v[1], sg * v[0] + cg * v[1], v[2]])
            v = v + np.array([_gp_x, _gp_y, _gp_z], dtype=float)
            return v

        def _globalize_dir(d):
            v = np.asarray(d, dtype=float).reshape(3)
            if abs(global_rot_z_deg) > 0.01:
                g_rad = np.radians(global_rot_z_deg)
                cg, sg = np.cos(g_rad), np.sin(g_rad)
                v = np.array([cg * v[0] - sg * v[1], sg * v[0] + cg * v[1], v[2]])
            return v

        draw_groups = []
        owner = selected_owner[0]
        if owner is not None:
            draw_groups = list(_owner_groups(owner))
        for gi, group in enumerate(draw_groups):
            if not _guide_is_enabled(group):
                continue
            guide = group['guide']
            origin = _as_vec3(guide.get('origin', (0, 0, 0)))
            axis = _as_vec3(guide.get('axis', (0, 0, 1)))
            an = np.linalg.norm(axis)
            if an < 1e-12:
                continue
            axis = axis / an
            circles = guide.get('circles') or []
            for ci, circ in enumerate(circles):
                segs = _circle_line_segments_m(
                    _globalize_cm(circ.get('center', origin)),
                    float(circ.get('radius', 0.0)),
                    _globalize_dir(circ.get('normal', axis)),
                    n_seg=64,
                )
                if segs.shape[0] == 0:
                    continue
                handle = server.scene.add_line_segments(
                    f"/guides/group_{group.get('id', gi)}/circle_{ci}",
                    points=segs,
                    colors=(0.55, 0.75, 0.95),
                    line_width=1.5,
                )
                guide_handles.append(handle)
            ts = [float(np.dot(_as_vec3(c.get('center', origin)) - origin, axis)) for c in circles] or [0.0]
            t_lo, t_hi = min(ts) - 3.0, max(ts) + 3.0
            if abs(t_hi - t_lo) < 4.0:
                t_lo, t_hi = -8.0, 8.0
            p0 = _globalize_cm(origin + axis * t_lo) / 100.0
            p1 = _globalize_cm(origin + axis * t_hi) / 100.0
            axis_h = server.scene.add_line_segments(
                f"/guides/group_{group.get('id', gi)}/axis",
                points=np.array([[p0, p1]]),
                colors=(1.0, 0.85, 0.2),
                line_width=3.0,
            )
            guide_handles.append(axis_h)

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
                'rotation': None,
            })
        
        # Add abs2 and abs3 at origin with offsets
        if absorbers_enable.value:
            # Abs2 with rotation
            abs_cx = 0.0 + abs2_off_x.value
            abs_cy = 0.0 + abs2_off_y.value
            abs_cz = 0.0 + abs2_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs2_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })
            
            # Abs3 with rotation
            abs_cx = 0.0 + abs3_off_x.value
            abs_cy = 0.0 + abs3_off_y.value
            abs_cz = 0.0 + abs3_off_z.value
            half_length_x = 5.0 / 2.0
            half_width_y = 1.5 / 2.0
            half_thickness_z = 3.0 / 2.0
            # Convert rotation angle to quaternion (rotation around Z axis)
            angle_rad = np.radians(abs3_rot_z.value)
            qw = np.cos(angle_rad / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(angle_rad / 2)
            absorbers.append({
                'center': (abs_cx, abs_cy, abs_cz),
                'half_sizes': (half_length_x, half_width_y, half_thickness_z),
                'rotation': (qw, qx, qy, qz),
            })

        # ── Apply global Z rotation to absorbers ──
        if abs(global_rot_z_deg) > 0.01:
            for a in absorbers:
                cx, cy, cz = a['center']
                new_cx = cg * cx - sg * cy
                new_cy = sg * cx + cg * cy
                a['center'] = (new_cx, new_cy, cz)
                # Compose global rotation with existing quaternion rotation
                if a.get('rotation') is not None:
                    qw0, qx0, qy0, qz0 = a['rotation']
                    # Quaternion for Rg around Z: (cos(a/2), 0, 0, sin(a/2))
                    half = g_rad / 2.0
                    gqw, gqx, gqy, gqz = np.cos(half), 0.0, 0.0, np.sin(half)
                    # q_new = q_global * q_existing
                    nw = gqw*qw0 - gqx*qx0 - gqy*qy0 - gqz*qz0
                    nx = gqw*qx0 + gqx*qw0 + gqy*qz0 - gqz*qy0
                    ny = gqw*qy0 - gqx*qz0 + gqy*qw0 + gqz*qx0
                    nz = gqw*qz0 + gqx*qy0 - gqy*qx0 + gqz*qw0
                    a['rotation'] = (nw, nx, ny, nz)
                else:
                    half = g_rad / 2.0
                    a['rotation'] = (np.cos(half), 0.0, 0.0, np.sin(half))

        # Draw absorber boxes (red) in the scene
        for idx, a in enumerate(absorbers):
            cx, cy, cz = a['center']
            hx, hy, hz = a['half_sizes']
            rot = a.get('rotation', None)
            # Viser add_box dimensions are in meters (x,y,z)
            dims = ((hx * 2) / 100.0, (hy * 2) / 100.0, (hz * 2) / 100.0)
            pos_m = (cx / 100.0, cy / 100.0, cz / 100.0)
            if rot is not None:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                    wxyz=rot,
                )
            else:
                handle = server.scene.add_box(
                    f"/absorbers/abs_{idx}",
                    dimensions=dims,
                    color=(1.0, 0.0, 0.0),
                    position=pos_m,
                )
            absorber_handles.append(handle)

        # Draw LEDs as squares with center source (if enabled)
        if show_led_markers.value:
            for i, led in enumerate(leds):
                led_idx = getattr(led, 'led_index', i)
                led_enabled = not (hasattr(led, 'enabled') and not led.enabled)
                
                # Build local coordinate system for LED
                # Use square_normal (original direction) for mesh if beam_tilt is applied
                square_dir = getattr(led, 'mesh_normal', led.direction)
                z_axis = square_dir / np.linalg.norm(square_dir)
                
                # Use row_direction as reference for consistent square orientation across all rows
                row_dir = getattr(led, 'row_direction', None)
                if row_dir is not None:
                    # y_axis aligned with row direction (direction along the row of LEDs)
                    y_axis = row_dir / np.linalg.norm(row_dir)
                    # Make y_axis perpendicular to z_axis (Gram-Schmidt)
                    y_axis = y_axis - z_axis * np.dot(y_axis, z_axis)
                    if np.linalg.norm(y_axis) < 0.01:  # Nearly parallel, use fallback
                        if abs(z_axis[2]) < 0.9:
                            x_axis = np.cross(z_axis, [0, 0, 1])
                        else:
                            x_axis = np.cross(z_axis, [0, 1, 0])
                        x_axis = x_axis / np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                    else:
                        y_axis = y_axis / np.linalg.norm(y_axis)
                        x_axis = np.cross(y_axis, z_axis)
                else:
                    # Fallback for backwards compatibility
                    if abs(z_axis[2]) < 0.9:
                        x_axis = np.cross(z_axis, [0, 0, 1])
                    else:
                        x_axis = np.cross(z_axis, [0, 1, 0])
                    x_axis = x_axis / np.linalg.norm(x_axis)
                    y_axis = np.cross(z_axis, x_axis)
                
                # Create rotation matrix from local axes to world axes
                # Square default orientation: thin in X, extends in Y and Z
                # We want: thin along z_axis (LED direction), extends along x_axis and y_axis
                # Rotation matrix: columns are the target axes in world coordinates
                rot_matrix = np.column_stack([z_axis, x_axis, y_axis])
                
                # Convert rotation matrix to quaternion (wxyz format)
                # Using Shepperd's method for numerical stability
                trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    w = 0.25 / s
                    x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
                    y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
                    z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
                else:
                    if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
                        w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                        x = 0.25 * s
                        y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                        z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                    elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
                        w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                        x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                        y = 0.25 * s
                        z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
                        w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                        x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                        y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                        z = 0.25 * s
                quat_wxyz = np.array([w, x, y, z])
                
                # Square size from LED width (in cm, convert to meters)
                square_size = led.width / 100.0  # LED width in cm converted to meters
                square_thickness = 0.0002  # Very thin (0.2mm)
                dims = (square_thickness, square_size, square_size)
                
                # White color: bright if enabled, dim if disabled; cyan if this panel is selected
                square_color = (1.0, 1.0, 1.0) if led_enabled else (0.3, 0.3, 0.3)
                _owner = getattr(led, 'owner', None)
                if selected_owner[0] is not None and _owner == selected_owner[0]:
                    square_color = (0.15, 1.0, 1.0) if led_enabled else (0.08, 0.45, 0.45)
                
                # Draw square base with rotation
                handle = server.scene.add_box(
                    f"/leds/led_{led_idx}_base",
                    dimensions=dims,
                    color=square_color,
                    position=tuple(led.position / 100.0),  # Convert cm to m for viser
                    wxyz=tuple(quat_wxyz),
                )
                led_handles.append(handle)
                if _owner is not None:
                    def _make_owner_click(own):
                        def _on_click(_event):
                            _just_clicked_mesh[0] = True
                            select_panel(own)
                        return _on_click
                    handle.on_click(_make_owner_click(_owner))
                
                # Draw small center source sphere (only if enabled)
                if led_enabled:
                    handle = server.scene.add_icosphere(
                        f"/leds/led_{led_idx}_source",
                        radius=0.001,  # Very small 1mm source
                        color=led.color,
                        position=tuple(led.position / 100.0),
                    )
                    led_handles.append(handle)
                    if _owner is not None:
                        handle.on_click(_make_owner_click(_owner))

        # Prepare STL mesh data for ray tracing visualization if enabled
        stl_mesh_for_raytracing = None
        if stl_absorber_enable.value and stl_mesh_data[0] is not None:
            mesh_ref = stl_mesh_data[0]
            transform = _build_stl_transform(stl_scale, stl_rot_x, stl_rot_y, stl_rot_z, stl_pos_x, stl_pos_y, stl_pos_z)
            # Apply global Z rotation to STL transform
            if abs(global_rot_z_deg) > 0.01:
                T_global = np.eye(4)
                T_global[:3, :3] = Rg
                transform = T_global @ transform
            stl_mesh_for_raytracing = {
                'vertices': mesh_ref.vertices,
                'faces': mesh_ref.faces,
                'transform': transform
            }

        # Draw rays (toggleable)
        if show_rays_output.value:
            for i, led in enumerate(leds):
                if hasattr(led, 'enabled') and not led.enabled:
                    continue
                vis_rays = led.get_visualization_rays(ray_length)

                for j, (pos, direction) in enumerate(vis_rays):
                    # Calculate end point, clipping at absorbers and wall
                    # Rays have infinite length until they hit something
                    
                    # Check absorbers first via box intersection
                    t_abs_min = None
                    if absorbers is not None:
                        for a in absorbers:
                            t_hit = ray_box_intersection(pos, direction, a)
                            if t_hit is not None and t_hit > 0:
                                if t_abs_min is None or t_hit < t_abs_min:
                                    t_abs_min = t_hit
                    
                    # Check STL mesh intersection
                    if stl_mesh_for_raytracing is not None:
                        t_stl = _ray_mesh_intersection(pos, direction, stl_mesh_for_raytracing)
                        if t_stl is not None and t_stl > 0:
                            if t_abs_min is None or t_stl < t_abs_min:
                                t_abs_min = t_stl

                    # Clip at wall(s) - if room mode, check all 5 walls
                    t_wall = None
                    if room_mode_enable.value:
                        # Check all 5 room walls
                        front_dist = room_front_dist.value
                        side_dist = room_side_dist.value
                        top_bottom_dist = room_top_bottom_dist.value
                        
                        wall_intersections = []
                        # Front wall
                        if direction[0] != 0:
                            t = (front_dist - pos[0]) / direction[0]
                            if t > 0:
                                wall_intersections.append(t)
                        # Left wall
                        if direction[1] != 0:
                            t = (-side_dist - pos[1]) / direction[1]
                            if t > 0:
                                wall_intersections.append(t)
                        # Right wall
                        if direction[1] != 0:
                            t = (side_dist - pos[1]) / direction[1]
                            if t > 0:
                                wall_intersections.append(t)
                        # Top wall
                        if direction[2] != 0:
                            t = (top_bottom_dist - pos[2]) / direction[2]
                            if t > 0:
                                wall_intersections.append(t)
                        # Bottom wall
                        if direction[2] != 0:
                            t = (-top_bottom_dist - pos[2]) / direction[2]
                            if t > 0:
                                wall_intersections.append(t)
                        
                        if wall_intersections:
                            t_wall = min(wall_intersections)
                    else:
                        # Single front wall only
                        if direction[0] != 0:
                            t_wall = (wall_dist - pos[0]) / direction[0]

                    # Choose nearest positive intersection (absorber before wall)
                    t_clip = None
                    if t_abs_min is not None and t_abs_min > 0:
                        t_clip = t_abs_min
                    if t_wall is not None and t_wall > 0:
                        if t_clip is None or t_wall < t_clip:
                            t_clip = t_wall

                    # Use intersection point, or very far if no intersection
                    if t_clip is not None:
                        end = pos + direction * t_clip
                    else:
                        end = pos + direction * 1000.0  # 10 meters if no intersection

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
                        
                        uniformity = float(ray_uniformity_slider.value)
                        n = _get_effective_n(led, uniformity)
                        
                        _vis_angle = getattr(led, 'ext_lens_angle', None) or led.viewing_angle
                        max_theta = np.radians(_vis_angle / 2.0)
                        cos_max = np.cos(max_theta)
                        cos_theta_sampled = 1.0 - u1 * (1.0 - cos_max)
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

                        # Compute nearest intersection with absorbers or wall
                        # Rays have infinite length until they hit something
                        t_abs_min = None
                        if absorbers is not None:
                            for a in absorbers:
                                t_hit = ray_box_intersection(led.position, world_dir, a)
                                if t_hit is not None and t_hit > 0:
                                    if t_abs_min is None or t_hit < t_abs_min:
                                        t_abs_min = t_hit
                        
                        # Check STL mesh intersection
                        if stl_mesh_for_raytracing is not None:
                            t_stl = _ray_mesh_intersection(led.position, world_dir, stl_mesh_for_raytracing)
                            if t_stl is not None and t_stl > 0:
                                if t_abs_min is None or t_stl < t_abs_min:
                                    t_abs_min = t_stl

                        t_wall = None
                        if room_mode_enable.value:
                            # Check all 5 room walls
                            front_dist = room_front_dist.value
                            side_dist = room_side_dist.value
                            top_bottom_dist = room_top_bottom_dist.value
                            
                            wall_intersections = []
                            # Front wall
                            if world_dir[0] != 0:
                                t = (front_dist - led.position[0]) / world_dir[0]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Left wall
                            if world_dir[1] != 0:
                                t = (-side_dist - led.position[1]) / world_dir[1]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Right wall
                            if world_dir[1] != 0:
                                t = (side_dist - led.position[1]) / world_dir[1]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Top wall
                            if world_dir[2] != 0:
                                t = (top_bottom_dist - led.position[2]) / world_dir[2]
                                if t > 0:
                                    wall_intersections.append(t)
                            # Bottom wall
                            if world_dir[2] != 0:
                                t = (-top_bottom_dist - led.position[2]) / world_dir[2]
                                if t > 0:
                                    wall_intersections.append(t)
                            
                            if wall_intersections:
                                t_wall = min(wall_intersections)
                        else:
                            # Single front wall only
                            if world_dir[0] != 0:
                                t_wall = (wall_dist - led.position[0]) / world_dir[0]

                        t_clip = None
                        if t_abs_min is not None and t_abs_min > 0:
                            t_clip = t_abs_min
                        if t_wall is not None and t_wall > 0:
                            if t_clip is None or t_wall < t_clip:
                                t_clip = t_wall

                        # Use intersection point, or very far if no intersection
                        if t_clip is not None:
                            end = led.position + world_dir * t_clip
                        else:
                            end = led.position + world_dir * 1000.0  # 10 meters if no intersection

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
            # Use correct wall distance based on mode
            if room_mode_enable.value:
                wall_dist = room_front_dist.value
            else:
                wall_dist = wall_dist_slider.value
            
            cam_x = camera_pos_x.value
            cam_y = camera_pos_y.value

            # Footprint of the (possibly pitched) camera FOV on the wall.
            # With pitch != 0 this is a trapezoid, not a rectangle.
            z_bot_cm, z_top_cm, w_bot_cm, w_top_cm = _camera_fov_wall_trapezoid(
                wall_dist - cam_x, camera_pitch.value,
                camera_fov_h.value, camera_fov_v.value,
            )
            z_bot = z_bot_cm / 100.0  # metres
            z_top = z_top_cm / 100.0
            w_bot = w_bot_cm / 100.0
            w_top = w_top_cm / 100.0
            y_c = cam_y / 100.0  # camera Y offset shifts the footprint sideways
            wall_x = wall_dist / 100.0 - 0.008  # Slightly in front of wall

            y_bot_l, y_bot_r = y_c - w_bot, y_c + w_bot
            y_top_l, y_top_r = y_c - w_top, y_c + w_top

            # In room mode, clamp the FOV footprint to the front wall boundaries
            if room_mode_enable.value:
                max_half_w = room_side_dist.value / 100.0        # wall half-width in m
                max_half_h = room_top_bottom_dist.value / 100.0  # wall half-height in m
                y_bot_l = float(np.clip(y_bot_l, -max_half_w, max_half_w))
                y_bot_r = float(np.clip(y_bot_r, -max_half_w, max_half_w))
                y_top_l = float(np.clip(y_top_l, -max_half_w, max_half_w))
                y_top_r = float(np.clip(y_top_r, -max_half_w, max_half_w))
                z_bot = float(np.clip(z_bot, -max_half_h, max_half_h))
                z_top = float(np.clip(z_top, -max_half_h, max_half_h))
            
            # Four corner lines
            corners = [
                [[wall_x, y_bot_l, z_bot], [wall_x, y_bot_r, z_bot]],  # Bottom
                [[wall_x, y_bot_r, z_bot], [wall_x, y_top_r, z_top]],  # Right
                [[wall_x, y_top_r, z_top], [wall_x, y_top_l, z_top]],  # Top
                [[wall_x, y_top_l, z_top], [wall_x, y_bot_l, z_bot]],  # Left
            ]
            
            handle = server.scene.add_line_segments(
                "/camera/fov_border",
                points=np.array(corners),
                colors=(0.0, 1.0, 0.0),  # Green
                line_width=6.0,  # Thicker lines
            )
            camera_fov_handles.append(handle)

            # Main-camera marker + optical-axis vector (like the VIO cameras)
            cam_pos_cm = np.array([cam_x, cam_y, 0.0], dtype=float)
            cam_axis = vio_optical_axis(camera_pitch.value, 0.0)
            cam_pos_m = cam_pos_cm / 100.0
            cam_axis_end_m = (cam_pos_cm + cam_axis * 8.0) / 100.0
            cam_sph = server.scene.add_icosphere(
                "/camera/marker",
                radius=0.006,
                color=(0.0, 1.0, 0.0),
                position=tuple(cam_pos_m + cam_axis * 0.008),
            )
            camera_fov_handles.append(cam_sph)
            cam_axis_h = server.scene.add_line_segments(
                "/camera/axis",
                points=np.array([[cam_pos_m, cam_axis_end_m]]),
                colors=(0.0, 1.0, 0.0),
                line_width=4.0,
            )
            camera_fov_handles.append(cam_axis_h)

        # Draw VIO fisheye FOV footprints on the wall(s).
        # In room mode the footprint is projected on ALL room walls (front,
        # left, right, top, bottom, and back if shown), not just the front one.
        if show_vio_fov.value:
            # Wall patches: (axis, plane_coord_cm, u_min, u_max, v_min, v_max, inward_sign)
            # axis 0=x, 1=y, 2=z; (u, v) are the two remaining axes in
            # ascending order; inward_sign offsets the overlay into the room.
            if room_mode_enable.value:
                _fd = room_front_dist.value
                _sd = room_side_dist.value
                _td = room_top_bottom_dist.value
                # Same extent as draw_room_walls: 2.5x depth behind the front wall
                _x_back_edge = _fd - (_fd - circle_center_slider.value) * 2.5
                if show_back_wall.value:
                    _x_back_edge = max(_x_back_edge, -room_back_dist.value)
                vio_walls = [
                    (0, _fd, -_sd, _sd, -_td, _td, -1.0),   # front
                    (1, -_sd, _x_back_edge, _fd, -_td, _td, +1.0),  # left
                    (1, _sd, _x_back_edge, _fd, -_td, _td, -1.0),   # right
                    (2, _td, _x_back_edge, _fd, -_sd, _sd, -1.0),   # top
                    (2, -_td, _x_back_edge, _fd, -_sd, _sd, +1.0),  # bottom
                ]
                if show_back_wall.value:
                    vio_walls.append((0, -room_back_dist.value, -_sd, _sd, -_td, _td, +1.0))
            else:
                _half = wall_view_size.value / 2.0
                vio_walls = [(0, wall_dist_slider.value, -_half, _half, -_half, _half, -1.0)]

            vio_hfov, vio_vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
            cam_pos = np.array([vio_pos_x.value, vio_pos_y.value, vio_pos_z.value], dtype=float)
            axis_len_cm = 8.0
            marker_specs = [
                (1, vio_cam1_pitch.value, vio_cam1_yaw.value, (1.0, 0.0, 1.0), 0.008),  # magenta
                (2, vio_cam2_pitch.value, vio_cam2_yaw.value, (0.0, 1.0, 1.0), 0.012),  # cyan
            ]
            for cam_id, pitch, yaw, color, inset_m in marker_specs:
                all_verts, all_faces, all_segs = [], [], []
                n_verts = 0
                for w_axis, plane_cm, u_min, u_max, v_min, v_max, inward in vio_walls:
                    # Skip walls the camera is not on the interior side of
                    if (cam_pos[w_axis] - plane_cm) * inward <= 1e-6:
                        continue
                    mask, us, vs = rasterize_fisheye_fov_on_plane(
                        cam_pos, pitch, yaw, vio_hfov, vio_vfov,
                        w_axis, plane_cm, u_min, u_max, v_min, v_max, n_grid=90,
                    )
                    plane_m = plane_cm / 100.0 + inward * inset_m
                    verts, faces, segs = fov_plane_mask_to_quads_and_contour(
                        mask, us, vs, w_axis, plane_m
                    )
                    if len(faces) > 0:
                        all_verts.append(verts)
                        all_faces.append(faces + np.uint32(n_verts))
                        n_verts += len(verts)
                    if len(segs) > 0:
                        all_segs.append(segs)
                if vio_fill_fov.value and all_faces:
                    fill = server.scene.add_mesh_simple(
                        name=f"/vio_cam{cam_id}/fov_fill",
                        vertices=np.concatenate(all_verts, axis=0),
                        faces=np.concatenate(all_faces, axis=0),
                        color=tuple(int(c * 255) for c in color),
                        opacity=0.28,
                        flat_shading=True,
                        side="double",
                    )
                    vio_fov_handles.append(fill)
                if all_segs:
                    handle = server.scene.add_line_segments(
                        f"/vio_cam{cam_id}/fov_border",
                        points=np.concatenate(all_segs, axis=0),
                        colors=color,
                        line_width=4.0,
                    )
                    vio_fov_handles.append(handle)

                axis = vio_optical_axis(pitch, yaw)
                pos_m = cam_pos / 100.0
                axis_end_m = (cam_pos + axis * axis_len_cm) / 100.0
                marker_pos = pos_m + axis * 0.008
                sph = server.scene.add_icosphere(
                    f"/vio_cam{cam_id}/marker",
                    radius=0.006,
                    color=color,
                    position=tuple(marker_pos),
                )
                vio_fov_handles.append(sph)
                axis_h = server.scene.add_line_segments(
                    f"/vio_cam{cam_id}/axis",
                    points=np.array([[pos_m, axis_end_m]]),
                    colors=color,
                    line_width=4.0,
                )
                vio_fov_handles.append(axis_h)

    @server.scene.on_click()
    def _on_scene_background_click(_event):
        if designer_mode[0]:
            _just_clicked_mesh[0] = False
            return
        if _just_clicked_mesh[0]:
            _just_clicked_mesh[0] = False
            return
        if selected_owner[0] is not None:
            select_panel(None)

    # Add static elements
    # Wall (at x = wall_dist)
    wall_dist_init = wall_dist_slider.value
    wall_size_init = wall_view_size.value / 100.0  # cm -> m
    wall_handle = server.scene.add_box(
        "/wall",
        dimensions=(0.01, wall_size_init, wall_size_init),
        color=(0.5, 0.5, 0.5),
        position=(wall_dist_init / 100.0, 0.0, 0.0),
    )

    # Grid on XY plane (millimeter resolution)
    grid_points = []
    for i in range(-10, 11):
        grid_points.append([[-1.0, i * 0.01, 0], [1.0, i * 0.01, 0]])  # 1mm spacing
        grid_points.append([[i * 0.01, -1.0, 0], [i * 0.01, 1.0, 0]])  # 1mm spacing

    static_scene_handles.append(server.scene.add_line_segments(
        "/grid",
        points=np.array(grid_points),
        colors=(0.3, 0.3, 0.3),  # Single color for all segments
        line_width=1.0,
    ))

    # Origin axes
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/x",
        points=np.array([[[0, 0, 0], [0.5, 0, 0]]]),
        colors=(1.0, 0.0, 0.0),
        line_width=3.0,
    ))
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/y",
        points=np.array([[[0, 0, 0], [0, 0.5, 0]]]),
        colors=(0.0, 1.0, 0.0),
        line_width=3.0,
    ))
    static_scene_handles.append(server.scene.add_line_segments(
        "/axes/z",
        points=np.array([[[0, 0, 0], [0, 0, 0.5]]]),
        colors=(0.0, 0.0, 1.0),
        line_width=3.0,
    ))

    # Callback to update wall position and size
    def update_wall():
        nonlocal wall_handle
        if room_mode_enable.value:
            # In room mode, don't update main wall
            return
        wall_dist = wall_dist_slider.value
        wall_size_m = wall_view_size.value / 100.0  # cm -> m
        try:
            wall_handle.remove()
        except (AttributeError, KeyError):
            pass
        wall_handle = server.scene.add_box(
            "/wall",
            dimensions=(0.01, wall_size_m, wall_size_m),
            color=(0.5, 0.5, 0.5),
            position=(wall_dist / 100.0, 0.0, 0.0),
        )

    # Function to update cell area info
    def update_cell_area_info():
        grid_size = int(intensity_grid_size.value)
        wall_size_cm = int(wall_view_size.value)
        cell_size_cm = wall_size_cm / grid_size
        cell_area_cm2 = cell_size_cm * cell_size_cm
        cell_area_m2 = cell_area_cm2 / 10000.0  # Convert cm² to m²
        
        cell_area_html.content = (
            f"<div style='font-family: sans-serif; font-size: 11px; color: #666; margin-top: -8px; margin-bottom: 8px;'>"
            f"Cell: {cell_size_cm:.2f} cm × {cell_size_cm:.2f} cm = {cell_area_cm2:.2f} cm² ({cell_area_m2:.6f} m²)"
            "</div>"
        )
    
    # Initial cell area update
    update_cell_area_info()

    # Register callbacks
    viewing_angle_slider.on_update(lambda _: update_scene())
    diffuser_enable_chk.on_update(lambda _: update_scene())
    diffuser_angle_slider.on_update(lambda _: update_scene())
    rot_front_pos.on_update(lambda _: update_scene())
    rot_front_neg.on_update(lambda _: update_scene())
    rot_side_pos.on_update(lambda _: update_scene())
    rot_side_neg.on_update(lambda _: update_scene())
    rot_y_front_pos.on_update(lambda _: update_scene())
    rot_y_front_neg.on_update(lambda _: update_scene())
    rot_y_side_pos.on_update(lambda _: update_scene())
    rot_y_side_neg.on_update(lambda _: update_scene())
    # Group position offset callbacks
    offset_front_pos_x.on_update(lambda _: update_scene())
    offset_front_pos_y.on_update(lambda _: update_scene())
    offset_front_pos_z.on_update(lambda _: update_scene())
    offset_front_neg_x.on_update(lambda _: update_scene())
    offset_front_neg_y.on_update(lambda _: update_scene())
    offset_front_neg_z.on_update(lambda _: update_scene())
    offset_side_pos_x.on_update(lambda _: update_scene())
    offset_side_pos_y.on_update(lambda _: update_scene())
    offset_side_pos_z.on_update(lambda _: update_scene())
    offset_side_neg_x.on_update(lambda _: update_scene())
    offset_side_neg_y.on_update(lambda _: update_scene())
    offset_side_neg_z.on_update(lambda _: update_scene())
    radius_slider.on_update(lambda _: update_scene())
    circle_center_slider.on_update(lambda _: update_scene())
    def _on_global_rotation_change(_):
        """Handle global rotation slider: update LEDs immediately, debounce mesh update."""
        update_scene()
        # Schedule mesh update with debounce to avoid recomputing on every tick
        import threading
        if hasattr(_on_global_rotation_change, '_timer') and _on_global_rotation_change._timer is not None:
            _on_global_rotation_change._timer.cancel()
        def _deferred_mesh():
            try:
                update_stl_mesh(skip_lighting=True)
            except Exception:
                pass
        _on_global_rotation_change._timer = threading.Timer(0.3, _deferred_mesh)
        _on_global_rotation_change._timer.start()
    _on_global_rotation_change._timer = None
    global_rotation_z_slider.on_update(_on_global_rotation_change)
    global_pos_x_slider.on_update(_on_global_rotation_change)
    global_pos_y_slider.on_update(_on_global_rotation_change)
    global_pos_z_slider.on_update(_on_global_rotation_change)
    ray_length_slider.on_update(lambda _: update_scene())
    led_lumens_slider.on_update(lambda _: None)  # No auto-update, use manual button
    show_random_rays.on_update(lambda _: update_scene())
    show_rays_output.on_update(lambda _: update_scene())
    show_led_markers.on_update(lambda _: update_scene())
    show_intensity_map.on_update(lambda _: None if _mode_toggle_syncing[0] else update_intensity_map())
    row1_chk.on_update(lambda _: update_scene())
    row2_chk.on_update(lambda _: update_scene())
    row3_chk.on_update(lambda _: update_scene())
    row4_chk.on_update(lambda _: update_scene())
    absorbers_enable.on_update(lambda _: (update_scene(), update_ui_visibility()))
    show_camera_fov.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_h.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_fov_v.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pos_y.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    camera_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    show_vio_fov.on_update(lambda _: update_scene())
    vio_fill_fov.on_update(lambda _: update_scene())
    vio_pos_x.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_pos_y.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_pos_z.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam1_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam1_yaw.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam2_pitch.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_cam2_yaw.on_update(lambda _: (update_scene(), _refresh_uniformity()))
    vio_long_fov.on_update(lambda _: (_refresh_vio_fov_label(), update_scene(), _refresh_uniformity()))
    vio_landscape.on_update(lambda _: (_refresh_vio_fov_label(), update_scene(), _refresh_uniformity()))
    abs0_off_x.on_update(lambda _: update_scene())
    abs0_off_y.on_update(lambda _: update_scene())
    abs0_off_z.on_update(lambda _: update_scene())
    abs1_off_x.on_update(lambda _: update_scene())
    abs1_off_y.on_update(lambda _: update_scene())
    abs1_off_z.on_update(lambda _: update_scene())
    abs2_off_x.on_update(lambda _: update_scene())
    abs2_off_y.on_update(lambda _: update_scene())
    abs2_off_z.on_update(lambda _: update_scene())
    abs2_rot_z.on_update(lambda _: update_scene())
    abs3_off_x.on_update(lambda _: update_scene())
    abs3_off_y.on_update(lambda _: update_scene())
    abs3_off_z.on_update(lambda _: update_scene())
    abs3_rot_z.on_update(lambda _: update_scene())
    intensity_rays_slider.on_update(lambda _: None)  # No auto-update - manual button only
    ray_uniformity_slider.on_update(lambda _: None)  # No auto-update for expensive params
    intensity_threshold_slider.on_update(lambda _: _refresh_uniformity())
    uniformity_percentile_slider.on_update(lambda _: _refresh_uniformity())
    intensity_grid_size.on_update(lambda _: update_cell_area_info())  # Update cell area when resolution changes
    wall_view_size.on_update(lambda _: update_cell_area_info())  # Update cell area when wall size changes
    
    # Room mode callback - draw/clear room walls when toggled
    def on_room_mode_toggle(_):
        nonlocal wall_handle, intensity_handles
        if room_mode_enable.value:
            # Hide main wall and show room walls
            try:
                wall_handle.remove()
            except (KeyError, AttributeError):
                pass
            for handle in intensity_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            intensity_handles = []
            _mode_toggle_syncing[0] = True
            try:
                show_intensity_map.value = False
            finally:
                _mode_toggle_syncing[0] = False
            legend_html.content = (
                "<div style='font-family: sans-serif;'>"
                "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                "<div style='color:#888;font-size:12px;'>Room Mode is active. Enable 'Show Room Intensity' and click 'Update Room Intensity'.</div>"
                "</div>"
            )
            draw_room_walls()
        else:
            # Clear room intensity handles
            for handle in room_intensity_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            room_intensity_handles.clear()
            # Clear room wall handles
            for handle in room_wall_handles:
                try:
                    handle.remove()
                except KeyError:
                    pass
            room_wall_handles.clear()
            _mode_toggle_syncing[0] = True
            try:
                show_room_intensity.value = False
            finally:
                _mode_toggle_syncing[0] = False
            _last_room_cache['grids'] = None
            # Restore main wall
            wall_dist = wall_dist_slider.value
            wall_size_m = wall_view_size.value / 100.0
            wall_handle = server.scene.add_box(
                "/wall",
                dimensions=(0.01, wall_size_m, wall_size_m),
                color=(0.5, 0.5, 0.5),
                position=(wall_dist / 100.0, 0.0, 0.0),
            )
            if _last_intensity_cache['grid'] is not None:
                _refresh_uniformity()
            else:
                legend_html.content = (
                    "<div style='font-family: sans-serif;'>"
                    "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
                    "<div style='color:#888;font-size:12px;'>Enable 'Show intensity on wall' and click 'Update Intensity Map' to see the legend</div>"
                    "</div>"
                )
    
    room_mode_enable.on_update(on_room_mode_toggle)
    show_room_walls.on_update(lambda _: draw_room_walls())
    show_room_intensity.on_update(
        lambda _: None if _mode_toggle_syncing[0] else (
            (update_room_intensity_map() if (room_mode_enable.value and show_room_intensity.value) else draw_room_walls())
            if room_mode_enable.value else None
        )
    )
    room_front_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_side_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_top_bottom_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    show_back_wall.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    room_back_dist.on_update(lambda _: (draw_room_walls(), update_scene()) if room_mode_enable.value else None)
    wall_view_size.on_update(lambda _: (update_wall(), update_cell_area_info(), update_scene()))  # Update wall size, cell area, VIO FOV clip

    def on_wall_dist_change(_):
        """Clear stale intensity map when wall distance changes."""
        # Remove old intensity visualization (values are no longer valid)
        for h in intensity_handles:
            try:
                h.remove()
            except KeyError:
                pass
        intensity_handles.clear()
        # Update legend to inform user that recalculation is needed
        legend_html.content = (
            "<div style='font-family: sans-serif;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Intensity legend</div>"
            "<div style='color:#F0AD4E;font-size:12px;'>⚠ Wall distance changed.<br>Click 'Update Intensity Map' to recalculate.</div>"
            "</div>"
        )
        update_wall()
        update_scene()

    wall_dist_slider.on_update(on_wall_dist_change)
    
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
                update_ui_visibility()
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
                update_ui_visibility()
            return handler
        btn.on_click(make_row_handler(group_idx, row_idx))
    
    # Individual LED buttons
    for led_idx, btn in led_buttons.items():
        def make_led_handler(l_idx):
            def handler(_):
                led_states[l_idx] = not led_states[l_idx]
                print(f"LED {l_idx} toggled to {led_states[l_idx]}")
                update_scene()
                update_ui_visibility()
            return handler
        btn.on_click(make_led_handler(led_idx))
    
    # Button for manual intensity map update
    update_intensity_button.on_click(lambda _: update_intensity_map())
    
    # Button for exporting lux matrix
    export_lux_matrix_button.on_click(lambda _: export_lux_matrix())
    
    # Button for running benchmark (multi-distance)
    run_benchmark_button.on_click(lambda _: run_benchmark())
    
    # Buttons for CSV pattern import
    csv_import_btn.on_click(lambda _: import_csv_pattern())
    csv_clear_btn.on_click(lambda _: clear_csv_pattern())

    # Button for capturing FOV intensity image
    capture_fov_btn.on_click(lambda _: capture_camera_fov_image())
    
    # Button for room intensity map update
    update_room_button.on_click(lambda _: update_room_intensity_map())

    # --- Panel system (see ui/panels.py) ---
    _panels_ns = _panels.build(_SimpleNamespace(
        _ELIOS3_SLOTS=_ELIOS3_SLOTS,
        _Rz_matrix=_Rz_matrix,
        _clear_mirror_state=_clear_mirror_state,
        _mirror_primary=_mirror_primary,
        _panel_dropdowns=_panel_dropdowns,
        _panel_slot_data=_panel_slot_data,
        custom_groups=custom_groups,
        custom_groups_folder=custom_groups_folder,
        custom_groups_templates_dir=custom_groups_templates_dir,
        individual_leds=individual_leds,
        individual_leds_folder=individual_leds_folder,
        loading_in_progress=loading_in_progress,
        next_custom_group_id=next_custom_group_id,
        next_individual_led_id=next_individual_led_id,
        select_panel=select_panel,
        selected_owner=selected_owner,
        server=server,
        show_led_markers=show_led_markers,
        tab_panels=tab_panels,
        template_folders=template_folders,
        update_scene=update_scene,
    ))
    _clear_panel_slot = _panels_ns._clear_panel_slot
    _load_template_into_slot = _panels_ns._load_template_into_slot
    create_custom_group = _panels_ns.create_custom_group
    create_individual_led = _panels_ns.create_individual_led
    load_custom_group_from_template = _panels_ns.load_custom_group_from_template
    load_template_as_individual_leds = _panels_ns.load_template_as_individual_leds
    # --- Saved-configuration I/O (see ui/config_io.py) ---
    _config_io_ns = _config_io.build(_SimpleNamespace(
        _ELIOS3_SLOTS=_ELIOS3_SLOTS,
        _clear_mirror_state=_clear_mirror_state,
        _enable_mirror_for=_enable_mirror_for,
        _mirror_primary=_mirror_primary,
        _panel_dropdowns=_panel_dropdowns,
        _panel_slot_data=_panel_slot_data,
        _refresh_vio_fov_label=_refresh_vio_fov_label,
        abs0_off_x=abs0_off_x,
        abs0_off_y=abs0_off_y,
        abs0_off_z=abs0_off_z,
        abs1_off_x=abs1_off_x,
        abs1_off_y=abs1_off_y,
        abs1_off_z=abs1_off_z,
        abs2_off_x=abs2_off_x,
        abs2_off_y=abs2_off_y,
        abs2_off_z=abs2_off_z,
        abs2_rot_z=abs2_rot_z,
        abs3_off_x=abs3_off_x,
        abs3_off_y=abs3_off_y,
        abs3_off_z=abs3_off_z,
        abs3_rot_z=abs3_rot_z,
        absorbers_enable=absorbers_enable,
        absorbers_folder=absorbers_folder,
        base_groups_active=base_groups_active,
        circle_center_slider=circle_center_slider,
        clear_stl_model=clear_stl_model,
        create_custom_group=create_custom_group,
        create_individual_led=create_individual_led,
        current_config_name=current_config_name,
        custom_groups=custom_groups,
        global_pos_x_slider=global_pos_x_slider,
        global_pos_y_slider=global_pos_y_slider,
        global_pos_z_slider=global_pos_z_slider,
        global_rotation_z_slider=global_rotation_z_slider,
        group_colors_hex=group_colors_hex,
        individual_leds=individual_leds,
        led_buttons=led_buttons,
        led_config_folder=led_config_folder,
        led_states=led_states,
        load_stl_file=load_stl_file,
        loading_in_progress=loading_in_progress,
        offset_front_neg_x=offset_front_neg_x,
        offset_front_neg_y=offset_front_neg_y,
        offset_front_neg_z=offset_front_neg_z,
        offset_front_pos_x=offset_front_pos_x,
        offset_front_pos_y=offset_front_pos_y,
        offset_front_pos_z=offset_front_pos_z,
        offset_side_neg_x=offset_side_neg_x,
        offset_side_neg_y=offset_side_neg_y,
        offset_side_neg_z=offset_side_neg_z,
        offset_side_pos_x=offset_side_pos_x,
        offset_side_pos_y=offset_side_pos_y,
        offset_side_pos_z=offset_side_pos_z,
        project_loaded=project_loaded,
        radius_slider=radius_slider,
        rot_front_neg=rot_front_neg,
        rot_front_pos=rot_front_pos,
        rot_side_neg=rot_side_neg,
        rot_side_pos=rot_side_pos,
        rot_y_front_neg=rot_y_front_neg,
        rot_y_front_pos=rot_y_front_pos,
        rot_y_side_neg=rot_y_side_neg,
        rot_y_side_pos=rot_y_side_pos,
        row1_chk=row1_chk,
        row2_chk=row2_chk,
        row3_chk=row3_chk,
        row4_chk=row4_chk,
        select_panel=select_panel,
        server=server,
        show_led_markers=show_led_markers,
        show_vio_fov=show_vio_fov,
        stl_absorber_enable=stl_absorber_enable,
        stl_file_path=stl_file_path,
        stl_mesh_data=stl_mesh_data,
        stl_opacity=stl_opacity,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
        stl_visible=stl_visible,
        stl_wireframe=stl_wireframe,
        tab_panels=tab_panels,
        template_folders=template_folders,
        update_scene=update_scene,
        viewing_angle_slider=viewing_angle_slider,
        vio_cam1_pitch=vio_cam1_pitch,
        vio_cam1_yaw=vio_cam1_yaw,
        vio_cam2_pitch=vio_cam2_pitch,
        vio_cam2_yaw=vio_cam2_yaw,
        vio_fill_fov=vio_fill_fov,
        vio_landscape=vio_landscape,
        vio_long_fov=vio_long_fov,
        vio_pos_x=vio_pos_x,
        vio_pos_y=vio_pos_y,
        vio_pos_z=vio_pos_z,
    ))
    _absorber_config = _config_io_ns._absorber_config
    apply_config = _config_io_ns.apply_config
    get_current_config = _config_io_ns.get_current_config
    new_project = _config_io_ns.new_project
    update_all_led_buttons = _config_io_ns.update_all_led_buttons
    update_ui_visibility = _config_io_ns.update_ui_visibility
    # --- Optimize tab (see ui/optimize_tab.py) ---
    _optimize_tab_ns = _optimize_tab.build(_SimpleNamespace(
        _project_root=_project_root,
        apply_config=apply_config,
        camera_fov_h=camera_fov_h,
        camera_fov_v=camera_fov_v,
        camera_pitch=camera_pitch,
        camera_pos_x=camera_pos_x,
        camera_pos_y=camera_pos_y,
        config_dir=config_dir,
        config_dropdown=config_dropdown,
        current_config_name=current_config_name,
        custom_groups=custom_groups,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        get_available_configs=get_available_configs,
        get_current_config=get_current_config,
        led_lumens_slider=led_lumens_slider,
        project_loaded=project_loaded,
        ray_uniformity_slider=ray_uniformity_slider,
        save_name_input=save_name_input,
        server=server,
        show_intensity_map=show_intensity_map,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        tab_optim=tab_optim,
        uniformity_percentile_slider=uniformity_percentile_slider,
        update_intensity_map=update_intensity_map,
        vio_cam1_pitch=vio_cam1_pitch,
        vio_cam1_yaw=vio_cam1_yaw,
        vio_cam2_pitch=vio_cam2_pitch,
        vio_cam2_yaw=vio_cam2_yaw,
        vio_landscape=vio_landscape,
        vio_long_fov=vio_long_fov,
        vio_pos_x=vio_pos_x,
        vio_pos_y=vio_pos_y,
        vio_pos_z=vio_pos_z,
        wall_dist_slider=wall_dist_slider,
        wall_view_size=wall_view_size,
    ))
    # Capture default values so reset restores them
    defaults = {
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
    update_ui_visibility()  # Set initial UI visibility indicators
    if room_mode_enable.value:
        draw_room_walls()

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
