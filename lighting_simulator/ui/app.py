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
from lighting_simulator.domain.led import apply_operating_mode
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
from lighting_simulator.ui import scene_view as _scene_view
from lighting_simulator.ui import intensity_map as _intensity_map
from lighting_simulator.ui import fov_capture as _fov_capture
from lighting_simulator.ui import exports as _exports
from lighting_simulator.ui import config_io as _config_io
from lighting_simulator.ui import panels as _panels
from lighting_simulator.analysis.uniformity import compute_uniformity_html as _compute_uniformity_html
from lighting_simulator.scene.absorbers import build_elios_absorbers, rotate_absorbers_z
from lighting_simulator.scene.builder import apply_diffuser, apply_global_transform
from lighting_simulator.scene.step_import import STEP_MM_TO_CM, is_step_file, load_step_mesh
from lighting_simulator.scene.stl import (
    _rot4_x, _rot4_y, _rot4_z,
    global_z_rotation_4x4,
    stl_mesh_data as stl_mesh_data_payload,
    stl_transform,
)

# Suppress viser warnings about removing already-removed nodes
warnings.filterwarnings("ignore", message="Attempted to remove already removed node")

_VIEW_FLIGHT = "Flight (continuous)"
_VIEW_FLASH = "Flash (pulse)"


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

    def flash_lumens():
        return float(flash_current_input.value) * float(led_voltage_input.value) * float(led_efficacy_input.value)

    def apply_view_mode(leds):
        """Apply the selected operating mode (LED roles) to freshly built LEDs, in place."""
        flash = view_mode_dropdown.value == _VIEW_FLASH
        return apply_operating_mode(leds, flash, flash_lumens() if flash else None)

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
                            'led_states': group['led_states'][:],
                            'led_roles': list(group.get('led_roles') or ['both'] * len(group['led_states'])),
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
                            'role': led.get('role', 'both'),
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
        view_mode_dropdown = server.gui.add_dropdown(
            "Operating mode", options=[_VIEW_FLIGHT, _VIEW_FLASH], initial_value=_VIEW_FLIGHT,
            hint="Flight: VIO + Both LEDs at their continuous flux, Flash-only LEDs off. "
                 "Flash: Flash + Both LEDs at the flash current (Display tab), VIO LEDs continuous. "
                 "Set roles per LED in the Selected panel or the Panel Designer.",
        )
        server.gui.add_html(
            "<div style='font-size:10px;color:#aaa;margin:-4px 0 6px;line-height:1.6;'>3-D LED markers: "
            "<span style='color:#FFFFFF;'>■</span> Both &nbsp;"
            "<span style='color:#1E90FF;'>■</span> VIO &nbsp;"
            "<span style='color:#FF8C00;'>■</span> Flash &nbsp;"
            "<span style='color:#666;'>■</span> dim = idle in this mode &nbsp;"
            "<span style='color:#2a2a2a;background:#555;'>■</span> dark grey = off &nbsp;"
            "<span style='color:#FFF21A;'>■</span> selected panel</div>"
        )
        mode_lumens_html = server.gui.add_html("")
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
        vio_occupancy_lux = server.gui.add_number(
            "VIO occupancy min lux", initial_value=0, min=0, max=100000, step=10,
            hint="VIO FOV Lighting Occupancy counts cells at or above this lux (0 = any light). Legend updates instantly.",
        )
        cell_area_html = server.gui.add_html(
            "<div style='font-family: sans-serif; font-size: 11px; color: #666; margin-top: -8px; margin-bottom: 8px;'>"
            "Cell area: calculating..."
            "</div>"
        )
        cell_readout_chk = server.gui.add_checkbox(
            "Cell readout on click", initial_value=False,
            hint="Click a cell of the wall or room intensity map in the 3-D view to show its lux value here.",
        )
        cell_readout_html = server.gui.add_html("")
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
            "LED lumens (lm/LED)", min=10, max=1000, step=10, initial_value=168,
            hint="Continuous flux used in the 'Flight' operating mode (VIO + Both LEDs). "
                 "Panels / LEDs with a lumens override use their own value instead.",
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Electrical (roles / flash):</b>")
        led_voltage_input = server.gui.add_number("LED forward voltage (V)", 6.0, min=1.0, max=60.0, step=0.1)
        led_efficacy_input = server.gui.add_number("Efficacy (lm/W)", 180.0, min=10.0, max=400.0, step=5.0)
        flash_current_input = server.gui.add_number(
            "Flash current per LED (A)", 13.0, min=0.1, max=50.0, step=0.1,
            hint="Used by the 'Flash (pulse)' operating mode: lm = I · V · efficacy for Flash / Both LEDs",
        )

        def _refresh_mode_lumens_html(_=None):
            v, eff, i_fl = float(led_voltage_input.value), float(led_efficacy_input.value), float(flash_current_input.value)
            lm_cont = float(led_lumens_slider.value)
            i_cont = lm_cont / (v * eff) if v * eff > 0 else 0.0
            mode_lumens_html.content = (
                "<div style='font-size:11px;color:#bbb;margin:-2px 0 6px;line-height:1.5;'>"
                f"<b>Flight</b> (VIO + Both): <b>{lm_cont:,.0f} lm</b>/LED ≈ {i_cont:.2f} A continuous "
                "<span style='color:#888;'>(Display → LED lumens; panel / LED overrides win)</span><br>"
                f"<b>Flash</b> (Flash + Both): <b>{flash_lumens():,.0f} lm</b>/LED = {i_fl:g} A × {v:g} V × {eff:g} lm/W "
                "<span style='color:#888;'>(Display → Electrical); VIO LEDs stay at their flight flux</span></div>"
            )

        for _h in (led_lumens_slider, led_voltage_input, led_efficacy_input, flash_current_input):
            _h.on_update(_refresh_mode_lumens_html)
        _refresh_mode_lumens_html()
        
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
        _stl_folder = server.gui.add_folder("3D Models (STL / STEP)")
    with _stl_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Import 3D CAD models</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:6px;'>"
                            "STEP keeps the CAD units and origin (true size, scale 0.1 = mm→cm). "
                            "STL has no units: it is centred and auto-fitted to 70 cm; set Scale to 0.1 for a mm export.</div>")
        stl_file_path = server.gui.add_text("Model file path (.stl / .step)", initial_value=r"C:\Users\gianmatteo.marietti_\Downloads\109045 E3 CAGE ASSEMBLY_Coarse.STL")
        stl_load_button = server.gui.add_button("📂 Load model", color="#4CAF50")
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
            """Load an STL or STEP model into the scene. Uses a numpy binary cache for fast reloads."""
            file_path = stl_file_path.value.strip()
            if not file_path:
                print("⚠️ Please enter a file path")
                return
            
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                return
            
            is_step = is_step_file(file_path)
            try:
                t_start = time.perf_counter()
                cache_path = _get_stl_cache_path(file_path)
                
                # Try loading from numpy binary cache first (10-50x faster)
                if os.path.exists(cache_path):
                    print(f"Loading {'STEP' if is_step else 'STL'} from cache: {os.path.basename(file_path)}")
                    cached = np.load(cache_path)
                    mesh = trimesh.Trimesh(
                        vertices=cached['vertices'],
                        faces=cached['faces'],
                        vertex_normals=cached['vertex_normals'],
                        process=False  # Skip expensive validation since we know data is good
                    )
                    t_load = time.perf_counter()
                    print(f"  Cache loaded in {t_load - t_start:.2f}s")
                elif is_step:
                    print(f"Loading STEP file: {file_path} (tessellating, will cache)")
                    mesh = load_step_mesh(file_path)
                    t_load = time.perf_counter()
                    print(f"  STEP tessellated in {t_load - t_start:.2f}s")
                    try:
                        np.savez(cache_path, vertices=mesh.vertices.astype(np.float32), faces=mesh.faces,
                                 vertex_normals=mesh.vertex_normals.astype(np.float32))
                    except Exception as ce:
                        print(f"  Warning: could not save cache: {ce}")
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
                
                if is_step:
                    # STEP is unit-aware: OpenCascade delivers mm, and the modelled origin is kept
                    ideal_scale = STEP_MM_TO_CM
                    stl_scale.value = ideal_scale
                    units_note = "mm (from STEP)"
                else:
                    # STL has no units: fit the largest dimension to 70 cm as a starting point
                    target_size_cm = 70.0
                    max_dimension = np.max(size)
                    ideal_scale = 1.0
                    if max_dimension > 0:
                        ideal_scale = target_size_cm / max_dimension
                        stl_scale.value = ideal_scale
                    units_note = "unknown (STL) — auto-fitted to 70 cm"
                
                # Update info
                info_text = (
                    f"<div style='font-family: sans-serif; font-size: 11px; color: #4CAF50;'>"
                    f"✓ Model loaded<br>"
                    f"Vertices: {num_vertices:,}<br>"
                    f"Faces: {num_faces:,}<br>"
                    f"Original size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} [{units_note}]<br>"
                    f"Scaled size: {size[0]*ideal_scale:.1f} × {size[1]*ideal_scale:.1f} × {size[2]*ideal_scale:.1f} cm"
                    f"</div>"
                )
                stl_info_html.content = info_text
                
                t_total = time.perf_counter() - t_start
                print(f"✓ {'STEP' if is_step else 'STL'} loaded: {num_vertices:,} vertices, {num_faces:,} faces ({t_total:.2f}s total)")
                update_stl_mesh(skip_lighting=True)
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
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
            "Front wall distance (cm)", min=20, max=300, step=10, initial_value=200
        )
        room_side_dist = server.gui.add_slider(
            "Side walls distance (cm)", min=20, max=300, step=10, initial_value=200
        )
        room_top_bottom_dist = server.gui.add_slider(
            "Top/Bottom walls distance (cm)", min=20, max=300, step=10, initial_value=200
        )
        show_back_wall = server.gui.add_checkbox("Show Back Wall", initial_value=False)
        room_back_dist = server.gui.add_slider(
            "Back wall distance (cm)", min=20, max=300, step=10, initial_value=50
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
        show_tilt_fovs = server.gui.add_checkbox(
            "Show ±tilt FOVs (up / down uniformity)", initial_value=False,
            hint="Two copies of this camera pitched up and down by the angle below. Their footprints are drawn "
                 "and the legend gains a separate uniformity for each.",
        )
        tilt_fov_deg = server.gui.add_slider("Tilt FOV angle (°)", min=5, max=85, step=5, initial_value=45)
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


    _scene_view_late = _SimpleNamespace()  # filled after ui.scene_view.build()

    def update_scene(*a, **k):
        return _scene_view_late.update_scene(*a, **k)

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
    # --- Wall intensity map (see ui/intensity_map.py) ---
    _intensity_map_ns = _intensity_map.build(_SimpleNamespace(
        _absorber_config=_absorber_config,
        _expand_mirror_configs=_expand_mirror_configs,
        _panel_slot_data=_panel_slot_data,
        apply_view_mode=apply_view_mode,
        bw_scale_chk=bw_scale_chk,
        calibration_factor_slider=calibration_factor_slider,
        camera_fov_h=camera_fov_h,
        camera_fov_v=camera_fov_v,
        camera_pitch=camera_pitch,
        camera_pos_x=camera_pos_x,
        camera_pos_y=camera_pos_y,
        cell_readout_chk=cell_readout_chk,
        cell_readout_html=cell_readout_html,
        circle_center_slider=circle_center_slider,
        custom_groups=custom_groups,
        custom_reflectance_slider=custom_reflectance_slider,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        global_pos_x_slider=global_pos_x_slider,
        global_pos_y_slider=global_pos_y_slider,
        global_pos_z_slider=global_pos_z_slider,
        global_rotation_z_slider=global_rotation_z_slider,
        individual_leds=individual_leds,
        intensity_grid_size=intensity_grid_size,
        intensity_handles=intensity_handles,
        intensity_rays_slider=intensity_rays_slider,
        intensity_threshold_slider=intensity_threshold_slider,
        led_lumens_slider=led_lumens_slider,
        led_states=led_states,
        legend_html=legend_html,
        legend_max_input=legend_max_input,
        max_bounces_slider_room=max_bounces_slider_room,
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
        radius_slider=radius_slider,
        ray_uniformity_slider=ray_uniformity_slider,
        reflections_enable=reflections_enable,
        room_mode_enable=room_mode_enable,
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
        server=server,
        show_intensity_map=show_intensity_map,
        show_tilt_fovs=show_tilt_fovs,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
        tilt_fov_deg=tilt_fov_deg,
        uniformity_percentile_slider=uniformity_percentile_slider,
        view_mode_dropdown=view_mode_dropdown,
        viewing_angle_slider=viewing_angle_slider,
        vio_cam1_pitch=vio_cam1_pitch,
        vio_cam1_yaw=vio_cam1_yaw,
        vio_cam2_pitch=vio_cam2_pitch,
        vio_cam2_yaw=vio_cam2_yaw,
        vio_landscape=vio_landscape,
        vio_long_fov=vio_long_fov,
        vio_occupancy_lux=vio_occupancy_lux,
        vio_pos_x=vio_pos_x,
        vio_pos_y=vio_pos_y,
        vio_pos_z=vio_pos_z,
        wall_dist_slider=wall_dist_slider,
        wall_view_size=wall_view_size,
    ))
    _build_current_leds_and_absorbers = _intensity_map_ns._build_current_leds_and_absorbers
    _build_lux_legend_html = _intensity_map_ns._build_lux_legend_html
    read_cell_at_ray = _intensity_map_ns.read_cell_at_ray
    _last_intensity_cache = _intensity_map_ns._last_intensity_cache
    _last_room_cache = _intensity_map_ns._last_room_cache
    _mode_toggle_syncing = _intensity_map_ns._mode_toggle_syncing
    _refresh_uniformity = _intensity_map_ns._refresh_uniformity
    _room_metrics_html = _intensity_map_ns._room_metrics_html
    compute_room_intensity = _intensity_map_ns.compute_room_intensity
    compute_wall_intensity = _intensity_map_ns.compute_wall_intensity
    intensity_to_color = _intensity_map_ns.intensity_to_color
    update_intensity_map = _intensity_map_ns.update_intensity_map
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
    # --- Main-camera FOV capture (see ui/fov_capture.py) ---
    _fov_capture_ns = _fov_capture.build(_SimpleNamespace(
        _expand_mirror_configs=_expand_mirror_configs,
        _panel_slot_data=_panel_slot_data,
        apply_view_mode=apply_view_mode,
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
        calibration_factor_slider=calibration_factor_slider,
        camera_fov_h=camera_fov_h,
        camera_fov_v=camera_fov_v,
        camera_pitch=camera_pitch,
        camera_pos_x=camera_pos_x,
        camera_pos_y=camera_pos_y,
        circle_center_slider=circle_center_slider,
        custom_groups=custom_groups,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        individual_leds=individual_leds,
        intensity_rays_slider=intensity_rays_slider,
        intensity_to_color=intensity_to_color,
        led_lumens_slider=led_lumens_slider,
        led_states=led_states,
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
        radius_slider=radius_slider,
        ray_uniformity_slider=ray_uniformity_slider,
        room_front_dist=room_front_dist,
        room_mode_enable=room_mode_enable,
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
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
        viewing_angle_slider=viewing_angle_slider,
        wall_dist_slider=wall_dist_slider,
    ))
    capture_camera_fov_image = _fov_capture_ns.capture_camera_fov_image
    # --- Export helpers (see ui/exports.py) ---
    _exports_ns = _exports.build(_SimpleNamespace(
        current_leds=current_leds,
        individual_leds=individual_leds,
    ))
    export_custom_group_dxf = _exports_ns.export_custom_group_dxf
    export_individual_leds_simple = _exports_ns.export_individual_leds_simple
    export_leds_to_stl = _exports_ns.export_leds_to_stl
    # --- 3-D scene (see ui/scene_view.py) ---
    _scene_view_ns = _scene_view.build(_SimpleNamespace(
        _ELIOS3_SLOTS=_ELIOS3_SLOTS,
        _clear_mirror_state=_clear_mirror_state,
        _clear_panel_slot=_clear_panel_slot,
        _enable_mirror_for=_enable_mirror_for,
        _expand_mirror_configs=_expand_mirror_configs,
        _inspector_handles=_inspector_handles,
        _inspector_syncing=_inspector_syncing,
        _just_clicked_mesh=_just_clicked_mesh,
        _last_intensity_cache=_last_intensity_cache,
        _last_room_cache=_last_room_cache,
        _mirror_counterpart_name=_mirror_counterpart_name,
        _mirror_primary=_mirror_primary,
        _mode_toggle_syncing=_mode_toggle_syncing,
        _owner_display_name=_owner_display_name,
        _owner_groups=_owner_groups,
        _owner_individual_leds=_owner_individual_leds,
        _panel_dropdowns=_panel_dropdowns,
        _panel_slot_data=_panel_slot_data,
        _refresh_uniformity=_refresh_uniformity,
        _refresh_vio_fov_label=_refresh_vio_fov_label,
        _select_panel_impl=_select_panel_impl,
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
        absorber_handles=absorber_handles,
        absorbers_enable=absorbers_enable,
        apply_view_mode=apply_view_mode,
        flash_lumens=flash_lumens,
        camera_fov_h=camera_fov_h,
        camera_fov_handles=camera_fov_handles,
        camera_fov_v=camera_fov_v,
        camera_pitch=camera_pitch,
        camera_pos_x=camera_pos_x,
        camera_pos_y=camera_pos_y,
        capture_camera_fov_image=capture_camera_fov_image,
        capture_fov_btn=capture_fov_btn,
        cell_area_html=cell_area_html,
        circle_center_slider=circle_center_slider,
        clear_csv_pattern=clear_csv_pattern,
        create_custom_group=create_custom_group,
        csv_clear_btn=csv_clear_btn,
        csv_import_btn=csv_import_btn,
        current_leds=current_leds,
        custom_groups=custom_groups,
        designer_gizmo=designer_gizmo,
        designer_led_nodes=designer_led_nodes,
        designer_mode=designer_mode,
        designer_scene_handles=designer_scene_handles,
        designer_state=designer_state,
        designer_syncing=designer_syncing,
        designer_ui_handles=designer_ui_handles,
        designer_widget_refs=designer_widget_refs,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        draw_room_walls=draw_room_walls,
        export_lux_matrix=export_lux_matrix,
        export_lux_matrix_button=export_lux_matrix_button,
        get_available_templates=get_available_templates,
        global_pos_x_slider=global_pos_x_slider,
        global_pos_y_slider=global_pos_y_slider,
        global_pos_z_slider=global_pos_z_slider,
        global_rotation_z_slider=global_rotation_z_slider,
        group_buttons=group_buttons,
        group_colors_hex=group_colors_hex,
        guide_handles=guide_handles,
        import_csv_pattern=import_csv_pattern,
        imported_csv_handles=imported_csv_handles,
        individual_leds=individual_leds,
        inspector_tab=inspector_tab,
        intensity_grid_size=intensity_grid_size,
        intensity_handles=intensity_handles,
        intensity_rays_slider=intensity_rays_slider,
        intensity_threshold_slider=intensity_threshold_slider,
        led_buttons=led_buttons,
        led_handles=led_handles,
        led_lumens_slider=led_lumens_slider,
        led_states=led_states,
        legend_html=legend_html,
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
        open_panel_designer_btn=open_panel_designer_btn,
        radius_slider=radius_slider,
        ray_handles=ray_handles,
        ray_length_slider=ray_length_slider,
        read_cell_at_ray=read_cell_at_ray,
        ray_uniformity_slider=ray_uniformity_slider,
        room_back_dist=room_back_dist,
        room_front_dist=room_front_dist,
        room_intensity_handles=room_intensity_handles,
        room_mode_enable=room_mode_enable,
        room_side_dist=room_side_dist,
        room_top_bottom_dist=room_top_bottom_dist,
        room_wall_handles=room_wall_handles,
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
        row_buttons=row_buttons,
        run_benchmark=run_benchmark,
        run_benchmark_button=run_benchmark_button,
        save_custom_group_template=save_custom_group_template,
        select_panel=select_panel,
        selected_owner=selected_owner,
        server=server,
        show_back_wall=show_back_wall,
        show_camera_fov=show_camera_fov,
        show_intensity_map=show_intensity_map,
        show_led_markers=show_led_markers,
        show_random_rays=show_random_rays,
        show_rays_output=show_rays_output,
        show_room_intensity=show_room_intensity,
        show_room_walls=show_room_walls,
        show_tilt_fovs=show_tilt_fovs,
        show_vio_fov=show_vio_fov,
        static_scene_handles=static_scene_handles,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        stl_mesh_handle=stl_mesh_handle,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
        template_dropdown=template_dropdown,
        tilt_fov_deg=tilt_fov_deg,
        uniformity_percentile_slider=uniformity_percentile_slider,
        update_all_led_buttons=update_all_led_buttons,
        update_intensity_button=update_intensity_button,
        update_intensity_map=update_intensity_map,
        update_room_button=update_room_button,
        update_room_intensity_map=update_room_intensity_map,
        update_stl_mesh=update_stl_mesh,
        update_ui_visibility=update_ui_visibility,
        view_mode_dropdown=view_mode_dropdown,
        flash_current_input=flash_current_input,
        led_voltage_input=led_voltage_input,
        led_efficacy_input=led_efficacy_input,
        viewing_angle_slider=viewing_angle_slider,
        vio_cam1_pitch=vio_cam1_pitch,
        vio_cam1_yaw=vio_cam1_yaw,
        vio_cam2_pitch=vio_cam2_pitch,
        vio_cam2_yaw=vio_cam2_yaw,
        vio_fill_fov=vio_fill_fov,
        vio_fov_handles=vio_fov_handles,
        vio_landscape=vio_landscape,
        vio_long_fov=vio_long_fov,
        vio_occupancy_lux=vio_occupancy_lux,
        vio_pos_x=vio_pos_x,
        vio_pos_y=vio_pos_y,
        vio_pos_z=vio_pos_z,
        wall_dist_slider=wall_dist_slider,
        wall_view_size=wall_view_size,
    ))
    update_scene = _scene_view_ns.update_scene
    update_wall = _scene_view_ns.update_wall
    _scene_view_late.update_scene = _scene_view_ns.update_scene
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
        flash_current_input=flash_current_input,
        get_available_configs=get_available_configs,
        get_current_config=get_current_config,
        led_efficacy_input=led_efficacy_input,
        led_lumens_slider=led_lumens_slider,
        led_voltage_input=led_voltage_input,
        project_loaded=project_loaded,
        ray_uniformity_slider=ray_uniformity_slider,
        save_name_input=save_name_input,
        server=server,
        show_intensity_map=show_intensity_map,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        tab_optim=tab_optim,
        tilt_fov_deg=tilt_fov_deg,
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
