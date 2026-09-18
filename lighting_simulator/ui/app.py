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
from lighting_simulator.ui import panels_tab as _panels_tab
from lighting_simulator.ui.layout_state import LayoutState
from lighting_simulator.analysis.uniformity import compute_uniformity_html as _compute_uniformity_html
from lighting_simulator.scene.builder import apply_diffuser
from lighting_simulator.scene.layout import (
    Layout, Platform, convert_v1, layout_to_v1, load_layout, load_platform, save_json,
)
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
    print("  1. Create a 🆕 New Project (empty, then add panels in the Panels & LEDs tab)")
    print("  2. Or 📂 Load an existing configuration (e.g., Elios 3)")
    print("\n💡 All LEDs are disabled until you load or create a project.")
    print("="*60 + "\n")

    # --- Configuration Management ---
    # Prefer the launch directory (so a bundled exe finds its data next to it);
    # otherwise fall back to the project root so it works from any cwd.
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _data_dir(name):
        return name if os.path.isdir(name) else os.path.join(_project_root, name)

    config_dir = _data_dir("layouts")          # schema-v2 layouts (legacy configs/*.json still load)
    platforms_dir = _data_dir("platforms")
    legacy_config_dir = _data_dir("configs")
    templates_dir = _data_dir("templates")     # panel templates (schema-v2 layouts, panel-local LEDs)
    for _d in (config_dir, platforms_dir, templates_dir):
        os.makedirs(_d, exist_ok=True)
    print(f"  📁  Layouts: {os.path.abspath(config_dir)}   platforms: {os.path.abspath(platforms_dir)}")

    # The scene: one Layout (panels of panel-local LEDs) + selection, see ui/layout_state.py
    state = LayoutState()

    # Flag to track if a project is loaded
    project_loaded = [False]  # Use list for mutability in nested functions
    current_config_name = [""]  # Track which configuration is loaded
    loading_in_progress = [False]  # Flag to prevent callbacks during config loading

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

    _just_clicked_mesh = [False]

    def flash_lumens():
        return float(flash_lumens_input.value)

    def apply_view_mode(leds):
        """Apply the selected operating mode (LED roles) to freshly built LEDs, in place."""
        flash = view_mode_dropdown.value == _VIEW_FLASH
        return apply_operating_mode(leds, flash, flash_lumens() if flash else None)

    # --- GUI Controls ---
    with tab_project:
        _project_folder = server.gui.add_folder("Project Management")
    with _project_folder:
        server.gui.add_html("<div style='font-weight:600;margin-bottom:6px;'>Start a new project or load existing</div>")
        
        new_project_btn = server.gui.add_button("🆕 New Project (Empty)", color="#4CAF50")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        
        def get_available_configs():
            names = {f[:-5] for f in os.listdir(config_dir) if f.lower().endswith(".json")}
            if os.path.isdir(legacy_config_dir) and os.path.abspath(legacy_config_dir) != os.path.abspath(config_dir):
                names |= {f"{f[:-5]} (legacy)" for f in os.listdir(legacy_config_dir) if f.lower().endswith(".json")}
            return sorted(names, key=str.lower)

        def _config_path(name):
            if name.endswith(" (legacy)"):
                return os.path.join(legacy_config_dir, f"{name[:-9]}.json")
            return os.path.join(config_dir, f"{name}.json")

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
            options=["Layout", "Panel template (selected panel)"],
            initial_value="Layout",
            hint="Layout: the whole scene (panels + flux + platform link) into layouts/. "
                 "Panel template: only the selected panel, with LEDs in panel coordinates, into templates/."
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
            path = _config_path(name)
            if os.path.exists(path):
                load_layout_file(path)
                save_name_input.value = name.replace(" (legacy)", "")
                print(f"  💡 Modify and click 'Save Project' to update the layout")

        def load_layout_file(path):
            """Load a v2 layout (or legacy v1 config) file into the scene."""
            lay = load_layout(path, default_lumens=float(led_lumens_slider.value))
            print(f"Loading layout: {lay.name} ({len(lay.panels)} panel(s), "
                  f"{sum(1 for p in lay.panels if p.mirror)} mirrored)")
            current_config_name[0] = lay.name or os.path.splitext(os.path.basename(path))[0]
            apply_layout(lay, platforms_dir=platforms_dir)
            _sync_platform_dropdown()
            led_lumens_slider.value = float(lay.flux.vio_lumens)
            if lay.flux.flash_lumens:
                flash_lumens_input.value = float(lay.flux.flash_lumens)
            print(f"✓ Layout loaded: {lay.name}")

        def current_layout(name=None):
            """The scene as a v2 Layout (flux from the Display tab, platform link or inline platform)."""
            lay = copy.deepcopy(state.layout)
            lay.flux.vio_lumens = float(led_lumens_slider.value)
            lay.flux.flash_lumens = float(flash_lumens_input.value)
            if state.platform_name:
                lay.platform = state.platform_name
            else:
                inline = convert_v1(get_current_config()).platform
                lay.platform = inline if isinstance(inline, Platform) else None
            if name:
                lay.name = name
            return lay

        @save_project_btn.on_click
        def _(_):
            name = save_name_input.value.strip()
            if not name:
                print("Error: Please enter a project name.")
                return
            if save_type_dropdown.value == "Layout":
                path = os.path.join(config_dir, f"{name.lower().replace(' ', '_')}.json")
                save_json(current_layout(name), path)
                current_config_name[0] = name
                print(f"✓ Layout saved: {path}")
                config_dropdown.options = get_available_configs()
            else:
                panel = state.selected_panel()
                if panel is None:
                    print("Error: select a panel (click it in the 3-D view) to save it as a template.")
                    return
                _panels_ns.save_panel_as_template(panel, name)
                _panels_ns.refresh_templates()

    with global_tab:
        server.gui.add_markdown("**Geometry**")
        wall_dist_slider = server.gui.add_slider(
            "Wall distance (cm)", min=10, max=1500, step=5, initial_value=50
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
        led_lumens_slider = server.gui.add_number(
            "Flight / VIO flux (lm/LED)", 168.0, min=1.0, max=100000.0, step=1.0,
            hint="Continuous flux of every LED in the 'Flight' operating mode (VIO + Both LEDs). "
                 "Saved in the layout as flux.vio_lumens.",
        )
        flash_lumens_input = server.gui.add_number(
            "Flash flux (lm/LED)", 14040.0, min=1.0, max=1000000.0, step=100.0,
            hint="Pulse flux of Flash / Both LEDs in the 'Flash' operating mode. Saved as flux.flash_lumens.",
        )
        server.gui.add_html("<hr style='margin:8px 0;'><b>Electrical helper (optional):</b>"
                            "<div style='color:#888;font-size:11px;'>Sets the flash flux from a current: "
                            "lm = I · V · efficacy.</div>")
        led_voltage_input = server.gui.add_number("LED forward voltage (V)", 6.0, min=1.0, max=60.0, step=0.1)
        led_efficacy_input = server.gui.add_number("Efficacy (lm/W)", 180.0, min=10.0, max=400.0, step=5.0)
        flash_current_input = server.gui.add_number("Flash current per LED (A)", 13.0, min=0.1, max=50.0, step=0.1)
        apply_flash_current_btn = server.gui.add_button("→ Set flash flux from current")

        @apply_flash_current_btn.on_click
        def _(_):
            flash_lumens_input.value = (float(flash_current_input.value) * float(led_voltage_input.value)
                                        * float(led_efficacy_input.value))

        def _refresh_mode_lumens_html(_=None):
            v, eff = float(led_voltage_input.value), float(led_efficacy_input.value)
            lm_cont = float(led_lumens_slider.value)
            i_cont = lm_cont / (v * eff) if v * eff > 0 else 0.0
            i_fl = flash_lumens() / (v * eff) if v * eff > 0 else 0.0
            mode_lumens_html.content = (
                "<div style='font-size:11px;color:#bbb;margin:-2px 0 6px;line-height:1.5;'>"
                f"<b>Flight</b> (VIO + Both): <b>{lm_cont:,.0f} lm</b>/LED (≈ {i_cont:.2f} A at {v:g} V, {eff:g} lm/W)<br>"
                f"<b>Flash</b> (Flash + Both): <b>{flash_lumens():,.0f} lm</b>/LED (≈ {i_fl:.1f} A); "
                "VIO LEDs stay at their flight flux</div>"
            )

        for _h in (led_lumens_slider, led_voltage_input, led_efficacy_input, flash_lumens_input):
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

    # 3D Model Import (STL files)
    stl_mesh_handle = [None]  # Store mesh handle for removal/update
    stl_mesh_data = [None]  # Store loaded trimesh object

    with tab_advanced:
        _stl_folder = server.gui.add_folder("Platform (frame model & VIO cameras)")
    with _stl_folder:
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:6px;'>A platform = the drone: its "
                            "CAD frame (STL/STEP, display + occlusion) and the VIO camera poses (FOV tab). Layouts link "
                            "to a platform by name so several designs share one.</div>")

        def get_available_platforms():
            return sorted((f[:-5] for f in os.listdir(platforms_dir) if f.lower().endswith(".json")), key=str.lower)

        _NO_PLATFORM = "(none / inline)"
        platform_dropdown = server.gui.add_dropdown("Platform", options=[_NO_PLATFORM] + get_available_platforms(),
                                                    initial_value=_NO_PLATFORM)
        load_platform_btn = server.gui.add_button("📂 Load platform")
        platform_name_input = server.gui.add_text("Save platform as", initial_value="")
        save_platform_btn = server.gui.add_button("💾 Save platform (frame + VIO cameras)")

        def _sync_platform_dropdown():
            platform_dropdown.options = [_NO_PLATFORM] + get_available_platforms()
            want = state.platform_name or _NO_PLATFORM
            platform_dropdown.value = want if want in platform_dropdown.options else _NO_PLATFORM

        server.gui.add_html("<hr style='margin:8px 0;'><div style='font-weight:600;margin-bottom:6px;'>Frame model (STL / STEP)</div>")
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:6px;'>"
                            "STEP keeps the CAD units and origin (true size, scale 0.1 = mm→cm). "
                            "STL has no units: it is centred and auto-fitted to 70 cm; set Scale to 0.1 for a mm export.</div>")
        stl_file_path = server.gui.add_text("Model file path (.stl / .step)", initial_value="")
        stl_load_button = server.gui.add_button("📂 Load model", color="#4CAF50")
        stl_clear_button = server.gui.add_button("🗑️ Clear Model", color="#FF5555")
        
        server.gui.add_html("<hr style='margin:8px 0;'>")
        stl_absorber_enable = server.gui.add_checkbox("Occludes light (shadow test)", initial_value=True)
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>When enabled, the frame blocks light rays in every simulation</div>")
        
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
                    
                    # Combined transform: Scale -> Rotate -> Translate -> cm-to-meters
                    # Instead of building 4 separate matrices and multiplying, build directly
                    R = T_rot[:3, :3]
                    scale_m = scale * 0.01  # scale * cm_to_meters
                    
                    # Transform vertices: v' = (R * scale_m) @ v + translate_m
                    RS = R * scale_m  # 3x3 scaled rotation
                    translate_m = np.array([pos_x, pos_y, pos_z]) * 0.01  # cm to meters
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
    imported_csv_handles = []
    
    # Store current LED objects (for reuse in room intensity calculation)
    current_leds = []

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
    
    with tab_panels:
        export_folder = server.gui.add_folder("Export", expand_by_default=False)
    with export_folder:
        server.gui.add_html("<div style='color:#888;font-size:11px;margin-bottom:8px;'>"
                            "World-space LED list (JSON), cover panel with one planar face per LED (STEP), "
                            "and a 2-D CNC cutting file of the selected panel (DXF).</div>")
        export_individual_leds_btn = server.gui.add_button("💾 Export LEDs to JSON", color="#4CAF50")
        export_stl_btn = server.gui.add_button("📦 Export Panel STEP", color="#FF9800")
        export_cnc_btn = server.gui.add_button("🔩 Export CNC DXF (selected panel)", color="#2196F3")

        @export_individual_leds_btn.on_click
        def _(_):
            export_individual_leds_simple()

        @export_stl_btn.on_click
        def _(_):
            export_leds_to_stl()

        @export_cnc_btn.on_click
        def _(_):
            export_custom_group_dxf()

    _scene_view_late = _SimpleNamespace()  # filled after ui.scene_view.build()

    def update_scene(*a, **k):
        return _scene_view_late.update_scene(*a, **k)

    # --- Panels & LEDs tab + Selected inspector (see ui/panels_tab.py) ---
    enter_designer_ref = [lambda index=None: print("Designer not ready yet")]
    _panels_ns = _panels_tab.build(_SimpleNamespace(
        enter_designer=enter_designer_ref,
        flash_lumens=flash_lumens,
        inspector_tab=inspector_tab,
        led_lumens_slider=led_lumens_slider,
        server=server,
        state=state,
        tab_panels=tab_panels,
        templates_dir=templates_dir,
    ))
    populate_inspector = _panels_ns.populate_inspector
    clear_inspector = _panels_ns.clear_inspector
    save_panel_as_template = _panels_ns.save_panel_as_template
    # --- Saved-configuration I/O (see ui/config_io.py) ---
    _config_io_ns = _config_io.build(_SimpleNamespace(
        state=state,
        _refresh_vio_fov_label=_refresh_vio_fov_label,
        clear_stl_model=clear_stl_model,
        current_config_name=current_config_name,
        load_stl_file=load_stl_file,
        loading_in_progress=loading_in_progress,
        project_loaded=project_loaded,
        server=server,
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
        update_scene=update_scene,
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
    apply_config = _config_io_ns.apply_config
    apply_layout = _config_io_ns.apply_layout
    apply_platform_cfg = _config_io_ns.apply_platform_cfg
    get_current_config = _config_io_ns.get_current_config
    new_project = _config_io_ns.new_project
    update_ui_visibility = _config_io_ns.update_ui_visibility

    @load_platform_btn.on_click
    def _(_):
        name = platform_dropdown.value
        if not name or name == _NO_PLATFORM:
            state.platform_name = None
            print("Platform link cleared: the frame / VIO settings will be saved inline with the layout.")
            return
        plat = load_platform(os.path.join(platforms_dir, f"{name}.json"))
        loading_in_progress[0] = True
        try:
            apply_platform_cfg(layout_to_v1(Layout(platform=plat), platform=plat))
        finally:
            loading_in_progress[0] = False
        state.platform_name = name
        update_scene()
        print(f"✓ Platform loaded: {name}")

    @save_platform_btn.on_click
    def _(_):
        name = platform_name_input.value.strip() or state.platform_name
        if not name:
            print("Error: enter a platform name.")
            return
        lay = convert_v1(get_current_config())
        plat = lay.platform if isinstance(lay.platform, Platform) else Platform()
        plat.name = name
        path = os.path.join(platforms_dir, f"{name.lower().replace(' ', '_')}.json")
        save_json(plat, path)
        state.platform_name = os.path.splitext(os.path.basename(path))[0]
        _sync_platform_dropdown()
        print(f"✓ Platform saved: {path}")
    # --- Wall intensity map (see ui/intensity_map.py) ---
    _intensity_map_ns = _intensity_map.build(_SimpleNamespace(
        state=state,
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
        custom_reflectance_slider=custom_reflectance_slider,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        intensity_grid_size=intensity_grid_size,
        intensity_handles=intensity_handles,
        intensity_rays_slider=intensity_rays_slider,
        intensity_threshold_slider=intensity_threshold_slider,
        led_lumens_slider=led_lumens_slider,
        legend_html=legend_html,
        legend_max_input=legend_max_input,
        max_bounces_slider_room=max_bounces_slider_room,
        ray_uniformity_slider=ray_uniformity_slider,
        reflections_enable=reflections_enable,
        room_mode_enable=room_mode_enable,
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
        compute_room_intensity=compute_room_intensity,
        current_leds=current_leds,
        intensity_rays_slider=intensity_rays_slider,
        intensity_to_color=intensity_to_color,
        legend_html=legend_html,
        legend_max_input=legend_max_input,
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
        state=state,
        apply_view_mode=apply_view_mode,
        calibration_factor_slider=calibration_factor_slider,
        camera_fov_h=camera_fov_h,
        camera_fov_v=camera_fov_v,
        camera_pitch=camera_pitch,
        camera_pos_x=camera_pos_x,
        camera_pos_y=camera_pos_y,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        intensity_rays_slider=intensity_rays_slider,
        intensity_to_color=intensity_to_color,
        led_lumens_slider=led_lumens_slider,
        ray_uniformity_slider=ray_uniformity_slider,
        room_front_dist=room_front_dist,
        room_mode_enable=room_mode_enable,
        stl_absorber_enable=stl_absorber_enable,
        stl_mesh_data=stl_mesh_data,
        stl_pos_x=stl_pos_x,
        stl_pos_y=stl_pos_y,
        stl_pos_z=stl_pos_z,
        stl_rot_x=stl_rot_x,
        stl_rot_y=stl_rot_y,
        stl_rot_z=stl_rot_z,
        stl_scale=stl_scale,
        wall_dist_slider=wall_dist_slider,
    ))
    capture_camera_fov_image = _fov_capture_ns.capture_camera_fov_image
    # --- Export helpers (see ui/exports.py) ---
    _exports_ns = _exports.build(_SimpleNamespace(
        current_leds=current_leds,
        state=state,
    ))
    export_custom_group_dxf = _exports_ns.export_custom_group_dxf
    export_individual_leds_simple = _exports_ns.export_individual_leds_simple
    export_leds_to_stl = _exports_ns.export_leds_to_stl
    # --- 3-D scene (see ui/scene_view.py) ---
    _scene_view_ns = _scene_view.build(_SimpleNamespace(
        state=state,
        populate_inspector=populate_inspector,
        clear_inspector=clear_inspector,
        save_panel_as_template=save_panel_as_template,
        enter_designer_ref=enter_designer_ref,
        _just_clicked_mesh=_just_clicked_mesh,
        _last_intensity_cache=_last_intensity_cache,
        _last_room_cache=_last_room_cache,
        _mode_toggle_syncing=_mode_toggle_syncing,
        _refresh_uniformity=_refresh_uniformity,
        _refresh_vio_fov_label=_refresh_vio_fov_label,
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
        clear_csv_pattern=clear_csv_pattern,
        csv_clear_btn=csv_clear_btn,
        csv_import_btn=csv_import_btn,
        current_leds=current_leds,
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
        guide_handles=guide_handles,
        import_csv_pattern=import_csv_pattern,
        imported_csv_handles=imported_csv_handles,
        inspector_tab=inspector_tab,
        intensity_grid_size=intensity_grid_size,
        intensity_handles=intensity_handles,
        intensity_rays_slider=intensity_rays_slider,
        intensity_threshold_slider=intensity_threshold_slider,
        led_handles=led_handles,
        led_lumens_slider=led_lumens_slider,
        legend_html=legend_html,
        loading_in_progress=loading_in_progress,
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
        run_benchmark=run_benchmark,
        run_benchmark_button=run_benchmark_button,
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
        tilt_fov_deg=tilt_fov_deg,
        uniformity_percentile_slider=uniformity_percentile_slider,
        update_intensity_button=update_intensity_button,
        update_intensity_map=update_intensity_map,
        update_room_button=update_room_button,
        update_room_intensity_map=update_room_intensity_map,
        update_stl_mesh=update_stl_mesh,
        update_ui_visibility=update_ui_visibility,
        view_mode_dropdown=view_mode_dropdown,
        flash_lumens_input=flash_lumens_input,
        led_voltage_input=led_voltage_input,
        led_efficacy_input=led_efficacy_input,
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
        state=state,
        diffuser_angle_slider=diffuser_angle_slider,
        diffuser_enable_chk=diffuser_enable_chk,
        diffuser_transmission_slider=diffuser_transmission_slider,
        flash_lumens_input=flash_lumens_input,
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

    # Any change of the layout (panel added / moved / toggled ...) redraws the 3-D scene
    state.on_change(lambda what: update_scene())

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

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
