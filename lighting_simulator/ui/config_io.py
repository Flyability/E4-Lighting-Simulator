"""Layout <-> GUI: apply a layout (or legacy v1 config) to the scene state, platform (STL / VIO) I/O, new project."""
from types import SimpleNamespace
import os

from lighting_simulator.scene.layout import Layout, convert_v1, is_v2, layout_from_dict, layout_to_v1


def build(ctx):
    state = ctx.state
    _refresh_vio_fov_label = ctx._refresh_vio_fov_label
    clear_stl_model = ctx.clear_stl_model
    current_config_name = ctx.current_config_name
    load_stl_file = ctx.load_stl_file
    loading_in_progress = ctx.loading_in_progress
    project_loaded = ctx.project_loaded
    show_vio_fov = ctx.show_vio_fov
    stl_absorber_enable = ctx.stl_absorber_enable
    stl_file_path = ctx.stl_file_path
    stl_mesh_data = ctx.stl_mesh_data
    stl_opacity = ctx.stl_opacity
    stl_pos_x = ctx.stl_pos_x
    stl_pos_y = ctx.stl_pos_y
    stl_pos_z = ctx.stl_pos_z
    stl_rot_x = ctx.stl_rot_x
    stl_rot_y = ctx.stl_rot_y
    stl_rot_z = ctx.stl_rot_z
    stl_scale = ctx.stl_scale
    stl_visible = ctx.stl_visible
    stl_wireframe = ctx.stl_wireframe
    update_scene = ctx.update_scene
    vio_cam1_pitch = ctx.vio_cam1_pitch
    vio_cam1_yaw = ctx.vio_cam1_yaw
    vio_cam2_pitch = ctx.vio_cam2_pitch
    vio_cam2_yaw = ctx.vio_cam2_yaw
    vio_fill_fov = ctx.vio_fill_fov
    vio_landscape = ctx.vio_landscape
    vio_long_fov = ctx.vio_long_fov
    vio_pos_x = ctx.vio_pos_x
    vio_pos_y = ctx.vio_pos_y
    vio_pos_z = ctx.vio_pos_z

    def get_current_config():
        """The scene as a v1 runtime dict (optimiser / legacy exporters), frame + VIO from the GUI."""
        cfg = state.v1_config()
        cfg['stl_model'] = {
            "file_path": stl_file_path.value, "absorber_enable": stl_absorber_enable.value,
            "visible": stl_visible.value, "scale": stl_scale.value,
            "position": [stl_pos_x.value, stl_pos_y.value, stl_pos_z.value],
            "rotation": [stl_rot_x.value, stl_rot_y.value, stl_rot_z.value],
            "opacity": stl_opacity.value, "wireframe": stl_wireframe.value,
        } if stl_mesh_data[0] is not None else None
        cfg['vio_cameras'] = {
            "show": show_vio_fov.value, "fill": vio_fill_fov.value,
            "position": [vio_pos_x.value, vio_pos_y.value, vio_pos_z.value],
            "cam1_pitch": vio_cam1_pitch.value, "cam1_yaw": vio_cam1_yaw.value,
            "cam2_pitch": vio_cam2_pitch.value, "cam2_yaw": vio_cam2_yaw.value,
            "long_fov": vio_long_fov.value, "landscape": vio_landscape.value,
        }
        cfg['platform_name'] = state.platform_name
        return cfg

    def apply_platform_cfg(cfg):
        """STL frame model + VIO camera poses from a v1-style config section."""
        # Load STL model configuration if present
        stl_cfg = cfg.get("stl_model")
        if stl_cfg:
            # Clear existing model first
            clear_stl_model()
            
            # Load file path and try to load the model
            file_path = stl_cfg.get('file_path', '')
            if file_path and os.path.exists(file_path):
                stl_file_path.value = file_path
                load_stl_file()  # Load the mesh
                
                # Apply saved settings
                stl_absorber_enable.value = stl_cfg.get('absorber_enable', True)
                stl_visible.value = stl_cfg.get('visible', True)
                stl_scale.value = stl_cfg.get('scale', 1.0)
                
                position = stl_cfg.get('position', [0, 0, 0])
                stl_pos_x.value = position[0]
                stl_pos_y.value = position[1]
                stl_pos_z.value = position[2]
                
                rotation = stl_cfg.get('rotation', [0, 0, 0])
                stl_rot_x.value = rotation[0]
                stl_rot_y.value = rotation[1]
                stl_rot_z.value = rotation[2]
                
                stl_opacity.value = stl_cfg.get('opacity', 0.8)
                stl_wireframe.value = stl_cfg.get('wireframe', False)
                
                print(f"✓ STL model loaded from config: {os.path.basename(file_path)}")
            else:
                if file_path:
                    print(f"⚠️ STL file not found: {file_path}")
        else:
            # No STL model in config, clear any existing model
            clear_stl_model()
        

        vio_cfg = cfg.get("vio_cameras")
        if vio_cfg:
            show_vio_fov.value = vio_cfg.get("show", True)
            vio_fill_fov.value = vio_cfg.get("fill", False)
            pos = vio_cfg.get("position", [3.0, 0.0, 0.0])
            vio_pos_x.value = pos[0]
            vio_pos_y.value = pos[1] if len(pos) > 1 else 0.0
            vio_pos_z.value = pos[2] if len(pos) > 2 else 0.0
            vio_cam1_pitch.value = vio_cfg.get("cam1_pitch", 45)
            vio_cam1_yaw.value = vio_cfg.get("cam1_yaw", 0)
            vio_cam2_pitch.value = vio_cfg.get("cam2_pitch", -45)
            vio_cam2_yaw.value = vio_cfg.get("cam2_yaw", 0)
            vio_long_fov.value = vio_cfg.get("long_fov", 170)
            vio_landscape.value = vio_cfg.get("landscape", True)
            _refresh_vio_fov_label()
        

    def apply_layout(layout: Layout, platforms_dir="platforms"):
        """Replace the scene with ``layout`` and apply its platform (frame + VIO) to the GUI."""
        loading_in_progress[0] = True
        state.loading = True
        try:
            apply_platform_cfg(layout_to_v1(layout, platforms_dir=platforms_dir))
        finally:
            state.loading = False
            loading_in_progress[0] = False
        state.set_layout(layout)
        project_loaded[0] = True
        update_scene()

    def apply_config(cfg, platforms_dir="platforms"):
        """Apply a saved config dict (schema v2 layout or legacy v1)."""
        layout = layout_from_dict(cfg) if is_v2(cfg) else convert_v1(cfg)
        apply_layout(layout, platforms_dir=platforms_dir)

    def update_ui_visibility():
        return None

    def new_project():
        """Empty scene, no frame model, default VIO poses."""
        print("Creating new empty project...")
        current_config_name[0] = ""
        loading_in_progress[0] = True
        try:
            clear_stl_model()
            stl_file_path.value = ""
            stl_absorber_enable.value = True
            stl_visible.value = True
            stl_scale.value = 1.0
            for h in (stl_pos_x, stl_pos_y, stl_pos_z, stl_rot_x, stl_rot_y, stl_rot_z):
                h.value = 0.0
            stl_opacity.value = 0.8
            stl_wireframe.value = False
        finally:
            loading_in_progress[0] = False
        state.clear()
        project_loaded[0] = True
        update_scene()
        print("✓ New empty project created - add panels from the Panels & LEDs tab")

    return SimpleNamespace(apply_platform_cfg=apply_platform_cfg, apply_config=apply_config, apply_layout=apply_layout,
                           get_current_config=get_current_config, new_project=new_project,
                           update_ui_visibility=update_ui_visibility)
