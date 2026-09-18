"""Sphere mode: illuminance on a sphere of editable radius around the drone (equal-distance VIO coverage view).

``build(ctx)`` adds the "Sphere Mode" folder to the Advanced tab, draws the coloured sphere from
the inside (single-sided, so it never hides the drone when viewed from outside) and writes the
legend + main-camera / VIO-camera coverage metrics.
"""
from types import SimpleNamespace
import time

import numpy as np
import trimesh
from trimesh.visual import ColorVisuals

from lighting_simulator.analysis.uniformity import compute_uniformity_html
from lighting_simulator.camera.fov import points_in_fisheye_fov, points_in_pinhole_fov, vio_hfov_vfov_deg
from lighting_simulator.simulation.sphere import (
    SphereSettings, compute_sphere_illuminance, sphere_cell_vertices, sphere_cells,
)

_DIM = 0.3  # brightness of cells outside the VIO FOV when dimming is on


def build(ctx):
    server = ctx.server
    tab_advanced = ctx.tab_advanced
    legend_html = ctx.legend_html
    legend_max_input = ctx.legend_max_input
    intensity_to_color = ctx.intensity_to_color
    _build_lux_legend_html = ctx._build_lux_legend_html
    _build_current_leds_and_absorbers = ctx._build_current_leds_and_absorbers
    _emission_settings = ctx._emission_settings
    uniformity_percentile_slider = ctx.uniformity_percentile_slider
    intensity_threshold_slider = ctx.intensity_threshold_slider
    vio_occupancy_lux = ctx.vio_occupancy_lux
    view_mode_dropdown = ctx.view_mode_dropdown
    hide_wall = ctx.hide_wall
    update_wall = ctx.update_wall
    camera_pos_x, camera_pos_y, camera_pitch = ctx.camera_pos_x, ctx.camera_pos_y, ctx.camera_pitch
    camera_fov_h, camera_fov_v = ctx.camera_fov_h, ctx.camera_fov_v
    vio_pos_x, vio_pos_y, vio_pos_z = ctx.vio_pos_x, ctx.vio_pos_y, ctx.vio_pos_z
    vio_cam1_pitch, vio_cam1_yaw = ctx.vio_cam1_pitch, ctx.vio_cam1_yaw
    vio_cam2_pitch, vio_cam2_yaw = ctx.vio_cam2_pitch, ctx.vio_cam2_yaw
    vio_long_fov, vio_landscape = ctx.vio_long_fov, ctx.vio_landscape

    handles = []
    cache = {'lux': None, 'settings': None, 'mode': None}

    with tab_advanced:
        folder = server.gui.add_folder("Sphere Mode (equal-distance coverage)", expand_by_default=False)
    with folder:
        server.gui.add_html(
            "<div style='color:#888;font-size:11px;margin-bottom:6px;'>Illuminance on a sphere centred on the "
            "drone: every cell is at the same distance, so the map shows the angular light distribution "
            "(VIO coverage all around) without the 1/d² fall-off of flat walls. Analytic direct light, "
            "shadowed by the frame model when it occludes; no reflections. Drawn from the inside — orbit "
            "the view into the sphere or look through it from outside.</div>")
        sphere_enable = server.gui.add_checkbox("Enable Sphere Mode", initial_value=False)
        sphere_radius = server.gui.add_number("Sphere radius (cm)", 300.0, min=20.0, max=5000.0, step=10.0)
        sphere_cells_n = server.gui.add_slider("Cells around (longitude)", min=24, max=360, step=12, initial_value=90,
                                               hint="Latitude rows = half of this. 90 → 4 050 cells of ≈ 21 cm at R = 300 cm.")
        sphere_dim_outside = server.gui.add_checkbox("Dim cells outside the VIO FOV", initial_value=True)
        update_btn = server.gui.add_button("Update Sphere Intensity", color="#FFA500")
        info_html = server.gui.add_html("")

    def _settings():
        return SphereSettings(radius_cm=float(sphere_radius.value), n_phi=int(sphere_cells_n.value))

    def _clear():
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()

    def _vio_masks(centers):
        cam = np.array([vio_pos_x.value, vio_pos_y.value, vio_pos_z.value], dtype=float)
        hfov, vfov = vio_hfov_vfov_deg(vio_long_fov.value, vio_landscape.value)
        m1 = points_in_fisheye_fov(cam, vio_cam1_pitch.value, vio_cam1_yaw.value, hfov, vfov, centers)
        m2 = points_in_fisheye_fov(cam, vio_cam2_pitch.value, vio_cam2_yaw.value, hfov, vfov, centers)
        return m1, m2

    def _main_cam_mask(centers):
        cam = np.array([camera_pos_x.value, camera_pos_y.value, 0.0], dtype=float)
        return points_in_pinhole_fov(cam, camera_pitch.value, camera_fov_h.value, camera_fov_v.value, centers)

    def _draw_placeholder():
        _clear()
        if not sphere_enable.value:
            return
        s = _settings()
        quads = sphere_cell_vertices(SphereSettings(radius_cm=s.radius_cm, n_phi=36))
        segs = np.concatenate([quads[..., [0, 1], :], quads[..., [0, 3], :]], axis=2).reshape(-1, 2, 3)
        handles.append(server.scene.add_line_segments("/sphere/wire", points=segs.astype(np.float32),
                                                      colors=(0.35, 0.35, 0.35), line_width=1.0))

    def _render():
        """Colour the cached lux onto the sphere and write the legend (cheap; no recomputation)."""
        _clear()
        lux, s = cache['lux'], cache['settings']
        if lux is None or not sphere_enable.value:
            return
        centers, _, areas = sphere_cells(s)
        m1, m2 = _vio_masks(centers)
        union = m1 | m2
        max_lux = float(lux.max()) if lux.size else 0.0
        cap = float(legend_max_input.value)
        scale_max = cap if max_lux <= cap else max_lux
        n_t, n_p = lux.shape
        rgb = np.array([[intensity_to_color(v, scale_max) for v in row] for row in lux], dtype=float)
        if sphere_dim_outside.value:
            rgb[~union] *= _DIM
        quads = sphere_cell_vertices(s)  # (n_t, n_p, 4, 3) m, inward winding
        tris = np.concatenate([quads[..., [0, 1, 2], :], quads[..., [0, 2, 3], :]], axis=2)  # (n_t, n_p, 6, 3)
        verts = tris.reshape(-1, 3).astype(np.float32)
        faces = np.arange(len(verts), dtype=np.uint32).reshape(-1, 3)
        rgba = np.concatenate([(rgb * 255).astype(np.uint8), np.full((n_t, n_p, 1), 255, np.uint8)], axis=-1)
        vcol = np.repeat(rgba.reshape(-1, 1, 4), 6, axis=1).reshape(-1, 4)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=vcol)
        handles.append(server.scene.add_mesh_trimesh("/sphere/intensity", mesh=mesh))

        # ---- legend + metrics
        pct = float(uniformity_percentile_slider.value)
        thr = float(vio_occupancy_lux.value)
        good = lux >= thr if thr > 0 else lux > 0
        html = _build_lux_legend_html(max_lux, scale_max, float(areas.mean()), cell_caption="lm/cell avg")
        html += (f"<div style='font-size:11px;color:#bbb;margin:6px 0 -2px;'>Sphere R = <b>{s.radius_cm:g} cm</b>, "
                 f"{n_t}×{n_p} cells, operating mode <b>{cache['mode']}</b>. Whole sphere: Eavg "
                 f"{float((lux * areas).sum() / areas.sum()):,.1f} lx, lit {100 * float(areas[lux > 0].sum() / areas.sum()):.0f}% of the area.</div>")
        mc = _main_cam_mask(centers)
        html += (compute_uniformity_html(lux[mc].reshape(1, -1), min_percentile=pct, title="Main camera FOV on the sphere")
                 if mc.any() else "") or "<div style='color:#888;font-size:12px;margin-top:8px;'>Main camera FOV: no lit cells</div>"

        def _row(label, color, mask):
            a = areas[mask].sum()
            if a <= 0:
                return f"<tr><td style='padding:1px 6px 1px 0;color:{color};'>{label}</td><td colspan=3>—</td></tr>"
            cov = 100.0 * areas[mask & good].sum() / a
            vals = lux[mask]
            eavg = float((vals * areas[mask]).sum() / a)
            lit = vals[vals > 0]
            emin = float(np.percentile(lit, pct)) if lit.size and pct > 0 else (float(lit.min()) if lit.size else 0.0)
            u0 = emin / eavg if eavg > 0 else 0.0
            c = "#4CAF50" if cov >= 50.0 else "#FF9800"
            return (f"<tr><td style='padding:1px 6px 1px 0;color:{color};'>{label}</td>"
                    f"<td style='color:{c};font-weight:700;'>{cov:.1f}%</td><td>{eavg:,.1f} lx</td><td>{u0:.2f}</td></tr>")

        html += (
            "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
            "<div style='font-weight:600;margin-bottom:4px;'>VIO FOV coverage on the sphere</div>"
            f"<div style='color:#888;font-size:10px;margin-bottom:6px;'>Area share of FOV cells "
            f"{'at or above ' + format(thr, '.0f') + ' lx' if thr > 0 else 'with any light'}; "
            f"Eavg area-weighted; U0 = E{'P' + format(pct, 'g') + '%' if pct > 0 else 'min'}/Eavg over lit cells.</div>"
            "<table style='font-size:11px;color:#ccc;border-collapse:collapse;width:100%;'>"
            "<tr style='color:#888;'><td></td><td>coverage</td><td>Eavg</td><td>U0</td></tr>"
            + _row("Cam 1 (up)", "#ff00ff", m1) + _row("Cam 2 (down)", "#00ffff", m2)
            + _row("Union (Cam 1 ∪ Cam 2)", "#ddd", union)
            + "</table></div>")
        legend_html.content = html

    def update_sphere_intensity():
        if not sphere_enable.value:
            print("Sphere Mode is off — enable it first.")
            return
        leds, _, stl_payload = _build_current_leds_and_absorbers()
        active = [led for led in leds if getattr(led, 'enabled', True)]
        if not active:
            print("Sphere: no active LEDs")
            return
        s = _settings()
        t0 = time.perf_counter()
        lux = compute_sphere_illuminance(active, s, _emission_settings(), stl_mesh_data=stl_payload)
        dt = time.perf_counter() - t0
        cache.update(lux=lux, settings=s, mode=view_mode_dropdown.value)
        _, _, areas = sphere_cells(s)
        print(f"=== SPHERE MODE === R={s.radius_cm:g} cm, {lux.shape[0]}×{lux.shape[1]} cells, {len(active)} LEDs, "
              f"{dt:.2f}s | on sphere {(lux * areas).sum():,.1f} lm, peak {lux.max():,.1f} lx")
        info_html.content = (f"<div style='font-size:11px;color:#4CAF50;'>✓ {lux.shape[0]}×{lux.shape[1]} cells in {dt:.2f}s"
                             f" — peak {lux.max():,.0f} lx, {(lux * areas).sum():,.0f} lm on the sphere</div>")
        _render()

    def _on_toggle(_):
        if sphere_enable.value:
            hide_wall()
            if cache['lux'] is not None and cache['settings'] is not None:
                _render()
            else:
                _draw_placeholder()
                legend_html.content = ("<div style='font-family: sans-serif;'><div style='font-weight:600;margin-bottom:6px;'>"
                                       "Intensity legend</div><div style='color:#888;font-size:12px;'>Sphere Mode is active. "
                                       "Click 'Update Sphere Intensity'.</div></div>")
        else:
            _clear()
            update_wall()

    def _stale(_):
        if sphere_enable.value and cache['lux'] is not None:
            cache['lux'] = None
            _draw_placeholder()
            info_html.content = "<div style='font-size:11px;color:#F0AD4E;'>⚠ Sphere changed — click 'Update Sphere Intensity'.</div>"

    sphere_enable.on_update(_on_toggle)
    update_btn.on_click(lambda _: update_sphere_intensity())
    sphere_radius.on_update(_stale)
    sphere_cells_n.on_update(_stale)
    view_mode_dropdown.on_update(_stale)
    ctx.state.on_change(lambda what: _stale(None))
    for h in (sphere_dim_outside, legend_max_input, intensity_threshold_slider, uniformity_percentile_slider, vio_occupancy_lux,
              vio_pos_x, vio_pos_y, vio_pos_z, vio_cam1_pitch, vio_cam1_yaw, vio_cam2_pitch, vio_cam2_yaw,
              vio_long_fov, vio_landscape, camera_pos_x, camera_pos_y, camera_pitch, camera_fov_h, camera_fov_v):
        h.on_update(lambda _: _render() if (sphere_enable.value and cache['lux'] is not None) else None)

    return SimpleNamespace(sphere_enable=sphere_enable, update_sphere_intensity=update_sphere_intensity,
                           render_sphere=_render)
