"""Export helpers: world-space LED JSON, LED STEP and panel DXF.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import json
import os
import time
import numpy as np
import trimesh
from lighting_simulator.domain.geometry import as_vec3 as _as_vec3


def build(ctx):
    current_leds = ctx.current_leds
    state = ctx.state

    def export_individual_leds_simple():
        """Export every LED of the scene in world coordinates (position, normal, size, beam) as JSON."""
        leds = [led for led in current_leds if getattr(led, 'enabled', True)]
        if not leds:
            print("⚠️ No LEDs in the scene to export")
            return

        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)

        leds_export = []
        for i, led in enumerate(leds):
            n = np.asarray(getattr(led, 'mesh_normal', led.direction), dtype=float)
            leds_export.append({
                "id": i,
                "on": bool(getattr(led, 'led_on', True)),
                "role": getattr(led, 'role', 'both'),
                "position_cm": [float(v) for v in led.position],
                "normal": [float(v) for v in n],
                "beam_direction": [float(v) for v in led.direction],
                "size_cm": float(led.width),
                "beam_angle_deg": float(led.viewing_angle),
                "lumens": None if led.lumens is None else float(led.lumens),
            })

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(export_dir, f"leds_world_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump({
                "format_version": "2.0",
                "description": "All scene LEDs in world coordinates (cm), after panel placement and mirroring",
                "export_date": timestamp,
                "num_leds": len(leds_export),
                "leds": leds_export,
            }, f, indent=2)
        print(f"✓ Exported {len(leds_export)} LED(s) to: {filepath}")
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
        
        # Selected panel's LEDs if any, else every enabled LED in the scene
        sel = state.selected
        custom_leds = [
            led for led in current_leds
            if getattr(led, 'enabled', True)
            and (sel is None or getattr(led, 'owner', None) == ('panel', sel))
        ]
        if not custom_leds:
            print("⚠️ No active LEDs to export (select a panel or enable some LEDs).")
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
    


    return SimpleNamespace(export_custom_group_dxf=export_custom_group_dxf, export_individual_leds_simple=export_individual_leds_simple, export_leds_to_stl=export_leds_to_stl)
