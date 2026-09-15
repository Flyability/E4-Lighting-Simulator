"""CSV pattern import (benchmark / FOV captures), lux-matrix export and the multi-distance benchmark.

Extracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and
callbacks it needs and returns the closures main() keeps using.
"""
from types import SimpleNamespace
import os
import time as _time
import numpy as np
import trimesh
from lighting_simulator.analysis.uniformity import compute_uniformity_html as _compute_uniformity_html


def build(ctx):
    _build_current_leds_and_absorbers = ctx._build_current_leds_and_absorbers
    _last_intensity_cache = ctx._last_intensity_cache
    compute_wall_intensity = ctx.compute_wall_intensity
    csv_diff_html = ctx.csv_diff_html
    csv_import_path = ctx.csv_import_path
    csv_import_status = ctx.csv_import_status
    csv_legend_html = ctx.csv_legend_html
    csv_legend_max_input = ctx.csv_legend_max_input
    imported_csv_handles = ctx.imported_csv_handles
    intensity_grid_size = ctx.intensity_grid_size
    intensity_rays_slider = ctx.intensity_rays_slider
    intensity_to_color = ctx.intensity_to_color
    room_front_dist = ctx.room_front_dist
    room_mode_enable = ctx.room_mode_enable
    server = ctx.server
    wall_dist_slider = ctx.wall_dist_slider
    wall_view_size = ctx.wall_view_size

    def _parse_benchmark_csv(filepath):
        """Parse a benchmark CSV file (NORMAL + UNIFORMITY sections).
        Supports both the internal text export and spreadsheet CSV formats.
        Extracts DFRobot lux values when available, falls back to Simulator.
        Returns (grid_2d, wall_size_cm, description, y_min, y_max, x_min, x_max) or raises ValueError."""
        with open(filepath, 'r') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        if not lines:
            raise ValueError("Empty file")

        if lines[0].startswith("Camera FOV Intensity Image"):
            return _parse_fov_intensity_csv(lines)

        # Detect if this is a comma-separated spreadsheet CSV
        is_csv = any(',' in l for l in lines[:20])

        # Helper: extract non-empty cells from a CSV line
        def _csv_cells(line):
            return [c.strip() for c in line.split(',')]

        # Helper: find a cell matching a pattern in a line (returns cell index or -1)
        def _find_cell(cells, pattern):
            pat_lower = pattern.lower()
            for idx, c in enumerate(cells):
                if pat_lower in c.lower():
                    return idx
            return -1

        # Helper: extract numeric values from cells starting at given index
        def _extract_nums(cells, start):
            vals = []
            for c in cells[start:]:
                c = c.strip()
                if not c:
                    continue
                try:
                    vals.append(float(c))
                except ValueError:
                    break
            return vals

        # Find UNIFORMITY section
        unif_start = None
        for i, line in enumerate(lines):
            text = line.strip().replace(',', ' ').strip()
            cells = _csv_cells(line) if is_csv else [line.strip()]
            for c in cells:
                if 'UNIFORMITY' in c.upper():
                    unif_start = i
                    break
            if unif_start is not None:
                break
        if unif_start is None:
            raise ValueError("No UNIFORMITY section found in benchmark file")

        # Parse Y blocks
        y_positions_mm = []
        lux_rows = []
        x_scan_mm = None
        i = unif_start + 1

        while i < len(lines):
            if is_csv:
                cells = _csv_cells(lines[i])
            else:
                cells = [lines[i].strip()]

            # Look for Y label in any cell
            y_cell_idx = -1
            for ci, c in enumerate(cells):
                c_stripped = c.strip()
                if c_stripped.startswith('Y ') or c_stripped.startswith('Y+') or c_stripped.startswith('Y-'):
                    y_cell_idx = ci
                    break
            if y_cell_idx >= 0:
                y_label = cells[y_cell_idx].strip()
                if 'center' in y_label.lower():
                    y_mm = 0
                else:
                    # Extract number: "Y +200", "Y -100", "Y+200"
                    import re as _re
                    nums = _re.findall(r'[+-]?\d+', y_label)
                    if nums:
                        y_mm = int(nums[0])
                    else:
                        i += 1
                        continue
                y_positions_mm.append(y_mm)

                # Scan forward for X scan, Simulator, DFrobot lux lines
                x_line_found = False
                sim_vals = None
                dfr_vals = None
                j = i + 1
                search_limit = min(j + 8, len(lines))
                while j < search_limit:
                    if is_csv:
                        jcells = _csv_cells(lines[j])
                    else:
                        jcells = lines[j].strip().split()

                    joined = ' '.join(c.strip() for c in jcells).lower()

                    if 'x scan' in joined or 'x_scan' in joined:
                        # Extract X positions
                        if is_csv:
                            xci = _find_cell(jcells, 'scan')
                            if xci >= 0:
                                x_vals = _extract_nums(jcells, xci + 1)
                            else:
                                x_vals = _extract_nums(jcells, 0)
                        else:
                            x_vals = []
                            for t in jcells:
                                try:
                                    x_vals.append(int(t))
                                except ValueError:
                                    continue
                        if x_vals and x_scan_mm is None:
                            x_scan_mm = [int(v) for v in x_vals]
                        x_line_found = True

                    elif 'simulator' in joined:
                        if is_csv:
                            sci = _find_cell(jcells, 'simulator')
                            if sci >= 0:
                                sim_vals = _extract_nums(jcells, sci + 1)
                        else:
                            sv = []
                            for t in jcells:
                                try:
                                    sv.append(float(t))
                                except ValueError:
                                    continue
                            sim_vals = sv

                    elif 'dfrobot' in joined or 'df robot' in joined or 'dfr' in joined.replace(' ', ''):
                        if is_csv:
                            dci = _find_cell(jcells, 'lux')
                            if dci < 0:
                                dci = _find_cell(jcells, 'dfrobot')
                                if dci < 0:
                                    dci = _find_cell(jcells, 'DFR')
                            if dci >= 0:
                                dfr_vals = _extract_nums(jcells, dci + 1)
                            else:
                                dfr_vals = _extract_nums(jcells, 0)
                        else:
                            dv = []
                            for t in jcells:
                                try:
                                    dv.append(float(t))
                                except ValueError:
                                    continue
                            dfr_vals = dv

                    # Stop when we have both or hit next Y block or empty block
                    if dfr_vals is not None and sim_vals is not None:
                        break
                    j += 1

                # Prefer DFRobot lux; fall back to Simulator
                chosen = dfr_vals if dfr_vals else sim_vals
                if chosen:
                    lux_rows.append(chosen)
                else:
                    # Remove the Y position since we found no data
                    y_positions_mm.pop()

            i += 1

        if not lux_rows or x_scan_mm is None:
            raise ValueError("Could not parse UNIFORMITY data")

        # Build dense grid via interpolation
        from scipy.interpolate import RegularGridInterpolator

        y_cm = [y / 10.0 for y in y_positions_mm]  # e.g. [20, 10, 0, -10, -20]
        x_cm = [x / 10.0 for x in x_scan_mm]       # e.g. [-40, -30, ..., 40]

        # Sort y ascending for interpolator
        sorted_pairs = sorted(zip(y_cm, lux_rows), key=lambda p: p[0])
        y_sorted = [p[0] for p in sorted_pairs]
        data_sorted = [p[1] for p in sorted_pairs]
        data_2d = np.array(data_sorted, dtype=np.float64)  # shape (ny, nx)

        # Target dense grid covering the data range
        y_min, y_max = y_sorted[0], y_sorted[-1]
        x_min, x_max = x_cm[0], x_cm[-1]
        wall_size_cm = max(abs(x_max - x_min), abs(y_max - y_min))
        dense_n = max(50, int(wall_size_cm))  # ~1cm resolution

        dense_y = np.linspace(y_min, y_max, dense_n)
        dense_x = np.linspace(x_min, x_max, dense_n)

        interp = RegularGridInterpolator(
            (np.array(y_sorted), np.array(x_cm)), data_2d,
            method='linear', bounds_error=False, fill_value=0.0
        )
        yy, xx = np.meshgrid(dense_y, dense_x, indexing='ij')
        grid = interp((yy, xx))
        grid = np.clip(grid, 0, None)

        desc = f"DFRobot lux – UNIFORMITY ({len(y_positions_mm)} Y × {len(x_scan_mm)} X)"
        return grid, wall_size_cm, desc, y_min, y_max, x_min, x_max

    def _parse_fov_intensity_csv(lines):
        """Parse a Camera FOV Intensity Image CSV.
        Returns (grid_2d, wall_size_cm, description, y_min, y_max, x_min, x_max)."""
        # Header parsing
        fov_w_cm = None
        fov_h_cm = None
        grid_start = None
        for i, line in enumerate(lines):
            if line.startswith("FOV Width"):
                fov_w_cm = float(line.split(",")[1])
            elif line.startswith("FOV Height"):
                fov_h_cm = float(line.split(",")[1])
            elif line.startswith("Intensity Grid"):
                grid_start = i + 2  # skip blank line after header
                break

        if fov_w_cm is None or fov_h_cm is None or grid_start is None:
            raise ValueError("Cannot parse FOV intensity CSV header")

        # Parse grid data
        rows = []
        for i in range(grid_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",") if v.strip()]
            if vals:
                rows.append(vals)

        grid = np.array(rows, dtype=np.float64)
        wall_size_cm = max(fov_w_cm, fov_h_cm)
        half_w = fov_w_cm / 2.0
        half_h = fov_h_cm / 2.0
        # FOV grid stores lumens per cell — convert to lux (lm/m²)
        cell_w_cm = fov_w_cm / grid.shape[1]
        cell_h_cm = fov_h_cm / grid.shape[0]
        cell_area_m2 = (cell_w_cm / 100.0) * (cell_h_cm / 100.0)
        if cell_area_m2 > 0:
            grid = grid / cell_area_m2

        desc = f"FOV Intensity ({grid.shape[0]}×{grid.shape[1]})"
        return grid, wall_size_cm, desc, -half_h, half_h, -half_w, half_w

    def _render_imported_csv_on_wall(grid, wall_size_cm, y_min, y_max, x_min, x_max):
        """Render an imported lux grid on the wall, similar to update_intensity_map."""

        # Clear previous
        for h in imported_csv_handles:
            try:
                h.remove()
            except (KeyError, Exception):
                pass
        imported_csv_handles.clear()
        if room_mode_enable.value:
            wall_dist = room_front_dist.value
        else:
            wall_dist = wall_dist_slider.value

        _csv_cap = float(csv_legend_max_input.value)

        max_lux = float(np.max(grid))
        if max_lux <= 0:
            print("[CSV Import] All values are zero — nothing to display.")
            return

        color_scale_max = _csv_cap if max_lux <= _csv_cap else max_lux
        nrows, ncols = grid.shape

        cell_h_cm = (y_max - y_min) / nrows
        cell_w_cm = (x_max - x_min) / ncols

        x_pos = wall_dist / 100.0 - 0.006  # slightly in front (different offset from sim)

        vertices_list = []
        faces_list = []
        colors_list = []
        vert_idx = 0
        gap = 0.025

        for gz in range(nrows):
            for gy in range(ncols):
                intensity = grid[gz, gy]
                if intensity > 0:
                    color = intensity_to_color(intensity, color_scale_max)
                    color_uint8 = [int(c * 255) for c in color] + [255]

                    # Map grid cell to wall coordinates (in meters)
                    y_center = (x_min + gy * cell_w_cm + cell_w_cm / 2) / 100.0
                    z_center = (y_min + gz * cell_h_cm + cell_h_cm / 2) / 100.0
                    half_cell_y = (cell_w_cm / 100.0) * 0.5 * (1.0 - gap)
                    half_cell_z = (cell_h_cm / 100.0) * 0.5 * (1.0 - gap)

                    v0 = [x_pos, y_center - half_cell_y, z_center - half_cell_z]
                    v1 = [x_pos, y_center + half_cell_y, z_center - half_cell_z]
                    v2 = [x_pos, y_center + half_cell_y, z_center + half_cell_z]
                    v3 = [x_pos, y_center - half_cell_y, z_center + half_cell_z]

                    vertices_list.extend([v0, v1, v2, v3])
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

            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
            from trimesh.visual import ColorVisuals
            mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=colors_np)

            handle = server.scene.add_mesh_trimesh(
                name="/imported_csv_pattern",
                mesh=mesh,
                visible=True,
            )
            imported_csv_handles.append(handle)

        print(f"[CSV Import] Rendered {vert_idx // 4} cells, max {max_lux:.1f} lux")
        return max_lux

    def import_csv_pattern():
        """Import and render a CSV pattern file on the wall."""
        filepath = csv_import_path.value.strip()
        if not filepath:
            csv_import_status.content = "<div style='font-size:11px;color:#f44;'>⚠ Enter a file path first</div>"
            return
        if not os.path.isfile(filepath):
            csv_import_status.content = f"<div style='font-size:11px;color:#f44;'>⚠ File not found: {filepath}</div>"
            return
        try:
            result = _parse_benchmark_csv(filepath)
            grid, wall_size_cm, desc = result[0], result[1], result[2]
            y_min, y_max, x_min, x_max = result[3], result[4], result[5], result[6]
            max_lux = _render_imported_csv_on_wall(grid, wall_size_cm, y_min, y_max, x_min, x_max)
            csv_import_status.content = (
                f"<div style='font-size:11px;color:#4CAF50;'>"
                f"✓ {desc}<br>Peak: {max_lux:.1f} lux | Grid: {grid.shape[0]}×{grid.shape[1]}"
                f"</div>"
            )
            # Compute stats from the imported CSV values
            csv_vals = grid[grid > 0] if np.any(grid > 0) else grid.ravel()
            csv_min = float(np.min(csv_vals))
            csv_max = float(np.max(csv_vals))
            csv_avg = float(np.mean(csv_vals))
            csv_diff_html.content = (
                "<div style='font-family:sans-serif;margin-top:6px;padding:6px;background:#1a1a2e;border-radius:4px;'>"
                "<div style='font-weight:600;font-size:12px;color:#e0e0e0;margin-bottom:4px;'>CSV Stats (lux)</div>"
                f"<div style='font-size:11px;color:#90caf9;'>Min: <b>{csv_min:.1f}</b></div>"
                f"<div style='font-size:11px;color:#ef9a9a;'>Max: <b>{csv_max:.1f}</b></div>"
                f"<div style='font-size:11px;color:#fff59d;'>Avg: <b>{csv_avg:.1f}</b></div>"
                "</div>"
            )
            # Build color/value legend (fixed / auto, like main intensity legend)
            _csv_cap = float(csv_legend_max_input.value)
            csv_color_scale = _csv_cap if max_lux <= _csv_cap else max_lux
            if csv_color_scale <= _csv_cap:
                _csv_step = max(1, _csv_cap / 10)
                legend_vals = np.arange(0, _csv_cap + 1, _csv_step)
                mode_label = "FIXED"
            else:
                legend_vals = np.linspace(0, csv_color_scale, 11)
                mode_label = "AUTO"
            scale_label = f"(scale 0–{int(csv_color_scale)} lx, {mode_label})"
            legend_lines = [
                "<div style='font-family:sans-serif;margin-top:8px;'>",
                "<div style='font-weight:600;margin-bottom:2px;font-size:12px;'>CSV Pattern Legend (lux)</div>",
                f"<div style='color:#888;font-size:10px;margin-bottom:4px;'>{scale_label} — peak {max_lux:.0f} lx</div>",
            ]
            for lux_val in reversed(legend_vals):
                color = intensity_to_color(lux_val, csv_color_scale)
                hex_c = "#%02x%02x%02x" % tuple(int(255 * c) for c in color)
                legend_lines.append(
                    f"<div style='display:flex;align-items:center;margin:1px 0;'>"
                    f"<div style='width:18px;height:10px;background:{hex_c};margin-right:6px;"
                    f"border:1px solid #333;'></div>"
                    f"<span style='font-size:11px;'>{lux_val:.0f} lx</span></div>"
                )
            legend_lines.append("</div>")
            csv_legend_html.content = "".join(legend_lines)
        except Exception as e:
            csv_import_status.content = f"<div style='font-size:11px;color:#f44;'>⚠ Error: {e}</div>"
            csv_legend_html.content = ""
            csv_diff_html.content = ""
            print(f"[CSV Import] Error: {e}")

    def clear_csv_pattern():
        """Remove imported CSV pattern from the wall."""
        for h in imported_csv_handles:
            try:
                h.remove()
            except (KeyError, Exception):
                pass
        imported_csv_handles.clear()
        csv_import_status.content = "<div style='font-size:11px;color:#888;'>No file imported</div>"
        csv_legend_html.content = ""
        csv_diff_html.content = ""
        print("[CSV Import] Cleared imported pattern.")

    def export_lux_matrix():
        """Export a text file with lux values sampled every 10cm in ±40cm range on Y and Z axes."""
        cache = _last_intensity_cache
        if cache['grid'] is None:
            print("[Export] No intensity data available. Run 'Update Intensity Map' first.")
            return

        grid = cache['grid']
        wall_size_cm = cache['wall_size_cm']
        wall_dist = cache['wall_dist']
        grid_size = grid.shape[0]
        cell_size = wall_size_cm / grid_size
        half_size = wall_size_cm / 2.0

        # Sample positions: -40 to +40 cm, step 10 cm
        sample_positions = list(range(-40, 41, 10))  # [-40, -30, ..., 0, ..., 30, 40]

        # Build the lux matrix: rows = Z (top to bottom), cols = Y (left to right)
        lux_matrix = []
        for z_cm in reversed(sample_positions):  # top to bottom
            row = []
            for y_cm in sample_positions:
                # Convert cm to grid index
                gy = int((y_cm + half_size) / cell_size)
                gz = int((z_cm + half_size) / cell_size)
                if 0 <= gy < grid_size and 0 <= gz < grid_size:
                    row.append(grid[gz, gy])
                else:
                    row.append(0.0)
            lux_matrix.append(row)

        # Write to file
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"lux_matrix_{ts}.txt"
        filepath = os.path.join(export_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"Lux Matrix Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Wall distance: {wall_dist:.1f} cm\n")
            f.write(f"Grid resolution: {grid_size}x{grid_size}\n")
            f.write(f"Wall size: {wall_size_cm:.1f} cm\n")
            f.write(f"Sample range: +-40 cm, step 10 cm\n")
            f.write(f"Rows: Z axis (top +40 to bottom -40)\n")
            f.write(f"Columns: Y axis (left -40 to right +40)\n")
            f.write("\n")

            # Header row with Y positions
            header = "Z\\Y(cm)" + "\t" + "\t".join(f"{y:+d}" for y in sample_positions)
            f.write(header + "\n")

            # Data rows
            for i, z_cm in enumerate(reversed(sample_positions)):
                row_str = f"{z_cm:+d}" + "\t" + "\t".join(f"{v:.1f}" for v in lux_matrix[i])
                f.write(row_str + "\n")

        print(f"[Export] Lux matrix saved to {filepath}")

    def run_benchmark():
        """Run benchmark: center lux at multiple distances + uniformity grid at 50cm.
        Exports a CSV file matching the NORMAL + UNIFORMITY format."""
        import time as _time
        from datetime import datetime

        print("\n=== STARTING BENCHMARK ===")
        t0 = _time.perf_counter()

        leds, absorbers, stl_mesh_for_raytracing = _build_current_leds_and_absorbers()
        grid_size = int(intensity_grid_size.value)
        rays_per_pixel = int(intensity_rays_slider.value)

        # --- NORMAL: center lux at multiple distances ---
        normal_distances = [10, 20, 30, 50, 80]
        normal_center_lux = {}
        for dist in normal_distances:
            print(f"  [Benchmark] Computing center lux at {dist} cm...")
            # Wall size must cover at least +-40cm
            w_size = max(80, int(wall_view_size.value))
            grid, actual_ws = compute_wall_intensity(
                leds, dist, rays_per_pixel, grid_size, w_size,
                absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
            )
            grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
            center_idx = grid_size // 2
            normal_center_lux[dist] = grid[center_idx, center_idx]
            print(f"    Center lux: {normal_center_lux[dist]:.1f}")

        # --- UNIFORMITY at 50cm: grid scan ---
        unif_dist = 50
        print(f"  [Benchmark] Computing uniformity grid at {unif_dist} cm...")
        w_size = max(80, int(wall_view_size.value))
        unif_grid, actual_ws = compute_wall_intensity(
            leds, unif_dist, rays_per_pixel, grid_size, w_size,
            absorbers=absorbers, stl_mesh_data=stl_mesh_for_raytracing
        )
        unif_grid = np.nan_to_num(unif_grid, nan=0.0, posinf=0.0, neginf=0.0)

        cell_size = actual_ws / grid_size
        half_size = actual_ws / 2.0

        # X scan positions (horizontal on wall = Y axis): -400 to +400 mm step 100 = -40 to +40 cm step 10
        x_scan_mm = list(range(-400, 401, 100))
        x_scan_cm = [x / 10.0 for x in x_scan_mm]
        # Y positions (vertical on wall = Z axis): +200, +100, 0, -100, -200 mm
        y_positions_mm = [400, 300, 200, 100, 0, -100, -200, -300, -400]
        y_positions_cm = [y / 10.0 for y in y_positions_mm]

        uniformity_data = {}
        for y_mm, y_cm in zip(y_positions_mm, y_positions_cm):
            row_lux = []
            for x_mm, x_cm in zip(x_scan_mm, x_scan_cm):
                gy = min(int(round((x_cm + half_size) / cell_size)), grid_size - 1)
                gz = min(int(round((y_cm + half_size) / cell_size)), grid_size - 1)
                if 0 <= gy < grid_size and 0 <= gz < grid_size:
                    row_lux.append(unif_grid[gz, gy])
                else:
                    row_lux.append(0.0)
            uniformity_data[y_mm] = row_lux

        # --- Write CSV ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        filename = f"benchmark_{ts}.txt"
        filepath = os.path.join(export_dir, filename)

        COL1 = 16  # first column width
        COLN = 10  # data column width

        with open(filepath, 'w') as f:
            # NORMAL section
            f.write("NORMAL\n")
            f.write("DISTANCE cm".ljust(COL1) + "".join(str(d).rjust(COLN) for d in normal_distances) + "\n")
            f.write("Simulator".ljust(COL1) + "".join(f"{normal_center_lux[d]:.1f}".rjust(COLN) for d in normal_distances) + "\n")
            f.write("DFRobot lux\n")
            f.write("\n")

            # UNIFORMITY section
            f.write(f"UNIFORMITY\n")
            f.write(f"DISTANCE {unif_dist}CM\n")
            f.write("\n")

            for y_mm in y_positions_mm:
                if y_mm == 0:
                    label = "Y center"
                else:
                    label = f"Y {y_mm:+d}"
                f.write(f"{label}\n")
                f.write("X scan".ljust(COL1) + "".join(str(x).rjust(COLN) for x in x_scan_mm) + "\n")
                f.write("Simulator".ljust(COL1) + "".join(f"{v:.1f}".rjust(COLN) for v in uniformity_data[y_mm]) + "\n")
                f.write("DFRobot lux\n")
                f.write("\n")

        t1 = _time.perf_counter()
        print(f"=== BENCHMARK COMPLETE ({t1 - t0:.1f}s) ===")
        print(f"[Export] Benchmark saved to {filepath}")


    return SimpleNamespace(clear_csv_pattern=clear_csv_pattern, export_lux_matrix=export_lux_matrix, import_csv_pattern=import_csv_pattern, run_benchmark=run_benchmark)
