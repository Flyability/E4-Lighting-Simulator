"""PDF report of an optimisation run (A4, matplotlib ``Figure`` API only — no pyplot,
so it can be rendered from the optimiser's background thread).

Contents: run summary, problem definition, scoring formulas, convergence plots,
wall-illuminance images of the initial guess and the two best *distinct* designs,
LED layouts, and per-variable history / bound-usage charts.
"""

from __future__ import annotations

import datetime
import textwrap
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colormaps as plt_colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

A4 = (8.27, 11.69)
MARGIN = 0.07
DISTINCT_TOL = 0.02
"""Two designs are 'distinct' if some variable differs by > 2 % of its range."""


# --------------------------------------------------------------------------- text flow
class _Writer:
    """Flows text lines over as many A4 pages as needed."""

    def __init__(self, pdf, title):
        self.pdf = pdf
        self.title = title
        self.fig = None
        self.y = 0.0
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
        self.fig = Figure(figsize=A4, dpi=110)
        self.fig.text(MARGIN, 0.955, self.title, fontsize=15, weight="bold")
        self.fig.add_artist(_hline(0.945))
        self.y = 0.925

    def heading(self, text):
        self.gap(0.008)
        self.line(text, size=11.5, weight="bold")
        self.gap(0.003)

    def gap(self, dy=0.012):
        self.y -= dy

    def line(self, text="", size=9, weight="normal", family="sans-serif", color="black", indent=0.0, wrap=None):
        if wrap is None:
            wrap = self.chars_per_line(size, family, indent)
        chunks = textwrap.wrap(str(text), wrap) or [""]
        for chunk in chunks:
            step = 0.0155 * size / 9
            if self.y - step < 0.05:
                self._new_page()
            self.fig.text(MARGIN + indent, self.y, chunk, fontsize=size, weight=weight, family=family,
                          color=color, va="top")
            self.y -= step

    def math(self, latex, size=10.5, indent=0.03):
        tall = any(t in latex for t in (r"\frac", r"\sum", r"\#\{"))
        step = 0.036 if tall else 0.024
        if self.y - step < 0.05:
            self._new_page()
        self.fig.text(MARGIN + indent, self.y - 0.004, f"${latex}$", fontsize=size, va="top")
        self.y -= step

    def table(self, header, rows, widths, size=8):
        fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
        fit = self.chars_per_line(size, "monospace")
        total = sum(widths) + 2 * (len(widths) - 1)
        if total > fit:  # shrink the label column first, then the value columns
            widths = list(widths)
            widths[0] = max(12, widths[0] - (total - fit))
            total = sum(widths) + 2 * (len(widths) - 1)
            while total > fit:
                widths[1:] = [max(6, w - 1) for w in widths[1:]]
                total = sum(widths) + 2 * (len(widths) - 1)
            fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
        self.line(fmt.format(*[_clip(str(c), w) for c, w in zip(header, widths)]), family="monospace",
                  weight="bold", size=size, wrap=10_000)
        for r in rows:
            self.line(fmt.format(*[_clip(str(c), w) for c, w in zip(r, widths)]), family="monospace", size=size,
                      wrap=10_000)

    @staticmethod
    def chars_per_line(size, family, indent=0.0):
        """Characters that fit the printable width (monospace advance ≈ 0.6 em, sans ≈ 0.5 em)."""
        width_pt = (1 - 2 * MARGIN - indent) * A4[0] * 72
        em = 0.6 if family == "monospace" else 0.5
        return max(10, int(width_pt / (em * size)))

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.fig = None


def _hline(y):
    from matplotlib.lines import Line2D
    return Line2D([MARGIN, 1 - MARGIN], [y, y], color="#888", linewidth=0.8)


def _clip(s, w):
    return s if len(s) <= w else s[: max(0, w - 1)] + "…"


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


# --------------------------------------------------------------------------- selection
VERIFY_TOP_K = 5
VERIFY_RAY_FACTOR = 4


def _distinct_top(problem, records, k):
    """Indices of the ``k`` best records that are mutually distinct (see DISTINCT_TOL)."""
    order = sorted(range(len(records)), key=lambda i: records[i][1].score)
    lo = np.array([b[0] for b in problem.bounds], float)
    hi = np.array([b[1] for b in problem.bounds], float)
    span = np.where(hi > lo, hi - lo, 1.0)
    picked = []
    for i in order:
        if all(np.max(np.abs(records[i][2] - records[j][2]) / span) > DISTINCT_TOL for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    return picked


def _verifier(problem, factor):
    """Shallow copy of ``problem`` tracing ``factor``× more rays per pixel (fresh random rays)."""
    import copy
    from lighting_simulator.simulation.settings import WallSettings
    p = copy.copy(problem)
    p.walls = [WallSettings(wall_dist=w.wall_dist, grid_size=w.grid_size, wall_size=w.wall_size,
                            rays_per_pixel=int(w.rays_per_pixel * factor)) for w in problem.walls]
    return p


def _hard_min_u0(problem, ev):
    """Per-wall U0 with the hard minimum (what the UI legend shows), from the kept grids."""
    out = []
    for grid, mask in zip(_as_grid_list(ev), problem._fov_masks):
        lit = grid[mask]
        lit = lit[lit > 0]
        out.append(float(lit.min() / lit.mean() * 100) if lit.size else 0.0)
    return out


def _as_grid_list(ev):
    if ev.grid is None:
        return []
    return list(ev.grid) if isinstance(ev.grid, list) else [ev.grid]


# --------------------------------------------------------------------------- pages
def _page_summary(w: _Writer, problem, opt, designs, n_evals, elapsed, stopped, out_dir, logged_scores,
                  n_confirm=0):
    w.heading("Run")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    w.line(f"Generated {now}   —   output folder: {out_dir}")
    w.line(f"Method: {opt.method}   budget: {opt.max_evals} evaluations   performed: {n_evals}   "
           f"elapsed: {elapsed:.0f} s   backend: {'analytic (no ray tracing)' if problem.analytic else ('GPU' if problem.use_gpu else 'CPU')}"
           + (f"   new-best confirmations: {n_confirm}" if n_confirm else "")
           + ("   (stopped by user)" if stopped else ""))
    w.line(f"Decision variables: {problem.dim}   walls: "
           + ", ".join(f"{wl.wall_dist:g} cm ({wl.wall_size:g} cm wide)" for wl in problem.walls)
           + f"   eval grid: {problem.wall.grid_size}² cells, {problem.wall.rays_per_pixel} rays/pixel")
    w.line(f"Ray sampling is random: the {VERIFY_TOP_K} best distinct designs of the run were re-evaluated "
           f"with {VERIFY_RAY_FACTOR}× the rays and ranked by that verified score; both scores are shown.",
           size=8, color="#444")

    w.heading("Key results")
    labels = list(designs.keys())
    header = ["", *labels]
    widths = [40] + [16] * len(labels)
    evs = [designs[k][1] for k in labels]
    pct = problem.objective.min_percentile
    u0_label = f"U0 = P{pct:g}(E)/E_avg" if pct > 0 else "U0 = E_min/E_avg"
    t1_img = "flash image" if problem.flash_modes else "flight image"
    rows = [
        [f"Score, verified {VERIFY_RAY_FACTOR}× rays", *[f"{e.score:.4f}" for e in evs]],
        ["Score, logged during run", *[(f"{logged_scores[k]:.4f}" if logged_scores.get(k) is not None else "—")
                                       for k in labels]],
        [f"— T1: wall, main camera, {t1_img} —", *[""] * len(labels)],
        [f"{u0_label}, mean of walls", *[f"{e.uniformity_pct:.1f} %" for e in evs]],
        ["U0 hard-min per wall (UI legend)", *[" / ".join(f"{u:.0f}" for u in _hard_min_u0(problem, e)) + " %"
                                              for e in evs]],
        ["FOV coverage (worst wall)", *[f"{e.coverage*100:.0f} %" for e in evs]],
        ["E_avg in FOV (mean over walls)", *[f"{e.e_avg:,.0f} lx" for e in evs]],
    ]
    if evs[0].metrics is not None:
        rows += [
            ["E_min / E_max (last wall)", *[
                f"{e.metrics.e_min:,.0f} / {e.metrics.e_max:,.0f} lx" if e.metrics else "—" for e in evs]],
            ["ΔEV (stops)", *[f"{e.metrics.delta_ev:.2f}" if e.metrics else "—" for e in evs]],
        ]
    if getattr(problem, "_needs_vio", False):
        v = problem.vio
        rows.append([f"— T2: room {v.room_dist:g} cm, VIO cameras —", *[""] * len(labels)])
        for mode in problem.modes:
            if mode.vio_min_lux:
                rows.append([f"cells ≥ {mode.vio_min_lux:g} lx (target {mode.vio_min_fraction*100:.0f} %)", *[
                    (lambda f: f"{f*100:.0f} %" if f is not None else "—")(e.modes.get(mode.name, {}).get('vio_fraction'))
                    for e in evs]])
    if problem.objective.tilt_fov_deg:
        t = problem.objective.tilt_fov_deg
        rows.append([f"— T3: rooms, camera ±{t:g}°, w={problem.objective.tilt_fov_weight:g} —",
                     *[""] * len(labels)])
        rows.append([f"U0 pitched +{t:g}° (up), mean of rooms", *[f"{e.tilt.get('up', 0):.1f} %" for e in evs]])
        rows.append([f"U0 pitched −{t:g}° (down), mean of rooms", *[f"{e.tilt.get('down', 0):.1f} %" for e in evs]])
        for i, wl in enumerate(problem.walls):
            def _cell(e, keys, i=i):
                if i >= len(e.tilt_walls):
                    return "—"
                tw = e.tilt_walls[i]
                return " / ".join(f"{tw[k] * (100 if k.startswith('cov') else 1):.0f}" for k in keys) + " %"
            rows.append([f"  room {wl.wall_dist:g} cm: U0 up / down", *[_cell(e, ('up', 'down')) for e in evs]])
            rows.append([f"  room {wl.wall_dist:g} cm: coverage up / down", *[_cell(e, ('cov_up', 'cov_down')) for e in evs]])
    rows.append(["— Hardware —", *[""] * len(labels)])
    rows.append(["Active LEDs", *[str(e.n_active) for e in evs]])
    if problem.electrical:
        rows += [
            ["Drivers", *[str(e.n_drivers) for e in evs]],
            ["Continuous current", *[f"{e.total_current_a:.1f} A" for e in evs]],
        ]
    if any(e.electrical for e in evs):
        rows.append(["LED roles vio / flash / both", *[
            f"{e.electrical.get('n_vio', 0)} / {e.electrical.get('n_flash', 0)} / {e.electrical.get('n_both', 0)}"
            for e in evs]])
    if any('n_pulse_drivers' in e.electrical for e in evs):
        rows += [
            ["Pulse / continuous drivers", *[
                f"{e.electrical.get('n_pulse_drivers', 0)} / {e.electrical.get('n_cont_drivers', 0)}" for e in evs]],
            ["Peak current (flash)", *[f"{e.electrical.get('peak_current_a', 0):.1f} A" for e in evs]],
        ]
    for mode in problem.modes:
        lm = problem.flash_lumens if mode.is_flash else problem.flight_lumens
        rows.append([f"{mode.name}: flux per LED", *[f"{lm:,.0f} lm" if lm else "scene" for _ in evs]])
        rows.append([f"{mode.name}: E_avg in FOV", *[
            f"{e.modes.get(mode.name, {}).get('e_avg', 0):,.0f} lx" for e in evs]])
        rows.append([f"{mode.name}: LEDs lit", *[str(e.modes.get(mode.name, {}).get('n_leds', '—')) for e in evs]])
    pen_keys = sorted({k for e in evs for k in e.penalties})
    for k in pen_keys:
        rows.append([f"penalty: {k}", *[f"{e.penalties.get(k, 0.0):.4f}" for e in evs]])
    w.table(header, rows, widths)

    dead = [k for k in pen_keys
            if len(evs) > 1 and max(e.penalties.get(k, 0.0) for e in evs) > 1e-6
            and max(e.penalties.get(k, 0.0) for e in evs) - min(e.penalties.get(k, 0.0) for e in evs) < 1e-6]
    for k in dead:
        w.gap(0.004)
        w.line(f"⚠ penalty '{k}' is identical for every design ({evs[0].penalties[k]:.3f}): this target does not "
               "steer the search (unreachable or saturated). Relax it, add a variable that can act on it "
               "(e.g. the flux or the LED count), or drop it.", size=8.5, color="#b26a00")

    if len(evs) > 1 and evs[0].score > 0:
        gain = (evs[0].score - evs[1].score) / evs[0].score * 100
        w.gap()
        w.line(f"Best design improves the verified score by {gain:.1f} % over the initial configuration "
               f"({evs[0].score:.4f} → {evs[1].score:.4f}).", weight="bold")
    w.gap()
    w.line("The 'initial' column is the optimiser's starting point x₀: the loaded scene when an existing "
           "design is refined, or the centre of the search box for a generated layout. The two best designs "
           f"differ in at least one variable by more than {DISTINCT_TOL*100:.0f} % of its range. "
           "'U0 hard-min' uses the single darkest FOV cell (as the Intensity Map legend does); it is "
           "systematically lower and noisier than the percentile-based objective, especially on fine grids.",
           size=8, color="#444")


def _page_problem(w: _Writer, problem, designs):
    w.heading("Scene / evaluation settings")
    c = problem.camera
    w.line(f"Main camera: position ({c.pos_x:g}, {c.pos_y:g}) cm, pitch {c.pitch:g}°, FOV {c.fov_h:g}° × {c.fov_v:g}°")
    e = problem.emission
    w.line(f"Emission: default {e.default_lumens:g} lm/LED"
           + (f" (every LED forced to {problem.flight_lumens:,.0f} lm in flight)" if problem.flight_lumens else "")
           + f", focus factor {e.ray_uniformity:g}"
           + (";  diffuser " + f"{problem.diffuser[0]:g}° × {problem.diffuser[1]*100:.0f} %" if problem.diffuser else "")
           + (";  STL occluder active" if problem.stl_mesh is not None else ""))
    if problem.electrical:
        d = problem.driver
        w.line(f"Electrical model on — pulse driver: {d.voltage_v:g} V, {d.efficacy_lm_per_w:g} lm/W → "
               f"{d.lumens(1.0):,.0f} lm/A, max {d.max_current_a:g} A per LED, {d.leds_per_driver} LED(s) per driver")
        if problem.cont_driver is not problem.driver:
            cd = problem.cont_driver
            w.line(f"Continuous driver model ('vio' LEDs): max {cd.max_current_a:g} A per LED, "
                   f"{cd.leds_per_driver} LED(s) per driver")
    else:
        w.line("Electrical model off: fluxes are set per mode in lumens; no driver / current penalties.")
    if problem.vio is not None:
        v = problem.vio
        w.line(f"VIO cameras: at {_fmt(list(v.position))} cm, cam1 pitch {v.cam1_pitch:g}° yaw {v.cam1_yaw:g}°, "
               f"cam2 pitch {v.cam2_pitch:g}° yaw {v.cam2_yaw:g}°, long-side FOV {v.long_fov:g}° "
               f"({'landscape' if v.landscape else 'portrait'}); T2 room: five walls {v.room_dist:g} cm away "
               f"({2*v.room_dist/100:g} m across), {v.room_grid_size}² cells per wall")
    if problem.modes:
        w.heading("Operating modes")
        for m in problem.modes:
            if m.is_flash:
                src = f"{problem.flash_lumens:,.0f} lm" + (f" ({m.current_a:g} A)" if m.current_a is not None else "")
                parts = [f"pulse: flash + both LEDs at {src} per LED, vio LEDs continuous — this is the image judged "
                         "in T1 and T3"]
            else:
                src = f"{problem.flight_lumens:,.0f} lm per LED" if problem.flight_lumens else "scene flux"
                parts = [f"continuous: vio + both LEDs at {src}"
                         + (f" × {m.lumens_scale:g}" if m.lumens_scale != 1.0 else "")
                         + (" — judged in T2 only" if problem.flash_modes else " — the image judged in T1 and T3")]
            if m.min_avg_lux:
                parts.append(f"E_avg ≥ {m.min_avg_lux:,.0f} lx"
                             + (f" at {m.min_avg_lux_dist:g} cm" if m.min_avg_lux_dist else "") + f" (w={m.lux_weight:g})")
            if m.vio_min_lux:
                parts.append(f"T2: ≥ {m.vio_min_fraction*100:.0f} % of VIO FOV above {m.vio_min_lux:g} lx (w={m.vio_weight:g})")
            w.line(f"• {m.name}: " + "; ".join(parts))

    w.heading("Objective")
    o = problem.objective
    w.line(f"metric = {o.metric}, E_min percentile = {o.min_percentile:g} %, coverage weight = {o.coverage_weight:g}"
           + (f", min average lux = {o.min_avg_lux:,.0f} (w={o.lux_weight:g})" if o.min_avg_lux else "")
           + (f"; T3: ±{o.tilt_fov_deg:g}° tilt FOVs on 5-wall rooms at the wall distances, "
              f"{o.tilt_room_grid_size}² cells/wall, w={o.tilt_fov_weight:g}" if o.tilt_fov_deg else "; T3 off"))

    w.heading("Constraints (active)")
    cs = asdict(problem.constraints)
    active = {k: v for k, v in cs.items() if v not in (None, 0, 0.0, [], ()) and not k.endswith("_weight")
              and k not in ("beam_angle_axis", "symmetry_tol_cm")}
    if not active:
        w.line("none")
    for k, v in active.items():
        wkey = {"max_leds": "max_leds_weight", "max_drivers": "max_drivers_weight", "led_cost": None,
                "driver_cost": None, "max_total_current_a": "current_weight", "min_led_spacing_cm": "spacing_weight",
                "keep_out": "keep_out_weight", "min_beam_angle_deg": "beam_angle_weight",
                "symmetry_weight": None}.get(k)
        extra = f"  (weight {cs[wkey]:g})" if wkey else ""
        if k == "min_beam_angle_deg":
            extra += f"  axis {_fmt(list(cs['beam_angle_axis']))}"
        if k == "symmetry_weight":
            extra += f"  tolerance {cs['symmetry_tol_cm']:g} cm"
        w.line(f"• {k} = {_fmt(v if k != 'keep_out' else len(v))}{extra}")

    w.heading("Decision variables")
    labels = list(designs.keys())
    header = ["variable", "min", "max", "int", *labels]
    widths = [28, 8, 8, 3] + [11] * len(labels)
    rows = []
    for i, (name, (lo, hi), isint) in enumerate(zip(problem.names, problem.bounds, problem.integrality)):
        rows.append([name, _fmt(lo), _fmt(hi), "●" if isint else "",
                     *[_fmt(float(designs[k][0][i]), 4) for k in labels]])
    w.table(header, rows, widths)


def _page_method(w: _Writer, problem):
    o, c = problem.objective, problem.constraints
    has_flash = bool(problem.flash_modes)
    img = "flash image" if has_flash else "flight image"
    dists = ", ".join(f"{wl.wall_dist:g}" for wl in problem.walls)
    w.heading("Test cases")
    w.line("Each candidate decision vector x is decoded into a regular saved configuration (same JSON schema "
           "as the layouts) and the scene is built with the exact same code as the UI. The design is then "
           "judged on three independent test cases — each with its own geometry, operating mode and camera — "
           "plus hardware penalties. Two illuminance images exist when a flash mode is defined: the flight "
           "image (VIO + Both LEDs at their continuous flux) and the flash image (Flash + Both LEDs at the "
           "pulse flux, VIO LEDs continuous). The main camera only records during the flash, so T1 and T3 "
           "judge the flash image; the VIO cameras never see the flash, so T2 judges the flight image."
           if has_flash else
           "Each candidate decision vector x is decoded into a regular saved configuration (same JSON schema "
           "as the layouts) and the scene is built with the exact same code as the UI. The design is then "
           "judged on three independent test cases — each with its own geometry and camera — plus hardware "
           "penalties. No flash mode is defined, so every test case uses the single continuous image.")
    w.gap(0.006)
    w.table(
        ["", "geometry", "image", "camera", "metric", "weight"],
        [
            ["T1 inspection", f"flat wall @ {dists} cm", img, "main, untilted",
             f"{o.metric.upper()} + coverage" + (" + lux" if o.min_avg_lux or any(m.min_avg_lux for m in problem.modes) else ""),
             "1 (base)"],
            ["T2 VIO coverage",
             (f"5-wall room @ {problem.vio.room_dist:g} cm" if getattr(problem, "_needs_vio", False) else "—"),
             "flight image", "2 fisheye VIO",
             (f"cells ≥ {[m.vio_min_lux for m in problem.modes if m.vio_min_lux][0]:g} lx"
              if getattr(problem, "_needs_vio", False) else "off"),
             (f"{[m.vio_weight for m in problem.modes if m.vio_min_lux][0]:g}"
              if getattr(problem, "_needs_vio", False) else "—")],
            ["T3 tilted view", f"5-wall rooms @ {dists} cm" if o.tilt_fov_deg else "—", img,
             f"main, ±{o.tilt_fov_deg:g}°" if o.tilt_fov_deg else "—",
             f"{o.metric.upper()} + coverage" if o.tilt_fov_deg else "off",
             f"{o.tilt_fov_weight:g}" if o.tilt_fov_deg else "—"],
        ],
        [16, 30, 13, 15, 21, 8], size=7)
    w.gap(0.006)
    w.line("T1 — inspection image (primary).", weight="bold")
    w.line(f"Monte-Carlo ray tracing on a {problem.wall.grid_size}² grid at each wall distance "
           f"({problem.wall.rays_per_pixel} rays per cell shared by the active LEDs). Only the cells inside the "
           "main-camera FOV trapezoid (white dashed outline on the images) are kept; cells with E = 0 are "
           "ignored for the uniformity metric but counted for coverage. The score is the mean over distances.",
           size=8.5, indent=0.03)
    if getattr(problem, "_needs_vio", False):
        v = problem.vio
        w.line("T2 — VIO coverage.", weight="bold")
        w.line(f"Monte-Carlo trace of the flight image on the five walls (front, left, right, top, bottom — no "
               f"back wall) of a {2*v.room_dist/100:g} m room, {v.room_grid_size}² cells per wall, roughly the ray "
               "budget of one T1 wall. The union of the two 170° fisheye footprints (cyan outline) selects the "
               "cells; the metric is the share of those cells above the lux threshold, penalised through the "
               "matching percentile so it keeps a gradient far from the target. The main camera plays no role.",
               size=8.5, indent=0.03)
    if o.tilt_fov_deg:
        w.line("T3 — tilted inspection image (secondary).", weight="bold")
        w.line(f"A 5-wall room is placed at each T1 distance ({dists} cm) so the pitched camera sees the front "
               f"wall plus the ceiling or floor at the same working distance. The main camera is pitched "
               f"+{o.tilt_fov_deg:g}° and −{o.tilt_fov_deg:g}° (orange / pink dotted outlines); the {img} is "
               f"computed analytically — the exact expected value of the Monte-Carlo tracer, cos^n emission and "
               f"1/d² with a shadow test against absorbers and the CAD mesh — on the {o.tilt_room_grid_size}² "
               "cells inside those footprints only, so the term carries no ray noise and costs a fraction of a "
               "trace. Each footprint gets the same metric + coverage term as T1; the penalty is the mean over "
               f"distances and the two pitches, weighted {o.tilt_fov_weight:g}. Its U0 is inherently lower than "
               "T1's (the footprint spans surfaces at very different distances), which is why the weight is small.",
               size=8.5, indent=0.03)
    w.gap(0.006)
    w.line("Uniformity metrics over the lit cells of a footprint:", weight="bold")
    pct = f"P_{{{o.min_percentile:g}}}(E)" if o.min_percentile > 0 else r"\min(E)"
    w.math(rf"E_{{min}} = {pct},\quad E_{{avg}} = \overline{{E}},\quad E_{{max}} = \max(E)")
    w.math(r"U_0 = \frac{E_{min}}{E_{avg}},\qquad U_1 = \frac{E_{min}}{E_{max}},\qquad CV = \frac{\sigma_E}{E_{avg}}")
    if o.min_percentile > 0:
        w.line(f"E_min uses the {o.min_percentile:g}-th percentile of lit cells so a single Monte-Carlo-starved "
               "cell cannot dominate the metric.", size=8, color="#444", indent=0.03)
    w.gap(0.006)
    w.line("Score (minimised):", weight="bold")
    f = {"u0": r"1 - U_0", "u1": r"1 - U_1", "cv": r"CV"}[o.metric]
    w.math(rf"S(x) = T_1 + T_2 + T_3 + \sum_k P_k(x)")
    w.math(rf"T_1 = \frac{{1}}{{N_d}}\sum_{{d}} \left[{f}\right]_d + "
           rf"{o.coverage_weight:g}\,\left(1 - \min_d \mathrm{{cov}}_d\right)")
    if getattr(problem, "_needs_vio", False):
        w.math(r"T_2 = P_{vio}\quad\text{(see below)}")
    else:
        w.math(r"T_2 = 0\quad\text{(no VIO target)}")
    if o.tilt_fov_deg:
        w.math(rf"T_3 = {o.tilt_fov_weight:g}\cdot\frac{{1}}{{2N_d}}\sum_{{d}}\sum_{{\pm}}"
               rf"\left(\left[{f}\right]_{{d,\pm}} + {o.coverage_weight:g}\,(1 - \mathrm{{cov}}_{{d,\pm}})\right)")
    else:
        w.math(r"T_3 = 0\quad\text{(tilt FOVs off)}")
    w.line("cov is the fraction of footprint cells receiving any light. Every constraint is a soft penalty "
           "P_k ≥ 0 added to the score (no hard infeasibility), which keeps the landscape usable for "
           "population-based global search.", size=8.5, color="#444", indent=0.03)
    w.gap(0.006)
    w.line("Active penalty terms:", weight="bold")
    if o.min_avg_lux:
        w.math(rf"P_{{lux}} = {o.lux_weight:g}\,\max\!\left(0,\ \frac{{{o.min_avg_lux:g} - \min_d E_{{avg,d}}}}{{{o.min_avg_lux:g}}}\right)")
    if c.led_cost:
        w.math(rf"P_{{led}} = {c.led_cost:g}\cdot N_{{LED}}")
    if c.max_leds is not None:
        w.math(rf"P_{{maxled}} = {c.max_leds_weight:g}\cdot\max(0,\ N_{{LED}} - {c.max_leds})")
    if c.driver_cost and problem.electrical:
        w.math(rf"P_{{drv}} = {c.driver_cost:g}\cdot N_{{drv}},\qquad N_{{drv}} = \left\lceil N_{{LED}} / {problem.driver.leds_per_driver} \right\rceil")
    if c.max_drivers is not None and problem.electrical:
        w.math(rf"P_{{maxdrv}} = {c.max_drivers_weight:g}\cdot\max(0,\ N_{{drv}} - {c.max_drivers})")
    if c.max_total_current_a and problem.electrical:
        w.math(rf"P_{{I}} = {c.current_weight:g}\,\max\!\left(0,\ \frac{{\sum_i I_i - {c.max_total_current_a:g}}}{{{c.max_total_current_a:g}}}\right),"
               rf"\qquad I_i = \frac{{\Phi_i}}{{V\cdot\eta}}")
    if c.min_led_spacing_cm:
        w.math(rf"P_{{sp}} = {c.spacing_weight:g}\sum_{{i<j}} \max\!\left(0,\ 1 - \frac{{\|p_i - p_j\|}}{{{c.min_led_spacing_cm:g}}}\right)")
    if c.keep_out:
        w.math(rf"P_{{ko}} = {c.keep_out_weight:g}\sum_{{i\in box}} \min_a \frac{{h_a - |p_{{i,a}} - c_a|}}{{h_a}}")
    if c.min_beam_angle_deg:
        w.math(rf"P_{{ang}} = {c.beam_angle_weight:g}\cdot\frac{{1}}{{N}}\sum_i \max\!\left(0,\ 1 - \frac{{\angle(\hat d_i, \hat a)}}{{{c.min_beam_angle_deg:g}^\circ}}\right),"
               rf"\quad \hat a = {_fmt(list(c.beam_angle_axis))}")
    if c.symmetry_weight:
        w.math(rf"P_{{sym}} = {c.symmetry_weight:g}\cdot\frac{{\#\{{i : \min_j \|p_i - M p_j\| > {c.symmetry_tol_cm:g}\}}}}{{N}},"
               r"\quad M = \mathrm{diag}(1,-1,1)")
    for m in problem.modes:
        if m.min_avg_lux:
            w.math(rf"P_{{{m.name},lux}} = {m.lux_weight:g}\,\max\!\left(0,\ \frac{{{m.min_avg_lux:g} - E^{{({m.name})}}_{{avg}}}}{{{m.min_avg_lux:g}}}\right)")
        if m.vio_min_lux:
            w.math(rf"P_{{vio}} = {m.vio_weight:g}\,\max\!\left(0,\ \frac{{{m.vio_min_lux:g} - E^{{({m.name})}}_{{P{100*(1-m.vio_min_fraction):g}}}}}{{{m.vio_min_lux:g}}}\right)")
            w.line(f"E_P{100*(1-m.vio_min_fraction):g} is the {100*(1-m.vio_min_fraction):g}-th percentile of lux over the "
                   f"VIO-visible cells: requiring it to reach {m.vio_min_lux:g} lx is the same as requiring "
                   f"{m.vio_min_fraction*100:.0f} % of the VIO FOV above {m.vio_min_lux:g} lx, but the penalty keeps a "
                   "gradient when the target is still far away.", size=8, color="#444", indent=0.03)
    if has_flash:
        w.line("The flash image is traced, not rescaled: 'vio', 'both' and 'flash' LEDs are traced as separate "
               "groups (sharing the ray budget in proportion to their counts) and superposed with the pulse flux "
               f"Φ = {problem.flash_lumens:,.0f} lm"
               + (" (= I_flash·V·η)" if problem.electrical and problem.flash_modes[0].lumens is None else "")
               + " applied to the flash / both group.", size=8.5, color="#444", indent=0.03)
    w.gap(0.004)
    w.line("Optimiser:", weight="bold")
    w.line("Differential evolution (scipy) evolves a population inside the bounds and keeps improvements; "
           "integer variables (row/column counts, roles) are rounded before decoding. Nelder–Mead / random "
           "search are local alternatives. The best design is exported after every improvement.", size=8)
    if problem.analytic:
        w.line("Illuminance was computed analytically (closed-form direct light, exact expected value of the tracer): "
               "no Monte-Carlo noise, so re-evaluations return identical scores; wall reflections are not modelled.", size=8)
    else:
        w.line("Rays are random, so T1 and T2 carry Monte-Carlo noise (T3 is analytic). Safeguards: a candidate "
           "beating the current best is re-evaluated with fresh rays and the mean is kept (a lucky draw must be "
           f"lucky twice); for this report the best distinct designs are re-evaluated with {VERIFY_RAY_FACTOR}× "
           "the rays and ranked by that verified score. Dashed trend lines on the convergence page are centred "
           "moving averages over ~5 % of the run.", size=8)
    w.line("Sensitivity pages (end of report): Spearman rank correlations between every variable — plus derived "
           "layout features such as LED count, nearest-neighbour spacing and spread — and every score component, "
           "computed from the logged evaluations; a rank-linear surrogate ranks the variables and flags those the "
           "score does not react to; binned-median trend plots show non-monotonic effects.", size=8)


def _trend(y, frac=0.05, min_win=5):
    """Centred moving average over ~``frac`` of the run (same as the UI plot)."""
    y = np.asarray(y, float)
    win = max(min_win, int(len(y) * frac))
    if len(y) < 2:
        return y
    win = min(win, len(y))
    pad = np.pad(y, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(pad, np.ones(win) / win, mode="valid")


def _page_convergence(pdf, problem, records, designs):
    if len(records) < 2:
        return
    ev_no = np.array([r[0] for r in records], float)
    score = np.array([r[1].score for r in records])
    best = np.minimum.accumulate(score)
    uni = np.array([r[1].uniformity_pct for r in records])
    cov = np.array([r[1].coverage for r in records]) * 100
    eavg = np.array([r[1].e_avg for r in records])
    nled = np.array([r[1].n_active for r in records])
    ndrv = np.array([r[1].n_drivers for r in records])
    trend_kw = dict(color="#ff9800", lw=2, ls="--", label="trend (moving avg)")

    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle("Convergence", fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
    axs = fig.subplots(3, 2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.06, hspace=0.45, wspace=0.3)

    ax = axs[0, 0]
    ax.plot(ev_no, score, ".", ms=2, color="#999", label="evaluated")
    ax.plot(ev_no, _trend(score), **trend_kw)
    ax.plot(ev_no, best, color="#2e7d32", lw=2, label="best so far")
    ax.set_yscale("log")
    ax.set_title("Score"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)

    ax = axs[0, 1]
    ax.plot(ev_no, uni, ".", ms=2, color="#1565c0"); ax.plot(ev_no, _trend(uni), **trend_kw)
    ax.set_title("Uniformity U0 (%)"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)
    ax = axs[1, 0]
    ax.plot(ev_no, cov, ".", ms=2, color="#6a1b9a"); ax.plot(ev_no, _trend(cov), **trend_kw)
    ax.set_title("FOV coverage (%)"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)
    ax = axs[1, 1]
    ax.plot(ev_no, eavg, ".", ms=2, color="#ef6c00"); ax.plot(ev_no, _trend(eavg), color="#1565c0", lw=2, ls="--",
                                                            label="trend (moving avg)")
    ax.set_title("E_avg in FOV (lx)"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)
    ax = axs[2, 0]
    ax.plot(ev_no, nled, ".", ms=2, color="#333", label="LEDs")
    ax.plot(ev_no, _trend(nled), **trend_kw)
    if ndrv.any():
        ax.plot(ev_no, ndrv, ".", ms=2, color="#c62828", label="drivers")
    ax.set_title("Active LEDs / drivers" if ndrv.any() else "Active LEDs"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)

    ax = axs[2, 1]
    labels = list(designs.keys())
    keys = sorted({k for l in labels for k in designs[l][1].penalties} | {"metric"})
    bottoms = np.zeros(len(labels))
    cmap = ["#90a4ae", "#ef9a9a", "#ffcc80", "#a5d6a7", "#ce93d8", "#80deea", "#fff59d", "#bcaaa4", "#f48fb1", "#b0bec5"]
    for j, k in enumerate(keys):
        vals = []
        for l in labels:
            e = designs[l][1]
            vals.append(e.score - sum(e.penalties.values()) if k == "metric" else e.penalties.get(k, 0.0))
        vals = np.array(vals)
        ax.bar(labels, vals, bottom=bottoms, color=cmap[j % len(cmap)], label=k, edgecolor="white", lw=0.5)
        bottoms += vals
    ax.set_title("Score decomposition"); ax.legend(fontsize=6, ncol=2)
    ax.tick_params(axis="x", labelsize=7)
    for a in axs.flat:
        a.tick_params(labelsize=7); a.grid(alpha=0.25)
    pdf.savefig(fig)


TILT_UP_COLOR, TILT_DOWN_COLOR, VIO_COLOR = "#ffb74d", "#f06292", "cyan"


def _fov_polygon(problem, wall_dist, pitch_offset=0.0, color="white", ls="--"):
    z_bot, z_top, w_bot, w_top, y_c = problem.camera.trapezoid(wall_dist, pitch_offset)
    return Polygon([(y_c - w_bot, z_bot), (y_c + w_bot, z_bot), (y_c + w_top, z_top), (y_c - w_top, z_top)],
                   closed=True, fill=False, edgecolor=color, lw=1.2, ls=ls)


def _add_fov_outlines(ax, problem, wall_dist):
    """Main-camera footprint (T1 is judged inside it)."""
    ax.add_patch(_fov_polygon(problem, wall_dist))


def _fov_footer(problem, extra=""):
    return "Dashed white: main-camera FOV footprint (T1 metrics are computed inside it). " + extra


def _page_heatmaps(pdf, problem, designs):
    labels = list(designs.keys())
    n_rows = len(labels)
    cols = [(f"wall {wl.wall_dist:g} cm", wl.wall_size, i) for i, wl in enumerate(problem.walls)]
    n_cols = len(cols)
    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle("T1 — wall illuminance, main camera" + (" (flash image)" if problem.flash_modes else ""),
                 fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
    axs = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.35, wspace=0.35)

    for j, (title, wall_size, key) in enumerate(cols):
        grids = [(_as_grid_list(designs[l][1])[key] if _as_grid_list(designs[l][1]) else None) for l in labels]
        vmax = max([float(np.nanmax(g)) for g in grids if g is not None] + [1e-9])
        half = wall_size / 2
        im = None
        for i, l in enumerate(labels):
            ax = axs[i, j]
            g = grids[i]
            if g is None:
                ax.axis("off"); continue
            im = ax.imshow(g, origin="lower", extent=[-half, half, -half, half], cmap="inferno", vmin=0, vmax=vmax,
                           aspect="equal")
            _add_fov_outlines(ax, problem, problem.walls[key].wall_dist)
            fov = g[problem._fov_masks[key]]
            lit = fov[fov > 0]
            if lit.size:
                pct = problem.objective.min_percentile
                e_min = np.percentile(lit, pct) if pct > 0 else lit.min()
                sub = (f"U0 {e_min/lit.mean()*100:.0f} % (P{pct:g})  ·  hard-min {lit.min()/lit.mean()*100:.0f} %"
                       f"  ·  {lit.mean():,.0f} lx")
            else:
                sub = "unlit"
            ax.invert_xaxis()  # +Y is to the viewer's left when facing the wall
            ax.set_title((title + "\n" if i == 0 else "") + sub, fontsize=7)
            ax.tick_params(labelsize=6)
            if i == n_rows - 1:
                ax.set_xlabel("Y (cm)", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{l}\nZ (cm)", fontsize=8, weight="bold")
        if im is not None:
            cb = fig.colorbar(im, ax=axs[:, j].tolist(), orientation="horizontal", fraction=0.025, pad=0.06)
            cb.ax.tick_params(labelsize=6); cb.set_label("lux", fontsize=7)
    fig.text(MARGIN, 0.02, textwrap.fill(_fov_footer(
        problem, "Colour scale is shared per column. "
        f"Images use {VERIFY_RAY_FACTOR}× the run's rays per pixel."), 150), fontsize=7.5, color="#444", va="bottom")
    pdf.savefig(fig)


def _unfolded_room(settings, specs):
    """Place the five room walls around the front wall (cube net, seen from inside).

    Returns ``{wall: (transform, extent, H, V)}``: ``transform`` reorients a wall grid for
    ``imshow`` with ``extent`` (net coordinates, cm), ``H``/``V`` are the cell-centre net
    coordinates in the grid's own indexing.
    """
    from lighting_simulator.simulation.room_geometry import room_wall_cell_centers
    s = settings
    hy, hz = specs['front']['size_y'] / 2, specs['front']['size_z'] / 2
    fd = s.front_dist
    out = {}
    for name, spec in specs.items():
        if name == 'back':
            continue
        pts = room_wall_cell_centers(name, spec, s.front_dist, s.side_dist, s.top_bottom_dist)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        depth = spec.get('size_x', 0.0)
        if name == 'front':      # grid [Z, Y]
            tf, H, V, ext = (lambda g: g), y, z, [-hy, hy, -hz, hz]
        elif name == 'top':      # grid [Y, X] -> rows X, x = front_dist adjacent to the front wall
            tf, H, V, ext = (lambda g: g.T[::-1]), y, hz + (fd - x), [-hy, hy, hz, hz + depth]
        elif name == 'bottom':
            tf, H, V, ext = (lambda g: g.T), y, -hz - (fd - x), [-hy, hy, -hz - depth, -hz]
        elif name == 'left':     # y = -side_dist; grid [Z, X]
            tf, H, V, ext = (lambda g: g), -(hy + (fd - x)), z, [-hy - depth, -hy, -hz, hz]
        else:                    # right, y = +side_dist
            tf, H, V, ext = (lambda g: g[:, ::-1]), hy + (fd - x), z, [hy, hy + depth, -hz, hz]
        out[name] = (tf, ext, H, V)
    return out


def _draw_room_net(fig, axs, labels, grids, net, outlines, subtitles, cbar_label):
    """Unfolded-room images for up to four designs; ``grids[label] = {wall: grid}``."""
    vmax = max([float(np.nanmax(g[n])) for g in grids.values() for n in net if n in g] + [1e-9])
    im = None
    for ax, l in zip(axs, labels):
        for name, (tf, ext, H, V) in net.items():
            g = grids[l].get(name)
            if g is None:
                continue
            im = ax.imshow(tf(g), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=vmax, aspect="equal",
                           interpolation="nearest")
            ax.add_patch(Polygon([(ext[0], ext[2]), (ext[1], ext[2]), (ext[1], ext[3]), (ext[0], ext[3])],
                                 closed=True, fill=False, edgecolor="#888", lw=0.5))
            ax.text((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, name, ha="center", va="center", fontsize=6,
                    color="#bbb", alpha=0.8)
            for _, color, ls, masks in outlines:
                m = masks.get(name)
                if m is not None and m.any() and not m.all():
                    ax.contour(H, V, m.astype(float), levels=[0.5], colors=color, linewidths=0.9, linestyles=ls)
        span = max(abs(v) for e in (e for _, e, *_ in net.values()) for v in e)
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.invert_xaxis()
        ax.set_facecolor("#f2f2f2")
        ax.set_title(f"{l}" + ("\n" + subtitles[l] if subtitles.get(l) else ""), fontsize=8, weight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Y (cm, unfolded)", fontsize=7); ax.set_ylabel("Z (cm, unfolded)", fontsize=7)
    for ax in axs[len(labels):]:
        ax.axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axs.tolist(), orientation="horizontal", fraction=0.02, pad=0.06)
        cb.ax.tick_params(labelsize=6); cb.set_label(cbar_label, fontsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c if c != "white" else "#555", ls=ls, lw=1.2, label=lbl) for lbl, c, ls, _ in outlines]
    (axs[len(labels)] if len(labels) < len(axs) else fig).legend(
        handles=handles, fontsize=7, loc="center" if len(labels) < len(axs) else "lower right", frameon=False)


def _room_main_fov_masks(problem, settings, specs, pitch_offset=0.0):
    from lighting_simulator.camera.fov import points_in_pinhole_fov
    from lighting_simulator.simulation.room_geometry import room_wall_cell_centers
    cam = problem.camera
    cam_pos = np.array([cam.pos_x, cam.pos_y, 0.0])
    return {name: points_in_pinhole_fov(cam_pos, cam.pitch + pitch_offset, cam.fov_h, cam.fov_v,
                                        room_wall_cell_centers(name, spec, settings.front_dist, settings.side_dist,
                                                               settings.top_bottom_dist))
            for name, spec in specs.items()}


def _page_vio_room(pdf, problem, designs):
    """T2: unfolded VIO room (flight image) with the fisheye footprints."""
    labels = [l for l in designs if isinstance(designs[l][1].vio_grid, dict)]
    if not labels or not getattr(problem, "_needs_vio", False):
        return
    s = problem.vio.room_settings()
    specs = problem.vio.room_wall_specs()
    net = _unfolded_room(s, specs)
    outlines = [("main camera FOV (not judged here)", "white", "--", _room_main_fov_masks(problem, s, specs)),
                ("VIO fisheyes footprint (T2 metric)", VIO_COLOR, "-", problem._vio_masks)]
    flight_modes = [m for m in problem.modes if not m.is_flash and m.vio_min_lux]
    flight = flight_modes[0] if flight_modes else None
    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle(f"T2 — VIO room, flight image ({2 * problem.vio.room_dist / 100:g} m cube, unfolded)",
                 fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
    axs = fig.subplots(2, 2).ravel()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12, hspace=0.3, wspace=0.25)
    subtitles = {}
    for l in labels:
        ev = designs[l][1]
        if flight is not None and flight.name in ev.modes:
            frac = ev.modes[flight.name].get('vio_fraction') or 0.0
            subtitles[l] = (f"{frac * 100:.0f} % of VIO cells ≥ {flight.vio_min_lux:g} lx "
                            f"(target {flight.vio_min_fraction * 100:.0f} %)")
    _draw_room_net(fig, axs, labels, {l: designs[l][1].vio_grid for l in labels}, net, outlines, subtitles,
                   "lux (flight image)")
    fig.text(MARGIN, 0.02, textwrap.fill(
        "Cube net seen from inside the room: side / top / bottom walls fold out around the front wall (no back "
        "wall). +Y is to the viewer's left. Monte-Carlo trace of the VIO + Both LEDs at their continuous flux; only "
        "the cells inside the cyan fisheye footprints count. Grids are coarse on purpose.", 150),
        fontsize=7.5, color="#444", va="bottom")
    pdf.savefig(fig)


def _page_tilt_rooms(pdf, problem, designs):
    """T3: one unfolded room per wall distance, main camera pitched ±t (analytic image)."""
    labels = [l for l in designs if designs[l][1].tilt_grids]
    rooms = getattr(problem, "_tilt_rooms", None)
    if not labels or not rooms:
        return
    t = problem.objective.tilt_fov_deg
    img = "flash image" if problem.flash_modes else "flight image"
    for ri, room in enumerate(rooms):
        net = _unfolded_room(room.settings, room.specs)
        outlines = [("main camera untilted (T1, not judged here)", "white", "--",
                     _room_main_fov_masks(problem, room.settings, room.specs)),
                    (f"camera pitched +{t:g}° (T3 up)", TILT_UP_COLOR, ":", room.wall_masks('up')),
                    (f"camera pitched −{t:g}° (T3 down)", TILT_DOWN_COLOR, ":", room.wall_masks('down'))]
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle(f"T3 — room at {room.dist:g} cm, main camera ±{t:g}° ({img}, analytic)",
                     fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
        axs = fig.subplots(2, 2).ravel()
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12, hspace=0.3, wspace=0.25)
        subtitles = {}
        grids = {}
        for l in labels:
            ev = designs[l][1]
            grids[l] = ev.tilt_grids[ri] if ri < len(ev.tilt_grids) else {}
            if ri < len(ev.tilt_walls):
                tw = ev.tilt_walls[ri]
                subtitles[l] = (f"U0 ↑{tw['up']:.0f} % (cov {tw['cov_up'] * 100:.0f} %)  ·  "
                                f"↓{tw['down']:.0f} % (cov {tw['cov_down'] * 100:.0f} %)")
        _draw_room_net(fig, axs, labels, grids, net, outlines, subtitles, f"lux ({img})")
        fig.text(MARGIN, 0.02, textwrap.fill(
            f"Cube net seen from inside a 5-wall room {room.dist:g} cm from the rig (same distance as the T1 wall). "
            "Direct illuminance computed analytically on every cell for this picture; during optimisation only "
            "the cells inside the dotted footprints are evaluated. The pitched footprints span the front wall "
            "and the ceiling / floor, so their U0 is naturally lower than T1's.", 150),
            fontsize=7.5, color="#444", va="bottom")
        pdf.savefig(fig)


def _page_flash_heatmaps(pdf, problem, designs):
    """One page per wall: flight (continuous LEDs, reported only) next to the judged flash image."""
    labels = list(designs.keys())
    if not problem.flash_modes or not any(designs[l][1].flight_grid is not None for l in labels):
        return
    pct = problem.objective.min_percentile

    def _grid(ev, attr, wi):
        g = getattr(ev, attr)
        if g is None:
            return None
        return g[wi] if isinstance(g, list) else (g if wi == 0 else None)

    for wi, wl in enumerate(problem.walls):
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle(f"Flight vs. flash illuminance — wall {wl.wall_dist:g} cm", fontsize=15, weight="bold",
                     x=MARGIN, ha="left", y=0.965)
        axs = np.atleast_2d(fig.subplots(len(labels), 2, squeeze=False))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.35, wspace=0.3)
        half = wl.wall_size / 2
        cols = [("flight — VIO + Both LEDs, continuous flux (not judged)", "flight_grid"),
                (f"flash — Flash + Both LEDs at {problem.flash_lumens:,.0f} lm (+ VIO continuous) — T1", "grid")]
        for j, (title, attr) in enumerate(cols):
            grids = [_grid(designs[l][1], attr, wi) for l in labels]
            vmax = max([float(np.nanmax(g)) for g in grids if g is not None] + [1e-9])
            im = None
            for i, l in enumerate(labels):
                ax = axs[i, j]
                g = grids[i]
                if g is None:
                    ax.axis("off"); continue
                im = ax.imshow(g, origin="lower", extent=[-half, half, -half, half], cmap="inferno", vmin=0, vmax=vmax,
                               aspect="equal")
                _add_fov_outlines(ax, problem, wl.wall_dist)
                fov = g[problem._fov_masks[wi]]
                lit = fov[fov > 0]
                if lit.size:
                    e_min = np.percentile(lit, pct) if pct > 0 else lit.min()
                    sub = f"U0 {e_min / lit.mean() * 100:.0f} % (P{pct:g})  ·  E_avg {lit.mean():,.0f} lx"
                else:
                    sub = "unlit"
                el = designs[l][1].electrical
                n = (el.get('n_vio', 0) + el.get('n_both', 0)) if attr == "flight_grid" else (el.get('n_flash', 0) + el.get('n_both', 0))
                ax.invert_xaxis()
                ax.set_title((title + "\n" if i == 0 else "") + f"{sub}  ·  {n} LEDs lit", fontsize=7)
                ax.tick_params(labelsize=6)
                if i == len(labels) - 1:
                    ax.set_xlabel("Y (cm)", fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{l}\nZ (cm)", fontsize=8, weight="bold")
            if im is not None:
                cb = fig.colorbar(im, ax=axs[:, j].tolist(), orientation="horizontal", fraction=0.025, pad=0.06)
                cb.ax.tick_params(labelsize=6); cb.set_label("lux", fontsize=7)
        fig.text(MARGIN, 0.02, textwrap.fill(_fov_footer(
            problem, "Each column has its own colour scale (the flash image is typically 10–100× brighter)."), 150),
            fontsize=7.5, color="#444", va="bottom")
        pdf.savefig(fig)


def _page_layouts(pdf, problem, designs):
    labels = list(designs.keys())
    roles_on = bool(problem.flash_modes)
    role_color = {'vio': "#1e88e5", 'flash': "#fb8c00", 'both': "#8e24aa"}
    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle("LED layouts (active LEDs, arrows = beam axis" + (", colour = role)" if roles_on else ")"),
                 fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
    axs = np.atleast_2d(fig.subplots(len(labels), 2, squeeze=False))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.05, hspace=0.45, wspace=0.3)
    all_pos = []
    scenes = {}
    for l in labels:
        scene = problem.build_scene(designs[l][2])
        scenes[l] = scene
        all_pos += [np.asarray(led.position, float) for led in scene.active_leds]
    if not all_pos:
        return
    P = np.array(all_pos)
    lo, hi = P.min(0) - 4, P.max(0) + 4
    centre, span = (lo + hi) / 2, float(np.max(hi - lo))  # same square window for every view
    lo, hi = centre - span / 2, centre + span / 2
    for i, l in enumerate(labels):
        leds = scenes[l].active_leds
        pos = np.array([np.asarray(led.position, float) for led in leds]) if leds else np.zeros((0, 3))
        dirs = np.array([np.asarray(led.direction, float) for led in leds]) if leds else np.zeros((0, 3))
        lum = np.array([led.lumens for led in leds]) if leds else np.zeros(0)
        for j, (a, b, name_a, name_b) in enumerate([(0, 1, "X (cm)", "Y (cm)"), (1, 2, "Y (cm)", "Z (cm)")]):
            ax = axs[i, j]
            if len(pos):
                if roles_on:
                    from lighting_simulator.domain.led import led_role
                    colors = [role_color[led_role(led)] for led in leds]
                    ax.scatter(pos[:, a], pos[:, b], c=colors, s=28, edgecolor="k", lw=0.4, zorder=3)
                    if i == 0 and j == 1:
                        from matplotlib.lines import Line2D
                        ax.legend(handles=[Line2D([], [], marker='o', ls='', color=c, label=r) for r, c in role_color.items()],
                                  fontsize=6, loc="upper right")
                else:
                    sc = ax.scatter(pos[:, a], pos[:, b], c=lum, cmap="viridis", s=28, edgecolor="k", lw=0.4, zorder=3)
                    if i == 0 and j == 1:
                        cb = fig.colorbar(sc, ax=axs[:, 1].tolist(), fraction=0.03, pad=0.03); cb.set_label("lm / LED", fontsize=7)
                        cb.ax.tick_params(labelsize=6)
                ax.quiver(pos[:, a], pos[:, b], dirs[:, a], dirs[:, b], angles="xy", scale_units="xy", scale=0.35,
                          color="#c62828", width=0.005, zorder=2)
            ax.set_xlim(lo[a], hi[a]); ax.set_ylim(lo[b], hi[b]); ax.set_aspect("equal")
            if j == 1:
                ax.invert_xaxis()
            ax.set_xlabel(name_a, fontsize=7); ax.set_ylabel(name_b, fontsize=7)
            ax.set_title(f"{l} — {'top view' if j == 0 else 'front view (from the wall)'} — {len(leds)} LEDs",
                         fontsize=8)
            ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
    pdf.savefig(fig)


def _page_variable_bounds(pdf, problem, designs):
    names = problem.names
    lo = np.array([b[0] for b in problem.bounds], float)
    hi = np.array([b[1] for b in problem.bounds], float)
    span = np.where(hi > lo, hi - lo, 1.0)
    labels = list(designs.keys())
    per_page = 34
    colors = {"initial": "#9e9e9e", "best #1": "#2e7d32", "best #2": "#1565c0"}
    markers = {"initial": "o", "best #1": "*", "best #2": "s"}
    for start in range(0, len(names), per_page):
        idx = range(start, min(len(names), start + per_page))
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle("Optimised variables — position inside the search bounds", fontsize=15, weight="bold",
                     x=MARGIN, ha="left", y=0.965)
        ax = fig.add_axes([0.32, 0.06, 0.62, 0.85])
        for k, i in enumerate(idx):
            y = len(idx) - 1 - k
            ax.plot([0, 1], [y, y], color="#ddd", lw=4, solid_capstyle="round", zorder=1)
            for l in labels:
                v = (designs[l][0][i] - lo[i]) / span[i]
                ax.plot(v, y, markers.get(l, "o"), color=colors.get(l, "k"), ms=8 if l != "best #1" else 12,
                        zorder=3, label=l if k == 0 else None)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([f"{names[i]}   [{_fmt(lo[i])} … {_fmt(hi[i])}]" for i in reversed(list(idx))], fontsize=7,
                           family="monospace")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.7, len(idx) - 0.3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["min", "", "mid", "", "max"], fontsize=8)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
        fig.text(MARGIN, 0.02, "Values sitting on a bound suggest the search box may be too tight for that variable.",
                 fontsize=7.5, color="#444")
        pdf.savefig(fig)


def _page_variable_history(pdf, problem, records, designs):
    if len(records) < 5:
        return
    names = problem.names
    lo = np.array([b[0] for b in problem.bounds], float)
    hi = np.array([b[1] for b in problem.bounds], float)
    X = np.array([r[2] for r in records])
    ev_no = np.array([r[0] for r in records], float)
    score = np.array([r[1].score for r in records])
    order = np.argsort(-score)  # draw best (lowest) last
    best_x = designs["best #1"][0]
    ncol, nrow = 3, 6
    per_page = ncol * nrow
    for start in range(0, len(names), per_page):
        idx = list(range(start, min(len(names), start + per_page)))
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle("Optimised variables — explored values (colour = score, ★ = best)", fontsize=15,
                     weight="bold", x=MARGIN, ha="left", y=0.965)
        axs = np.atleast_2d(fig.subplots(nrow, ncol, squeeze=False))
        fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.05, hspace=0.6, wspace=0.3)
        sc = None
        for k, ax in enumerate(axs.flat):
            if k >= len(idx):
                ax.axis("off"); continue
            i = idx[k]
            sc = ax.scatter(ev_no[order], X[order, i], c=score[order], cmap="viridis_r", s=6, lw=0,
                            norm=_lognorm(score))
            ax.plot(ev_no[np.argmin(score)], best_x[i], "*", color="red", ms=10, mec="k", mew=0.5)
            ax.axhline(lo[i], color="#bbb", lw=0.8, ls=":"); ax.axhline(hi[i], color="#bbb", lw=0.8, ls=":")
            ax.set_title(names[i], fontsize=7.5)
            ax.tick_params(labelsize=6); ax.grid(alpha=0.2)
            pad = 0.05 * (hi[i] - lo[i] or 1)
            ax.set_ylim(lo[i] - pad, hi[i] + pad)
        if sc is not None:
            cax = fig.add_axes([0.91, 0.3, 0.015, 0.4])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label("score", fontsize=7); cb.ax.tick_params(labelsize=6)
        pdf.savefig(fig)


def _lognorm(score):
    from matplotlib.colors import LogNorm, Normalize
    s = score[np.isfinite(score) & (score > 0)]
    if s.size and s.max() / s.min() > 10:
        return LogNorm(vmin=s.min(), vmax=s.max())
    return Normalize(vmin=float(np.min(score)), vmax=float(np.max(score)))


# --------------------------------------------------------------------------- sensitivity
def _short(label, n=34):
    return label if len(label) <= n else label[: n - 1] + "…"


def _heatmap(ax, M, row_labels, col_labels, fontsize=6.5, annotate=True, mask_diagonal=False):
    M = np.array(M, dtype=float)
    if mask_diagonal:
        np.fill_diagonal(M, np.nan)  # self-correlation is always 1 and would dominate the colour scale
    img = np.ma.masked_invalid(M)
    cmap = plt_colormaps["coolwarm"].copy()
    cmap.set_bad("#e6e6e6")
    ax.imshow(img, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="left", fontsize=fontsize)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.tick_params(length=0)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=fontsize - 1,
                            color="white" if abs(v) > 0.6 else "black")
                elif not (mask_diagonal and i == j):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=fontsize - 1, color="#888")


def _page_sensitivity(pdf, problem, records):
    """Which inputs move the score: Spearman heatmap, surrogate ranking, partial dependence, co-variation."""
    if len(records) < 20 or problem.dim == 0:
        return
    from . import sensitivity as S
    res = S.analyse(problem, records)
    order = res.ranking()
    si = res.outcome_names.index('score')

    # ---- page 1: input × outcome correlation heatmap (rows ranked by |ρ(score)|)
    rows = order
    per_page = 38
    for start in range(0, len(rows), per_page):
        idx = rows[start:start + per_page]
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle("Sensitivity — rank correlation: inputs × score components",
                     fontsize=13, weight="bold", x=MARGIN, ha="left", y=0.965)
        ax = fig.add_axes([0.42, 0.12, 0.50, 0.74])
        labels = [("★ " if i >= res.n_vars else "") + _short(res.input_labels[i], 44) for i in idx]
        _heatmap(ax, res.corr[idx], labels, res.outcome_names)
        fig.text(MARGIN, 0.075, textwrap.fill(
            f"Spearman ρ over the {res.n_records} logged evaluations (derived layout features ★ on "
            f"{res.n_derived_records} of them). The score is minimised: ρ < 0 with 'score' means increasing the "
            "input tends to IMPROVE the design, ρ > 0 tends to hurt. For T1/T2/T3 columns (higher = better) the "
            "sign reads the other way. |ρ| ≈ 0 means the score does not react monotonically to that input over the "
            "explored range — either it does not matter, or its effect is non-monotonic (see the trend plots). "
            "Correlations are associations over the points the optimiser visited, not controlled experiments: "
            "the search oversamples good regions.", 135), fontsize=7.5, color="#444", va="top")
        pdf.savefig(fig)

    # ---- page 2: ranking bars (ρ with score and surrogate coefficient) + weak-variable list
    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle("Sensitivity — influence ranking", fontsize=14, weight="bold", x=MARGIN, ha="left", y=0.965)
    nv = res.n_vars
    var_order = [i for i in order if i < nv][:38]
    ax = fig.add_axes([0.40, 0.42, 0.55, 0.50])
    y = np.arange(len(var_order))
    rho = res.corr[var_order, si]
    coef = res.surrogate_coef[var_order]
    cmax = float(np.nanmax(np.abs(res.surrogate_coef))) or 1.0
    ax.barh(y + 0.2, np.nan_to_num(rho), height=0.38, color=np.where(rho < 0, "#2e7d32", "#c62828"), label="Spearman ρ with score")
    ax.barh(y - 0.2, coef / cmax, height=0.38, color="#607d8b", alpha=0.7,
            label=f"rank-linear surrogate coefficient (scaled, R² = {res.surrogate_r2:.2f})")
    ax.set_yticks(y); ax.set_yticklabels([_short(res.input_names[i], 40) for i in var_order], fontsize=6.5)
    ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-1.05, 1.05)
    ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("← larger value improves the score   |   larger value hurts →", fontsize=7.5)
    ax.set_title("Variables, most influential first", fontsize=9, loc="left")

    # derived features ranking
    feat_order = [i for i in order if i >= nv]
    if feat_order:
        ax2 = fig.add_axes([0.40, 0.20, 0.55, 0.14])
        y2 = np.arange(len(feat_order))
        r2 = res.corr[feat_order, si]
        ax2.barh(y2, np.nan_to_num(r2), height=0.6, color=np.where(r2 < 0, "#2e7d32", "#c62828"))
        ax2.set_yticks(y2); ax2.set_yticklabels([_short(res.input_labels[i], 48) for i in feat_order], fontsize=6.5)
        ax2.invert_yaxis(); ax2.axvline(0, color="k", lw=0.6); ax2.set_xlim(-1.05, 1.05)
        ax2.tick_params(labelsize=7); ax2.grid(axis="x", alpha=0.3)
        ax2.set_title("Derived layout features (ρ with score)", fontsize=9, loc="left")

    weak = res.weak_variables()
    lines = []
    if weak:
        lines.append("Candidates to freeze (score barely reacts, yet the best 20 % of designs still span > 60 % of "
                     "their range — the optimiser wanders in them for free):")
        for i in weak:
            lines.append(f"   • {res.input_names[i]}   ρ = {res.corr[i, si]:+.2f}, top-20 % range "
                         f"{res.top_range_frac[i]*100:.0f} % of the bounds")
    else:
        lines.append("No variable qualifies as clearly unneeded (|ρ| < 0.1 and still spanning > 60 % of its bounds "
                     "among the best 20 % of designs).")
    for n in res.notes:
        lines.append("⚠ " + n)
    fig.text(MARGIN, 0.155, "\n".join(textwrap.fill(l, 135, subsequent_indent="      ") for l in lines),
             fontsize=7.5, color="#444", va="top")
    pdf.savefig(fig)

    # ---- page 3: partial dependence (score vs input, binned median) for the most influential inputs
    top_inputs = [i for i in order if np.isfinite(res.corr[i, si])][:12]
    if top_inputs:
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle("Sensitivity — score vs. input, most influential first",
                     fontsize=13, weight="bold", x=MARGIN, ha="left", y=0.965)
        axs = np.atleast_2d(fig.subplots(4, 3, squeeze=False))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.06, hspace=0.55, wspace=0.3)
        score = res.Y[:, si]
        for ax, i in zip(axs.flat, top_inputs):
            x = res.X[:, i]
            ok = np.isfinite(x) & np.isfinite(score)
            ax.scatter(x[ok], score[ok], s=4, color="#9e9e9e", alpha=0.35, lw=0)
            cx, med, q1, q3 = S.binned_trend(x, score)
            if cx.size:
                ax.fill_between(cx, q1, q3, color="#ff9800", alpha=0.2, lw=0)
                ax.plot(cx, med, color="#e65100", lw=1.6, marker="o", ms=2.5)
            if np.all(np.isfinite(score[ok])) and score[ok].size and score[ok].max() / max(score[ok].min(), 1e-9) > 10:
                ax.set_yscale("log")
            ax.set_title(f"{_short(res.input_labels[i], 42)}   ρ = {res.corr[i, si]:+.2f}", fontsize=7)
            ax.tick_params(labelsize=6); ax.grid(alpha=0.25)
            if i < nv:
                lo, hi = problem.bounds[i]
                ax.axvline(lo, color="#bbb", lw=0.7, ls=":"); ax.axvline(hi, color="#bbb", lw=0.7, ls=":")
        for ax in list(axs.flat)[len(top_inputs):]:
            ax.axis("off")
        fig.text(MARGIN, 0.035, textwrap.fill(
            "Dots = evaluations, line = binned median score, band = interquartile range, dotted = bounds. "
            "A minimum inside the bounds is the useful operating range for that input; a flat median with a large "
            "|ρ| for another input signals interactions.", 135), fontsize=7.5, color="#444", va="top")
        pdf.savefig(fig)

    # ---- page 4: variable co-variation among the best designs
    if 2 <= nv <= 45:
        fig = Figure(figsize=A4, dpi=110)
        fig.suptitle(f"Sensitivity — variable co-variation in the best {int(res.top_mask.sum())} designs (top 20 %)",
                     fontsize=13, weight="bold", x=MARGIN, ha="left", y=0.965)
        ax = fig.add_axes([0.30, 0.14, 0.62, 0.66])
        vo = [i for i in order if i < nv]
        lab = [_short(res.input_names[i], 30) for i in vo]
        _heatmap(ax, res.top_corr[np.ix_(vo, vo)], lab, lab, fontsize=5.5, annotate=nv <= 24, mask_diagonal=True)
        fig.text(MARGIN, 0.085, textwrap.fill(
            "Spearman ρ between pairs of variables, restricted to the best-scoring fifth of the run. A strong "
            "|ρ| means the good designs trade one variable against the other (e.g. wider arc ↔ smaller tilt): "
            "they are not independent knobs and could be replaced by a single one. Near-zero everywhere means "
            "the optimum is well separated in each variable.", 135), fontsize=7.5, color="#444", va="top")
        pdf.savefig(fig)


# --------------------------------------------------------------------------- entry point
def write_report(problem, records, opt, out_dir, x0=None, elapsed=0.0, stopped=False, path=None, n_confirm=0):
    """Render ``<out_dir>/report.pdf`` and return its path.

    ``records`` is ``RunLogger.records``; ``x0`` the initial decision vector
    (defaults to ``problem.x0``). The best distinct designs are re-evaluated with
    more rays and ranked by that verified score before being compared.
    """
    out_dir = Path(out_dir)
    path = Path(path) if path else out_dir / "report.pdf"
    if not records:
        raise ValueError("no evaluations to report")
    x0 = np.asarray(problem.x0 if x0 is None else x0, float)
    verifier = _verifier(problem, VERIFY_RAY_FACTOR)

    candidates = _distinct_top(problem, records, VERIFY_TOP_K)
    verified = []  # (verified Evaluation, x, logged score)
    for i in candidates:
        x = records[i][2]
        verified.append((verifier.evaluate_config(problem.decode(x), keep_grid=True), x, records[i][1].score))
    verified.sort(key=lambda t: t[0].score)

    designs = {}  # label -> (x, Evaluation with grids, cfg)
    logged_scores = {}
    designs["initial"] = (x0, verifier.evaluate_config(problem.decode(x0), keep_grid=True), problem.decode(x0))
    logged_scores["initial"] = records[0][1].score if np.allclose(records[0][2], x0) else None
    for label, (ev, x, logged) in zip(("best #1", "best #2"), verified[:2]):
        designs[label] = (np.asarray(x, float), ev, problem.decode(x))
        logged_scores[label] = logged

    import json
    for label, fname in (("initial", "initial_config.json"), ("best #1", "best_config.json"),
                         ("best #2", "best2_config.json")):
        if label in designs:
            cfg = dict(designs[label][2])
            cfg['name'] = problem.name if label == "best #1" else f"{problem.name}_{fname[:-12]}"
            cfg['description'] = f"{label} (verified {VERIFY_RAY_FACTOR}x rays): {designs[label][1].summary()}"
            with open(out_dir / fname, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

    n_evals = records[-1][0]
    with PdfPages(path) as pdf:
        pdf.infodict().update({"Title": f"Optimisation report — {problem.name}", "Creator": "E4 Lighting Simulator"})
        w = _Writer(pdf, f"Optimisation report — {problem.name}")
        _page_summary(w, problem, opt, designs, n_evals, elapsed, stopped, out_dir, logged_scores, n_confirm)
        w.close()
        w = _Writer(pdf, "Problem definition")
        _page_problem(w, problem, designs)
        w.close()
        w = _Writer(pdf, "Method & formulas")
        _page_method(w, problem)
        w.close()
        _page_convergence(pdf, problem, records, designs)
        _page_heatmaps(pdf, problem, designs)
        _page_flash_heatmaps(pdf, problem, designs)
        _page_vio_room(pdf, problem, designs)
        _page_tilt_rooms(pdf, problem, designs)
        _page_layouts(pdf, problem, designs)
        _page_variable_bounds(pdf, problem, designs)
        _page_variable_history(pdf, problem, records, designs)
        _page_sensitivity(pdf, problem, records)
        w = _Writer(pdf, "Files")
        w.line("best_config.json — best design after verification, loadable in the UI (Project → Load) or via --evaluate")
        w.line("best2_config.json / initial_config.json — the other two designs compared in this report")
        w.line("history.csv — every logged evaluation with score, metrics and all variable values")
        w.line("summary.json — run metadata, best evaluation and best decision vector")
        for label, (x, ev, cfg) in designs.items():
            w.gap(0.006)
            w.line(f"{label}: {ev.summary()}", family="monospace", size=7.5)
        w.close()
    return path
