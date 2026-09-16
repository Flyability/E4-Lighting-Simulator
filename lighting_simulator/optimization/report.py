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
    if getattr(problem, "_needs_vio", False) and getattr(problem, "_vio_wall", None) is not None:
        v = problem._vio_wall
        p._vio_wall = WallSettings(wall_dist=v.wall_dist, grid_size=v.grid_size, wall_size=v.wall_size,
                                   rays_per_pixel=int(v.rays_per_pixel * factor))
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
           f"elapsed: {elapsed:.0f} s   backend: {'GPU' if problem.use_gpu else 'CPU'}"
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
    widths = [36] + [16] * len(labels)
    evs = [designs[k][1] for k in labels]
    pct = problem.objective.min_percentile
    u0_label = f"U0 = P{pct:g}(E)/E_avg" if pct > 0 else "U0 = E_min/E_avg"
    rows = [
        [f"Score, verified {VERIFY_RAY_FACTOR}× rays", *[f"{e.score:.4f}" for e in evs]],
        ["Score, logged during run", *[(f"{logged_scores[k]:.4f}" if logged_scores.get(k) is not None else "—")
                                       for k in labels]],
        [f"{u0_label}, mean of walls", *[f"{e.uniformity_pct:.1f} %" for e in evs]],
        ["U0 hard-min per wall (UI legend)", *[" / ".join(f"{u:.0f}" for u in _hard_min_u0(problem, e)) + " %"
                                              for e in evs]],
        ["FOV coverage", *[f"{e.coverage*100:.0f} %" for e in evs]],
        ["E_avg in FOV (mean over walls)", *[f"{e.e_avg:,.0f} lx" for e in evs]],
        ["Active LEDs", *[str(e.n_active) for e in evs]],
        ["Drivers", *[str(e.n_drivers) for e in evs]],
        ["Continuous current", *[f"{e.total_current_a:.1f} A" for e in evs]],
    ]
    if any(e.electrical for e in evs):
        rows += [
            ["LED roles vio / flash / both", *[
                f"{e.electrical.get('n_vio', 0)} / {e.electrical.get('n_flash', 0)} / {e.electrical.get('n_both', 0)}"
                for e in evs]],
            ["Pulse / continuous drivers", *[
                f"{e.electrical.get('n_pulse_drivers', 0)} / {e.electrical.get('n_cont_drivers', 0)}" for e in evs]],
            ["Peak current (flash)", *[f"{e.electrical.get('peak_current_a', 0):.1f} A" for e in evs]],
        ]
    if evs[0].metrics is not None:
        rows += [
            ["E_min / E_max (last wall)", *[
                f"{e.metrics.e_min:,.0f} / {e.metrics.e_max:,.0f} lx" if e.metrics else "—" for e in evs]],
            ["ΔEV (stops)", *[f"{e.metrics.delta_ev:.2f}" if e.metrics else "—" for e in evs]],
        ]
    if problem.objective.tilt_fov_deg:
        t = problem.objective.tilt_fov_deg
        rows.append([f"U0, camera pitched +{t:g}° (up)", *[f"{e.tilt.get('up', 0):.1f} %" for e in evs]])
        rows.append([f"U0, camera pitched −{t:g}° (down)", *[f"{e.tilt.get('down', 0):.1f} %" for e in evs]])
    for mode in problem.modes:
        rows.append([f"{mode.name}: E_avg", *[
            f"{e.modes.get(mode.name, {}).get('e_avg', 0):,.0f} lx" for e in evs]])
        if mode.is_flash or mode.uniformity_weight:
            rows.append([f"{mode.name}: U0 ({'pulse' if mode.is_flash else 'continuous'} LEDs)", *[
                (lambda u: f"{u:.1f} %" if u is not None else "—")(e.modes.get(mode.name, {}).get('uniformity_pct'))
                for e in evs]])
            rows.append([f"{mode.name}: LEDs lit", *[str(e.modes.get(mode.name, {}).get('n_leds', '—')) for e in evs]])
        if mode.vio_min_lux:
            rows.append([f"{mode.name}: VIO ≥{mode.vio_min_lux:g} lx", *[
                (lambda f: f"{f*100:.0f} %" if f is not None else "—")(e.modes.get(mode.name, {}).get('vio_fraction'))
                for e in evs]])
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
               "(e.g. drive current), or drop it.", size=8.5, color="#b26a00")

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
    w.line(f"Emission: default {e.default_lumens:g} lm/LED, focus factor {e.ray_uniformity:g}"
           + (";  diffuser " + f"{problem.diffuser[0]:g}° × {problem.diffuser[1]*100:.0f} %" if problem.diffuser else "")
           + (";  STL occluder active" if problem.stl_mesh is not None else ""))
    d = problem.driver
    w.line(f"Pulse driver model: {d.voltage_v:g} V, {d.efficacy_lm_per_w:g} lm/W → {d.lumens(1.0):,.0f} lm/A, "
           f"max {d.max_current_a:g} A per LED, {d.leds_per_driver} LED(s) per driver")
    if problem.cont_driver is not problem.driver:
        cd = problem.cont_driver
        w.line(f"Continuous driver model ('vio' LEDs): max {cd.max_current_a:g} A per LED, "
               f"{cd.leds_per_driver} LED(s) per driver")
    if problem.vio is not None:
        v = problem.vio
        where = (f"VIO room: six walls {v.room_dist:g} cm away ({2*v.room_dist/100:g} m across), "
                 f"{v.room_grid_size}² cells per wall" if v.geometry == "room" else
                 f"VIO wall at {v.wall_dist:g} cm, {v.wall_size:g} cm wide, {v.grid_size}² cells")
        w.line(f"VIO cameras: at {_fmt(list(v.position))} cm, cam1 pitch {v.cam1_pitch:g}° yaw {v.cam1_yaw:g}°, "
               f"cam2 pitch {v.cam2_pitch:g}° yaw {v.cam2_yaw:g}°, long-side FOV {v.long_fov:g}° "
               f"({'landscape' if v.landscape else 'portrait'}); {where}")
    if problem.modes:
        w.heading("Operating modes")
        for m in problem.modes:
            parts = [(f"pulse: flash + both LEDs at {m.current_a:g} A, vio LEDs continuous" if m.is_flash
                      else f"continuous: vio + both LEDs, flux × {m.lumens_scale:g}")]
            if m.uniformity_weight:
                parts.append(f"uniformity term w={m.uniformity_weight:g}")
            if m.min_avg_lux:
                parts.append(f"E_avg ≥ {m.min_avg_lux:,.0f} lx"
                             + (f" at {m.min_avg_lux_dist:g} cm" if m.min_avg_lux_dist else "") + f" (w={m.lux_weight:g})")
            if m.vio_min_lux:
                parts.append(f"≥ {m.vio_min_fraction*100:.0f} % of VIO FOV above {m.vio_min_lux:g} lx (w={m.vio_weight:g})")
            w.line(f"• {m.name}: " + "; ".join(parts))

    w.heading("Objective")
    o = problem.objective
    w.line(f"metric = {o.metric}, E_min percentile = {o.min_percentile:g} %, coverage weight = {o.coverage_weight:g}"
           + (f", min average lux = {o.min_avg_lux:,.0f} (w={o.lux_weight:g})" if o.min_avg_lux else "")
           + (f"; ±{o.tilt_fov_deg:g}° tilt FOVs (w={o.tilt_fov_weight:g}) on the "
              + ("VIO room walls" if getattr(problem, '_tilt_room', False) else "flat wall") if o.tilt_fov_deg else ""))

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
    w.heading("How a design is scored")
    w.line("Each candidate decision vector x is decoded into a regular saved configuration (same JSON schema "
           "as configs/*.json), the scene is built with the exact same code as the UI, and the wall "
           "illuminance E (lux) is ray-traced on a grid at every wall distance d in the list above. Only the "
           "cells inside the main-camera footprint (the FOV trapezoid, drawn in white on the images) are "
           "kept; cells with E = 0 are ignored for the uniformity metrics but counted for coverage.")
    w.gap(0.006)
    w.line("Uniformity metrics over the lit FOV cells:", weight="bold")
    pct = f"P_{{{o.min_percentile:g}}}(E)" if o.min_percentile > 0 else r"\min(E)"
    w.math(rf"E_{{min}} = {pct},\quad E_{{avg}} = \overline{{E}},\quad E_{{max}} = \max(E)")
    w.math(r"U_0 = \frac{E_{min}}{E_{avg}},\qquad U_1 = \frac{E_{min}}{E_{max}},\qquad CV = \frac{\sigma_E}{E_{avg}}")
    if o.min_percentile > 0:
        w.line(f"E_min uses the {o.min_percentile:g}-th percentile of lit cells so a single Monte-Carlo-starved "
               "cell cannot dominate the metric.", size=8, color="#444", indent=0.03)
    w.gap(0.006)
    w.line("Score (minimised):", weight="bold")
    f = {"u0": r"1 - U_0", "u1": r"1 - U_1", "cv": r"CV"}[o.metric]
    w.math(rf"S(x) = \frac{{1}}{{N_d}}\sum_{{d}} \left[{f}\right]_d \;+\; "
           rf"{o.coverage_weight:g}\,\left(1 - \min_d \mathrm{{cov}}_d\right) \;+\; \sum_k P_k(x)")
    w.line("cov_d is the fraction of FOV cells receiving any light at distance d. Every constraint is a soft "
           "penalty P_k ≥ 0 added to the score (no hard infeasibility), which keeps the landscape usable for "
           "population-based global search.", size=8.5, color="#444", indent=0.03)
    w.gap(0.006)
    w.line("Active penalty terms:", weight="bold")
    if o.min_avg_lux:
        w.math(rf"P_{{lux}} = {o.lux_weight:g}\,\max\!\left(0,\ \frac{{{o.min_avg_lux:g} - \min_d E_{{avg,d}}}}{{{o.min_avg_lux:g}}}\right)")
    if c.led_cost:
        w.math(rf"P_{{led}} = {c.led_cost:g}\cdot N_{{LED}}")
    if c.max_leds is not None:
        w.math(rf"P_{{maxled}} = {c.max_leds_weight:g}\cdot\max(0,\ N_{{LED}} - {c.max_leds})")
    if c.driver_cost:
        w.math(rf"P_{{drv}} = {c.driver_cost:g}\cdot N_{{drv}},\qquad N_{{drv}} = \left\lceil N_{{LED}} / {problem.driver.leds_per_driver} \right\rceil")
    if c.max_drivers is not None:
        w.math(rf"P_{{maxdrv}} = {c.max_drivers_weight:g}\cdot\max(0,\ N_{{drv}} - {c.max_drivers})")
    if c.max_total_current_a:
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
        s = rf"I_{{{m.name}}} / \bar I" if m.current_a is not None else f"{m.lumens_scale:g}"
        w.math(rf"\text{{mode {m.name}:}}\quad E^{{({m.name})}} = s\,E,\qquad s = {s}")
        if m.min_avg_lux:
            w.math(rf"P_{{{m.name},lux}} = {m.lux_weight:g}\,\max\!\left(0,\ \frac{{{m.min_avg_lux:g} - E^{{({m.name})}}_{{avg}}}}{{{m.min_avg_lux:g}}}\right)")
        if m.vio_min_lux:
            w.math(rf"P_{{{m.name},vio}} = {m.vio_weight:g}\,\max\!\left(0,\ \frac{{{m.vio_min_lux:g} - E^{{({m.name})}}_{{P{100*(1-m.vio_min_fraction):g}}}}}{{{m.vio_min_lux:g}}}\right)")
            w.line(f"E_P{100*(1-m.vio_min_fraction):g} is the {100*(1-m.vio_min_fraction):g}-th percentile of lux over the "
                   f"VIO-visible cells: requiring it to reach {m.vio_min_lux:g} lx is the same as requiring "
                   f"{m.vio_min_fraction*100:.0f} % of the VIO FOV above {m.vio_min_lux:g} lx, but the penalty keeps a "
                   "gradient when the target is still far away.", size=8, color="#444", indent=0.03)
    if problem.modes:
        w.line("Modes share the geometry and differ only by drive current; since luminous flux is linear in "
               "current (Φ = I·V·η), the traced grid is rescaled per mode instead of re-traced. Ī is the mean "
               "design current implied by the LED flux in the decoded configuration.", size=8.5, color="#444",
               indent=0.03)
    w.gap(0.004)
    w.line("Optimiser:", weight="bold")
    w.line("Differential evolution (scipy) evolves a population inside the bounds and keeps improvements; "
           "integer variables (on/off, row/column counts) are rounded before decoding. Nelder–Mead / random "
           "search are local alternatives. The best design is exported after every improvement.", size=8)
    w.line("Rays are random, so scores carry Monte-Carlo noise. Safeguards: a candidate beating the current "
           "best is re-evaluated with fresh rays and the mean is kept (a lucky draw must be lucky twice); for "
           f"this report the best distinct designs are re-evaluated with {VERIFY_RAY_FACTOR}× the rays and ranked "
           "by that verified score. Dashed trend lines on the convergence page are centred moving averages over "
           "~5 % of the run.", size=8)


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
    ax.set_title("Active LEDs / drivers"); ax.set_xlabel("evaluation"); ax.legend(fontsize=7)

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


def _fov_polygon(problem, wall_dist):
    z_bot, z_top, w_bot, w_top, y_c = problem.camera.trapezoid(wall_dist)
    return Polygon([(y_c - w_bot, z_bot), (y_c + w_bot, z_bot), (y_c + w_top, z_top), (y_c - w_top, z_top)],
                   closed=True, fill=False, edgecolor="white", lw=1.2, ls="--")


def _page_heatmaps(pdf, problem, designs):
    labels = list(designs.keys())
    n_rows = len(labels)
    cols = [(f"wall {wl.wall_dist:g} cm", wl.wall_size, i) for i, wl in enumerate(problem.walls)]
    has_vio = any(designs[l][1].vio_grid is not None for l in labels)
    vio_room = has_vio and getattr(problem, "_vio_room", False)
    if vio_room:
        cols.append((f"VIO room, front wall {problem.vio.room_dist:g} cm", 2 * problem.vio.room_dist, "vio"))
    elif has_vio:
        cols.append((f"VIO wall {problem.vio.wall_dist:g} cm", problem.vio.wall_size, "vio"))
    n_cols = len(cols)
    fig = Figure(figsize=A4, dpi=110)
    fig.suptitle("Wall illuminance — initial vs. best designs" + (" (flight mode)" if problem.flash_modes else ""),
                 fontsize=15, weight="bold", x=MARGIN, ha="left", y=0.965)
    axs = np.atleast_2d(fig.subplots(n_rows, n_cols, squeeze=False))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10, hspace=0.35, wspace=0.35)

    for j, (title, wall_size, key) in enumerate(cols):
        grids = []
        for l in labels:
            ev = designs[l][1]
            if key == "vio":
                g = ev.vio_grid.get('front') if isinstance(ev.vio_grid, dict) else ev.vio_grid
            else:
                g = _as_grid_list(ev)[key] if _as_grid_list(ev) else None
            grids.append(g)
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
            if key != "vio":
                ax.add_patch(_fov_polygon(problem, problem.walls[key].wall_dist))
                fov = g[problem._fov_masks[key]]
                lit = fov[fov > 0]
                if lit.size:
                    pct = problem.objective.min_percentile
                    e_min = np.percentile(lit, pct) if pct > 0 else lit.min()
                    sub = (f"U0 {e_min/lit.mean()*100:.0f} % (P{pct:g})  ·  hard-min {lit.min()/lit.mean()*100:.0f} %"
                           f"  ·  {lit.mean():,.0f} lx")
                else:
                    sub = "unlit"
            else:
                mask = problem._vio_mask
                if vio_room:
                    n_all = sum(int(np.count_nonzero(m)) for m in problem._vio_masks.values())
                    sub = f"{np.count_nonzero(mask)} VIO cells on front wall ({n_all} over 6 walls)"
                else:
                    sub = f"{np.count_nonzero(mask)} VIO cells"
                if mask.size and not mask.all():
                    ax.contour(np.linspace(-half, half, g.shape[1]), np.linspace(-half, half, g.shape[0]),
                               mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
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
    fig.text(MARGIN, 0.02, f"Dashed white: main-camera FOV footprint (metrics are computed inside it). "
             f"Cyan: VIO camera footprint. Colour scale is shared per column. Images use {VERIFY_RAY_FACTOR}× the "
             "run's rays per pixel.", fontsize=7.5, color="#444")
    pdf.savefig(fig)


def _page_flash_heatmaps(pdf, problem, designs):
    """One page per wall: flight (continuous LEDs) next to flash (pulse) illuminance for every design."""
    labels = list(designs.keys())
    if not problem.flash_modes or not any(designs[l][1].flash_grid is not None for l in labels):
        return
    flash_mode = problem.flash_modes[0]
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
        cols = [("flight — VIO + Both LEDs, continuous current", "grid"),
                (f"flash — Flash + Both LEDs at {flash_mode.current_a:g} A (+ VIO continuous)", "flash_grid")]
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
                ax.add_patch(_fov_polygon(problem, wl.wall_dist))
                fov = g[problem._fov_masks[wi]]
                lit = fov[fov > 0]
                if lit.size:
                    e_min = np.percentile(lit, pct) if pct > 0 else lit.min()
                    sub = f"U0 {e_min / lit.mean() * 100:.0f} % (P{pct:g})  ·  E_avg {lit.mean():,.0f} lx"
                else:
                    sub = "unlit"
                el = designs[l][1].electrical
                n = (el.get('n_vio', 0) + el.get('n_both', 0)) if attr == "grid" else (el.get('n_flash', 0) + el.get('n_both', 0))
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
        fig.text(MARGIN, 0.02, "Each column has its own colour scale (the flash image is typically 10–100× brighter). "
                 "Dashed white: main-camera FOV footprint.", fontsize=7.5, color="#444")
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
        _page_layouts(pdf, problem, designs)
        _page_variable_bounds(pdf, problem, designs)
        _page_variable_history(pdf, problem, records, designs)
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
