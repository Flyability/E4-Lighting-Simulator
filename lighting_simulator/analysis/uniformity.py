"""Illuminance uniformity metrics (the objective for LED-placement optimisation).

Metrics follow common illuminotechnical practice, computed over lit cells only
(zero-lux cells outside the light cone are ignored):

- U0 = Emin / Eavg  (general uniformity)
- U1 = Emin / Emax  (overall uniformity)
- CV = σ / Eavg     (coefficient of variation)
- ΔEV = log2(Emax·0.4) − log2(Emin·0.4)  (photographic stop range)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformityMetrics:
    e_max: float
    e_min: float
    e_avg: float
    sigma: float
    u0: float
    u1: float
    cv: float
    ev_max: float
    ev_min: float
    delta_ev: float
    n_cells: int

    @property
    def uniformity_pct(self):
        return self.u0 * 100.0

    def ev_classification(self):
        """(label, detail, colour) for the ΔEV perceptual class."""
        if self.delta_ev < 0.3:
            return "Absolute uniformity", "green-screen / archival grade", "#4CAF50"
        if self.delta_ev <= 1.0:
            return "Soft wall washing", "smooth gradient, great general lighting", "#2196F3"
        if self.delta_ev <= 2.0:
            return "Moderate gradient", "visible fall-off, acceptable for most uses", "#FF9800"
        return "Wall grazing", "dramatic gradient, good for textures", "#F44336"


def trapezoid_mask(grid_shape, wall_size_cm, fov_trapezoid):
    """Cells inside a pitched-camera footprint on the wall (see camera.fov).

    ``fov_trapezoid`` = (z_bottom, z_top, half_w_bottom, half_w_top, y_center) in cm.
    """
    z_bot, z_top, w_bot, w_top, y_center = fov_trapezoid
    gz, gy = grid_shape
    cell_cm = wall_size_cm / gy
    half_wall = wall_size_cm / 2.0
    z_centers = -half_wall + (np.arange(gz) + 0.5) * cell_cm
    y_centers = -half_wall + (np.arange(gy) + 0.5) * cell_cm
    z_span = max(z_top - z_bot, 1e-9)
    frac = np.clip((z_centers - z_bot) / z_span, 0.0, 1.0)
    half_w_at_z = w_bot + (w_top - w_bot) * frac
    in_z = (z_centers >= z_bot) & (z_centers <= z_top)
    return in_z[:, None] & (np.abs(y_centers - y_center)[None, :] <= half_w_at_z[:, None])


def crop_to_fov_rect(grid, wall_size_cm, fov_w_cm, fov_h_cm):
    """Sub-grid inside a centred ``fov_w × fov_h`` rectangle."""
    gz, gy = grid.shape
    cell_cm = wall_size_cm / gy
    half_wall = wall_size_cm / 2.0
    y_lo = int(max(0, (half_wall - fov_w_cm / 2.0) / cell_cm))
    y_hi = int(min(gy, (half_wall + fov_w_cm / 2.0) / cell_cm))
    z_lo = int(max(0, (half_wall - fov_h_cm / 2.0) / cell_cm))
    z_hi = int(min(gz, (half_wall + fov_h_cm / 2.0) / cell_cm))
    return grid[z_lo:z_hi, y_lo:y_hi]


def select_region(grid, fov_bounds=None, wall_size_cm=None, fov_trapezoid=None):
    """Apply the optional FOV crop/mask used by the UI before computing metrics."""
    grid = np.asarray(grid, dtype=float)
    if fov_trapezoid is not None and wall_size_cm is not None:
        return np.where(trapezoid_mask(grid.shape, wall_size_cm, fov_trapezoid), grid, 0.0)
    if fov_bounds is not None and wall_size_cm is not None:
        return crop_to_fov_rect(grid, wall_size_cm, *fov_bounds)
    return grid


def uniformity_metrics(values, min_percentile=0.0):
    """Metrics over the strictly-positive entries of ``values``; None if none are lit.

    ``min_percentile`` > 0 replaces the hard minimum with that percentile of the
    lit cells, making U0/U1 robust to single Monte Carlo-starved cells.
    """
    active = np.asarray(values, dtype=float)
    active = active[active > 0]
    if active.size == 0:
        return None
    e_max = float(active.max())
    e_min = float(np.percentile(active, min_percentile)) if min_percentile > 0 else float(active.min())
    e_avg = float(active.mean())
    sigma = float(active.std())
    ev_max = float(np.log2(e_max * 0.4)) if e_max * 0.4 > 0 else 0.0
    ev_min = float(np.log2(e_min * 0.4)) if e_min * 0.4 > 0 else 0.0
    return UniformityMetrics(
        e_max=e_max, e_min=e_min, e_avg=e_avg, sigma=sigma,
        u0=e_min / e_avg if e_avg > 0 else 0.0,
        u1=e_min / e_max if e_max > 0 else 0.0,
        cv=sigma / e_avg if e_avg > 0 else 0.0,
        ev_max=ev_max, ev_min=ev_min, delta_ev=ev_max - ev_min,
        n_cells=int(active.size),
    )


def uniformity_html(metrics: UniformityMetrics):
    """HTML card for the intensity legend panel."""
    label, detail, color = metrics.ev_classification()
    return (
        "<div style='font-family:sans-serif;margin-top:10px;padding:8px;border-top:1px solid #444;'>"
        "<div style='font-weight:600;margin-bottom:4px;'>Pattern Uniformity</div>"
        f"<div style='font-size:22px;font-weight:700;color:{color};margin:2px 0 6px;'>{metrics.uniformity_pct:.1f}%</div>"
        "<table style='font-size:11px;color:#ccc;border-collapse:collapse;width:100%;'>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>max</sub></td><td>{metrics.e_max:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>min</sub></td><td>{metrics.e_min:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>E<sub>avg</sub></td><td>{metrics.e_avg:.1f} lx</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>U<sub>0</sub> (E<sub>min</sub>/E<sub>avg</sub>)</td><td>{metrics.u0:.3f}</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>U<sub>1</sub> (E<sub>min</sub>/E<sub>max</sub>)</td><td>{metrics.u1:.3f}</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>CV (&sigma;/E<sub>avg</sub>)</td><td>{metrics.cv:.3f}</td></tr>"
        "<tr><td colspan='2' style='padding-top:4px;'></td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>EV<sub>max</sub></td><td>{metrics.ev_max:.2f} stops</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>EV<sub>min</sub></td><td>{metrics.ev_min:.2f} stops</td></tr>"
        f"<tr><td style='padding:1px 6px 1px 0;'>&Delta;EV</td><td><b>{metrics.delta_ev:.2f} stops</b></td></tr>"
        "</table>"
        f"<div style='margin-top:6px;padding:4px 6px;background:{color}22;border-left:3px solid {color};border-radius:2px;'>"
        f"<span style='font-weight:600;color:{color};'>{label}</span><br/>"
        f"<span style='font-size:10px;color:#aaa;'>{detail}</span>"
        "</div></div>"
    )


def compute_uniformity_html(grid, fov_bounds=None, wall_size_cm=None, fov_trapezoid=None):
    """Legacy one-shot helper: region selection + metrics + HTML ('' if unlit)."""
    metrics = uniformity_metrics(select_region(grid, fov_bounds, wall_size_cm, fov_trapezoid))
    return uniformity_html(metrics) if metrics else ""
