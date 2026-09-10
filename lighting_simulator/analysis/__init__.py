"""Post-processing of simulated illuminance grids."""

from .uniformity import (
    UniformityMetrics,
    compute_uniformity_html,
    select_region,
    uniformity_html,
    uniformity_metrics,
)

__all__ = [
    "UniformityMetrics",
    "compute_uniformity_html",
    "select_region",
    "uniformity_html",
    "uniformity_metrics",
]
