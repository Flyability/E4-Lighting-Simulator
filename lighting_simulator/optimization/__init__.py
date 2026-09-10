"""LED placement optimisation for wall-pattern uniformity.

Typical use::

    python -m lighting_simulator.optimization optimization_specs/elios3_duct_rings.json

or programmatically::

    from lighting_simulator.optimization import load_spec, run_spec
    spec, spec_dir = load_spec("optimization_specs/elios3_duct_rings.json")
    summary, best = run_spec(spec, spec_dir)
"""

from .problem import (
    CameraSpec,
    ConstraintSpec,
    Evaluation,
    ObjectiveSpec,
    Problem,
    load_spec,
    problem_from_spec,
)
from .runner import OptimizationStopped, OptimizerSpec, run, run_spec
from .variables import (
    BeamAngle,
    BeamTilts,
    Duct,
    DuctRingLayout,
    LedStates,
    PanelPose,
    variable_from_spec,
)

__all__ = [
    "BeamAngle", "BeamTilts", "CameraSpec", "ConstraintSpec", "Duct", "DuctRingLayout",
    "Evaluation", "LedStates", "ObjectiveSpec", "OptimizationStopped", "OptimizerSpec", "PanelPose", "Problem",
    "load_spec", "problem_from_spec", "run", "run_spec", "variable_from_spec",
]
