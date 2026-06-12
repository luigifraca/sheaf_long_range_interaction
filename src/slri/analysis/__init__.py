"""Post-training Jacobian, pathwise, geometry, and curvature analysis."""

from slri.analysis.influence import (
    aggregate_hop_influence,
    compute_target_jacobians,
)
from slri.analysis.pathwise import compute_pathwise_jacobian
from slri.analysis.runner import analyze_run

__all__ = [
    "aggregate_hop_influence",
    "analyze_run",
    "compute_pathwise_jacobian",
    "compute_target_jacobians",
]
