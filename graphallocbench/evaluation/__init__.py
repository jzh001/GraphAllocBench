"""Evaluation helpers public surface.

This subpackage bundles utilities for:
 - running experiments to gather objective vectors (``run_experiments``)
 - computing Pareto + scalar metrics (hypervolume, ordering, dominance, allocation)
 - analytical objective generation for small problems
 - heuristic / approximate metrics
 - statistics aggregation across problems / architectures
 - plotting helpers & selection utilities

All functions are re-exported here for convenience so users can write:

    from graphallocbench.evaluation import calculate_hypervolume
"""

from .inference import run_experiments
from .utils import (
    calculate_hypervolume,
    calculate_non_dominated,
    calculate_ordering_score,
    calculate_proportion_allocated,
    get_pareto_front_max,
)
from .model_utils import load_model
from .analytical import get_analytical_objectives
from .heuristics import (
    get_analytical_objectives_sample_increasing_fc,
    calculate_hypervolume_approx,
)
from .stats import (
    calculate_stats,
    calculate_all_stats,
    calculate_stats_more_objectives,
    calculate_stats_across_arch,
)
from .plot import (
    plot_summary_stats,
    plot_objectives_vs_production,
    plot_2D_pareto_front,
    plot_2D_pareto_fronts_all,
    plot_more_objectives,
    plot_stats_across_arch,
)
from .selection import generate_problem_summary, plot_problem_summary

__all__ = [
    # core experiment runner
    "run_experiments",
    # metrics
    "calculate_hypervolume",
    "calculate_non_dominated",
    "calculate_ordering_score",
    "calculate_proportion_allocated",
    "get_pareto_front_max",
    # model util
    "load_model",
    # analytical / heuristic
    "get_analytical_objectives",
    "get_analytical_objectives_sample_increasing_fc",
    "calculate_hypervolume_approx",
    # stats
    "calculate_stats",
    "calculate_all_stats",
    "calculate_stats_more_objectives",
    "calculate_stats_across_arch",
    # plotting
    "plot_summary_stats",
    "plot_objectives_vs_production",
    "plot_2D_pareto_front",
    "plot_2D_pareto_fronts_all",
    "plot_more_objectives",
    "plot_stats_across_arch",
    # selection utilities
    "generate_problem_summary",
    "plot_problem_summary",
]
