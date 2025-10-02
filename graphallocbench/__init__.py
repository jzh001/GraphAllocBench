"""GraphAllocBench public API."""

from .constants import (
    RL_Model,
    ALLOWED_GPUS,
    ALLOWED_DEVICES,
    VECENV_BATCH_SIZE,
    INFERENCE_DEVICE,
)
from .city_env.env_model import CityPlannerEnv

# Re-export selected evaluation helpers
from .evaluation.inference import run_experiments
from .evaluation.utils import (
    calculate_hypervolume,
    calculate_non_dominated,
    calculate_ordering_score,
    calculate_proportion_allocated,
    get_pareto_front_max,
)

__all__ = [
    "CityPlannerEnv",
    # constants
    "RL_Model",
    "ALLOWED_GPUS",
    "ALLOWED_DEVICES",
    "VECENV_BATCH_SIZE",
    "INFERENCE_DEVICE",
    # evaluation
    "run_experiments",
    "calculate_hypervolume",
    "calculate_non_dominated",
    "calculate_ordering_score",
    "calculate_proportion_allocated",
    "get_pareto_front_max",
]
