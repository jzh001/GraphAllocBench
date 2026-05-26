import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from scipy.stats import spearmanr
from ..city_env.env_model import CityPlannerEnv
from .inference import run_experiments
from .model_utils import load_model

def calculate_hypervolume(objectives):
    objectives = -np.array(objectives)
    ref_point_neg = np.array([1e-3] * objectives.shape[1])
    hypervolume_fn = HV(ref_point=ref_point_neg)
    return hypervolume_fn(objectives)

def calculate_non_dominated(objectives):
    # Negate objectives: NonDominatedSorting uses minimization, but objectives are maximized.
    fronts = NonDominatedSorting().do(-np.array(objectives), only_non_dominated_front=False)
    num_non_dominated = len(fronts[0])
    total = len(objectives)
    return num_non_dominated / total

def calculate_ordering_score(env: CityPlannerEnv,
                             model=None,
                             model_path=None,
                             n_iter = 10,
                             n_intervals = 10,
                             random_seed=42,
                             alpha = 1.0,
                             ):
    np.random.seed(random_seed)
    if model is None:
        model = load_model(model_path, env=env)
    all_scores = []
    preferences = []
    for i in range(env.n_objectives):
        for j in range(n_iter):
            preferences_template = np.random.dirichlet(np.ones(env.n_objectives) * alpha)
            for p in np.linspace(0, 1, n_intervals):
                pref = preferences_template.copy()
                pref[i] = 0
                pref_sum = pref.sum()
                if pref_sum > 0:
                    pref *= (1 - p) / pref_sum
                pref[i] = p
                preferences.append(pref)
    preferences = np.array(preferences)
    all_objectives, _ = run_experiments(env=env, model=model, preferences=preferences, deterministic=True, random_seed=random_seed)
    for i in range(env.n_objectives):
        for j in range(n_iter):
            start = (i * n_iter * n_intervals) + (j * n_intervals)
            end = start + n_intervals
            objective_i = all_objectives[start:end, i]
            if np.allclose(objective_i, objective_i[0]):
                score = 1.0
            else:
                sorted_indices = np.argsort(objective_i)
                sorted_objective_i = objective_i[sorted_indices]
                score = spearmanr(objective_i, sorted_objective_i).statistic
                score = (score + 1) / 2
            all_scores.append(score)
    return np.mean(all_scores)

def get_pareto_front_max(objectives: np.ndarray) -> np.ndarray:
    objectives_min = -objectives
    nds = NonDominatedSorting()
    pareto_indices = nds.do(objectives_min, only_non_dominated_front=True)
    return objectives[pareto_indices]

def calculate_proportion_allocated(env: CityPlannerEnv, model = None, seed = 42):
    objectives, alloc_matrices = run_experiments(env=env, model=model, n_iter = 100, random_seed=seed)
    total_unallocated = np.mean([np.sum(alloc[-1, :]) for alloc in alloc_matrices])
    return 1 - (total_unallocated / np.sum(env.avail_resources))
