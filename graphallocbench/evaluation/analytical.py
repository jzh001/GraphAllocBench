import numpy as np
import os
import pickle
from ..city_env.env_model import CityPlannerEnv
from .utils import get_pareto_front_max

def get_analytical_objectives(env: CityPlannerEnv,
                                check_production=True, 
                                use_cache = True,
                                directory = 'problems'):
    save_dir = f'cache/non_dominated_points/{directory}'
    cache_file = f'{save_dir}/{env.config_name}.npy'
    if os.path.exists(cache_file) and use_cache:
        return np.load(cache_file)
    max_total = max(env.avail_resources)
    all_objectives = []
    if check_production:
        for prod in np.ndindex(*(max_total + 1,) * env.demand_count):
            alloc_matrix = env.check_productions_possible(prod)
            if alloc_matrix is not None:
                all_objectives.append(env.get_objectives(np.array(prod), alloc_matrix))
    else:
        for total in range(0, max_total + 1):
            for prod in generate_simplex_grid(env.demand_count, total):
                all_objectives.append(env.get_objectives(prod, None))
    all_objectives = np.array(all_objectives)
    non_dominated_points = get_pareto_front_max(all_objectives)
    os.makedirs(save_dir, exist_ok=True)
    np.save(cache_file, non_dominated_points)
    return non_dominated_points

def get_analytical_objectives_increasing_fc(env: CityPlannerEnv,
                                use_cache = True,
                                directory = 'problems'):
    save_dir = f'cache/non_dominated_points_increasing_fc/{directory}'
    cache_file = f'{save_dir}/{env.config_name}.npy'
    if os.path.exists(cache_file) and use_cache:
        return np.load(cache_file)
    all_productions = generate_simplex_grid(env.demand_count, min(env.avail_resources))
    all_objectives = []
    for p in all_productions:
        all_objectives.append(env.get_objectives(p, None))
    non_dominated_points = get_pareto_front_max(all_objectives)
    os.makedirs(save_dir, exist_ok=True)
    np.save(cache_file, non_dominated_points)
    return non_dominated_points

def generate_simplex_grid(N, total=10, cache_dir="cache/simplex_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"simplex_N{N}_T{total}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    def integer_compositions(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(n + 1):
                for tail in integer_compositions(n - i, k - 1):
                    yield [i] + tail
    result = list(integer_compositions(total, N))
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
