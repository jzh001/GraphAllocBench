import numpy as np
import os
from scipy.stats import qmc

from city_env.env_model import ResourceManagementEnv
from evaluation.utils import get_pareto_front_max


def get_analytical_objectives_sample_increasing_fc(env: ResourceManagementEnv,
                                     num_samples=1000,
                                     use_cache=True,
                                     directory='problems'):
    """
    Sample integer-valued points from the N-dimensional simplex 
    (sum = min(env.avail_resources)) and compute the Pareto front.

    Args:
        env (ResourceManagementEnv): The environment instance.
        num_samples (int): Number of production vectors to sample.
        use_cache (bool): Whether to load/save from cache.
        directory (str): Subfolder for cache storage.
    """
    save_dir = f'cache/non_dominated_points_sample/{directory}'
    os.makedirs(save_dir, exist_ok=True)
    cache_file = f'{save_dir}/{env.config_name}_samples{num_samples}.npy'

    if os.path.exists(cache_file) and use_cache:
        return np.load(cache_file)

    N = env.demand_count
    total = min(env.avail_resources)

    # Integer simplex sampling: random compositions summing to `total`
    samples = []
    for _ in range(num_samples):
        # Pick N-1 random cut points in [0, total]
        cuts = np.sort(np.random.randint(0, total + 1, size=N-1))
        # The differences give parts that sum to `total`
        points = np.diff(np.concatenate(([0], cuts, [total])))
        samples.append(points)
    all_productions = np.array(samples, dtype=int)
    
    # Evaluate each sampled production vector
    all_objectives = []
    for p in all_productions:
        all_objectives.append(env.get_objectives(p, None))

    all_objectives = np.array(all_objectives)
    
    # Get non-dominated points
    non_dominated_points = get_pareto_front_max(all_objectives)

    np.save(cache_file, non_dominated_points)
    return non_dominated_points

def calculate_hypervolume_approx(objectives, num_samples=10000, ref_point=-0.1, random_seed=42, plot=False):
    np.random.seed(random_seed)
    objectives = np.asarray(objectives)
    pareto_front = get_pareto_front_max(objectives)

    n_objectives = pareto_front.shape[1]

    if np.isscalar(ref_point):
        ref_point_vec = np.full(n_objectives, ref_point)
    else:
        ref_point_vec = np.asarray(ref_point)

    if ref_point_vec.shape[0] != n_objectives:
        raise ValueError(f"Reference point dimension ({ref_point_vec.shape[0]}) does not match objective dimension ({n_objectives}).")

    ref_point_vec = ref_point_vec.astype(pareto_front.dtype)

    sampler = qmc.LatinHypercube(d=n_objectives, seed=random_seed)
    samples = sampler.random(n=num_samples)

    max_objectives = np.max(pareto_front, axis=0)

    valid_range = max_objectives - ref_point_vec
    if np.any(valid_range <= 0):
        return 0.0

    scaled_samples = ref_point_vec + samples * valid_range

    hypercube_volume = np.prod(valid_range)

    dominated_samples_count = 0
    for sample in scaled_samples:
        if np.any(np.all(pareto_front >= sample, axis=1)):
            dominated_samples_count += 1

    hypervolume = hypercube_volume * (dominated_samples_count / num_samples)

    # print(dominated_samples_count, hypercube_volume)

    return hypervolume
