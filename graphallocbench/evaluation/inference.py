from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import random

from ..city_env.env_model import CityPlannerEnv
from .model_utils import load_model
from ..constants import VECENV_BATCH_SIZE

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def run_experiments(env: CityPlannerEnv, 
                    model = None, 
                    preferences=None, 
                    n_iter=100, 
                    deterministic=True, 
                    n_envs=VECENV_BATCH_SIZE,
                    random_seed=42,
                    model_path = None,
                    ):
    np.random.seed(random_seed)
    random.seed(random_seed)

    vec_env = DummyVecEnv([lambda: CityPlannerEnv(env.config_path) for _ in range(n_envs)])

    if model is None:
        model = load_model(model_path, env=vec_env)

    all_objectives = []
    all_final_objectives = []
    all_allocations = []
    
    if preferences is not None:
        n_iter = len(preferences)
    else:
        preferences= np.random.dirichlet(np.ones(env.n_objectives), size=n_iter)
    
    for p_start in range(0, n_iter, n_envs):
        p_end = min(p_start + n_envs, n_iter)
        batch_preferences = preferences[p_start:p_end]
        
        for i, env_i in enumerate(vec_env.envs):
            if i < len(batch_preferences):
                env_i.fix_weights(batch_preferences[i])
        
        obs = vec_env.reset()
        
        done = [False] * len(batch_preferences)
        while not all(done):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, infos = vec_env.step(action)

            for i, info in enumerate(infos):
                if i >= len(batch_preferences):
                    continue
                if 'episode' in info:
                    total_reward_components = info['episode']['r']
                    final_reward_components = info['episode']['final_r']
                    all_objectives.append(total_reward_components)
                    all_final_objectives.append(final_reward_components)
                    all_allocations.append(info['final_allocation'].copy())

    all_objectives, all_final_objectives = np.array(all_objectives), np.array(all_final_objectives)
    
    return all_final_objectives, all_allocations
