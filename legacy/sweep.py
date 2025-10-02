import wandb
import yaml
import os
import numpy as np
import random
import torch
import pandas as pd
import subprocess
import argparse

from city_env.env_model import ResourceManagementEnv
from city_env.utils import WandbTrainingCallback
from constants import RL_Model

from evaluation.stats import calculate_stats
from city_env.architectures import architectures

ALLOWED_GPUS = [0,]
PROJECT_NAME = "GraphAllocBench-v2"

def set_seeds(seed):
    """Set seeds for all random number generators to ensure reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def sweep(sweep_config_path = 'sweep_config/hyperparam_search.yml', env_name = 'problem_0'):
    # Only used for agent subprocesses, not for creating sweeps anymore
    gpu_id = int(os.environ.get("SWEEP_GPU_ID", -1))

    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)

    sweep_config['parameters']['env_name']['value'] = env_name
    # print(sweep_config)
    # Only run agent, not sweep creation
    wandb.agent(os.environ["SWEEP_ID"], function=lambda: train(gpu_id))


def sweep_all(env_names, sweep_config_path='sweep_config/hyperparam_search.yml'):
    # For each env_name, create a sweep ONCE, then launch agents (one per GPU) for that sweep
    processes = []
    # Try to initialize the wandb API once to look for existing sweeps
    try:
        api = wandb.Api()
    except Exception:
        api = None

    # Determine project path (prefer explicit WANDB_ENTITY env var when present)
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_ORGANIZATION") or os.environ.get("WANDB_ORG")
    project_path = f"{entity}/{PROJECT_NAME}" if entity else PROJECT_NAME

    for env_name in env_names:
        # Load sweep config and set env_name
        with open(sweep_config_path, "r") as file:
            sweep_config = yaml.safe_load(file)
        sweep_config['parameters']['env_name']['value'] = env_name

        # Try to find an existing sweep with the same env_name parameter and resume it
        sweep_id = None
        if api is not None:
            try:
                # The public Api exposes sweeps via the Project object in this version.
                # Prefer passing project name and entity separately when available.
                proj = None
                try:
                    if entity:
                        proj = api.project(PROJECT_NAME, entity=entity)
                    else:
                        proj = api.project(PROJECT_NAME)
                except Exception:
                    # Fallback: try the combined path string
                    try:
                        proj = api.project(project_path)
                    except Exception:
                        proj = None

                if proj is None:
                    raise RuntimeError(f"Could not find project {project_path}")

                for s in proj.sweeps():
                    try:
                        params = s.config.get('parameters', {}) if s.config else {}
                        env_param = params.get('env_name', {})
                        # env_param may be a dict with a 'value' key or a direct value
                        existing_val = None
                        if isinstance(env_param, dict):
                            existing_val = env_param.get('value')
                        else:
                            existing_val = env_param
                        if existing_val == env_name:
                            sweep_id = s.id
                            print(f"Found existing sweep for {env_name}: {sweep_id}")
                            break
                    except Exception:
                        # Skip problematic entries
                        continue
            except Exception as e:
                print(f"Warning: could not query wandb sweeps for project '{project_path}': {e}")

        if sweep_id is None:
            # No existing sweep found, create a new one
            sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)
            print(f"Sweep created for {env_name}: {sweep_id}")
        else:
            print(f"Resuming sweep for {env_name}: {sweep_id}")

        for gpu_id in ALLOWED_GPUS:
            env = os.environ.copy()
            env["SWEEP_GPU_ID"] = str(gpu_id)
            env["SWEEP_ID"] = sweep_id
            # Ensure agent subprocesses know which WANDB project/entity to use so
            # they can find the sweep without a 404. Prefer explicit env vars,
            # fall back to api.default_entity if available.
            env["WANDB_PROJECT"] = PROJECT_NAME
            if entity:
                env["WANDB_ENTITY"] = entity
            else:
                try:
                    if api is not None and getattr(api, 'default_entity', None):
                        env["WANDB_ENTITY"] = api.default_entity
                except Exception:
                    pass
            for _ in range(env_names[env_name]):
                p = subprocess.Popen([
                    "python", __file__, "--sweep", "--env_name", env_name, "--sweep_config_path", sweep_config_path
                ], env=env)
                processes.append((env_name, gpu_id, p))
        print(f"Running agents for {env_name}")

    # Wait for all processes to finish
    for env_name, gpu_id, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"Sweep failed for env {env_name} on GPU {gpu_id} with exit code {ret}")

def train(gpu_id=None):
    wandb.init()
    config = wandb.config

    # Set all seeds at the beginning of training
    set_seeds(config['seed'])

    env = ResourceManagementEnv(config_path = f'config/problems/{config['env_name']}.yml')
    env.set_scalarization_method(config["scalarization_method"], smoothness=config['smoothness'])

    # Set device string
    device_str = f"cuda:{gpu_id}" if gpu_id is not None and gpu_id >= 0 else "cuda"

    model = RL_Model("MultiInputPolicy", 
            device=device_str, 
            env=env, verbose=0, 
            ent_coef=config['entropy'], 
            policy_kwargs=architectures[config['architecture_idx']], 
            learning_rate=config['lr'],
            n_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
            target_kl=config["target_kl"],
            clip_range=config["clip_range"],
            seed=config['seed']
            )
    
    # total_params = sum(p.numel() for p in model.policy.parameters())
    # print(f"Total number of parameters in the model: {total_params}")

    model.learn(total_timesteps=config['total_timesteps'], callback=WandbTrainingCallback(
        # save_path = f"models/problems/{config['env_name']}/arch{config['architecture_idx']}/{wandb.run.name}",
        # save_freq = 50000
        save_path=None,
        ))

    stats = calculate_stats(env, model)

    wandb.log(stats)

    # Add wandb config to stats
    stats.update({key: config[key] for key in config.keys()})

    stats['wandb_run_name'] = wandb.run.name

    # Create a DataFrame from the stats dictionary
    stats_df = pd.DataFrame([stats])

    # stats_dir = f'data/{wandb.run.sweep_id}'
    stats_dir = f'data/{PROJECT_NAME}'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    stats_path = os.path.join(stats_dir, f'stats_{config["env_name"]}.csv')

    # Append the DataFrame to a CSV file
    stats_df.to_csv(stats_path, mode='a', header=not os.path.exists(stats_path), index=False)

    print(f"{config["env_name"]} stats", stats)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run as a sweep agent')
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--sweep_config_path', type=str, default='sweep_config/hyperparam_search.yml')
    args = parser.parse_args()

    if args.sweep:
        # Run as a sweep agent for a specific env_name
        sweep(sweep_config_path=args.sweep_config_path, env_name=args.env_name)
    else:
        # Only the main process should call sweep_all
        sweep_all({
            "problem_0": 1, 
            "problem_1a": 1,
            "problem_1b": 1,
            "problem_1c": 1,
            "problem_2a": 1,
            "problem_2b": 1,
            "problem_2c": 1,
            "problem_3a": 2,
            "problem_3b": 2,
            "problem_4a": 2,
            "problem_4b": 2,
            "problem_5a": 2,
            "problem_5b": 2,
            "problem_5c": 2,
            "problem_5d": 2,
            "problem_5e": 2,
        })