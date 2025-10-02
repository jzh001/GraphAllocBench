import os
import yaml
import wandb
import numpy as np
import random
import torch
import pandas as pd
import subprocess
from typing import Dict

# Fallback to ensure relative imports work when executed as a script (not via -m)
if __package__ in (None, ""):
    import sys, pathlib
    pkg_parent = pathlib.Path(__file__).resolve().parents[1].parent  # repo root containing graphallocbench
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))
    __package__ = "graphallocbench.train_utils"

from ..city_env.env_model import CityPlannerEnv
from ..city_env.utils import WandbTrainingCallback
from ..city_env.architectures import architectures
from ..constants import RL_Model, ALLOWED_DEVICES
from ..evaluation.stats import calculate_stats

PROJECT_NAME_DEFAULT = "GraphAllocBench-v2"
PROJECT_NAME_ARCH_DEFAULT = "GraphAllocBench-GNN-v1-1"
ALLOWED_GPUS_DEFAULT = [0]
ARCH_RUNS_DEFAULT = {0: 2, 1: 2, 2: 5}


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _launch_agent_process(script_path: str, extra_args: list[str], env: Dict[str, str]):
    # Always invoke via module path so relative imports resolve
    return subprocess.Popen(["python", "-m", "graphallocbench.train_utils.sweep", *extra_args], env=env)


def sweep(sweep_config_path='sweep_config/hyperparam_search.yml', env_name='problem_0'):
    gpu_id = int(os.environ.get("SWEEP_GPU_ID", -1))
    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
    sweep_config['parameters']['env_name']['value'] = env_name
    wandb.agent(os.environ["SWEEP_ID"], function=lambda: _train_agent(gpu_id))


def sweep_all(env_names: Dict[str, int],
              sweep_config_path='sweep_config/hyperparam_search.yml',
              project_name: str = PROJECT_NAME_DEFAULT,
              allowed_gpus=None):
    if allowed_gpus is None:
        allowed_gpus = ALLOWED_GPUS_DEFAULT
    processes = []
    try:
        api = wandb.Api()
    except Exception:
        api = None
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_ORGANIZATION") or os.environ.get("WANDB_ORG")
    project_path = f"{entity}/{project_name}" if entity else project_name
    for env_name in env_names:
        with open(sweep_config_path, "r") as file:
            sweep_config = yaml.safe_load(file)
        sweep_config['parameters']['env_name']['value'] = env_name
        sweep_id = None
        if api is not None:
            try:
                proj = api.project(project_name, entity=entity) if entity else api.project(project_name)
                for s in proj.sweeps():
                    try:
                        params = s.config.get('parameters', {}) if s.config else {}
                        env_param = params.get('env_name', {})
                        existing_val = env_param.get('value') if isinstance(env_param, dict) else env_param
                        if existing_val == env_name:
                            sweep_id = s.id
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        for gpu_id in allowed_gpus:
            env = os.environ.copy()
            env["SWEEP_GPU_ID"] = str(gpu_id)
            env["SWEEP_ID"] = sweep_id
            env["WANDB_PROJECT"] = project_name
            if entity:
                env["WANDB_ENTITY"] = entity
            for _ in range(env_names[env_name]):
                p = _launch_agent_process(__file__, ["--sweep", "--env_name", env_name, "--sweep_config_path", sweep_config_path], env)
                processes.append((env_name, gpu_id, p))
    for env_name, gpu_id, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"Sweep failed for env {env_name} on GPU {gpu_id} with exit code {ret}")


def sweep_arch(sweep_config_path='sweep_config/hyperparam_search.yml', env_name='problem_0', arch_idx=0):
    gpu_id = int(os.environ.get("SWEEP_GPU_ID", -1))
    with open(sweep_config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
    sweep_config['parameters']['env_name']['value'] = env_name
    sweep_config['parameters']['architecture_idx']['value'] = arch_idx
    wandb.agent(os.environ["SWEEP_ID"], function=lambda: _train_agent(gpu_id))


def sweep_arch_all(env_names: Dict[str, int],
                   sweep_config_path='sweep_config/hyperparam_search.yml',
                   project_name: str = PROJECT_NAME_ARCH_DEFAULT,
                   arch_runs: Dict[int, int] | None = None,
                   allowed_gpus=None):
    if arch_runs is None:
        arch_runs = ARCH_RUNS_DEFAULT
    if allowed_gpus is None:
        allowed_gpus = ALLOWED_GPUS_DEFAULT
    processes = []
    try:
        api = wandb.Api()
    except Exception:
        api = None
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_ORGANIZATION") or os.environ.get("WANDB_ORG")
    for env_name in env_names:
        for arch_idx in range(len(architectures)):
            with open(sweep_config_path, "r") as file:
                sweep_config = yaml.safe_load(file)
            sweep_config['parameters']['env_name']['value'] = env_name
            sweep_config['parameters']['architecture_idx']['value'] = arch_idx
            sweep_id = None
            if api is not None:
                try:
                    proj = api.project(project_name, entity=entity) if entity else api.project(project_name)
                    for s in proj.sweeps():
                        try:
                            params = s.config.get('parameters', {}) if s.config else {}
                            env_param = params.get('env_name', {})
                            arch_param = params.get('architecture_idx', {})
                            existing_env = env_param.get('value') if isinstance(env_param, dict) else env_param
                            existing_arch = arch_param.get('value') if isinstance(arch_param, dict) else arch_param
                            if existing_env == env_name and int(existing_arch) == int(arch_idx):
                                sweep_id = s.id
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
            if sweep_id is None:
                sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
            for gpu_id in allowed_gpus:
                env = os.environ.copy()
                env["SWEEP_GPU_ID"] = str(gpu_id)
                env["SWEEP_ID"] = sweep_id
                env["WANDB_PROJECT"] = project_name
                if entity:
                    env["WANDB_ENTITY"] = entity
                num_runs = arch_runs.get(arch_idx, 1) * env_names[env_name]
                for _ in range(num_runs):
                    p = _launch_agent_process(__file__, ["--sweep_arch", "--env_name", env_name, "--arch_idx", str(arch_idx), "--sweep_config_path", sweep_config_path], env)
                    processes.append((env_name, arch_idx, gpu_id, p))
    for env_name, arch_idx, gpu_id, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"Sweep failed for env {env_name} arch {arch_idx} on GPU {gpu_id} with exit code {ret}")


def _train_agent(gpu_id=None):
    wandb.init()
    config = wandb.config
    set_seeds(config['seed'])
    # Resolve problem config path robustly (works whether invoked via -m or script, any CWD)
    from pathlib import Path
    def _resolve_problem_config(name: str):
        candidate_names = [
            Path(f'config/problems/{name}.yml'),  # original expectation (repo root)
            Path(__file__).resolve().parents[1].parent / 'config' / 'problems' / f'{name}.yml',  # inside installed package
        ]
        for c in candidate_names:
            if c.is_file():
                return str(c)
        # Last resort: search upward for graphallocbench/config/problems
        start = Path.cwd()
        for p in [start] + list(start.parents):
            alt = p / 'graphallocbench' / 'config' / 'problems' / f'{name}.yml'
            if alt.is_file():
                return str(alt)
        raise FileNotFoundError(f'Could not locate problem config for {name}. Looked in: ' + ', '.join(map(str, candidate_names)))

    env_cfg_path = _resolve_problem_config(config['env_name'])
    env = CityPlannerEnv(config_path=env_cfg_path)
    env.set_scalarization_method(config["scalarization_method"], smoothness=config['smoothness'])
    device_str = f"cuda:{gpu_id}" if gpu_id is not None and gpu_id >= 0 else (ALLOWED_DEVICES[0] if torch.cuda.is_available() else 'cpu')
    model = RL_Model(
        "MultiInputPolicy",
        device=device_str,
        env=env,
        verbose=0,
        ent_coef=config['entropy'],
        policy_kwargs=architectures[config['architecture_idx']],
        learning_rate=config['lr'],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        target_kl=config["target_kl"],
        clip_range=config["clip_range"],
        seed=config['seed']
    )
    model.learn(total_timesteps=config['total_timesteps'], callback=WandbTrainingCallback(save_path=None))
    stats = calculate_stats(env, model)
    wandb.log(stats)
    stats.update({key: config[key] for key in config.keys()})
    stats['wandb_run_name'] = wandb.run.name
    stats_df = pd.DataFrame([stats])
    stats_dir = f'data/{os.environ.get("WANDB_PROJECT", PROJECT_NAME_DEFAULT)}'
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f'stats_{config["env_name"]}.csv')
    stats_df.to_csv(stats_path, mode='a', header=not os.path.exists(stats_path), index=False)
    print(f"{config['env_name']} stats", stats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep_arch', action='store_true')
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--arch_idx', type=int, default=0)
    parser.add_argument('--sweep_config_path', type=str, default='sweep_config/hyperparam_search.yml')
    args = parser.parse_args()
    if args.sweep:
        sweep(args.sweep_config_path, args.env_name)
    elif args.sweep_arch:
        sweep_arch(args.sweep_config_path, args.env_name, args.arch_idx)
    else:
        print("Specify --sweep or --sweep_arch for agent mode, or use provided higher level functions.")
