import os
import yaml
import numpy as np
import wandb
from types import SimpleNamespace
from ..city_env.env_model import CityPlannerEnv
from ..city_env.utils import WandbTrainingCallback
from ..city_env.architectures import architectures
from stable_baselines3 import PPO
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import torch
import random
from ..constants import ALLOWED_DEVICES

warnings.filterwarnings("ignore", category=UserWarning)

def train(training_config_path, project_name='GraphAllocBench-v2', device='cuda', seed = 42):
    with open(training_config_path, "r") as file:
        training_config = yaml.safe_load(file)
    config_path = f"config/problems/{training_config['env_name']}.yml"
    env = CityPlannerEnv(config_path)
    env.set_scalarization_method(training_config["scalarization_method"], smoothness=training_config['smoothness'])
    save_path = f"models/{project_name}/arch-{training_config['architecture_idx']}/{training_config['env_name']}/{seed}"
    env.spec = SimpleNamespace(id='CityPlannerEnv-v0')
    run_name = f"config-{training_config['env_name']}"
    wandb.init(project=run_name)
    model = PPO("MultiInputPolicy",
                device=device,
                env=env, 
                verbose=0,
                ent_coef=training_config['entropy'],
                policy_kwargs=architectures[training_config["architecture_idx"]],
                learning_rate=training_config["lr"],
                n_steps=training_config["n_steps"],
                n_epochs=training_config["n_epochs"],
                target_kl=training_config["target_kl"],
                clip_range=training_config["clip_range"],
                seed = seed
                )
    model.learn(total_timesteps=training_config['total_timesteps'], 
                callback=WandbTrainingCallback(save_path))

def train_wrapper(config_path, project_name, device, seed=42):
    try:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if 'cuda' in device:
            if not torch.cuda.is_available():
                print(f"CUDA not available, falling back to CPU")
                device = 'cpu'
            else:
                device_idx = int(device.split(':')[1])
                if device_idx >= torch.cuda.device_count():
                    device = 'cuda:0'
                    device_idx = 0
                torch.cuda.set_device(device_idx)
                torch.cuda.manual_seed(seed)
                torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train(config_path, project_name, device, seed)
    except Exception as e:
        print(f"Error in train_wrapper: {e}")
        import traceback
        traceback.print_exc()

def train_parallel(train_config_paths: list, project_name='GraphAllocBench-v2', seeds=list(range(0, 5))):
    mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        available_devices = []
        for device in ALLOWED_DEVICES:
            try:
                device_idx = int(device.split(':')[1])
            except Exception:
                continue
            if device_idx < torch.cuda.device_count():
                available_devices.append(device)
        if not available_devices:
            available_devices = ['cpu']
    else:
        available_devices = ['cpu']
    print(f"Using devices: {available_devices}")
    print(f"Using seeds: {seeds}")
    config_seed_pairs = [(config_path, seed)
                         for config_path in train_config_paths
                         for seed in seeds]
    total_runs = len(config_seed_pairs)
    cpu_count = os.cpu_count() or 1
    if available_devices == ['cpu']:
        max_workers = min(total_runs, cpu_count)
    else:
        max_workers = min(total_runs, max(len(available_devices), cpu_count))
    ctx = mp.get_context('spawn')
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for i, (config_path, seed) in enumerate(config_seed_pairs):
            device = list(itertools.islice(itertools.cycle(available_devices), i, i+1))[0]
            futures.append(executor.submit(train_wrapper, config_path, project_name, device, seed))
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                print(f"Completed {i}/{total_runs}")
            except Exception as e:
                print(f"A training job failed: {e}")
    print("All training tasks completed!")
