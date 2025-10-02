import os
import yaml
import numpy as np
import wandb
from types import SimpleNamespace
from city_env.env_model import ResourceManagementEnv
from city_env.utils import WandbTrainingCallback
from city_env.architectures import architectures
from stable_baselines3 import PPO
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import torch
import random
from constants import ALLOWED_DEVICES

warnings.filterwarnings("ignore", category=UserWarning)



def train(training_config_path, project_name='GraphAllocBench-v2', device='cuda', seed = 42):
    with open(training_config_path, "r") as file:
        training_config = yaml.safe_load(file)
    
    config_path = f"config/problems/{training_config['env_name']}.yml"
    env = ResourceManagementEnv(config_path)
    env.set_scalarization_method(training_config["scalarization_method"], smoothness=training_config['smoothness'])
    save_path = f"models/{project_name}/arch-{training_config['architecture_idx']}/{training_config['env_name']}/{seed}"
    env.spec = SimpleNamespace(id='ResourceManagementEnv-v0')
    
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
    
    # total_params = sum(p.numel() for p in model.policy.parameters())
    # print(f"Total number of parameters in the model: {total_params}")
    
    model.learn(total_timesteps=training_config['total_timesteps'], 
                callback=WandbTrainingCallback(save_path))

def train_wrapper(config_path, project_name, device, seed=42):
    """Wrapper function that properly initializes CUDA in each process"""
    try:
        # Set seeds FIRST, before any other operations
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize CUDA properly in each process
        if 'cuda' in device:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                print(f"Process {os.getpid()}: CUDA not available, falling back to CPU")
                device = 'cpu'
            else:
                device_idx = int(device.split(':')[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"Process {os.getpid()}: Device {device} not available, using cuda:0")
                    device = 'cuda:0'
                    device_idx = 0
                
                # Set the device for this process
                torch.cuda.set_device(device_idx)
                
                # Set CUDA seed AFTER setting device
                torch.cuda.manual_seed(seed)
                # torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
                
                # Clear any cached memory
                torch.cuda.empty_cache()
                
                # Test device functionality
                try:
                    test_tensor = torch.tensor([1.0], device=device)
                    print(f"Process {os.getpid()}: Successfully initialized {device} with seed {seed}")
                    del test_tensor
                except Exception as e:
                    print(f"Process {os.getpid()}: Error testing device {device}: {e}")
                    device = 'cpu'
        else:
            print(f"Process {os.getpid()}: Using CPU with seed {seed}")
        
        # Ensure deterministic behavior (though this may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Now run the actual training
        train(config_path, project_name, device, seed)
        
    except Exception as e:
        print(f"Process {os.getpid()}: Error in train_wrapper: {e}")
        import traceback
        traceback.print_exc()

def train_parallel(train_config_paths: list, project_name='GraphAllocBench-v2', seeds=list(range(0, 5))):
    """Run training in parallel using a ProcessPoolExecutor with spawn context.

    This mirrors the approach used in run_finetune_parallel.py which showed
    better throughput for model.learn() by using a process pool.
    """

    # Use spawn method to avoid CUDA context sharing issues
    mp.set_start_method('spawn', force=True)

    # Validate available devices
    if torch.cuda.is_available():
        available_devices = []
        for device in ALLOWED_DEVICES:
            try:
                device_idx = int(device.split(':')[1])
            except Exception:
                continue
            if device_idx < torch.cuda.device_count():
                available_devices.append(device)
            else:
                print(f"Warning: {device} not available (only {torch.cuda.device_count()} CUDA devices found)")

        if not available_devices:
            print("No valid CUDA devices found, using CPU")
            available_devices = ['cpu']
    else:
        print("CUDA not available, using CPU")
        available_devices = ['cpu']

    print(f"Using devices: {available_devices}")
    print(f"Using seeds: {seeds}")

    # Create combinations of config paths and seeds
    config_seed_pairs = [(config_path, seed)
                         for config_path in train_config_paths
                         for seed in seeds]

    total_runs = len(config_seed_pairs)
    print(f"Total training runs: {total_runs} (configs: {len(train_config_paths)}, seeds per config: {len(seeds)})")

    # Choose reasonable max_workers. We want behavior similar to run_finetune_parallel
    # which uses a ProcessPoolExecutor default and can run many workers concurrently.
    # Use CPU count as an upper bound so multiple workers can share GPUs when desired
    # (this matches user-observed faster throughput). Don't exceed total_runs.
    cpu_count = os.cpu_count() or 1
    if available_devices == ['cpu']:
        max_workers = min(total_runs, cpu_count)
    else:
        # allow up to either the number of GPUs or the CPU count (whichever is larger)
        # so we don't artificially limit concurrency to the number of GPUs
        max_workers = min(total_runs, max(len(available_devices), cpu_count))

    # Use spawn context for ProcessPoolExecutor to match mp start method
    ctx = mp.get_context('spawn')

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit tasks, cycling devices across jobs
        for i, (config_path, seed) in enumerate(config_seed_pairs):
            device = list(itertools.islice(itertools.cycle(available_devices), i, i+1))[0]
            config_name = os.path.basename(config_path).replace('.yml', '')
            print(f"Submitting job {i+1}/{total_runs}: {config_name} seed={seed} device={device}")
            futures.append(executor.submit(train_wrapper, config_path, project_name, device, seed))

        # Wait for completion and report
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                print(f"Completed {i}/{total_runs}")
            except Exception as e:
                print(f"A training job failed: {e}")

    print("All training tasks completed!")

# Additional utility function for debugging
def check_cuda_setup():
    """Check CUDA setup and available devices"""
    print("=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print("=== End CUDA Check ===")

if __name__ == "__main__":
    # Run CUDA check before training
    check_cuda_setup()