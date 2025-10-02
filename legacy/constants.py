##### CONSTANTS #####
"""
This file contains global configurations, including the model type and hardware-based constants.

Training hyperparameters are defined separately in YAML files under train_config/
"""

from stable_baselines3 import PPO
import torch

RL_Model = PPO # PPO is best supported by this repository

ALLOWED_GPUS, ALLOWED_DEVICES = None, None

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    preferred_gpus = [0, 1, 2, 3, 4]
    # Use preferred GPUs if available
    ALLOWED_GPUS = [i for i in preferred_gpus if i < gpu_count]
    if not ALLOWED_GPUS:
        # Fallback to GPU 0 if no preferred GPUs are available
        ALLOWED_GPUS = [0] if gpu_count > 0 else []
else:
    ALLOWED_GPUS = []

if not ALLOWED_GPUS:
    ALLOWED_DEVICES = ["cpu"]
else:
    ALLOWED_DEVICES = [f"cuda:{i}" for i in ALLOWED_GPUS]

VECENV_BATCH_SIZE = 32 # tune this to speed up inference
INFERENCE_DEVICE = 'cuda'
