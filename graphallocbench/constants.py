"""Global constants for GraphAllocBench.

Separated so users can override after import if desired.
"""

from stable_baselines3 import PPO
import torch

# Primary RL algorithm (can be swapped by advanced users)
RL_Model = PPO

ALLOWED_GPUS, ALLOWED_DEVICES = None, None

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    preferred_gpus = [0, 1, 2, 3, 4]
    ALLOWED_GPUS = [i for i in preferred_gpus if i < gpu_count]
    if not ALLOWED_GPUS:
        ALLOWED_GPUS = [0] if gpu_count > 0 else []
else:
    ALLOWED_GPUS = []

if not ALLOWED_GPUS:
    ALLOWED_DEVICES = ["cpu"]
else:
    ALLOWED_DEVICES = [f"cuda:{i}" for i in ALLOWED_GPUS]

# Vectorized env batch size (used in evaluation batching)
VECENV_BATCH_SIZE = 32

# Device for model loading in evaluation utilities
INFERENCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "RL_Model",
    "ALLOWED_GPUS",
    "ALLOWED_DEVICES",
    "VECENV_BATCH_SIZE",
    "INFERENCE_DEVICE",
]
