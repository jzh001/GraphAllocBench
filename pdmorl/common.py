"""Shared helpers for running PD-MORL on GraphAllocBench."""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import sys

try:  # Prefer gymnasium but fall back to gym if needed
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium not installed in all setups
    import gym  # type: ignore
from gym import spaces

from graphallocbench.city_env.env_model import CityPlannerEnv


def flatten_city_observation(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert the CityPlanner dict observation into a flat float32 vector."""
    allocation = np.asarray(obs["allocation_matrix"], dtype=np.float32).flatten()
    requirements = np.asarray(obs["requirements_matrix"], dtype=np.float32).flatten()
    prefs = np.asarray(obs["prefs"], dtype=np.float32).flatten()
    return np.concatenate([allocation, requirements, prefs]).astype(np.float32)


def action_index_to_multidiscrete(index: int, demand_count: int) -> np.ndarray:
    """Map a discrete action index back to the [3, demand_count] MultiDiscrete action."""
    index = int(index)
    op = index // demand_count
    demand = index % demand_count
    if op >= 2:  # do-nothing branch ignores the demand index
        demand = 0
        op = 2
    return np.array([op, demand], dtype=np.int64)


def multidiscrete_to_action_index(action: Iterable[int], demand_count: int) -> int:
    add_remove, demand = list(action)
    add_remove = int(add_remove)
    demand = int(demand)
    add_remove = max(0, min(add_remove, 2))
    demand = max(0, min(demand, demand_count - 1))
    return add_remove * demand_count + demand


class GraphAllocDiscreteEnv(gym.Env):
    """Wrapper that exposes the CityPlannerEnv via a flat observation and discrete action space."""

    metadata = {"render.modes": []}

    def __init__(self, config_path: str):
        super().__init__()
        self.city_env = CityPlannerEnv(config_path)
        # Ensure we receive vector rewards for MORL
        self.city_env.scalar_rewards = False
        self.city_env.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.city_env.n_objectives,), dtype=np.float32
        )
        self.demand_count = self.city_env.demand_count
        self.action_space = spaces.Discrete(self.demand_count * 3)
        sample_obs, _ = self.city_env.reset()
        flat_obs = flatten_city_observation(sample_obs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32,
        )
        self.current_preference = None

    @property
    def n_objectives(self) -> int:
        return self.city_env.n_objectives

    @property
    def max_episode_steps(self) -> int:
        return self.city_env.max_steps

    def set_episode_preference(self, preference: np.ndarray) -> None:
        pref = np.asarray(preference, dtype=np.float32)
        pref_sum = float(pref.sum())
        if pref_sum <= 0:
            raise ValueError("Preference vector must have positive sum")
        pref /= pref_sum
        self.city_env.fix_weights(pref.tolist())
        self.current_preference = pref

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.city_env.reset(seed=seed)
        if self.current_preference is None:
            uniform = np.ones(self.n_objectives, dtype=np.float32) / self.n_objectives
            self.set_episode_preference(uniform)
        obs, _ = self.city_env.reset()
        flat_obs = flatten_city_observation(obs)
        return flat_obs

    def step(self, action: int):
        env_action = action_index_to_multidiscrete(action, self.demand_count)
        obs, reward_vec, terminated, truncated, info = self.city_env.step(env_action)
        flat_obs = flatten_city_observation(obs)
        done = bool(terminated or truncated)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)
        return flat_obs, reward_vec, done, info

    def render(self, mode="human"):
        return self.city_env.render(mode=mode)


class OrderingPolicyAdapter:
    """Adapter that exposes a PD-MORL agent via the SB3-style predict API."""

    def __init__(self, agent, device: torch.device, demand_count: int):
        self.agent = agent
        self.device = device
        self.demand_count = demand_count

    def predict(self, obs, deterministic: bool = True):
        if not isinstance(obs, dict):
            raise TypeError("CityPlanner observations are expected as a dict when using DummyVecEnv.")
        num_envs = obs["prefs"].shape[0]
        actions = []
        for idx in range(num_envs):
            single_obs = {k: obs[k][idx] for k in obs}
            flat_obs = flatten_city_observation(single_obs)
            pref = single_obs["prefs"]
            pref_tensor = torch.as_tensor(pref, dtype=torch.float32, device=self.device)
            action_idx = int(self.agent(flat_obs, pref_tensor, deterministic=deterministic))
            actions.append(action_index_to_multidiscrete(action_idx, self.demand_count))
        return np.stack(actions), None


_NP_ARRAY_PATCHED = False


def ensure_numpy_copy_patch() -> None:
    """Patch numpy.array(copy=False) to gracefully fall back when unavoidable."""

    global _NP_ARRAY_PATCHED
    if _NP_ARRAY_PATCHED:
        return

    original_array = np.array

    def safe_array(obj, *args, **kwargs):
        copy_flag = kwargs.get("copy", True)
        try:
            return original_array(obj, *args, **kwargs)
        except ValueError as exc:
            if copy_flag is False and "Unable to avoid copy" in str(exc):
                kwargs = dict(kwargs)
                kwargs.pop("copy", None)
                return original_array(obj, *args, **kwargs)
            raise

    np.array = safe_array  # type: ignore[assignment]
    _NP_ARRAY_PATCHED = True


def ensure_pymoo_factory() -> None:
    """Register pymoo.factory.get_performance_indicator when missing."""
    try:
        import pymoo.factory  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    from pymoo.indicators.hv import HV

    module = ModuleType("pymoo.factory")

    def get_performance_indicator(name: str, **kwargs):
        if name.lower() == "hv":
            return HV(**kwargs)
        raise NotImplementedError(
            "Only the 'hv' indicator is implemented in this compatibility shim."
        )

    module.get_performance_indicator = get_performance_indicator  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    batch_size: int = 128
    buffer_size: int = 200_000
    start_steps: int = 2_000
    eval_interval: int = 100_000
    epsilon_start: float = 0.2
    epsilon_final: float = 0.01
    epsilon_decay_steps: int = 50_000
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.02
    weight_batch_size: int = 3
    dirichlet_alpha: float = 0.5
    pref_grid_step: float = 0.05