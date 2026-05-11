"""Shared helpers for running Envelope Q-Learning on GraphAllocBench."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse env helpers from pdmorl to avoid duplication.
from pdmorl.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    flatten_city_observation,
    action_index_to_multidiscrete,
    ensure_pymoo_factory,
    ensure_numpy_copy_patch,
)

__all__ = [
    "GraphAllocDiscreteEnv",
    "flatten_city_observation",
    "action_index_to_multidiscrete",
    "ensure_pymoo_factory",
    "ensure_numpy_copy_patch",
    "ensure_numpy_float_patch",
    "GraphAllocEnvelopeCQN",
    "EnvelopePolicyAdapter",
    "EnvelopeTrainingConfig",
]


class GraphAllocEnvelopeCQN(nn.Module):
    """Fixed-width Q-network for Envelope Q-Learning on GraphAllocBench.

    Replaces the original EnvelopeLinearCQN whose hidden widths scale as
    multiples of (state_size + reward_size), producing models with 1–17M
    parameters.  This version uses a fixed hidden_size (default 256), matching
    PD-MORL's MO_DDQN architecture (~200K params across all problems).

    Interface is identical to EnvelopeLinearCQN:
      forward(state, preference, w_num=1) -> (hq, q)
        hq : (s_num, reward_size)   – envelope target
        q  : (batch, action_size, reward_size)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        reward_size: int,
        hidden_size: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.hidden_size = hidden_size

        inp = state_size + reward_size
        layers: list[nn.Module] = [nn.Linear(inp, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, action_size * reward_size))
        self.net = nn.Sequential(*layers)

    def H(self, Q: torch.Tensor, w: torch.Tensor, s_num: int, w_num: int) -> torch.Tensor:
        """Envelope operator (bool-mask, compatible with PyTorch >= 2.0)."""
        _LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        mask_idx = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num) for i in range(s_num)]
        ).type(_LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size)[mask_idx].view(-1, self.reward_size)
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1).view(-1, self.reward_size)
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).view(-1)
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        bool_mask = torch.zeros(prod.size(), dtype=torch.bool, device=prod.device)
        bool_mask.scatter_(1, inds.unsqueeze(1), True)
        bool_mask = bool_mask.view(-1, 1).repeat(1, self.reward_size)
        return reQ_ext.masked_select(bool_mask).view(-1, self.reward_size)

    def forward(
        self,
        state: torch.Tensor,
        preference: torch.Tensor,
        w_num: int = 1,
    ):
        s_num = int(preference.size(0) / w_num)
        x = torch.cat((state, preference), dim=1)
        q = self.net(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)
        hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)
        return hq, q


def ensure_numpy_float_patch() -> None:
    """Add np.float alias removed in NumPy >= 1.24 (needed by envelope/meta.py)."""
    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined]


class EnvelopePolicyAdapter:
    """Adapter exposing an Envelope MetaAgent via the SB3-style predict API.

    The GraphAllocBench evaluation utilities (run_experiments,
    calculate_ordering_score) call model.predict(obs_dict, deterministic=True)
    where obs_dict is a batched dict from a DummyVecEnv.  This adapter bridges
    that interface to MetaAgent.act().
    """

    def __init__(self, agent, demand_count: int):
        self.agent = agent
        self.demand_count = demand_count
        self._use_cuda = torch.cuda.is_available()

    def predict(self, obs, deterministic: bool = True):
        if not isinstance(obs, dict):
            raise TypeError("CityPlanner observations must be a dict (from DummyVecEnv).")
        num_envs = obs["prefs"].shape[0]
        actions = []
        for idx in range(num_envs):
            single_obs = {k: obs[k][idx] for k in obs}
            flat_obs = flatten_city_observation(single_obs)
            pref = np.asarray(single_obs["prefs"], dtype=np.float32)
            pref_tensor = torch.as_tensor(pref, dtype=torch.float32)
            if self._use_cuda:
                pref_tensor = pref_tensor.cuda()
            action_idx = self.agent.act(flat_obs, pref_tensor)
            actions.append(action_index_to_multidiscrete(action_idx, self.demand_count))
        return np.stack(actions), None


@dataclass
class EnvelopeTrainingConfig:
    """Default hyperparameters for Envelope Q-Learning on GraphAllocBench."""
    total_steps: int = 1_000_000
    mem_size: int = 10_000
    batch_size: int = 256
    eval_interval: int = 100_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon: float = 0.5
    epsilon_decay: bool = True
    weight_num: int = 32
    beta: float = 0.01
    homotopy: bool = True
    update_freq: int = 100
    optimizer: str = "Adam"
    dirichlet_alpha: float = 1.0
    pref_grid_step: float = 0.05
    hidden_size: int = 256
    n_layers: int = 3
