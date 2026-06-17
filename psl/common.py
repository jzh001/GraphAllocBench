"""Shared helpers for running PSL (Pareto Set Learning) on GraphAllocBench.

PSL-MORL (Liu et al., 2025, arXiv:2501.06773) learns a *hypernetwork* phi(w)
that maps a preference / decomposition weight vector ``w`` to (part of) the
parameters of a policy network. A single trained hypernetwork therefore yields a
preference-conditioned policy for *any* continuous preference at inference time,
which is exactly the PCPL paradigm GraphAllocBench targets.

The paper is algorithm-agnostic and instantiates the framework with DDQN for
discrete action spaces, which is what GraphAllocBench exposes through the
:class:`~pdmorl.common.GraphAllocDiscreteEnv` wrapper. We therefore implement a
scalar-Q Double-DQN whose final (head) layer weights are produced by the
hypernetwork from the preference vector, while a shared trunk encodes the state.

We deliberately reuse the existing baseline plumbing:

* :class:`pdmorl.common.GraphAllocDiscreteEnv` -- flat obs + discrete actions +
  vector rewards.
* :func:`pdmorl.common.flatten_city_observation` /
  :func:`pdmorl.common.action_index_to_multidiscrete`.
* :class:`pdmorl.common.OrderingPolicyAdapter` -- exposes a callable agent via the
  SB3-style ``.predict`` API for the standard evaluation pipeline. A
  :class:`PSLAgent` is callable with the same ``(flat_obs, pref, deterministic)``
  signature, so it plugs straight in.
* :func:`graphallocbench.city_env.scalarize.scalarize` -- linear and Smooth
  Tchebycheff scalarizations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graphallocbench.city_env.scalarize import scalarize

# Re-export the shared wrapper helpers so PSL code can import everything from one
# place, and so the evaluation adapter is available without duplication.
from pdmorl.common import (  # noqa: F401
    GraphAllocDiscreteEnv,
    OrderingPolicyAdapter,
    action_index_to_multidiscrete,
    ensure_numpy_copy_patch,
    ensure_pymoo_factory,
    flatten_city_observation,
    multidiscrete_to_action_index,
)


# --------------------------------------------------------------------------- #
# Scalarization                                                               #
# --------------------------------------------------------------------------- #
def scalarize_psl(
    reward_vec: np.ndarray,
    preference: np.ndarray,
    method: str = "linear",
    ideal_point: Optional[np.ndarray] = None,
    smoothness: float = 0.01,
) -> float:
    """Scalarize a multi-objective reward vector under a preference.

    ``linear`` reproduces the PSL paper's linearized utility ``u = w . J`` and
    needs no ideal point. ``smooth_tchebycheff`` matches PCPL-PPO and requires a
    (running) ideal-point estimate; the nadir is the origin because all
    GraphAllocBench objectives are non-negative by construction.
    """
    reward_vec = np.asarray(reward_vec, dtype=np.float64)
    preference = np.asarray(preference, dtype=np.float64)
    if method == "linear":
        return float(np.sum(preference * reward_vec))
    if ideal_point is None:
        # Fall back gracefully before any ideal-point estimate exists.
        ideal_point = np.maximum(reward_vec, 1e-3)
    return float(
        scalarize(
            weights=preference,
            objectives_components=reward_vec,
            ideal_point=np.asarray(ideal_point, dtype=np.float64),
            nadir_point=np.zeros_like(reward_vec),
            scalarization_method=method,
            normalize=True,
            smoothness=smoothness,
        )
    )


# --------------------------------------------------------------------------- #
# Hypernetwork Q-network                                                       #
# --------------------------------------------------------------------------- #
class PSLHyperQNet(nn.Module):
    """Preference-conditioned scalar Q-network.

    A shared trunk ``theta_1`` encodes the (flat) state. A hypernetwork
    ``phi(w)`` produces the weights/bias of the final linear head ``theta_2``
    from the preference vector, so different preferences yield different policies
    while sharing the state encoder -- the partial-parameter scheme described in
    PSL-MORL.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_objectives: int,
        hidden_dim: int = 128,
        hyper_hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.hidden_dim = hidden_dim

        # Shared state encoder (theta_1).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Hypernetwork phi(w): preference -> flattened head parameters (theta_2).
        self._head_w_size = hidden_dim * n_actions
        self._head_b_size = n_actions
        self.hyper = nn.Sequential(
            nn.Linear(n_objectives, hyper_hidden_dim),
            nn.SiLU(),
            nn.Linear(hyper_hidden_dim, self._head_w_size + self._head_b_size),
        )

    def forward(self, obs: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        """Return scalar Q-values of shape ``[B, n_actions]``.

        ``obs``: ``[B, obs_dim]``; ``pref``: ``[B, n_objectives]``.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if pref.dim() == 1:
            pref = pref.unsqueeze(0)
        batch = obs.shape[0]

        h = self.trunk(obs)  # [B, hidden]
        params = self.hyper(pref)  # [B, hidden*n_actions + n_actions]
        w = params[:, : self._head_w_size].view(batch, self.hidden_dim, self.n_actions)
        b = params[:, self._head_w_size :]  # [B, n_actions]
        q = torch.bmm(h.unsqueeze(1), w).squeeze(1) + b  # [B, n_actions]
        return q


# --------------------------------------------------------------------------- #
# Replay buffer                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: np.ndarray
    next_state: np.ndarray
    done: bool
    preference: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.stack([t.state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.stack([t.reward for t in batch]).astype(np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        prefs = np.stack([t.preference for t in batch]).astype(np.float32)
        return states, actions, rewards, next_states, dones, prefs

    def __len__(self) -> int:
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #
class PSLAgent:
    """Hypernetwork Double-DQN agent.

    Callable as ``agent(flat_obs, pref_tensor, deterministic=True) -> action_idx``
    so it works directly with :class:`pdmorl.common.OrderingPolicyAdapter` for
    evaluation.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_objectives: int,
        device: torch.device,
        hidden_dim: int = 128,
        hyper_hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        scalarization: str = "linear",
        smoothness: float = 0.01,
    ):
        self.device = device
        self.n_actions = n_actions
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.tau = tau
        self.scalarization = scalarization
        self.smoothness = smoothness

        self.online = PSLHyperQNet(
            obs_dim, n_actions, n_objectives, hidden_dim, hyper_hidden_dim
        ).to(device)
        self.target = PSLHyperQNet(
            obs_dim, n_actions, n_objectives, hidden_dim, hyper_hidden_dim
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        # Running ideal-point estimate (per-objective max) for Tchebycheff.
        self.ideal = np.full(n_objectives, 1e-3, dtype=np.float64)

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def __call__(self, flat_obs, pref, deterministic: bool = True) -> int:
        obs_t = torch.as_tensor(np.asarray(flat_obs, dtype=np.float32), device=self.device)
        pref_t = torch.as_tensor(np.asarray(pref, dtype=np.float32), device=self.device)
        q = self.online(obs_t, pref_t)  # [1, n_actions]
        return int(torch.argmax(q, dim=1).item())

    def act_epsilon_greedy(self, flat_obs, pref, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        return self(flat_obs, pref, deterministic=True)

    # -- ideal-point tracking ---------------------------------------------- #
    def update_ideal(self, reward_vec: np.ndarray) -> None:
        self.ideal = np.maximum(self.ideal, np.asarray(reward_vec, dtype=np.float64))

    # -- learning ----------------------------------------------------------- #
    def _scalarize_batch(self, rewards: np.ndarray, prefs: np.ndarray) -> np.ndarray:
        if self.scalarization == "linear":
            return np.sum(prefs * rewards, axis=1)
        out = np.empty(rewards.shape[0], dtype=np.float64)
        for i in range(rewards.shape[0]):
            out[i] = scalarize_psl(
                rewards[i],
                prefs[i],
                method=self.scalarization,
                ideal_point=self.ideal,
                smoothness=self.smoothness,
            )
        return out

    def learn(self, buffer: ReplayBuffer, batch_size: int) -> float:
        states, actions, rewards, next_states, dones, prefs = buffer.sample(batch_size)

        scalar_r = self._scalarize_batch(rewards, prefs).astype(np.float32)

        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        prefs_t = torch.as_tensor(prefs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(scalar_r, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        # Q(s, w, a)
        q = self.online(states_t, prefs_t).gather(1, actions_t).squeeze(1)

        # Double-DQN target: online net selects the next action, target evaluates.
        with torch.no_grad():
            next_actions = torch.argmax(self.online(next_states_t, prefs_t), dim=1, keepdim=True)
            next_q = self.target(next_states_t, prefs_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # Soft target update.
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * op.data)

        return float(loss.item())

    # -- checkpoint --------------------------------------------------------- #
    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "ideal": self.ideal,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.ideal = np.asarray(state.get("ideal", self.ideal), dtype=np.float64)


@dataclass
class PSLTrainingConfig:
    total_steps: int = 1_000_000
    batch_size: int = 64
    buffer_size: int = 50_000
    start_steps: int = 1_000
    learn_every: int = 1
    eval_interval: int = 100_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 100_000
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    hidden_size: int = 128
    hyper_hidden_size: int = 64
    dirichlet_alpha: float = 1.0
    scalarization: str = "linear"
    smoothness: float = 0.01
    n_partitions: int = 12
