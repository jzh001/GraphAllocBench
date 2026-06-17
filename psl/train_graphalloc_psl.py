#!/usr/bin/env python3
"""Train a PSL (Pareto Set Learning) baseline on GraphAllocBench problems.

This is a from-scratch reimplementation of PSL-MORL (Liu et al., 2025,
arXiv:2501.06773) -- the paper released no official code. The framework learns a
hypernetwork ``phi(w)`` that maps a preference vector to (part of) the policy
parameters, yielding a single preference-conditioned model usable for any
continuous preference at inference. We instantiate it with the paper's discrete
backbone (Double-DQN) on top of the existing ``GraphAllocDiscreteEnv`` wrapper.

The training objective follows the paper: ``max_phi  E_{w ~ Lambda} [u(J, w)]``
where ``u`` is the scalarized utility. ``--scalarization linear`` reproduces the
paper's linearized utility ``u = w . J`` (default); ``smooth_tchebycheff`` matches
PCPL-PPO and is provided for parity on non-convex fronts.

Example:
    uv run python psl/train_graphalloc_psl.py \
        --config graphallocbench/config/problems/problem_0.yml \
        --seeds 0 --total-steps 1000000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure the repository root (one level up from this file) is importable.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psl.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    OrderingPolicyAdapter,
    PSLAgent,
    PSLTrainingConfig,
    ReplayBuffer,
    Transition,
    ensure_numpy_copy_patch,
    ensure_pymoo_factory,
)
from graphallocbench.evaluation import utils as eval_utils  # noqa: E402
from graphallocbench.evaluation.inference import run_experiments  # noqa: E402
from graphallocbench.evaluation.analytical import get_analytical_objectives  # noqa: E402
from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def sample_preference(rng: np.random.Generator, n_objectives: int, alpha: float) -> np.ndarray:
    pref = rng.dirichlet(np.ones(n_objectives) * alpha).astype(np.float32)
    return pref / pref.sum()


def prepare_preferences(n_objectives: int, n_partitions: int) -> np.ndarray:
    from pymoo.util.ref_dirs import get_reference_directions

    return get_reference_directions(
        "das-dennis", n_objectives, n_partitions=n_partitions
    ).astype(np.float32)


def evaluate_agent(
    agent: PSLAgent,
    config_path: str,
    preferences: np.ndarray,
    device: torch.device,
) -> dict:
    """Compute HV / normalized HV / PNDS / OS using the official eval pipeline."""
    ordering_env = CityPlannerEnv(config_path)
    adapter = OrderingPolicyAdapter(agent, device, ordering_env.demand_count)
    final_objectives, _ = run_experiments(
        env=ordering_env,
        model=adapter,
        preferences=preferences,
        deterministic=True,
        random_seed=42,
    )
    objectives = np.asarray(final_objectives, dtype=np.float32)
    hv = float(eval_utils.calculate_hypervolume(objectives))
    ideal_objectives = (
        get_analytical_objectives(env=ordering_env) if ordering_env.demand_count < 6 else None
    )
    ideal_hv = (
        float(eval_utils.calculate_hypervolume(ideal_objectives))
        if ideal_objectives is not None
        else 0.0
    )
    normalized_hv = hv / ideal_hv if ideal_hv > 0 else 0.0
    pnd = float(eval_utils.calculate_non_dominated(objectives))
    ordering = float(
        eval_utils.calculate_ordering_score(env=ordering_env, model=adapter, random_seed=42)
    )
    return {
        "hypervolume": hv,
        "normalized_hypervolume": normalized_hv,
        "percent_non_dominated": pnd,
        "ordering_score": ordering,
        "objectives": objectives,
    }


def save_checkpoint(agent: PSLAgent, save_dir: Path, step: int, problem: str, seed: int, args) -> Path:
    ckpt_dir = save_dir / problem / f"seed-{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"psl_step_{step}.pt"
    torch.save({"state_dict": agent.state_dict(), "train_args": vars(args)}, path)
    return path


def train_single_seed(args: argparse.Namespace, seed: int) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")
    device = torch.device("cuda" if use_cuda else "cpu")

    env = GraphAllocDiscreteEnv(args.config)
    agent = PSLAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        n_objectives=env.n_objectives,
        device=device,
        hidden_dim=args.hidden_size,
        hyper_hidden_dim=args.hyper_hidden_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        scalarization=args.scalarization,
        smoothness=args.smoothness,
    )
    buffer = ReplayBuffer(args.buffer_size)
    pref_grid = prepare_preferences(env.n_objectives, args.n_partitions)

    problem_name = Path(args.config).stem
    if getattr(args, "save_logs", True):
        writer_dir = Path(args.log_dir) / problem_name / f"seed-{seed}"
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir))
    else:
        writer = _NoOpWriter()

    pref = sample_preference(rng, env.n_objectives, args.dirichlet_alpha)
    env.set_episode_preference(pref)
    state = env.reset()
    episode_reward = np.zeros(env.n_objectives, dtype=np.float32)
    episode_len = 0
    start_time = time.time()

    for step in tqdm(
        range(1, args.total_steps + 1),
        desc=f"seed {seed}",
        unit="step",
        leave=False,
        total=args.total_steps,
    ):
        if state is None:
            pref = sample_preference(rng, env.n_objectives, args.dirichlet_alpha)
            env.set_episode_preference(pref)
            state = env.reset()

        frac = min(1.0, step / max(1, args.epsilon_decay))
        epsilon = args.epsilon_start + frac * (args.epsilon_final - args.epsilon_start)

        if step < args.start_steps:
            action = int(env.action_space.sample())
        else:
            action = agent.act_epsilon_greedy(state, pref, epsilon)

        next_state, reward_vec, done, info = env.step(action)
        agent.update_ideal(reward_vec)
        buffer.push(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=action,
                reward=np.asarray(reward_vec, dtype=np.float32),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
                preference=pref.copy(),
            )
        )

        state = next_state
        episode_reward += reward_vec
        episode_len += 1

        if len(buffer) >= max(args.batch_size, args.start_steps) and step % args.learn_every == 0:
            loss = agent.learn(buffer, args.batch_size)
            writer.add_scalar("train/loss", loss, step)

        if done or episode_len >= env.max_episode_steps:
            writer.add_scalar("train/episode_length", episode_len, step)
            for idx, value in enumerate(episode_reward):
                writer.add_scalar(f"train/objective_{idx}", value, step)
            pref = sample_preference(rng, env.n_objectives, args.dirichlet_alpha)
            env.set_episode_preference(pref)
            state = env.reset()
            episode_reward[:] = 0
            episode_len = 0

        if step % args.eval_interval == 0:
            metrics = evaluate_agent(agent, args.config, pref_grid, device)
            writer.add_scalar("eval/hypervolume", metrics["hypervolume"], step)
            writer.add_scalar("eval/normalized_hypervolume", metrics["normalized_hypervolume"], step)
            writer.add_scalar("eval/percent_non_dominated", metrics["percent_non_dominated"], step)
            writer.add_scalar("eval/ordering_score", metrics["ordering_score"], step)

            out_dir = Path(args.save_dir) / problem_name / f"seed-{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_ref = ""
            if getattr(args, "save_checkpoints", True):
                checkpoint_ref = str(
                    save_checkpoint(agent, Path(args.save_dir), step, problem_name, seed, args)
                )
            serializable = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()
            }
            with open(out_dir / f"metrics_step_{step}.json", "w", encoding="utf-8") as fh:
                json.dump({"step": step, **serializable, "checkpoint": checkpoint_ref}, fh, indent=2)
            tqdm.write(
                f"[seed={seed} step={step}] HV={metrics['hypervolume']:.4f} | "
                f"NormHV={metrics['normalized_hypervolume']:.4f} | "
                f"PND={metrics['percent_non_dominated']:.4f} | "
                f"Order={metrics['ordering_score']:.4f}"
            )

    elapsed_min = (time.time() - start_time) / 60
    tqdm.write(f"[seed={seed}] Training completed in {elapsed_min:.2f} minutes.")
    if getattr(args, "save_checkpoints", True):
        final_path = save_checkpoint(agent, Path(args.save_dir), args.total_steps, problem_name, seed, args)
        tqdm.write(f"[seed={seed}] Final checkpoint saved to {final_path}")
    writer.close()


def train(args: argparse.Namespace) -> None:
    ensure_numpy_copy_patch()
    ensure_pymoo_factory()
    seed_list = args.seeds if args.seeds else [args.seed]
    for seed in seed_list:
        print(f"Starting PSL training for seed {seed} on {Path(args.config).stem}")
        train_single_seed(args, seed)


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "graphallocbench" / "config" / "problems" / "problem_0.yml"
    cfg = PSLTrainingConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=str(default_config),
                        help="Path to a GraphAlloc problem config.")
    parser.add_argument("--scalarization", choices=["linear", "smooth_tchebycheff"],
                        default=cfg.scalarization,
                        help="Utility used for the DDQN reward (linear is faithful to PSL).")
    parser.add_argument("--smoothness", type=float, default=cfg.smoothness,
                        help="Smoothness for smooth_tchebycheff scalarization.")
    parser.add_argument("--total-steps", type=int, default=cfg.total_steps)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--buffer-size", type=int, default=cfg.buffer_size)
    parser.add_argument("--start-steps", type=int, default=cfg.start_steps)
    parser.add_argument("--learn-every", type=int, default=cfg.learn_every)
    parser.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--epsilon-start", type=float, default=cfg.epsilon_start)
    parser.add_argument("--epsilon-final", type=float, default=cfg.epsilon_final)
    parser.add_argument("--epsilon-decay", type=int, default=cfg.epsilon_decay_steps)
    parser.add_argument("--gamma", type=float, default=cfg.gamma)
    parser.add_argument("--learning-rate", type=float, default=cfg.lr)
    parser.add_argument("--tau", type=float, default=cfg.tau)
    parser.add_argument("--hidden-size", type=int, default=cfg.hidden_size)
    parser.add_argument("--hyper-hidden-size", type=int, default=cfg.hyper_hidden_size)
    parser.add_argument("--dirichlet-alpha", type=float, default=cfg.dirichlet_alpha)
    parser.add_argument("--n-partitions", type=int, default=cfg.n_partitions)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Fallback seed when --seeds is empty.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                        help="List of seeds to train (overrides --seed).")
    parser.add_argument("--log-dir", type=str, default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--save-dir", type=str, default=str(SCRIPT_DIR / "checkpoints"))
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    Path(parsed_args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)
    train(parsed_args)
