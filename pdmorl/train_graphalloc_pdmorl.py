#!/usr/bin/env python3
"""Train PD-MORL (MO-DDQN-HER) on GraphAllocBench problems."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque, namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Ensure the repository root (one level up from this file) is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdmorl.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    OrderingPolicyAdapter,
    TrainingConfig,
    ensure_pymoo_factory,
    ensure_numpy_copy_patch,
)
from graphallocbench.evaluation import utils as eval_utils  # noqa: E402
from graphallocbench.evaluation.inference import run_experiments  # noqa: E402
from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402

# Locate the cloned PD-MORL repository (cloned under pdmorl/external by default)
DEFAULT_PD_MORL_ROOT = SCRIPT_DIR / "external" / "PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm"


def _extend_sys_path(pdmorl_root: Path) -> None:
    if not pdmorl_root.exists():
        raise FileNotFoundError(
            f"Could not find PD-MORL repo at {pdmorl_root}. "
            "Clone https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm "
            "under pdmorl/external or pass --pdmorl-root."
        )
    if str(pdmorl_root) not in sys.path:
        sys.path.insert(0, str(pdmorl_root))


# Import PD-MORL modules after extending sys.path

def _import_pdmorl_modules():  # pragma: no cover - thin wrapper around imports
    ensure_numpy_copy_patch()
    ensure_pymoo_factory()
    from lib.models.networks import MO_DDQN  # type: ignore
    from lib.common_ptan import actions as ptan_actions  # type: ignore
    from lib.common_ptan import agent as ptan_agent  # type: ignore

    return MO_DDQN, ptan_actions, ptan_agent


Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "terminal", "preference"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, experience: Experience) -> None:
        self.storage.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        idxs = np.random.choice(len(self.storage), batch_size, replace=True)
        return [self.storage[i] for i in idxs]


def sample_preference(rng: np.random.Generator, n_objectives: int, alpha: float) -> np.ndarray:
    pref = rng.dirichlet(np.ones(n_objectives) * alpha).astype(np.float32)
    return pref / pref.sum()


def make_algo_namespace(env: GraphAllocDiscreteEnv, args: argparse.Namespace) -> SimpleNamespace:
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if args.device.startswith("cuda") and not use_cuda:
        print("CUDA requested but not available. Falling back to CPU.")
    device = torch.device("cuda" if use_cuda else "cpu")
    algo_args = SimpleNamespace(
        scenario_name=f"graphalloc-{Path(args.config).stem}",
        cuda=use_cuda,
        device=device,
        obs_shape=env.observation_space.shape[0],
        action_shape=env.action_space.n,
        reward_size=env.n_objectives,
        weight_num=args.weight_batch,
        tau=args.tau,
        lr=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        layer_N=args.layer_n,
        hidden_size=args.hidden_size,
    )
    return algo_args, device


def prepare_preferences(args: argparse.Namespace, n_objectives: int) -> np.ndarray:
    grid = np.arange(0, 1 + args.pref_grid_step, args.pref_grid_step)
    mesh = np.array(np.meshgrid(*([grid] * n_objectives))).T.reshape(-1, n_objectives)
    mesh = mesh[np.isclose(mesh.sum(axis=1), 1.0)]
    if len(mesh) == 0:
        mesh = np.eye(n_objectives, dtype=np.float32)
    return mesh.astype(np.float32)


def evaluate_agent(
    agent,
    config_path: str,
    preferences: np.ndarray,
    device: torch.device,
    seed: int,
) -> dict:
    ordering_env = CityPlannerEnv(config_path)
    ordering_adapter = OrderingPolicyAdapter(agent, device, ordering_env.demand_count)
    # Use official evaluation helper so we get final-step objectives under the
    # exact max_steps specified by the problem config (e.g., problem_0.yml).
    final_objectives, _ = run_experiments(
        env=ordering_env,
        model=ordering_adapter,
        preferences=preferences,
        deterministic=True,
        random_seed=seed,
    )
    objectives = np.asarray(final_objectives, dtype=np.float32)
    hv = float(eval_utils.calculate_hypervolume(objectives))
    pnd = float(eval_utils.calculate_non_dominated(objectives))
    ordering = float(
        eval_utils.calculate_ordering_score(
            env=ordering_env,
            model=ordering_adapter,
            n_iter=5,
            n_intervals=5,
            random_seed=seed,
        )
    )
    return {
        "hypervolume": hv,
        "percent_non_dominated": pnd,
        "ordering_score": ordering,
        "objectives": objectives,
    }


def save_checkpoint(model, save_dir: Path, step: int, args: argparse.Namespace) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"ddqn_step_{step}.pt"
    torch.save({"state_dict": model.state_dict(), "train_args": vars(args)}, checkpoint_path)
    return checkpoint_path


def train(args: argparse.Namespace) -> None:
    _extend_sys_path(Path(args.pdmorl_root))
    MO_DDQN, ptan_actions, ptan_agent = _import_pdmorl_modules()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = GraphAllocDiscreteEnv(args.config)
    algo_args, device = make_algo_namespace(env, args)
    net = MO_DDQN(algo_args)
    action_selector = ptan_actions.EpsilonGreedyActionSelector(epsilon=args.epsilon_start)
    agent = ptan_agent.MO_DDQN_HER(net, action_selector, device, algo_args)

    def _normalize_preferences(prefs):
        if isinstance(prefs, torch.Tensor):
            data = prefs.detach().cpu().numpy()
        else:
            data = np.asarray(prefs, dtype=np.float32)
        norms = np.linalg.norm(data, ord=1, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return data / norms

    agent.interp = _normalize_preferences

    pref_grid = prepare_preferences(args, env.n_objectives)

    buffer = ReplayBuffer(args.buffer_size)
    writer = SummaryWriter(args.log_dir)
    env.set_episode_preference(sample_preference(rng, env.n_objectives, args.dirichlet_alpha))
    state = env.reset()
    episode_reward = np.zeros(env.n_objectives, dtype=np.float32)
    episode_len = 0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        if state is None:
            state = env.reset()
        pref = env.current_preference
        pref_tensor = torch.as_tensor(pref, dtype=torch.float32, device=device)
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = int(agent(state, pref_tensor, deterministic=False))
        next_state, reward_vec, done, info = env.step(action)
        buffer.add(Experience(state, action, reward_vec, next_state, done, pref.copy()))
        state = next_state
        episode_reward += reward_vec
        episode_len += 1

        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            agent.learn(batch, writer)

        frac = min(1.0, step / max(1, args.epsilon_decay))
        action_selector.epsilon = args.epsilon_start + frac * (args.epsilon_final - args.epsilon_start)

        if done or episode_len >= env.max_episode_steps:
            writer.add_scalar("train/episode_length", episode_len, step)
            for idx, value in enumerate(episode_reward):
                writer.add_scalar(f"train/objective_{idx}", value, step)
            env.set_episode_preference(sample_preference(rng, env.n_objectives, args.dirichlet_alpha))
            state = env.reset()
            episode_reward[:] = 0
            episode_len = 0

        if step % args.eval_interval == 0:
            metrics = evaluate_agent(agent, args.config, pref_grid, device, args.seed)
            writer.add_scalar("eval/hypervolume", metrics["hypervolume"], step)
            writer.add_scalar("eval/percent_non_dominated", metrics["percent_non_dominated"], step)
            writer.add_scalar("eval/ordering_score", metrics["ordering_score"], step)
            save_path = save_checkpoint(agent.net, Path(args.save_dir), step, args)
            serializable_metrics = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()
            }
            with open(Path(args.save_dir) / f"metrics_step_{step}.json", "w", encoding="utf-8") as fh:
                json.dump({"step": step, **serializable_metrics, "checkpoint": str(save_path)}, fh, indent=2)
            print(
                f"[step={step}] HV={metrics['hypervolume']:.4f} | "
                f"PND={metrics['percent_non_dominated']:.4f} | "
                f"Order={metrics['ordering_score']:.4f}"
            )

    elapsed_min = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_min:.2f} minutes.")
    final_path = save_checkpoint(agent.net, Path(args.save_dir), args.total_steps, args)
    print(f"Final checkpoint saved to {final_path}")
    writer.close()


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "graphallocbench" / "config" / "problems" / "problem_0.yml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to a GraphAlloc problem config.")
    parser.add_argument("--pdmorl-root", type=str, default=str(DEFAULT_PD_MORL_ROOT), help="Path to the PD-MORL repository root.")
    parser.add_argument("--total-steps", type=int, default=TrainingConfig.total_steps)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--buffer-size", type=int, default=TrainingConfig.buffer_size)
    parser.add_argument("--start-steps", type=int, default=TrainingConfig.start_steps)
    parser.add_argument("--eval-interval", type=int, default=TrainingConfig.eval_interval)
    parser.add_argument("--epsilon-start", type=float, default=TrainingConfig.epsilon_start)
    parser.add_argument("--epsilon-final", type=float, default=TrainingConfig.epsilon_final)
    parser.add_argument("--epsilon-decay", type=int, default=TrainingConfig.epsilon_decay_steps)
    parser.add_argument("--gamma", type=float, default=TrainingConfig.gamma)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.lr)
    parser.add_argument("--tau", type=float, default=TrainingConfig.tau)
    parser.add_argument("--weight-batch", type=int, default=TrainingConfig.weight_batch_size)
    parser.add_argument("--dirichlet-alpha", type=float, default=TrainingConfig.dirichlet_alpha)
    parser.add_argument("--pref-grid-step", type=float, default=TrainingConfig.pref_grid_step)
    parser.add_argument("--layer-n", type=int, default=2, help="Hidden layer count for MO-DDQN.")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden width for MO-DDQN layers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--save-dir", type=str, default=str(SCRIPT_DIR / "checkpoints"))
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    Path(parsed_args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)
    train(parsed_args)
