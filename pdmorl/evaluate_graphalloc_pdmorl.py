#!/usr/bin/env python3
"""Evaluate a PD-MORL checkpoint on GraphAllocBench metrics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdmorl.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    OrderingPolicyAdapter,
    ensure_pymoo_factory,
    ensure_numpy_copy_patch,
)
from graphallocbench.evaluation import utils as eval_utils  # noqa: E402
from graphallocbench.evaluation.inference import run_experiments  # noqa: E402
from graphallocbench.evaluation.analytical import get_analytical_objectives  # noqa: E402
from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402

DEFAULT_PD_MORL_ROOT = SCRIPT_DIR / "external" / "PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm"
DATA_DIR = SCRIPT_DIR / "data"
REFERENCE_DIRS = [
    DATA_DIR / "GraphAllocBench-v2",
    REPO_ROOT / "graphallocbench" / "examples" / "data" / "GraphAllocBench-v2",
]


def _extend_sys_path(pdmorl_root: Path) -> None:
    if not pdmorl_root.exists():
        raise FileNotFoundError(
            f"Could not find PD-MORL repo at {pdmorl_root}. "
            "Clone https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm "
            "under pdmorl/external or pass --pdmorl-root."
        )
    if str(pdmorl_root) not in sys.path:
        sys.path.insert(0, str(pdmorl_root))


def _import_pdmorl_modules():  # pragma: no cover
    ensure_numpy_copy_patch()
    ensure_pymoo_factory()
    from lib.models.networks import MO_DDQN  # type: ignore
    from lib.common_ptan import actions as ptan_actions  # type: ignore
    from lib.common_ptan import agent as ptan_agent  # type: ignore

    return MO_DDQN, ptan_actions, ptan_agent


def make_algo_namespace(env: GraphAllocDiscreteEnv, args: argparse.Namespace) -> tuple[SimpleNamespace, torch.device]:
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
        weight_num=1,
        tau=args.tau,
        lr=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        layer_N=args.layer_n,
        hidden_size=args.hidden_size,
    )
    return algo_args, device


def prepare_preferences(step: float, n_objectives: int, n_partitions: int = 12) -> np.ndarray:
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions).astype(np.float32)


def build_agent(args: argparse.Namespace, env: GraphAllocDiscreteEnv, checkpoint_path: Path):
    _extend_sys_path(Path(args.pdmorl_root))
    MO_DDQN, ptan_actions, ptan_agent = _import_pdmorl_modules()
    algo_args, device = make_algo_namespace(env, args)
    net = MO_DDQN(algo_args)
    action_selector = ptan_actions.EpsilonGreedyActionSelector(epsilon=0.0)
    agent = ptan_agent.MO_DDQN_HER(net, action_selector, device, algo_args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    agent.net.load_state_dict(state_dict)
    agent.tgt_net.load_state_dict(state_dict)

    def _normalize_preferences(prefs):
        if isinstance(prefs, torch.Tensor):
            data = prefs.detach().cpu().numpy()
        else:
            data = np.asarray(prefs, dtype=np.float32)
        norms = np.linalg.norm(data, ord=1, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return data / norms

    agent.interp = _normalize_preferences
    return agent, device


def evaluate(agent, config_path: str, preferences: np.ndarray, device: torch.device, seed: int) -> dict:
    ordering_env = CityPlannerEnv(config_path)
    ordering_adapter = OrderingPolicyAdapter(agent, device, ordering_env.demand_count)
    final_objectives, _ = run_experiments(
        env=ordering_env,
        model=ordering_adapter,
        preferences=preferences,
        deterministic=True,
        random_seed=42,
    )
    objectives = np.asarray(final_objectives, dtype=np.float32)
    hv = float(eval_utils.calculate_hypervolume(objectives))
    ideal_objectives = (
        get_analytical_objectives(env=ordering_env) if ordering_env.demand_count < 6 else None
    )
    ideal_hv = float(eval_utils.calculate_hypervolume(ideal_objectives)) if ideal_objectives is not None else 0.0
    normalized_hv = hv / ideal_hv if ideal_hv > 0 else 0.0
    pnd = float(eval_utils.calculate_non_dominated(objectives))
    ordering = float(
        eval_utils.calculate_ordering_score(
            env=ordering_env,
            model=ordering_adapter,
            random_seed=42,
        )
    )
    return {
        "hypervolume": hv,
        "normalized_hypervolume": normalized_hv,
        "percent_non_dominated": pnd,
        "ordering_score": ordering,
        "objectives": objectives,
    }



def _write_metrics_csv(
    args: argparse.Namespace,
    config_name: str,
    seed: int,
    metrics: dict,
    checkpoint_path: Path,
) -> None:
    csv_path = Path(args.csv_output) if args.csv_output else DATA_DIR / "pdmorl_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    hv = float(metrics["hypervolume"])
    fieldnames = [
        "problem",
        "config",
        "checkpoint",
        "hypervolume",
        "normalized_hv",
        "percent_non_dominated",
        "ordering_score",
        "pref_grid_step",
        "tau",
        "learning_rate",
        "gamma",
        "batch_size",
        "seed",
        "timestamp",
    ]
    row = {
        "problem": config_name,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "hypervolume": hv,
        "normalized_hv": float(metrics["normalized_hypervolume"]),
        "percent_non_dominated": float(metrics["percent_non_dominated"]),
        "ordering_score": float(metrics["ordering_score"]),
        "pref_grid_step": args.pref_grid_step,
        "tau": args.tau,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat(),
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _resolve_checkpoint_path(base_path: str, seed: int) -> Path:
    if "{seed}" in base_path:
        candidate = base_path.replace("{seed}", str(seed))
    else:
        candidate = base_path
    path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for seed {seed} not found at {path}")
    return path


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "graphallocbench" / "config" / "problems" / "problem_0.yml"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default=str(default_config), help="Problem configuration path.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path (or template with {seed}) to a saved PD-MORL checkpoint.",
    )
    parser.add_argument("--pdmorl-root", type=str, default=str(DEFAULT_PD_MORL_ROOT))
    parser.add_argument("--pref-grid-step", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0, help="Legacy single-seed flag (used when --seeds is omitted).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="List of seeds to evaluate (overrides --seed).",
    )
    default_output = DATA_DIR / "evaluation"
    parser.add_argument("--output", type=str, default="", help="Optional custom JSON output path.")
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-n", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--csv-output", type=str, default="", help="Optional path for the CSV metrics file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = GraphAllocDiscreteEnv(args.config)
    prefs = prepare_preferences(args.pref_grid_step, env.n_objectives)
    seed_list = args.seeds if args.seeds else [args.seed]
    results = []
    config_name = Path(args.config).stem
    for seed in seed_list:
        checkpoint_path = _resolve_checkpoint_path(args.checkpoint, seed)
        agent, device = build_agent(args, env, checkpoint_path)
        metrics = evaluate(agent, args.config, prefs, device, seed)
        serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
        results.append({"seed": seed, **serializable})
        _write_metrics_csv(args, config_name, seed, metrics, checkpoint_path)
        print(
            f"[seed={seed}] Hypervolume={metrics['hypervolume']:.4f} | "
            f"NormHV={metrics['normalized_hypervolume']:.4f} | "
            f"PND={metrics['percent_non_dominated']:.4f} | "
            f"Order={metrics['ordering_score']:.4f}"
        )

    seed_label = (
        f"seeds_{'_'.join(map(str, seed_list))}"
        if args.seeds
        else f"seed_{seed_list[0]}"
    )
    default_json = (DATA_DIR / "evaluation" / f"{config_name}_{seed_label}.json").resolve()
    out_path = Path(args.output) if args.output else default_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "pref_grid_step": args.pref_grid_step,
        "seeds": seed_list,
        "results": results,
    }
    summary_keys = [
        "hypervolume",
        "normalized_hypervolume",
        "percent_non_dominated",
        "ordering_score",
    ]
    avg_metrics = {
        key: float(np.mean([result[key] for result in results]))
        for key in summary_keys
        if results
    }
    if avg_metrics:
        payload["average_metrics"] = avg_metrics
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved aggregated results to {out_path} and appended rows to the CSV metrics file")
    if avg_metrics:
        print(
            "Average metrics across seeds | "
            + " | ".join(
                f"{key}={value:.4f}" for key, value in avg_metrics.items()
            )
        )
