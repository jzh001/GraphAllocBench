#!/usr/bin/env python3
"""Evaluate an Envelope Q-Learning checkpoint on GraphAllocBench metrics."""
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

from envelope.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    GraphAllocEnvelopeCQN,
    EnvelopePolicyAdapter,
    ensure_pymoo_factory,
    ensure_numpy_copy_patch,
    ensure_numpy_float_patch,
)
from envelope.train_graphalloc_envelope import (  # noqa: E402
    _extend_sys_path,
    _import_envelope_modules,
    _create_patched_agent_class,
    _build_meta_args,
    _evaluate_agent,
)
from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402

DEFAULT_ENVELOPE_ROOT = SCRIPT_DIR / "external" / "MORL"
DATA_DIR = SCRIPT_DIR / "data"


def _prepare_preferences(step: float, n_objectives: int, n_partitions: int = 12) -> np.ndarray:
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions).astype(np.float32)


def build_agent(args: argparse.Namespace, env: GraphAllocDiscreteEnv, checkpoint_path: Path):
    """Load a saved checkpoint and return a ready-to-evaluate (non-training) agent."""
    _extend_sys_path(Path(args.envelope_root))
    MetaAgent_cls, get_new_model = _import_envelope_modules()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint.get("arch", {})
    state_size = arch.get("state_size", env.observation_space.shape[0])
    action_size = arch.get("action_size", env.action_space.n)
    reward_size = arch.get("reward_size", env.n_objectives)
    hidden_size = arch.get("hidden_size", 256)
    n_layers = arch.get("n_layers", 3)

    model = GraphAllocEnvelopeCQN(state_size, action_size, reward_size, hidden_size, n_layers)

    # Minimal meta_args for evaluation (no decay needed, epsilon=0)
    meta_args = SimpleNamespace(
        gamma=0.99,
        epsilon=0.0,
        epsilon_decay=False,
        episode_num=1,
        mem_size=1,
        batch_size=1,
        weight_num=1,
        beta=0.01,
        homotopy=False,
        optimizer="Adam",
        lr=1e-3,
        update_freq=9999,
    )
    AgentCls = _create_patched_agent_class(MetaAgent_cls)
    agent = AgentCls(model, meta_args, is_train=False)
    agent.model_.load_state_dict(checkpoint["state_dict"])
    agent.model.load_state_dict(checkpoint["state_dict"])
    return agent


def _write_metrics_csv(
    args: argparse.Namespace,
    config_name: str,
    seed: int,
    metrics: dict,
    checkpoint_path: Path,
) -> None:
    csv_path = Path(args.csv_output) if args.csv_output else DATA_DIR / "envelope_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem", "config", "checkpoint",
        "hypervolume", "normalized_hv", "percent_non_dominated", "ordering_score",
        "pref_grid_step", "seed", "timestamp",
    ]
    row = {
        "problem": config_name,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "hypervolume": float(metrics["hypervolume"]),
        "normalized_hv": float(metrics["normalized_hypervolume"]),
        "percent_non_dominated": float(metrics["percent_non_dominated"]),
        "ordering_score": float(metrics["ordering_score"]),
        "pref_grid_step": args.pref_grid_step,
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat(),
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _resolve_checkpoint(base_path: str, seed: int) -> Path:
    candidate = base_path.replace("{seed}", str(seed)) if "{seed}" in base_path else base_path
    path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for seed {seed} not found at {path}")
    return path


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "graphallocbench" / "config" / "problems" / "problem_0.yml"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--checkpoint", required=True, help="Path (or {seed} template) to checkpoint.")
    p.add_argument("--envelope-root", default=str(DEFAULT_ENVELOPE_ROOT))
    p.add_argument("--pref-grid-step", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", default="", help="Optional JSON output path.")
    p.add_argument("--csv-output", default="", help="Optional CSV output path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = GraphAllocDiscreteEnv(args.config)
    prefs = _prepare_preferences(args.pref_grid_step, env.n_objectives)
    seed_list = args.seeds if args.seeds else [args.seed]
    config_name = Path(args.config).stem
    results = []

    for seed in seed_list:
        ckpt_path = _resolve_checkpoint(args.checkpoint, seed)
        agent = build_agent(args, env, ckpt_path)
        metrics = _evaluate_agent(agent, args.config, prefs, seed)
        serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
        results.append({"seed": seed, **serial})
        _write_metrics_csv(args, config_name, seed, metrics, ckpt_path)
        print(
            f"[seed={seed}] HV={metrics['hypervolume']:.4f} | "
            f"NormHV={metrics['normalized_hypervolume']:.4f} | "
            f"PND={metrics['percent_non_dominated']:.4f} | "
            f"Order={metrics['ordering_score']:.4f}"
        )

    seed_label = f"seeds_{'_'.join(map(str, seed_list))}" if len(seed_list) > 1 else f"seed_{seed_list[0]}"
    default_json = (DATA_DIR / "evaluation" / f"{config_name}_{seed_label}.json").resolve()
    out_path = Path(args.output) if args.output else default_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_keys = ["hypervolume", "normalized_hypervolume", "percent_non_dominated", "ordering_score"]
    avg = {k: float(np.mean([r[k] for r in results])) for k in summary_keys if results}
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "pref_grid_step": args.pref_grid_step,
        "seeds": seed_list,
        "results": results,
        "average_metrics": avg,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved aggregated results to {out_path}")
    if avg:
        print("Average | " + " | ".join(f"{k}={v:.4f}" for k, v in avg.items()))
