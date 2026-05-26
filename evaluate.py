#!/usr/bin/env python3
"""Evaluate trained PCPL-PPO checkpoints and write stats CSVs.

Main benchmark (problems 0–5)  →  graphallocbench/examples/data/pcpl_stats.csv
GNN problems (6a/6b/6c)         →  graphallocbench/examples/data/GraphAllocBench-GNN-v1-1/pcpl_stats_gnn.csv

Usage (from repo root):
    python evaluate.py                          # all 16 main problems, seeds 0-4
    python evaluate.py --problems problem_0     # single problem
    python evaluate.py --seeds 0 1 2            # specific seeds
    python evaluate.py --gnn                    # also evaluate GNN problems
    python evaluate.py --gnn --no-main          # GNN only
    python evaluate.py --skip-done              # skip rows already in output CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

CSV_DEFAULT = (
    REPO_ROOT / "graphallocbench" / "examples" / "data"
    / "GraphAllocBench-v2" / "best_hyperparameters.csv"
)
OUTPUT_DEFAULT = REPO_ROOT / "graphallocbench" / "examples" / "data" / "pcpl_stats.csv"

ALL_PROBLEMS = [
    "problem_0",
    "problem_1a", "problem_1b", "problem_1c",
    "problem_2a", "problem_2b", "problem_2c",
    "problem_3a", "problem_3b",
    "problem_4a", "problem_4b",
    "problem_5a", "problem_5b", "problem_5c", "problem_5d", "problem_5e",
]

GNN_PROBLEMS = ["problem_6a", "problem_6b", "problem_6c"]
GNN_ARCHS = [0, 2, 3]
GNN_PROJECT = "GraphAllocBench-GNN-v1-1"
GNN_DEFAULT_STEPS = 3_000_000
GNN_OUTPUT_DEFAULT = (
    REPO_ROOT / "graphallocbench" / "examples" / "data"
    / "GraphAllocBench-GNN-v1-1" / "pcpl_stats_gnn.csv"
)


def _model_path(project: str, arch_idx: int, problem: str, seed: int, steps: int) -> Path:
    return (
        REPO_ROOT / "models" / project
        / f"arch-{arch_idx}" / problem / str(seed)
        / f"model_step_{steps}.zip"
    )


def _config_path(problem: str) -> Path:
    return REPO_ROOT / "graphallocbench" / "config" / "problems" / f"{problem}.yml"


def _already_in_csv(csv_path: Path, problem: str, seed: int, arch_idx: int | None = None) -> bool:
    """Return True if this (problem, seed[, arch_idx]) row is already in the CSV."""
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    mask = (df.get("env_name", df.get("problem", pd.Series(dtype=str))) == problem) & (df["seed"] == seed)
    if arch_idx is not None and "arch_idx" in df.columns:
        mask &= df["arch_idx"] == arch_idx
    return bool(mask.any())


def evaluate_main(args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate the main benchmark (arch-0, problems 0–5) → pcpl_stats.csv."""
    from graphallocbench.city_env.env_model import CityPlannerEnv
    from graphallocbench.evaluation.stats import calculate_stats

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: hyperparameters CSV not found:\n  {csv_path}")

    df_params = pd.read_csv(csv_path)
    problems = args.problems or ALL_PROBLEMS
    df_params = df_params[df_params["env_name"].isin(problems)]
    if df_params.empty:
        print("No matching problems found in CSV — skipping main benchmark evaluation.")
        return pd.DataFrame()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in tqdm(df_params.iterrows(), total=len(df_params), desc="Main benchmark"):
        problem = row["env_name"]
        arch_idx = int(row["architecture_idx"])
        env = CityPlannerEnv(str(_config_path(problem)))

        for seed in args.seeds:
            if args.skip_done and _already_in_csv(output, problem, seed):
                tqdm.write(f"  SKIP (exists): {problem}  seed={seed}")
                continue

            model_path = _model_path(args.project, arch_idx, problem, seed, args.total_steps)
            if not model_path.exists():
                tqdm.write(f"  SKIP (no ckpt): {model_path}")
                continue

            stats = calculate_stats(env=env, model_path=str(model_path), n_partitions=args.n_partitions)
            result = row.to_dict()
            result["seed"] = seed
            result.update(stats)
            rows.append(result)

    if not rows:
        print("No new main-benchmark rows to write.")
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)

    if output.exists() and args.skip_done:
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_df], ignore_index=True).sort_values(
            by=["env_name", "seed"]
        ).reset_index(drop=True)
        combined.to_csv(output, index=False)
    else:
        new_df.sort_values(by=["env_name", "seed"]).reset_index(drop=True).to_csv(output, index=False)

    print(f"Wrote {len(rows)} new rows → {output}")
    return new_df


def evaluate_gnn(args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate GNN problems (6a/6b/6c) across archs → pcpl_stats_gnn.csv."""
    from graphallocbench.city_env.env_model import CityPlannerEnv
    from graphallocbench.evaluation.stats import calculate_stats

    gnn_problems = args.gnn_problems or GNN_PROBLEMS
    gnn_archs = args.gnn_archs

    output = Path(args.gnn_output) if args.gnn_output else GNN_OUTPUT_DEFAULT
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in gnn_problems:
        cfg = str(_config_path(problem))
        env = None  # created once per problem
        for arch_idx in tqdm(gnn_archs, desc=f"GNN {problem}"):
            for seed in args.seeds:
                if args.skip_done and _already_in_csv(output, problem, seed, arch_idx):
                    tqdm.write(f"  SKIP (exists): {problem}  arch={arch_idx}  seed={seed}")
                    continue

                model_path = _model_path(GNN_PROJECT, arch_idx, problem, seed, args.gnn_total_steps)
                if not model_path.exists():
                    tqdm.write(f"  SKIP (no ckpt): {model_path}")
                    continue

                if env is None:
                    env = CityPlannerEnv(cfg)

                stats = calculate_stats(env=env, model_path=str(model_path), n_partitions=args.n_partitions)
                rows.append({
                    "problem": problem,
                    "arch_idx": arch_idx,
                    "seed": seed,
                    **stats,
                })

    if not rows:
        print("No new GNN rows to write.")
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)

    if output.exists() and args.skip_done:
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_df], ignore_index=True).sort_values(
            by=["problem", "arch_idx", "seed"]
        ).reset_index(drop=True)
        combined.to_csv(output, index=False)
    else:
        new_df.sort_values(by=["problem", "arch_idx", "seed"]).reset_index(drop=True).to_csv(output, index=False)

    print(f"Wrote {len(rows)} new GNN rows → {output}")
    return new_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── main benchmark ────────────────────────────────────────────────────────
    p.add_argument("--csv",         default=str(CSV_DEFAULT))
    p.add_argument("--project",     default="GraphAllocBench-v2")
    p.add_argument("--problems",    nargs="*", default=None)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--output",      default=str(OUTPUT_DEFAULT),
                   help="Output CSV for main benchmark (default: graphallocbench/examples/data/pcpl_stats.csv)")

    # ── GNN ───────────────────────────────────────────────────────────────────
    p.add_argument("--gnn",              action="store_true")
    p.add_argument("--gnn-problems",     nargs="*", default=None)
    p.add_argument("--gnn-archs",        nargs="+", type=int, default=GNN_ARCHS)
    p.add_argument("--gnn-total-steps",  type=int, default=GNN_DEFAULT_STEPS)
    p.add_argument("--gnn-output",       default="",
                   help="Output CSV for GNN (default: …/GraphAllocBench-GNN-v1-1/pcpl_stats_gnn.csv)")
    p.add_argument("--no-main",          action="store_true",
                   help="Skip main benchmark evaluation")

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--seeds",          nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--n-partitions",   type=int, default=12,
                   help="Das-Dennis partitions for preference grid (default: 12)")
    p.add_argument("--skip-done",      action="store_true", default=False,
                   help="Skip rows already present in the output CSV")
    p.add_argument("--workers",        type=int, default=None,
                   help="(ignored — evaluation always runs sequentially)")
    p.add_argument("--wandb-mode",     default=None,
                   help="(ignored — evaluation does not use W&B)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_main:
        evaluate_main(args)

    if args.gnn:
        evaluate_gnn(args)

    if args.no_main and not args.gnn:
        print("Nothing to evaluate (--no-main set and --gnn not set).")


if __name__ == "__main__":
    main()
