#!/usr/bin/env python3
"""Launch PCPL-PPO full training from best_hyperparameters.csv.

Reads per-problem hyperparameters from the sweep CSV, writes temporary YAML
configs, and trains all requested problems × seeds with a bounded process pool.

Ctrl-C (or SIGTERM) cleanly terminates all worker processes immediately.

Main benchmark (problems 0–5):
    python train.py                          # all 16 problems, seeds 0-4, 1M steps, 2 workers
    python train.py --workers 4              # run up to 4 jobs in parallel
    python train.py --seeds 0 1 2            # specific seeds
    python train.py --problems problem_0 problem_1a
    python train.py --total-steps 500000
    python train.py --wandb-mode disabled    # skip W&B logging

GNN problems (6a/6b/6c, archs 0/2/3, 3M steps):
    python train.py --gnn                    # all GNN problems × all archs
    python train.py --gnn --gnn-problems problem_6a problem_6b
    python train.py --gnn --gnn-archs 0 2   # specific architectures only

Both together:
    python train.py --gnn --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import yaml
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

CSV_DEFAULT = (
    REPO_ROOT
    / "graphallocbench"
    / "examples"
    / "data"
    / "GraphAllocBench-v2"
    / "best_hyperparameters.csv"
)

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
GNN_CONFIG_DIR = REPO_ROOT / "graphallocbench" / "train_config" / GNN_PROJECT
GNN_DEFAULT_STEPS = 3_000_000

# Module-level pool reference so the signal handler can reach it.
_pool: "mp.pool.Pool | None" = None


def _sigint_handler(sig, frame) -> None:  # noqa: ANN001
    """On Ctrl-C / SIGTERM: terminate all worker processes immediately."""
    print("\n[train.py] Interrupted — terminating all worker processes...", flush=True)
    if _pool is not None:
        _pool.terminate()
        _pool.join()
    sys.exit(130)  # conventional exit code for SIGINT


def _final_checkpoint(project: str, arch_idx: int, problem: str, seed: int, steps: int) -> Path:
    return (
        REPO_ROOT / "models" / project
        / f"arch-{arch_idx}" / problem / str(seed)
        / f"model_step_{steps}.zip"
    )


def _build_config_yaml(row: "pd.Series", total_steps: int, tmpdir: str) -> str:
    cfg = {
        "env_name":             row["env_name"],
        "architecture_idx":     int(row["architecture_idx"]),
        "scalarization_method": row["scalarization_method"],
        "smoothness":           float(row["smoothness"]),
        "lr":                   float(row["lr"]),
        "entropy":              float(row["entropy"]),
        "n_steps":              int(row["n_steps"]),
        "n_epochs":             int(row["n_epochs"]),
        "target_kl":            float(row["target_kl"]),
        "clip_range":           float(row["clip_range"]),
        "total_timesteps":      total_steps,
    }
    path = os.path.join(tmpdir, f"{row['env_name']}.yml")
    with open(path, "w") as fh:
        yaml.dump(cfg, fh)
    return path


# ── worker wrapper (must be top-level for pickle) ────────────────────────────

def _safe_train(args_tuple: tuple) -> tuple:
    """Wraps train_wrapper; returns (cfg_path, seed, project, error_or_None)."""
    cfg_path, project, device, seed = args_tuple
    # Workers inherit the parent's SIGINT handler.  Reset to default so that
    # a stray signal kills the worker cleanly rather than running our handler.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        from graphallocbench.train_utils.general import train_wrapper
        train_wrapper(cfg_path, project, device, seed)
        return (cfg_path, seed, project, None)
    except Exception as exc:  # noqa: BLE001
        return (cfg_path, seed, project, str(exc))


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── main benchmark ────────────────────────────────────────────────────────
    p.add_argument("--csv", default=str(CSV_DEFAULT),
                   help="Path to best_hyperparameters.csv")
    p.add_argument("--project",     default="GraphAllocBench-v2")
    p.add_argument("--problems",    nargs="*", default=None,
                   help="Subset of main-benchmark problems to train (default: all 16)")
    p.add_argument("--total-steps", type=int, default=1_000_000)

    # ── GNN problems ──────────────────────────────────────────────────────────
    p.add_argument("--gnn", action="store_true",
                   help="Train GNN problems (6a/6b/6c) across archs 0/2/3")
    p.add_argument("--gnn-problems", nargs="*", default=None,
                   help="Subset of GNN problems (default: all 3)")
    p.add_argument("--gnn-archs",    nargs="+", type=int, default=GNN_ARCHS,
                   help="GNN architecture indices to train (default: 0 2 3)")
    p.add_argument("--gnn-total-steps", type=int, default=GNN_DEFAULT_STEPS,
                   help="Training steps for GNN problems (default: 3M)")
    p.add_argument("--no-main", action="store_true",
                   help="Skip main-benchmark problems (useful when only --gnn is wanted)")

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--seeds",       nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--workers",     type=int, default=2,
                   help="Max parallel training processes (default: 2)")
    p.add_argument("--skip-done",   action="store_true", default=True,
                   help="Skip runs where the final checkpoint already exists (default: on)")
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    p.add_argument("--wandb-mode",  default=None,
                   choices=["online", "offline", "disabled"],
                   help="Override WANDB_MODE env var")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global _pool

    # Install signal handlers before spawning anything.
    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    args = parse_args()

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    tmpdir = tempfile.mkdtemp(prefix="graphalloc_train_")
    work: list[tuple[str, int, str]] = []   # (yaml_path, seed, project)
    skipped: list[str] = []

    # ── main benchmark (CSV-driven) ───────────────────────────────────────────
    if not args.no_main:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            sys.exit(f"ERROR: hyperparameters CSV not found:\n  {csv_path}")

        df = pd.read_csv(csv_path)
        problems = args.problems or ALL_PROBLEMS
        df = df[df["env_name"].isin(problems)]

        missing = set(problems) - set(df["env_name"])
        if missing:
            print(f"WARNING: no CSV entry for: {sorted(missing)}")

        for _, row in df.iterrows():
            arch = int(row["architecture_idx"])
            problem = row["env_name"]
            done_seeds = [
                s for s in args.seeds
                if args.skip_done
                and _final_checkpoint(args.project, arch, problem, s, args.total_steps).exists()
            ]
            todo_seeds = [s for s in args.seeds if s not in done_seeds]

            if not todo_seeds:
                skipped.append(problem)
                continue

            if done_seeds:
                print(f"  {problem}: skipping seeds {done_seeds}, training {todo_seeds}")

            cfg_path = _build_config_yaml(row, args.total_steps, tmpdir)
            for seed in todo_seeds:
                work.append((cfg_path, seed, args.project))

    # ── GNN problems (YAML-driven) ────────────────────────────────────────────
    if args.gnn:
        gnn_problems = args.gnn_problems or GNN_PROBLEMS
        for problem in gnn_problems:
            for arch_idx in args.gnn_archs:
                yaml_path = GNN_CONFIG_DIR / f"train_{problem}_arch-{arch_idx}.yml"
                if not yaml_path.exists():
                    print(f"WARNING: GNN config not found: {yaml_path}")
                    continue

                done_seeds = [
                    s for s in args.seeds
                    if args.skip_done
                    and _final_checkpoint(GNN_PROJECT, arch_idx, problem, s, args.gnn_total_steps).exists()
                ]
                todo_seeds = [s for s in args.seeds if s not in done_seeds]

                label = f"{problem}/arch-{arch_idx}"
                if not todo_seeds:
                    skipped.append(label)
                    continue

                if done_seeds:
                    print(f"  {label}: skipping seeds {done_seeds}, training {todo_seeds}")

                cfg_path = str(yaml_path)
                if args.gnn_total_steps != GNN_DEFAULT_STEPS:
                    with open(yaml_path) as fh:
                        gnn_cfg = yaml.safe_load(fh)
                    gnn_cfg["total_timesteps"] = args.gnn_total_steps
                    tmp = os.path.join(tmpdir, f"{problem}_arch{arch_idx}.yml")
                    with open(tmp, "w") as fh:
                        yaml.dump(gnn_cfg, fh)
                    cfg_path = tmp

                for seed in todo_seeds:
                    work.append((cfg_path, seed, GNN_PROJECT))

    # ── summary ───────────────────────────────────────────────────────────────
    if skipped:
        print(f"Fully done (skipping): {skipped}\n")

    if not work:
        print("Nothing to train — all checkpoints already exist.")
        return

    total = len(work)
    unique_cfgs = len(set(cfg for cfg, _, _ in work))
    print(f"=== GraphAllocBench PCPL-PPO Training ===")
    print(f"  Runs     : {total}  ({unique_cfgs} config(s) × seeds)")
    print(f"  Workers  : {args.workers}  (at most {args.workers} jobs running at once)")
    if not args.no_main:
        print(f"  Steps    : {args.total_steps:,}  (main benchmark)")
    if args.gnn:
        print(f"  Steps    : {args.gnn_total_steps:,}  (GNN problems)")
    if args.wandb_mode:
        print(f"  W&B mode : {args.wandb_mode}")
    print(f"  (Ctrl-C will terminate all workers immediately)")
    print()

    # ── device ───────────────────────────────────────────────────────────────
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── bounded process pool ──────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    work_args = [(cfg, project, device, seed) for cfg, seed, project in work]

    # maxtasksperchild=1: give every (problem, seed) task a fresh process so
    # wandb / env / torch state from one run can never leak into the next.
    _pool = ctx.Pool(processes=args.workers, maxtasksperchild=1)
    completed = 0
    try:
        for cfg_path, seed, project, exc in _pool.imap_unordered(_safe_train, work_args):
            problem = Path(cfg_path).stem
            completed += 1
            if exc is None:
                print(f"[{completed}/{total}] Done  : {problem}  seed={seed}  project={project}")
            else:
                print(f"[{completed}/{total}] FAILED: {problem}  seed={seed}  project={project}  — {exc}")
        _pool.close()
        _pool.join()
    except KeyboardInterrupt:
        # Shouldn't normally reach here (signal handler exits first),
        # but act as a safety net.
        print("\nInterrupted — terminating workers...")
        _pool.terminate()
        _pool.join()
        sys.exit(130)

    print(f"\nAll {total} runs finished.")


if __name__ == "__main__":
    main()
