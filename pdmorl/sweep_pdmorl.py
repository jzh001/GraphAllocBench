#!/usr/bin/env python3
"""Comprehensive hyperparameter search for PD-MORL on GraphAllocBench.

Workflow
--------
Comparison budget (matches PCPL):
  - Final training: step budget chosen by the sweep (500K or 1M), matching the
                    PCPL evaluated checkpoint at 1M steps.
  - Search phase:   total_steps is part of the search space [500_000, 1_000_000];
                    each trial samples its own step budget alongside other hparams.
  - Network:        MO_DDQN with layer_N=3, hidden_size=256 — 5 linear layers at
                    256 width, matching the 5-layer depth of PCPL's feature
                    extractor + policy head (3 layers at 128 + 2 layers at 256).

1. Run random hyperparameter search (one seed per trial) per problem::

       python pdmorl/sweep_pdmorl.py search \\
           --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \\
           --trials 10

2. Inspect the best configurations found::

       python pdmorl/sweep_pdmorl.py report

3. Train 5 seeds with the best per-problem configuration at full step budget::

       python pdmorl/sweep_pdmorl.py train \\
           --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \\
           --seeds 0 1 2 3 4

4. Generate the summary table for inclusion in the paper::

       python pdmorl/sweep_pdmorl.py summarize

The ``search`` phase saves per-trial results under ``pdmorl/data/sweep/`` and
checkpoints under ``pdmorl/checkpoints_sweep/``.  The ``train`` phase writes
final models to the canonical ``pdmorl/checkpoints/`` directory and appends
rows to ``pdmorl/data/pdmorl_stats.csv``, keeping it compatible with the
existing ``evaluate_graphalloc_pdmorl.py`` and ``plot.py`` scripts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm as _tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PD_MORL_ROOT = (
    SCRIPT_DIR
    / "external"
    / "PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm"
)

# ── Problem catalogue ────────────────────────────────────────────────────────

PROBLEM_CONFIGS: Dict[str, str] = {
    name: str(
        REPO_ROOT / "graphallocbench" / "config" / "problems" / f"problem_{name}.yml"
    )
    for name in [
        "0",
        "1a", "1b", "1c",
        "2a", "2b", "2c",
        "3a", "3b",
        "4a", "4b",
        "5a", "5b", "5c", "5d", "5e",
        "6a", "6b", "6c",
    ]
}

DEFAULT_PROBLEMS = [
    "0",
    "1a", "1b", "1c",
    "2a", "2b", "2c",
    "3a", "3b",
    "4a", "4b",
    "5a", "5b", "5c", "5d", "5e",
]

# ── Hyperparameter search space ──────────────────────────────────────────────

SEARCH_SPACE: Dict[str, List] = {
    # All four values produced best configs across problems in prior runs
    "lr":            [1e-4, 3e-4, 6e-4, 1e-3],
    # 0.95 won on 10/16 problems, 0.99 on 6/16 — both necessary
    "gamma":         [0.95, 0.99],
    # 32 won on problems 0 and 4b; 256 won on 5a/5b/5e — full range needed
    "batch_size":    [32, 64, 128, 256],
    # 0.05 won on 1b/2b/4a/5c/5d; 0.01 on 2c/4b/5a — both removed prematurely
    "tau":           [0.001, 0.005, 0.01, 0.05],
    # Episodes are 30-50 steps; 5K-20K covers 100-660 episodes — sufficient diversity
    "buffer_size":   [5_000, 10_000, 20_000],
    "epsilon_start": [0.5, 0.8, 1.0],
    # Small action space (6-15) needs limited exploration; >200K wastes budget
    "epsilon_decay": [100_000, 200_000],
    "weight_batch":  [3, 5, 7],
    "total_steps":   [500_000, 1_000_000],
}

SWEEP_SEARCH_CSV_FIELDNAMES = [
    "problem",
    "trial",
    "seed",
    "search_steps",
    "lr",
    "gamma",
    "batch_size",
    "tau",
    "buffer_size",
    "epsilon_start",
    "epsilon_final",
    "epsilon_decay",
    "weight_batch",
    "total_steps",
    "hypervolume",
    "normalized_hv",
    "percent_non_dominated",
    "ordering_score",
    "timestamp",
]

FULL_TRAIN_CSV_FIELDNAMES = [
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

# ── Paths ────────────────────────────────────────────────────────────────────

SWEEP_DATA_DIR = SCRIPT_DIR / "data" / "sweep"
SWEEP_CKPT_DIR = SCRIPT_DIR / "checkpoints_sweep"
BEST_CONFIGS_PATH = SWEEP_DATA_DIR / "best_configs.json"
CANONICAL_CKPT_DIR = SCRIPT_DIR / "checkpoints"
CANONICAL_CSV = SCRIPT_DIR / "data" / "pdmorl_stats.csv"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_device(requested: str) -> str:
    """Return a device string valid for the MO-DDQN-HER library.

    The upstream PD-MORL library controls device placement via a boolean
    ``cuda`` flag and has no MPS code path.  Only ``cuda`` (when a CUDA GPU is
    present) and ``cpu`` are supported.  Any other value is silently mapped to
    ``cpu`` with a warning printed here so the caller does not have to repeat
    the logic.
    """
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if requested != "cpu":
        print(
            f"Device '{requested}' is not supported by the MO-DDQN-HER library "
            "(only 'cuda' and 'cpu' are). Using CPU."
        )
        return "cpu"
    return "cpu"


def _extend_sys_path(pdmorl_root: Path) -> None:
    if not pdmorl_root.exists():
        raise FileNotFoundError(
            f"Could not find PD-MORL repo at {pdmorl_root}. "
            "Clone https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm "
            "under pdmorl/external or pass --pdmorl-root."
        )
    if str(pdmorl_root) not in sys.path:
        sys.path.insert(0, str(pdmorl_root))


def _import_pdmorl_modules():
    from pdmorl.common import ensure_numpy_copy_patch, ensure_pymoo_factory  # noqa: F401
    ensure_numpy_copy_patch()
    ensure_pymoo_factory()
    from lib.models.networks import MO_DDQN  # type: ignore
    from lib.common_ptan import actions as ptan_actions  # type: ignore
    from lib.common_ptan import agent as ptan_agent  # type: ignore
    from lib.common_ptan import experience as ptan_experience  # type: ignore
    return MO_DDQN, ptan_actions, ptan_agent, ptan_experience


def _sample_hyperparams(rng: random.Random) -> Dict:
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def _build_train_args(
    config_path: str,
    hparams: Dict,
    *,
    total_steps: int,
    eval_interval: int,
    seed: int,
    save_dir: str,
    log_dir: str,
    device: str,
    pdmorl_root: str,
    pref_grid_step: float = 0.05,
    epsilon_final: float = 0.05,
    start_steps: int = 64,
    layer_n: int = 3,
    hidden_size: int = 256,
) -> Namespace:
    """Construct an argparse.Namespace compatible with train_single_seed."""
    return Namespace(
        config=config_path,
        pdmorl_root=pdmorl_root,
        total_steps=total_steps,
        batch_size=hparams["batch_size"],
        buffer_size=hparams["buffer_size"],
        start_steps=start_steps,
        eval_interval=eval_interval,
        epsilon_start=hparams["epsilon_start"],
        epsilon_final=epsilon_final,
        epsilon_decay=hparams["epsilon_decay"],
        gamma=hparams["gamma"],
        learning_rate=hparams["lr"],
        tau=hparams["tau"],
        weight_batch=hparams["weight_batch"],
        dirichlet_alpha=1.0,
        pref_grid_step=pref_grid_step,
        layer_n=layer_n,
        hidden_size=hidden_size,
        device=device,
        seed=seed,
        seeds=[seed],
        log_dir=log_dir,
        save_dir=save_dir,
        save_checkpoints=False,
        save_logs=False,
    )


def _run_training(args: Namespace, seed: int) -> Optional[Dict]:
    """Train a single seed and return the metrics written by the training loop.

    Returns the metrics dict from the saved JSON if evaluation occurred, else None.
    """
    from pdmorl.train_graphalloc_pdmorl import train_single_seed  # noqa: F401

    _extend_sys_path(Path(args.pdmorl_root))
    MO_DDQN, ptan_actions, ptan_agent, ptan_experience = _import_pdmorl_modules()
    train_single_seed(args, seed, MO_DDQN, ptan_actions, ptan_agent, ptan_experience)

    # Retrieve the metrics JSON saved by the training loop at the eval step.
    problem_name = Path(args.config).stem
    metrics_path = (
        Path(args.save_dir)
        / problem_name
        / f"seed-{seed}"
        / f"metrics_step_{args.total_steps}.json"
    )
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _append_search_csv(
    problem: str,
    trial: int,
    seed: int,
    search_steps: int,
    hparams: Dict,
    metrics: Dict,
) -> None:
    csv_path = SWEEP_DATA_DIR / f"{problem}_search.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    row = {
        "problem": problem,
        "trial": trial,
        "seed": seed,
        "search_steps": search_steps,
        "lr": hparams["lr"],
        "gamma": hparams["gamma"],
        "batch_size": hparams["batch_size"],
        "tau": hparams["tau"],
        "buffer_size": hparams["buffer_size"],
        "epsilon_start": hparams["epsilon_start"],
        "epsilon_final": 0.05,
        "epsilon_decay": hparams["epsilon_decay"],
        "weight_batch": hparams["weight_batch"],
        "total_steps": hparams["total_steps"],
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_SEARCH_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_search_results(problem: str) -> List[Dict]:
    csv_path = SWEEP_DATA_DIR / f"{problem}_search.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _best_config_for_problem(rows: List[Dict]) -> Optional[Dict]:
    """Return the row with the highest normalized_hv (or raw hv as fallback)."""
    if not rows:
        return None

    def _score(row: Dict) -> float:
        nhv = float(row.get("normalized_hv", 0.0))
        if nhv > 0:
            return nhv
        return float(row.get("hypervolume", 0.0))

    return max(rows, key=_score)


def _save_best_configs(best: Dict[str, Dict]) -> None:
    SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_CONFIGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)


def _load_best_configs() -> Dict[str, Dict]:
    if not BEST_CONFIGS_PATH.exists():
        return {}
    with open(BEST_CONFIGS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _append_full_train_csv(
    problem: str,
    config_path: str,
    checkpoint_path: str,
    hparams: Dict,
    metrics: Dict,
    seed: int,
    pref_grid_step: float = 0.05,
    csv_path: Optional[Path] = None,
) -> None:
    out = csv_path or CANONICAL_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    row = {
        "problem": problem,
        "config": Path(config_path).name,
        "checkpoint": checkpoint_path,
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "pref_grid_step": pref_grid_step,
        "tau": hparams.get("tau", 0.005),
        "learning_rate": hparams.get("lr", 3e-4),
        "gamma": hparams.get("gamma", 0.99),
        "batch_size": hparams.get("batch_size", 64),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FULL_TRAIN_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Subcommand: search ───────────────────────────────────────────────────────


def _worker_initializer() -> None:
    """Per-worker setup called once when each worker process starts.

    - Ignores SIGINT so that Ctrl+C is caught only by the main process, which
      then terminates workers explicitly and exits cleanly.
    - Redirects stdout and stderr to /dev/null at the OS file-descriptor level.
      This suppresses per-step tqdm bars, tqdm.write() calls, print() calls,
      and any C-level output from PyTorch/libraries — none of which should reach
      the terminal and corrupt the main-process progress bar.  All meaningful
      information (metrics, exceptions) is returned via the result dict.
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)  # redirect stdout (fd 1)
    os.dup2(devnull, 2)  # redirect stderr (fd 2)
    os.close(devnull)


def _run_trial_worker(spec: Dict) -> Dict:
    """Run one hyperparameter trial in a worker process and return its result.

    This function must be a module-level callable so that ProcessPoolExecutor
    can pickle it for dispatch to worker processes.  It never writes to any
    shared file; results are returned to the main process which serialises all
    CSV writes, eliminating race conditions entirely.
    """
    import traceback

    error: Optional[str] = None
    try:
        trial_steps = spec["hparams"]["total_steps"]
        train_args = _build_train_args(
            config_path=spec["config_path"],
            hparams=spec["hparams"],
            total_steps=trial_steps,
            eval_interval=trial_steps,  # single eval at end of trial
            seed=spec["seed"],
            save_dir=spec["save_dir"],
            log_dir=spec["log_dir"],
            device=spec["device"],
            pdmorl_root=spec["pdmorl_root"],
        )
        metrics = _run_training(train_args, seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None

    if metrics is None:
        metrics = {
            "hypervolume": 0.0,
            "normalized_hypervolume": 0.0,
            "percent_non_dominated": 0.0,
            "ordering_score": 0.0,
        }

    return {
        "problem": spec["problem"],
        "trial_idx": spec["trial_idx"],
        "hparams": spec["hparams"],
        "metrics": metrics,
        "error": error,
    }


def _handle_trial_result(result: Dict, args: Namespace) -> None:
    """Persist and print one completed trial result.

    Always called from the main process so CSV appends are sequential and
    require no locking.
    """
    problem = result["problem"]
    trial_idx = result["trial_idx"]
    hparams = result["hparams"]
    metrics = result["metrics"]
    error = result.get("error")

    if error:
        _tqdm.write(f"  [ERROR] problem={problem} trial={trial_idx + 1}:\n{error.rstrip()}")

    _append_search_csv(
        problem=problem,
        trial=trial_idx,
        seed=args.search_seed_trial,
        search_steps=hparams["total_steps"],
        hparams=hparams,
        metrics=metrics,
    )
    _tqdm.write(
        f"  [problem={problem} trial={trial_idx + 1}/{args.trials}] "
        f"HV={metrics.get('hypervolume', 0):.4f} | "
        f"NormHV={metrics.get('normalized_hypervolume', 0):.4f} | "
        f"PNDS={metrics.get('percent_non_dominated', 0):.4f} | "
        f"Order={metrics.get('ordering_score', 0):.4f}"
    )


def cmd_search(args: Namespace) -> None:
    """Run random hyperparameter search across specified problems."""
    _extend_sys_path(Path(args.pdmorl_root))
    device = _resolve_device(args.device)
    problems = args.problems or DEFAULT_PROBLEMS

    # Build the set of already-completed (problem, trial_idx) pairs.  Keying
    # by pair (rather than a per-problem count) means resuming after a parallel
    # run works correctly even when trials finished out of order.
    completed: set = set()
    for problem in problems:
        for row in _load_search_results(problem):
            completed.add((row["problem"], int(row["trial"])))

    # Assemble the full list of pending trial specs, skipping completed ones.
    pending: List[Dict] = []
    for problem in problems:
        config_path = PROBLEM_CONFIGS.get(problem)
        if config_path is None or not Path(config_path).exists():
            print(f"[SKIP] Config not found for problem '{problem}': {config_path}")
            continue
        for trial_idx in range(args.trials):
            if (problem, trial_idx) in completed:
                continue
            trial_rng = random.Random(args.search_seed * 10_000 + trial_idx)
            hparams = _sample_hyperparams(trial_rng)
            pending.append({
                "problem": problem,
                "trial_idx": trial_idx,
                "config_path": config_path,
                "hparams": hparams,
                "search_steps": args.search_steps,
                "seed": args.search_seed_trial,
                "device": device,
                "pdmorl_root": args.pdmorl_root,
                "log_dir": str(
                    SCRIPT_DIR / "runs_sweep" / f"problem_{problem}" / f"trial_{trial_idx}"
                ),
                # Each trial gets its own checkpoint subdirectory so concurrent
                # workers never write to the same path.
                "save_dir": str(SWEEP_CKPT_DIR / f"trial_{trial_idx}"),
            })

    print(
        f"Running hyperparameter search: {len(problems)} problem(s), "
        f"up to {args.trials} trial(s) each, steps sampled per trial from {SEARCH_SPACE['total_steps']}. "
        f"{len(pending)} new trial(s) to run ({args.workers} worker(s))."
    )

    if args.workers == 1:
        # Sequential path — no subprocess overhead, simpler stack traces.
        # Per-step tqdm bars from train_single_seed are visible and useful here.
        for spec in pending:
            result = _run_trial_worker(spec)
            _handle_trial_result(result, args)
    else:
        # Parallel path.
        #
        # Design:
        #   • Workers run in separate processes; all CSV/JSON writes are done by
        #     the main process via _handle_trial_result (no locks needed).
        #   • _worker_initializer disables SIGINT and per-step tqdm in workers
        #     so Ctrl+C is handled cleanly and output is not garbled.
        #   • A single trial-level tqdm bar in the main process replaces the
        #     multiple per-step bars that would otherwise jitter and interleave.
        #   • Ctrl+C cancels pending futures, shuts down the pool, and exits
        #     with code 130 after saving all results that already completed.
        from concurrent.futures import ProcessPoolExecutor, as_completed

        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_initializer,
        )
        futures: Dict = {}
        _interrupted = False
        try:
            for spec in pending:
                futures[pool.submit(_run_trial_worker, spec)] = spec

            not_done: set = set(futures)
            with _tqdm(
                total=len(pending),
                desc="Search trials",
                unit="trial",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(futures):
                    spec = futures[future]
                    not_done.discard(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "problem": spec["problem"],
                            "trial_idx": spec["trial_idx"],
                            "hparams": spec["hparams"],
                            "metrics": {
                                "hypervolume": 0.0,
                                "normalized_hypervolume": 0.0,
                                "percent_non_dominated": 0.0,
                                "ordering_score": 0.0,
                            },
                            "error": str(exc),
                        }
                    _handle_trial_result(result, args)
                    pbar.update(1)
                    if not_done:
                        remaining_problems = sorted(
                            {futures[f]["problem"] for f in not_done}
                        )
                        sample = remaining_problems[:4]
                        ellipsis = "…" if len(remaining_problems) > 4 else ""
                        pbar.set_postfix_str(
                            f"remaining={len(not_done)} "
                            f"[{','.join(sample)}{ellipsis}]"
                        )
        except KeyboardInterrupt:
            _interrupted = True
            _tqdm.write(
                "\nKeyboardInterrupt — terminating workers…"
            )
            # Cancel futures that have not started yet.
            for f in futures:
                f.cancel()
            # Forcefully terminate processes that are actively running a trial.
            # pool._processes is a {pid: Process} dict maintained by the executor.
            try:
                for proc in pool._processes.values():
                    proc.terminate()
            except AttributeError:
                pass
        finally:
            pool.shutdown(wait=False)

        if _interrupted:
            _tqdm.write(
                "Workers terminated. Completed trial results are saved "
                "and resumable (re-run to continue)."
            )
            sys.exit(130)

    # Update best-configs JSON once all trials are done.
    _update_best_configs(problems)
    print(f"\nSearch complete. Best configs saved to {BEST_CONFIGS_PATH}")


def _update_best_configs(problems: List[str]) -> None:
    existing = _load_best_configs()
    for problem in problems:
        rows = _load_search_results(problem)
        best = _best_config_for_problem(rows)
        if best is not None:
            existing[problem] = best
    _save_best_configs(existing)


# ── Subcommand: report ───────────────────────────────────────────────────────


def cmd_report(args: Namespace) -> None:
    """Show the best hyperparameter configuration found per problem."""
    problems = args.problems or DEFAULT_PROBLEMS
    _update_best_configs(problems)
    best_configs = _load_best_configs()

    if not best_configs:
        print("No search results found. Run `sweep_pdmorl.py search` first.")
        return

    hparam_keys = ["lr", "gamma", "batch_size", "tau", "buffer_size",
                   "epsilon_start", "epsilon_decay", "weight_batch", "total_steps"]

    header = f"{'Problem':<12} {'NormHV':>8} {'PNDS':>8} {'Order':>8}  " + \
             "  ".join(f"{k}" for k in hparam_keys)
    print("\n" + header)
    print("-" * len(header))

    for problem in problems:
        cfg = best_configs.get(problem)
        if cfg is None:
            print(f"{'problem_' + problem:<12} {'N/A':>8}")
            continue
        nhv = float(cfg.get("normalized_hv", 0))
        pnds = float(cfg.get("percent_non_dominated", 0))
        order = float(cfg.get("ordering_score", 0))
        hvals = "  ".join(f"{cfg.get(k, 'N/A')}" for k in hparam_keys)
        print(f"{'problem_' + problem:<12} {nhv:>8.4f} {pnds:>8.4f} {order:>8.4f}  {hvals}")

    print(f"\nFull best configs: {BEST_CONFIGS_PATH}")


# ── Subcommand: train ────────────────────────────────────────────────────────


def _run_seed_worker(spec: Dict) -> Dict:
    """Run full training for one (problem, seed) pair in a worker process.

    Mirrors _run_trial_worker: never writes to shared files, returns everything
    the main process needs to persist results.
    """
    import traceback

    error: Optional[str] = None
    try:
        metrics = _run_training(spec["train_args"], seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None

    return {
        "problem": spec["problem"],
        "problem_name": spec["problem_name"],
        "config_path": spec["config_path"],
        "hparams": spec["hparams"],
        "seed": spec["seed"],
        "metrics": metrics,
        "error": error,
    }


def _handle_seed_result(result: Dict, args: Namespace) -> None:
    """Persist and print one completed (problem, seed) result.

    Always called from the main process so CSV appends are sequential.
    """
    problem = result["problem"]
    problem_name = result["problem_name"]
    seed = result["seed"]
    metrics = result["metrics"]
    error = result.get("error")

    if error:
        _tqdm.write(f"  [ERROR] problem={problem} seed={seed}:\n{error.rstrip()}")

    if metrics is None:
        _tqdm.write(
            f"  [WARN] No metrics for problem={problem} seed={seed}. Skipping CSV write."
        )
        return

    csv_path = Path(args.csv_output) if args.csv_output else None
    _append_full_train_csv(
        problem=problem_name,
        config_path=result["config_path"],
        checkpoint_path="",  # checkpoints not saved by sweep
        hparams=result["hparams"],
        metrics=metrics,
        seed=seed,
        csv_path=csv_path,
    )
    _tqdm.write(
        f"  [problem={problem} seed={seed}] "
        f"HV={metrics.get('hypervolume', 0):.4f} | "
        f"NormHV={metrics.get('normalized_hypervolume', 0):.4f} | "
        f"PNDS={metrics.get('percent_non_dominated', 0):.4f} | "
        f"Order={metrics.get('ordering_score', 0):.4f}"
    )


def cmd_train(args: Namespace) -> None:
    """Train best-found hyperparameters for multiple seeds at full step budget."""
    _extend_sys_path(Path(args.pdmorl_root))
    device = _resolve_device(args.device)

    problems = args.problems or DEFAULT_PROBLEMS
    seeds = args.seeds
    best_configs = _load_best_configs()

    # Build the set of already-completed (problem_name, seed) pairs from the
    # output CSV so that interrupted runs can be resumed without re-training.
    csv_out = Path(args.csv_output) if args.csv_output else CANONICAL_CSV
    completed: set = set()
    if csv_out.exists():
        with open(csv_out, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    completed.add((row["problem"], int(float(row["seed"]))))
                except (KeyError, ValueError):
                    pass

    # Build the full list of (problem, seed) specs up front.
    pending: List[Dict] = []
    for problem in problems:
        config_path = PROBLEM_CONFIGS.get(problem)
        if config_path is None or not Path(config_path).exists():
            print(f"[SKIP] Config not found for problem '{problem}': {config_path}")
            continue

        raw_cfg = best_configs.get(problem)
        if raw_cfg is not None:
            hparams = {
                "lr":            float(raw_cfg["lr"]),
                "gamma":         float(raw_cfg.get("gamma", 0.99)),
                "batch_size":    int(float(raw_cfg["batch_size"])),
                "tau":           float(raw_cfg["tau"]),
                "buffer_size":   int(float(raw_cfg["buffer_size"])),
                "epsilon_start": float(raw_cfg["epsilon_start"]),
                "epsilon_decay": int(float(raw_cfg["epsilon_decay"])),
                "weight_batch":  int(float(raw_cfg["weight_batch"])),
                "total_steps":   int(float(raw_cfg.get("total_steps", args.total_steps))),
            }
            source = "sweep-tuned"
        else:
            hparams = {
                "lr": 3e-4, "gamma": 0.99, "batch_size": 64, "tau": 0.005,
                "buffer_size": 10_000, "epsilon_start": 0.8,
                "epsilon_decay": 100_000, "weight_batch": 3,
                "total_steps": args.total_steps,
            }
            source = "defaults (no sweep results found)"

        print(
            f"  problem={problem} [{source}]: "
            + ", ".join(f"{k}={v}" for k, v in hparams.items())
        )

        for seed in seeds:
            problem_name = Path(config_path).stem
            if (problem_name, seed) in completed:
                print(f"  [SKIP] problem={problem} seed={seed} already in {csv_out.name}")
                continue

            train_steps = hparams["total_steps"]
            train_args = _build_train_args(
                config_path=config_path,
                hparams=hparams,
                total_steps=train_steps,
                eval_interval=train_steps,  # evaluate once at the end, same as search
                seed=seed,
                save_dir=str(CANONICAL_CKPT_DIR),
                log_dir=str(SCRIPT_DIR / "runs" / f"problem_{problem}"),
                device=device,
                pdmorl_root=args.pdmorl_root,
            )
            pending.append({
                "problem": problem,
                "problem_name": Path(config_path).stem,
                "config_path": config_path,
                "hparams": hparams,
                "seed": seed,
                "train_args": train_args,
            })

    print(
        f"\nTraining {len(pending)} (problem × seed) combinations "
        f"({args.workers} worker(s))."
    )

    if args.workers == 1:
        for spec in pending:
            result = _run_seed_worker(spec)
            _handle_seed_result(result, args)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_initializer,
        )
        futures: Dict = {}
        _interrupted = False
        try:
            for spec in pending:
                futures[pool.submit(_run_seed_worker, spec)] = spec

            not_done: set = set(futures)
            with _tqdm(
                total=len(pending),
                desc="Training runs",
                unit="run",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(futures):
                    spec = futures[future]
                    not_done.discard(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "problem": spec["problem"],
                            "problem_name": spec["problem_name"],
                            "config_path": spec["config_path"],
                            "hparams": spec["hparams"],
                            "seed": spec["seed"],
                            "metrics": None,
                            "error": str(exc),
                        }
                    _handle_seed_result(result, args)
                    pbar.update(1)
                    if not_done:
                        remaining_problems = sorted(
                            {futures[f]["problem"] for f in not_done}
                        )
                        sample = remaining_problems[:4]
                        ellipsis = "…" if len(remaining_problems) > 4 else ""
                        pbar.set_postfix_str(
                            f"remaining={len(not_done)} "
                            f"[{','.join(sample)}{ellipsis}]"
                        )
        except KeyboardInterrupt:
            _interrupted = True
            _tqdm.write("\nKeyboardInterrupt — terminating workers…")
            for f in futures:
                f.cancel()
            try:
                for proc in pool._processes.values():
                    proc.terminate()
            except AttributeError:
                pass
        finally:
            pool.shutdown(wait=False)

        if _interrupted:
            _tqdm.write(
                "Workers terminated. Completed run results are saved "
                "and resumable (re-run to continue)."
            )
            sys.exit(130)

    print(f"\nTraining complete. Results appended to {csv_out}")


# ── Subcommand: summarize ─────────────────────────────────────────────────────


def _load_pdmorl_csv(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _aggregate_by_problem(rows: List[Dict]) -> Dict[str, Dict]:
    """Return mean metrics per problem label."""
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        problem = row.get("problem", "")
        buckets[problem].append(row)

    result = {}
    for problem, bucket in buckets.items():
        def _mean(key: str) -> float:
            vals = [float(r[key]) for r in bucket if r.get(key) not in (None, "")]
            return float(np.mean(vals)) if vals else float("nan")

        def _std(key: str) -> float:
            vals = [float(r[key]) for r in bucket if r.get(key) not in (None, "")]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        result[problem] = {
            "n_seeds": len(bucket),
            "normalized_hv_mean": _mean("normalized_hv"),
            "normalized_hv_std": _std("normalized_hv"),
            "pnds_mean": _mean("percent_non_dominated"),
            "pnds_std": _std("percent_non_dominated"),
            "ordering_mean": _mean("ordering_score"),
            "ordering_std": _std("ordering_score"),
        }
    return result


def _format_val(mean: float, std: float, bold: bool = False) -> str:
    s = f"{mean:.3f} ± {std:.3f}"
    return f"**{s}**" if bold else s


def cmd_summarize(args: Namespace) -> None:
    """Aggregate results and print a Markdown comparison table (paper-ready)."""
    pdmorl_csv = Path(args.pdmorl_csv)
    pcpl_csv = Path(args.pcpl_csv)

    pdmorl_rows = _load_pdmorl_csv(pdmorl_csv)
    pcpl_rows = _load_pdmorl_csv(pcpl_csv)

    if not pdmorl_rows:
        print(f"[WARN] No PD-MORL results found at {pdmorl_csv}.")
    if not pcpl_rows:
        print(f"[WARN] No PCPL results found at {pcpl_csv}.")

    pdmorl_agg = _aggregate_by_problem(pdmorl_rows)

    # PCPL CSV has different column names; normalise them.
    pcpl_rows_norm = []
    for row in pcpl_rows:
        pcpl_rows_norm.append({
            "problem": "problem_" + row.get("env_name", "").split("_")[1]
            if "_" in row.get("env_name", "")
            else row.get("env_name", ""),
            "normalized_hv": row.get("normalized_hv", row.get("normalized_hv", "")),
            "percent_non_dominated": row.get("non_dominated", ""),
            "ordering_score": row.get("ordering", ""),
        })
    pcpl_agg = _aggregate_by_problem(pcpl_rows_norm)

    # Ordered list of problems for the table.
    problem_order = [
        "problem_0",
        "problem_1a", "problem_1b", "problem_1c",
        "problem_2a", "problem_2b", "problem_2c",
        "problem_3a", "problem_3b",
        "problem_4a", "problem_4b",
        "problem_5a", "problem_5b", "problem_5c", "problem_5d", "problem_5e",
    ]

    header = (
        "| Problem | Method | Normalized HV | PNDS | Ordering Score |"
    )
    sep = "|---------|--------|--------------|------|----------------|"
    lines = [header, sep]

    for prob in problem_order:
        label = prob.replace("problem_", "Problem ")

        pd_data = pdmorl_agg.get(prob)
        pc_data = pcpl_agg.get(prob)

        if pd_data is None and pc_data is None:
            continue

        def _entry(data: Optional[Dict], bold_hv: bool, bold_pnds: bool, bold_ord: bool) -> str:
            if data is None:
                return "| — | — | — "
            nhv = _format_val(data["normalized_hv_mean"], data["normalized_hv_std"], bold_hv)
            pnds = _format_val(data["pnds_mean"], data["pnds_std"], bold_pnds)
            order = _format_val(data["ordering_mean"], data["ordering_std"], bold_ord)
            return f"| {nhv} | {pnds} | {order} "

        # Determine which method wins each metric.
        pd_nhv = pd_data["normalized_hv_mean"] if pd_data else float("-inf")
        pc_nhv = pc_data["normalized_hv_mean"] if pc_data else float("-inf")
        pd_pnds = pd_data["pnds_mean"] if pd_data else float("-inf")
        pc_pnds = pc_data["pnds_mean"] if pc_data else float("-inf")
        pd_ord = pd_data["ordering_mean"] if pd_data else float("-inf")
        pc_ord = pc_data["ordering_mean"] if pc_data else float("-inf")

        pd_bold_hv = pd_nhv >= pc_nhv
        pd_bold_pnds = pd_pnds >= pc_pnds
        pd_bold_ord = pd_ord >= pc_ord

        n_pd = pd_data["n_seeds"] if pd_data else 0
        n_pc = pc_data["n_seeds"] if pc_data else 0

        pd_line = (
            f"| {label} | PD-MORL (n={n_pd}) "
            + _entry(pd_data, pd_bold_hv, pd_bold_pnds, pd_bold_ord) + "|"
        )
        pc_line = (
            f"| | PCPL (n={n_pc}) "
            + _entry(pc_data, not pd_bold_hv, not pd_bold_pnds, not pd_bold_ord) + "|"
        )
        lines.append(pd_line)
        lines.append(pc_line)

    table = "\n".join(lines)
    print("\n" + table + "\n")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print(f"Saved Markdown table to {out_md}")

    # Also print a condensed version matching Table 4 in the paper (mean only, no std).
    _print_paper_table(problem_order, pdmorl_agg, pcpl_agg)


def _print_paper_table(
    problem_order: List[str],
    pdmorl_agg: Dict,
    pcpl_agg: Dict,
) -> None:
    """Print a compact table matching Table 4 format in the paper."""
    header = (
        "| Problem | Normalized HV | | PNDS | | Ordering Score | |"
    )
    subheader = "|---------|:---:|:---:|:---:|:---:|:---:|:---:|"
    method_row = "| | PD-MORL | PCPL | PD-MORL | PCPL | PD-MORL | PCPL |"
    print("\n── Compact table (mean across seeds, matches paper Table 4 format) ──\n")
    print(header)
    print(subheader)
    print(method_row)

    for prob in problem_order:
        pd_d = pdmorl_agg.get(prob)
        pc_d = pcpl_agg.get(prob)

        def _m(data: Optional[Dict], key_mean: str) -> str:
            if data is None:
                return "—"
            val = data.get(key_mean, float("nan"))
            return f"{val:.3f}" if not np.isnan(val) else "—"

        pd_nhv = _m(pd_d, "normalized_hv_mean")
        pc_nhv = _m(pc_d, "normalized_hv_mean")
        pd_pnds = _m(pd_d, "pnds_mean")
        pc_pnds = _m(pc_d, "pnds_mean")
        pd_ord = _m(pd_d, "ordering_mean")
        pc_ord = _m(pc_d, "ordering_mean")

        label = prob.replace("problem_", "Problem ")
        print(
            f"| {label} | {pd_nhv} | {pc_nhv} | {pd_pnds} | {pc_pnds} | {pd_ord} | {pc_ord} |"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pdmorl-root",
        type=str,
        default=str(DEFAULT_PD_MORL_ROOT),
        help="Path to the PD-MORL repository root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda / cpu).",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        metavar="PROBLEM",
        help=(
            "Problem IDs to process (e.g. 0 1a 1b 2a). "
            f"Defaults to the main benchmark set: {DEFAULT_PROBLEMS}."
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── search ──
    sp_search = subparsers.add_parser(
        "search",
        help="Run random hyperparameter search (short runs, single seed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_search)
    sp_search.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random hyperparameter trials per problem.",
    )
    sp_search.add_argument(
        "--search-steps",
        type=int,
        default=1_000_000,
        help="Training steps per trial.",
    )
    sp_search.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="RNG seed used to sample hyperparameter configurations.",
    )
    sp_search.add_argument(
        "--search-seed-trial",
        type=int,
        default=0,
        help="Environment / training seed used for every search trial.",
    )
    sp_search.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of trials to run concurrently. Each worker runs in its own "
            "process (ProcessPoolExecutor). Set to the number of physical CPU "
            "cores you want to dedicate; all file writes are serialised in the "
            "main process so there are no race conditions."
        ),
    )

    # ── report ──
    sp_report = subparsers.add_parser(
        "report",
        help="Print the best hyperparameter config per problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_report)

    # ── train ──
    sp_train = subparsers.add_parser(
        "train",
        help="Train with best-found hyperparameters for multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_args(sp_train)
    sp_train.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to train.",
    )
    sp_train.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="Training steps for each seed.",
    )
    sp_train.add_argument(
        "--eval-interval",
        type=int,
        default=200_000,
        help="Steps between intermediate evaluations during training.",
    )
    sp_train.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of (problem × seed) runs to execute concurrently. "
            "Each worker runs in its own process; all CSV writes are "
            "serialised in the main process so there are no race conditions."
        ),
    )
    sp_train.add_argument(
        "--csv-output",
        type=str,
        default="",
        help="Override output CSV path (default: pdmorl/data/pdmorl_stats.csv).",
    )

    # ── summarize ──
    sp_sum = subparsers.add_parser(
        "summarize",
        help="Aggregate results and print a Markdown comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp_sum.add_argument(
        "--pdmorl-csv",
        type=str,
        default=str(CANONICAL_CSV),
        help="Path to the PD-MORL per-seed results CSV.",
    )
    sp_sum.add_argument(
        "--pcpl-csv",
        type=str,
        default=str(SCRIPT_DIR / "data" / "pcpl_stats.csv"),
        help="Path to the PCPL per-seed results CSV.",
    )
    sp_sum.add_argument(
        "--output-md",
        type=str,
        default=str(SCRIPT_DIR / "data" / "sweep" / "comparison_table.md"),
        help="Where to write the Markdown comparison table.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
