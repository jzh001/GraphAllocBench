#!/usr/bin/env python3
"""Hyperparameter search + multi-problem training for the PSL baseline.

Mirrors ``pdmorl/sweep_pdmorl.py`` / ``envelope/sweep_envelope.py``: run a random
hyperparameter search per problem, inspect the winners, train the best config
across multiple seeds (with a single top-level progress bar), and summarize.

Workflow
--------
1. Search (random hyperparameter search, one seed per trial)::

       uv run python psl/sweep_psl.py search \
           --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \
           --trials 10 --workers 4

2. Report the best configuration found per problem::

       uv run python psl/sweep_psl.py report

3. Train the best per-problem config across seeds (top-level progress bar)::

       uv run python psl/sweep_psl.py train --seeds 0 1 2 3 4 --workers 4

4. Summarize the per-seed results into a mean +/- std table::

       uv run python psl/sweep_psl.py summarize

Notes
-----
* ``--scalarization`` (default ``linear``, faithful to the PSL paper) applies to
  every trial/run; ``smooth_tchebycheff`` is available for parity with PCPL on
  non-convex fronts. Search results are kept separate per scalarization.
* PSL is scoped to the main benchmark (problems 0-5).
* All outputs stay under ``psl/`` -- search artifacts under ``psl/data/sweep/`` and
  per-seed rows in ``psl/data/psl_stats.csv``. The shared result CSVs under
  ``graphallocbench/examples/data/`` are never written.
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

from psl.common import PSLTrainingConfig  # noqa: E402

# ── Problem catalogue (PSL is scoped to the main benchmark, problems 0-5) ──────

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
    ]
}

DEFAULT_PROBLEMS = list(PROBLEM_CONFIGS.keys())

# ── Hyperparameter search space ───────────────────────────────────────────────

SEARCH_SPACE: Dict[str, List] = {
    "lr":            [1e-4, 3e-4, 6e-4, 1e-3],
    "gamma":         [0.95, 0.99],
    "batch_size":    [32, 64, 128, 256],
    "tau":           [0.001, 0.005, 0.01, 0.05],
    "buffer_size":   [10_000, 50_000, 100_000],
    "epsilon_start": [0.5, 0.8, 1.0],
    "epsilon_decay": [100_000, 200_000],
    "hidden_size":   [128, 256],
    "total_steps":   [500_000, 1_000_000],
    # Only sampled when scalarization == "smooth_tchebycheff"; fixed at default otherwise.
    "smoothness":    [0.001, 0.01, 0.05, 0.1],
}

_HPARAM_KEYS = list(SEARCH_SPACE.keys())

SEARCH_CSV_FIELDNAMES = [
    "problem", "trial", "seed", "scalarization",
    *_HPARAM_KEYS,
    "hypervolume", "normalized_hv", "percent_non_dominated", "ordering_score",
    "timestamp",
]

FULL_TRAIN_CSV_FIELDNAMES = [
    "problem", "config", "checkpoint", "scalarization",
    "hypervolume", "normalized_hv", "percent_non_dominated", "ordering_score",
    "learning_rate", "gamma", "batch_size", "tau", "total_steps",
    "seed", "timestamp",
]

# ── Paths (all under psl/, never the shared examples/data CSVs) ────────────────

SWEEP_DATA_DIR = SCRIPT_DIR / "data" / "sweep"
SWEEP_CKPT_DIR = SCRIPT_DIR / "checkpoints_sweep"
CANONICAL_CKPT_DIR = SCRIPT_DIR / "checkpoints"
CANONICAL_CSV = SCRIPT_DIR / "data" / "psl_stats.csv"


def _best_configs_path(scalarization: str) -> Path:
    return SWEEP_DATA_DIR / f"best_configs_{scalarization}.json"


def _search_csv_path(problem: str, scalarization: str) -> Path:
    return SWEEP_DATA_DIR / f"{problem}_{scalarization}_search.csv"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested


def _sample_hyperparams(rng: random.Random, scalarization: str = "linear") -> Dict:
    hparams = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()
               if key != "smoothness"}
    if scalarization == "smooth_tchebycheff":
        hparams["smoothness"] = rng.choice(SEARCH_SPACE["smoothness"])
    else:
        hparams["smoothness"] = PSLTrainingConfig().smoothness
    return hparams


def _default_hparams() -> Dict:
    cfg = PSLTrainingConfig()
    return {
        "lr": cfg.lr,
        "gamma": cfg.gamma,
        "batch_size": cfg.batch_size,
        "tau": cfg.tau,
        "buffer_size": cfg.buffer_size,
        "epsilon_start": cfg.epsilon_start,
        "epsilon_decay": cfg.epsilon_decay_steps,
        "hidden_size": cfg.hidden_size,
        "total_steps": cfg.total_steps,
        "smoothness": cfg.smoothness,
    }


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
    scalarization: str,
    smoothness: float,
    n_partitions: int,
    save_checkpoints: bool,
    save_logs: bool,
) -> Namespace:
    """Construct an argparse.Namespace compatible with train_single_seed."""
    cfg = PSLTrainingConfig()
    return Namespace(
        config=config_path,
        scalarization=scalarization,
        smoothness=smoothness,
        total_steps=total_steps,
        batch_size=int(hparams["batch_size"]),
        buffer_size=int(hparams["buffer_size"]),
        start_steps=cfg.start_steps,
        learn_every=cfg.learn_every,
        eval_interval=eval_interval,
        epsilon_start=float(hparams["epsilon_start"]),
        epsilon_final=cfg.epsilon_final,
        epsilon_decay=int(hparams["epsilon_decay"]),
        gamma=float(hparams["gamma"]),
        learning_rate=float(hparams["lr"]),
        tau=float(hparams["tau"]),
        hidden_size=int(hparams["hidden_size"]),
        hyper_hidden_size=cfg.hyper_hidden_size,
        dirichlet_alpha=cfg.dirichlet_alpha,
        n_partitions=n_partitions,
        device=device,
        seed=seed,
        seeds=[seed],
        log_dir=log_dir,
        save_dir=save_dir,
        save_checkpoints=save_checkpoints,
        save_logs=save_logs,
    )


def _run_training(train_args: Namespace, seed: int) -> Optional[Dict]:
    """Train one seed and return the metrics JSON the training loop wrote."""
    from psl.train_graphalloc_psl import train_single_seed

    train_single_seed(train_args, seed)
    problem_name = Path(train_args.config).stem
    metrics_path = (
        Path(train_args.save_dir)
        / problem_name
        / f"seed-{seed}"
        / f"metrics_step_{train_args.total_steps}.json"
    )
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _empty_metrics() -> Dict:
    return {
        "hypervolume": 0.0,
        "normalized_hypervolume": 0.0,
        "percent_non_dominated": 0.0,
        "ordering_score": 0.0,
    }


# ── Worker setup (shared by search + train parallel paths) ────────────────────


def _worker_initializer() -> None:
    """Ignore SIGINT in workers and silence per-step output at the fd level.

    Keeps the main process's top-level progress bar clean; metrics and errors are
    returned via the result dict instead of printed from the worker.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _run_with_progress(pending, worker_fn, handler, args, desc, unit) -> None:
    """Run ``pending`` specs sequentially (workers==1) or in a pool with a bar."""
    if args.workers == 1:
        for spec in pending:
            handler(worker_fn(spec), args)
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer)
    futures: Dict = {}
    interrupted = False
    try:
        for spec in pending:
            futures[pool.submit(worker_fn, spec)] = spec
        not_done: set = set(futures)
        with _tqdm(total=len(pending), desc=desc, unit=unit, dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                spec = futures[future]
                not_done.discard(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {**spec, "metrics": None, "error": str(exc)}
                handler(result, args)
                pbar.update(1)
                if not_done:
                    remaining = sorted({futures[f]["problem"] for f in not_done})
                    sample = remaining[:4]
                    ellipsis = "…" if len(remaining) > 4 else ""
                    pbar.set_postfix_str(f"remaining={len(not_done)} [{','.join(sample)}{ellipsis}]")
    except KeyboardInterrupt:
        interrupted = True
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

    if interrupted:
        _tqdm.write("Workers terminated. Completed results are saved and resumable (re-run to continue).")
        sys.exit(130)


# ── Subcommand: search ─────────────────────────────────────────────────────────


def _append_search_csv(problem: str, trial: int, seed: int, scalarization: str,
                       hparams: Dict, metrics: Dict) -> None:
    csv_path = _search_csv_path(problem, scalarization)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    row = {
        "problem": problem, "trial": trial, "seed": seed, "scalarization": scalarization,
        **{k: hparams[k] for k in _HPARAM_KEYS},
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEARCH_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_search_results(problem: str, scalarization: str) -> List[Dict]:
    csv_path = _search_csv_path(problem, scalarization)
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _best_config_for_problem(rows: List[Dict]) -> Optional[Dict]:
    if not rows:
        return None

    def _score(row: Dict) -> float:
        nhv = float(row.get("normalized_hv", 0.0) or 0.0)
        return nhv if nhv > 0 else float(row.get("hypervolume", 0.0) or 0.0)

    return max(rows, key=_score)


def _update_best_configs(problems: List[str], scalarization: str) -> None:
    path = _best_configs_path(scalarization)
    existing: Dict[str, Dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
    for problem in problems:
        best = _best_config_for_problem(_load_search_results(problem, scalarization))
        if best is not None:
            existing[problem] = best
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)


def _run_trial_worker(spec: Dict) -> Dict:
    import traceback

    error: Optional[str] = None
    try:
        steps = int(spec["hparams"]["total_steps"])
        train_args = _build_train_args(
            spec["config_path"], spec["hparams"],
            total_steps=steps, eval_interval=steps, seed=spec["seed"],
            save_dir=spec["save_dir"], log_dir=spec["log_dir"], device=spec["device"],
            scalarization=spec["scalarization"],
            smoothness=float(spec["hparams"].get("smoothness", spec["smoothness"])),
            n_partitions=spec["n_partitions"], save_checkpoints=False, save_logs=False,
        )
        metrics = _run_training(train_args, seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None
    return {**spec, "metrics": metrics if metrics is not None else _empty_metrics(),
            "error": error}


def _handle_trial_result(result: Dict, args: Namespace) -> None:
    if result.get("error"):
        _tqdm.write(f"  [ERROR] problem={result['problem']} trial={result['trial_idx'] + 1}:\n{result['error'].rstrip()}")
    m = result["metrics"]
    _append_search_csv(result["problem"], result["trial_idx"], args.search_seed_trial,
                       args.scalarization, result["hparams"], m)
    _tqdm.write(
        f"  [problem={result['problem']} trial={result['trial_idx'] + 1}/{args.trials}] "
        f"HV={m.get('hypervolume', 0):.4f} | NormHV={m.get('normalized_hypervolume', 0):.4f} | "
        f"PNDS={m.get('percent_non_dominated', 0):.4f} | Order={m.get('ordering_score', 0):.4f}"
    )


def cmd_search(args: Namespace) -> None:
    args.device = _resolve_device(args.device)
    problems = args.problems or DEFAULT_PROBLEMS

    completed: set = set()
    for problem in problems:
        for row in _load_search_results(problem, args.scalarization):
            completed.add((row["problem"], int(row["trial"])))

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
            hparams = _sample_hyperparams(trial_rng, args.scalarization)
            if args.search_steps:
                # Override the sampled budget (useful for fast searches / smoke tests).
                hparams["total_steps"] = args.search_steps
            pending.append({
                "problem": problem,
                "trial_idx": trial_idx,
                "config_path": config_path,
                "hparams": hparams,
                "seed": args.search_seed_trial,
                "device": args.device,
                "scalarization": args.scalarization,
                "smoothness": args.smoothness,
                "n_partitions": args.n_partitions,
                "log_dir": str(SCRIPT_DIR / "runs_sweep" / f"problem_{problem}" / f"trial_{trial_idx}"),
                "save_dir": str(SWEEP_CKPT_DIR / f"trial_{trial_idx}"),
            })

    print(
        f"Search [{args.scalarization}]: {len(problems)} problem(s), up to {args.trials} "
        f"trial(s) each, steps sampled per trial from {SEARCH_SPACE['total_steps']}. "
        f"{len(pending)} new trial(s) ({args.workers} worker(s))."
    )
    if not pending:
        print("Nothing to search -- all trials already completed.")
    else:
        _run_with_progress(pending, _run_trial_worker, _handle_trial_result, args,
                           desc="Search trials", unit="trial")

    _update_best_configs(problems, args.scalarization)
    print(f"\nSearch complete. Best configs saved to {_best_configs_path(args.scalarization)}")


# ── Subcommand: report ─────────────────────────────────────────────────────────


def cmd_report(args: Namespace) -> None:
    problems = args.problems or DEFAULT_PROBLEMS
    _update_best_configs(problems, args.scalarization)
    path = _best_configs_path(args.scalarization)
    if not path.exists():
        print(f"No search results for scalarization '{args.scalarization}'. Run `search` first.")
        return
    with open(path, encoding="utf-8") as fh:
        best = json.load(fh)

    header = f"{'Problem':<12} {'NormHV':>8} {'PNDS':>8} {'Order':>8}  " + "  ".join(_HPARAM_KEYS)
    print("\n" + header)
    print("-" * len(header))
    for problem in problems:
        cfg = best.get(problem)
        if cfg is None:
            print(f"{'problem_' + problem:<12} {'N/A':>8}")
            continue
        nhv = float(cfg.get("normalized_hv", 0) or 0)
        pnds = float(cfg.get("percent_non_dominated", 0) or 0)
        order = float(cfg.get("ordering_score", 0) or 0)
        hvals = "  ".join(f"{cfg.get(k, 'N/A')}" for k in _HPARAM_KEYS)
        print(f"{'problem_' + problem:<12} {nhv:>8.4f} {pnds:>8.4f} {order:>8.4f}  {hvals}")
    print(f"\nFull best configs: {path}")


# ── Subcommand: train ──────────────────────────────────────────────────────────


def _append_full_train_csv(problem_name: str, config_path: str, checkpoint_path: str,
                           scalarization: str, hparams: Dict, metrics: Dict, seed: int,
                           csv_path: Optional[Path] = None) -> None:
    out = csv_path or CANONICAL_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    row = {
        "problem": problem_name,
        "config": Path(config_path).name,
        "checkpoint": checkpoint_path,
        "scalarization": scalarization,
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "learning_rate": hparams.get("lr"),
        "gamma": hparams.get("gamma"),
        "batch_size": hparams.get("batch_size"),
        "tau": hparams.get("tau"),
        "total_steps": hparams.get("total_steps"),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FULL_TRAIN_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _coerce_hparams(raw_cfg: Optional[Dict], fallback_steps: int) -> Dict:
    if raw_cfg is None:
        h = _default_hparams()
        h["total_steps"] = fallback_steps
        return h
    return {
        "lr": float(raw_cfg["lr"]),
        "gamma": float(raw_cfg.get("gamma", 0.99)),
        "batch_size": int(float(raw_cfg["batch_size"])),
        "tau": float(raw_cfg["tau"]),
        "buffer_size": int(float(raw_cfg["buffer_size"])),
        "epsilon_start": float(raw_cfg["epsilon_start"]),
        "epsilon_decay": int(float(raw_cfg["epsilon_decay"])),
        "hidden_size": int(float(raw_cfg.get("hidden_size", 128))),
        "total_steps": int(float(raw_cfg.get("total_steps", fallback_steps))),
        "smoothness": float(raw_cfg.get("smoothness", PSLTrainingConfig().smoothness)),
    }


def _run_seed_worker(spec: Dict) -> Dict:
    import traceback

    error: Optional[str] = None
    try:
        metrics = _run_training(spec["train_args"], seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None
    return {**spec, "metrics": metrics, "error": error}


def _handle_seed_result(result: Dict, args: Namespace) -> None:
    problem, seed = result["problem"], result["seed"]
    metrics = result["metrics"]
    if result.get("error"):
        _tqdm.write(f"  [ERROR] problem={problem} seed={seed}:\n{result['error'].rstrip()}")
    if metrics is None:
        _tqdm.write(f"  [WARN] No metrics for problem={problem} seed={seed}. Skipping CSV write.")
        return
    csv_path = Path(args.csv_output) if args.csv_output else None
    _append_full_train_csv(result["problem_name"], result["config_path"], "",
                           args.scalarization, result["hparams"], metrics, seed, csv_path)
    _tqdm.write(
        f"  [problem={problem} seed={seed}] "
        f"HV={metrics.get('hypervolume', 0):.4f} | NormHV={metrics.get('normalized_hypervolume', 0):.4f} | "
        f"PNDS={metrics.get('percent_non_dominated', 0):.4f} | Order={metrics.get('ordering_score', 0):.4f}"
    )


def cmd_train(args: Namespace) -> None:
    args.device = _resolve_device(args.device)
    problems = args.problems or DEFAULT_PROBLEMS
    seeds = args.seeds

    best_path = _best_configs_path(args.scalarization)
    best_configs: Dict[str, Dict] = {}
    if best_path.exists():
        with open(best_path, encoding="utf-8") as fh:
            best_configs = json.load(fh)
    elif not args.allow_defaults:
        print(
            f"No best configs found at {best_path}.\n"
            f"Run `uv run python psl/sweep_psl.py search --scalarization {args.scalarization}` "
            f"first, or pass --allow-defaults to train with PSLTrainingConfig defaults."
        )
        return

    csv_out = Path(args.csv_output) if args.csv_output else CANONICAL_CSV
    completed: set = set()
    if csv_out.exists():
        with open(csv_out, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    completed.add((row["problem"], int(float(row["seed"]))))
                except (KeyError, ValueError):
                    pass

    pending: List[Dict] = []
    for problem in problems:
        config_path = PROBLEM_CONFIGS.get(problem)
        if config_path is None or not Path(config_path).exists():
            print(f"[SKIP] Config not found for problem '{problem}': {config_path}")
            continue
        raw_cfg = best_configs.get(problem)
        hparams = _coerce_hparams(raw_cfg, args.total_steps)
        source = "sweep-tuned" if raw_cfg is not None else "defaults (no sweep result)"
        print(f"  problem={problem} [{source}]: " + ", ".join(f"{k}={v}" for k, v in hparams.items()))

        problem_name = Path(config_path).stem
        for seed in seeds:
            if (problem_name, seed) in completed:
                print(f"  [SKIP] problem={problem} seed={seed} already in {csv_out.name}")
                continue
            steps = int(hparams["total_steps"])
            train_args = _build_train_args(
                config_path, hparams,
                total_steps=steps, eval_interval=steps, seed=seed,
                save_dir=str(CANONICAL_CKPT_DIR),
                log_dir=str(SCRIPT_DIR / "runs" / f"problem_{problem}"),
                device=args.device, scalarization=args.scalarization,
                smoothness=float(hparams.get("smoothness", args.smoothness)),
                n_partitions=args.n_partitions,
                save_checkpoints=not args.no_checkpoints, save_logs=not args.no_logs,
            )
            pending.append({
                "problem": problem, "problem_name": problem_name,
                "config_path": config_path, "hparams": hparams,
                "seed": seed, "train_args": train_args,
            })

    print(f"\nTraining {len(pending)} (problem x seed) combination(s) "
          f"[{args.scalarization} scalarization, {args.workers} worker(s)].")
    if not pending:
        print("Nothing to train -- all (problem, seed) pairs already in the CSV.")
        return

    _run_with_progress(pending, _run_seed_worker, _handle_seed_result, args,
                       desc="Training runs", unit="run")
    print(f"\nTraining complete. Results appended to {csv_out}")


# ── Subcommand: summarize ──────────────────────────────────────────────────────


def _aggregate_by_problem(rows: List[Dict]) -> Dict[str, Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("problem", "")].append(row)

    result: Dict[str, Dict] = {}
    for problem, bucket in buckets.items():
        def _stat(key: str, fn) -> float:
            vals = [float(r[key]) for r in bucket if r.get(key) not in (None, "")]
            return float(fn(vals)) if vals else float("nan")

        result[problem] = {
            "n_seeds": len(bucket),
            "nhv_mean": _stat("normalized_hv", np.mean),
            "nhv_std": _stat("normalized_hv", lambda v: np.std(v) if len(v) > 1 else 0.0),
            "pnds_mean": _stat("percent_non_dominated", np.mean),
            "pnds_std": _stat("percent_non_dominated", lambda v: np.std(v) if len(v) > 1 else 0.0),
            "ord_mean": _stat("ordering_score", np.mean),
            "ord_std": _stat("ordering_score", lambda v: np.std(v) if len(v) > 1 else 0.0),
        }
    return result


def cmd_summarize(args: Namespace) -> None:
    csv_path = Path(args.csv) if args.csv else CANONICAL_CSV
    if not csv_path.exists():
        print(f"No results found at {csv_path}. Run `sweep_psl.py train` first.")
        return
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    agg = _aggregate_by_problem(rows)

    order = [f"problem_{p}" for p in DEFAULT_PROBLEMS]
    lines = ["| Problem | n | Normalized HV | PNDS | Ordering Score |",
             "|---------|---|--------------|------|----------------|"]
    for prob in order:
        d = agg.get(prob)
        if d is None:
            continue
        lines.append(
            f"| {prob.replace('problem_', 'Problem ')} | {d['n_seeds']} | "
            f"{d['nhv_mean']:.3f} ± {d['nhv_std']:.3f} | "
            f"{d['pnds_mean']:.3f} ± {d['pnds_std']:.3f} | "
            f"{d['ord_mean']:.3f} ± {d['ord_std']:.3f} |"
        )
    table = "\n".join(lines)
    print("\n" + table + "\n")
    out_md = Path(args.output_md) if args.output_md else (SCRIPT_DIR / "data" / "psl_summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print(f"Saved Markdown table to {out_md}")


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problems", nargs="*", default=None, metavar="PROBLEM",
                        help=f"Problem IDs (e.g. 0 1c 3b). Default: {DEFAULT_PROBLEMS}.")
    parser.add_argument("--scalarization", choices=["linear", "smooth_tchebycheff"],
                        default=PSLTrainingConfig().scalarization,
                        help="Scalarization applied to every trial/run.")
    parser.add_argument("--smoothness", type=float, default=PSLTrainingConfig().smoothness)
    parser.add_argument("--n-partitions", type=int, default=PSLTrainingConfig().n_partitions)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    sp = sub.add_parser("search", help="Random hyperparameter search (one seed per trial).",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(sp)
    sp.add_argument("--trials", type=int, default=10, help="Random trials per problem.")
    sp.add_argument("--search-steps", type=int, default=0,
                    help="If >0, override each trial's sampled step budget (faster search).")
    sp.add_argument("--search-seed", type=int, default=0, help="RNG seed for sampling hparams.")
    sp.add_argument("--search-seed-trial", type=int, default=0, help="Training seed for each trial.")
    sp.add_argument("--workers", type=int, default=1, help="Concurrent trials (>1 shows top-level bar).")

    # report
    sp = sub.add_parser("report", help="Print best hyperparameter config per problem.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(sp)

    # train
    sp = sub.add_parser("train", help="Train best-found config across seeds (top-level progress bar).",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(sp)
    sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    sp.add_argument("--total-steps", type=int, default=PSLTrainingConfig().total_steps,
                    help="Fallback step budget when a problem has no sweep result / uses defaults.")
    sp.add_argument("--workers", type=int, default=1, help="Concurrent (problem x seed) runs.")
    sp.add_argument("--allow-defaults", action="store_true",
                    help="Train with PSLTrainingConfig defaults if no sweep results exist.")
    sp.add_argument("--no-checkpoints", action="store_true", help="Do not save model checkpoints.")
    sp.add_argument("--no-logs", action="store_true", help="Disable TensorBoard logging.")
    sp.add_argument("--csv-output", type=str, default="",
                    help="Override output CSV (default: psl/data/psl_stats.csv).")

    # summarize
    sp = sub.add_parser("summarize", help="Aggregate per-seed results into a mean+/-std table.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp.add_argument("--csv", type=str, default="", help="Results CSV (default: psl/data/psl_stats.csv).")
    sp.add_argument("--output-md", type=str, default="", help="Where to write the Markdown table.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"search": cmd_search, "report": cmd_report,
     "train": cmd_train, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    main()
