#!/usr/bin/env python3
"""Comprehensive hyperparameter search for Envelope Q-Learning on GraphAllocBench.

Workflow
--------
1. Random hyperparameter search (one seed per trial) per problem::

       python envelope/sweep_envelope.py search \\
           --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \\
           --trials 10

2. Inspect the best configurations found::

       python envelope/sweep_envelope.py report

3. Train 5 seeds with the best per-problem configuration::

       python envelope/sweep_envelope.py train \\
           --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \\
           --seeds 0 1 2 3 4

4. Generate the summary table for inclusion in the paper::

       python envelope/sweep_envelope.py summarize
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

DEFAULT_ENVELOPE_ROOT = SCRIPT_DIR / "external" / "MORL"

# ── Problem catalogue ────────────────────────────────────────────────────────

PROBLEM_CONFIGS: Dict[str, str] = {
    name: str(REPO_ROOT / "graphallocbench" / "config" / "problems" / f"problem_{name}.yml")
    for name in [
        "0",
        "1a", "1b", "1c",
        "2a", "2b", "2c",
        "3a", "3b",
        "4a", "4b",
        "5a", "5b", "5c", "5d", "5e",
        # 6a-c excluded: 100-demand graphs are too large for Envelope's MLP.
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
    "lr":          [1e-4, 3e-4, 6e-4, 1e-3],
    "gamma":       [0.95, 0.99],
    "batch_size":  [32, 64, 128, 256],
    "mem_size":    [5_000, 10_000, 20_000],
    "epsilon":     [0.3, 0.5, 0.8],
    "weight_num":  [8, 16, 32],
    "beta":        [0.01, 0.1, 0.3],
    "homotopy":    [True, False],
    "update_freq": [50, 100, 200],
    "optimizer":   ["Adam", "RMSprop"],
    "total_steps": [500_000, 1_000_000],
}

SWEEP_CSV_FIELDNAMES = [
    "problem", "trial", "seed", "search_steps",
    "lr", "gamma", "batch_size", "mem_size",
    "epsilon", "weight_num", "beta", "homotopy", "update_freq", "optimizer",
    "total_steps",
    "hypervolume", "normalized_hv", "percent_non_dominated", "ordering_score",
    "timestamp",
]

FULL_TRAIN_CSV_FIELDNAMES = [
    "problem", "config", "checkpoint",
    "hypervolume", "normalized_hv", "percent_non_dominated", "ordering_score",
    "pref_grid_step", "lr", "gamma", "batch_size", "seed", "timestamp",
]

# ── Paths ────────────────────────────────────────────────────────────────────

SWEEP_DATA_DIR = SCRIPT_DIR / "data" / "sweep"
SWEEP_CKPT_DIR = SCRIPT_DIR / "checkpoints_sweep"
BEST_CONFIGS_PATH = SWEEP_DATA_DIR / "best_configs.json"
CANONICAL_CKPT_DIR = SCRIPT_DIR / "checkpoints"
CANONICAL_CSV = SCRIPT_DIR / "data" / "envelope_stats.csv"

# ── Helpers ──────────────────────────────────────────────────────────────────


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
    envelope_root: str,
    pref_grid_step: float = 0.05,
    dirichlet_alpha: float = 1.0,
    epsilon_decay: bool = True,
) -> Namespace:
    return Namespace(
        config=config_path,
        envelope_root=envelope_root,
        total_steps=total_steps,
        mem_size=hparams["mem_size"],
        batch_size=hparams["batch_size"],
        eval_interval=eval_interval,
        gamma=hparams["gamma"],
        learning_rate=hparams["lr"],
        epsilon=hparams["epsilon"],
        epsilon_decay=epsilon_decay,
        weight_num=hparams["weight_num"],
        beta=hparams["beta"],
        homotopy=hparams["homotopy"],
        update_freq=hparams["update_freq"],
        optimizer=hparams["optimizer"],
        hidden_size=256,
        n_layers=3,
        dirichlet_alpha=dirichlet_alpha,
        pref_grid_step=pref_grid_step,
        seed=seed,
        seeds=[seed],
        log_dir=log_dir,
        save_dir=save_dir,
        save_checkpoints=False,
        save_logs=False,
    )


def _run_training(args: Namespace, seed: int) -> Optional[Dict]:
    from envelope.train_graphalloc_envelope import (
        train_single_seed,
        _extend_sys_path,
        _import_envelope_modules,
    )
    _extend_sys_path(Path(args.envelope_root))
    MetaAgent_cls, get_new_model = _import_envelope_modules()
    train_single_seed(args, seed, MetaAgent_cls, get_new_model)

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


def _append_search_csv(problem: str, trial: int, seed: int, hparams: Dict, metrics: Dict) -> None:
    csv_path = SWEEP_DATA_DIR / f"{problem}_search.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    row = {
        "problem": problem, "trial": trial, "seed": seed,
        "search_steps": hparams["total_steps"],
        "lr": hparams["lr"], "gamma": hparams["gamma"],
        "batch_size": hparams["batch_size"], "mem_size": hparams["mem_size"],
        "epsilon": hparams["epsilon"], "weight_num": hparams["weight_num"],
        "beta": hparams["beta"], "homotopy": hparams["homotopy"],
        "update_freq": hparams["update_freq"], "optimizer": hparams["optimizer"],
        "total_steps": hparams["total_steps"],
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_CSV_FIELDNAMES)
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
    if not rows:
        return None
    def _score(row: Dict) -> float:
        nhv = float(row.get("normalized_hv", 0.0))
        return nhv if nhv > 0 else float(row.get("hypervolume", 0.0))
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


def _update_best_configs(problems: List[str]) -> None:
    existing = _load_best_configs()
    for problem in problems:
        rows = _load_search_results(problem)
        best = _best_config_for_problem(rows)
        if best is not None:
            existing[problem] = best
    _save_best_configs(existing)


def _append_full_train_csv(
    problem: str,
    config_path: str,
    hparams: Dict,
    metrics: Dict,
    seed: int,
    csv_path: Optional[Path] = None,
) -> None:
    out = csv_path or CANONICAL_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    row = {
        "problem": problem, "config": Path(config_path).name, "checkpoint": "",
        "hypervolume": metrics.get("hypervolume", 0.0),
        "normalized_hv": metrics.get("normalized_hypervolume", 0.0),
        "percent_non_dominated": metrics.get("percent_non_dominated", 0.0),
        "ordering_score": metrics.get("ordering_score", 0.0),
        "pref_grid_step": 0.05,
        "lr": hparams.get("lr", 1e-3),
        "gamma": hparams.get("gamma", 0.99),
        "batch_size": hparams.get("batch_size", 256),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FULL_TRAIN_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Worker helpers ────────────────────────────────────────────────────────────


def _worker_initializer() -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def _run_trial_worker(spec: Dict) -> Dict:
    import traceback
    error: Optional[str] = None
    try:
        trial_steps = spec["hparams"]["total_steps"]
        train_args = _build_train_args(
            config_path=spec["config_path"],
            hparams=spec["hparams"],
            total_steps=trial_steps,
            eval_interval=trial_steps,
            seed=spec["seed"],
            save_dir=spec["save_dir"],
            log_dir=spec["log_dir"],
            envelope_root=spec["envelope_root"],
        )
        metrics = _run_training(train_args, seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None

    if metrics is None:
        metrics = {
            "hypervolume": 0.0, "normalized_hypervolume": 0.0,
            "percent_non_dominated": 0.0, "ordering_score": 0.0,
        }
    return {
        "problem": spec["problem"], "trial_idx": spec["trial_idx"],
        "hparams": spec["hparams"], "metrics": metrics, "error": error,
    }


def _handle_trial_result(result: Dict, args: Namespace) -> None:
    problem, trial_idx = result["problem"], result["trial_idx"]
    hparams, metrics, error = result["hparams"], result["metrics"], result.get("error")
    if error:
        _tqdm.write(f"  [ERROR] problem={problem} trial={trial_idx + 1}:\n{error.rstrip()}")
    _append_search_csv(problem, trial_idx, args.search_seed_trial, hparams, metrics)
    _tqdm.write(
        f"  [problem={problem} trial={trial_idx + 1}/{args.trials}] "
        f"HV={metrics.get('hypervolume', 0):.4f} | "
        f"NormHV={metrics.get('normalized_hypervolume', 0):.4f} | "
        f"PNDS={metrics.get('percent_non_dominated', 0):.4f} | "
        f"Order={metrics.get('ordering_score', 0):.4f}"
    )


def _run_seed_worker(spec: Dict) -> Dict:
    import traceback
    error: Optional[str] = None
    try:
        metrics = _run_training(spec["train_args"], seed=spec["seed"])
    except Exception:
        error = traceback.format_exc()
        metrics = None
    return {
        "problem": spec["problem"], "problem_name": spec["problem_name"],
        "config_path": spec["config_path"], "hparams": spec["hparams"],
        "seed": spec["seed"], "metrics": metrics, "error": error,
    }


def _handle_seed_result(result: Dict, args: Namespace) -> None:
    problem, seed = result["problem"], result["seed"]
    metrics, error = result["metrics"], result.get("error")
    if error:
        _tqdm.write(f"  [ERROR] problem={problem} seed={seed}:\n{error.rstrip()}")
    if metrics is None:
        _tqdm.write(f"  [WARN] No metrics for problem={problem} seed={seed}. Skipping.")
        return
    csv_path = Path(args.csv_output) if args.csv_output else None
    _append_full_train_csv(
        problem=result["problem_name"],
        config_path=result["config_path"],
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


# ── Subcommand: search ────────────────────────────────────────────────────────


def cmd_search(args: Namespace) -> None:
    problems = args.problems or DEFAULT_PROBLEMS

    completed: set = set()
    for problem in problems:
        for row in _load_search_results(problem):
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
            hparams = _sample_hyperparams(trial_rng)
            pending.append({
                "problem": problem,
                "trial_idx": trial_idx,
                "config_path": config_path,
                "hparams": hparams,
                "seed": args.search_seed_trial,
                "envelope_root": args.envelope_root,
                "log_dir": str(SCRIPT_DIR / "runs_sweep" / f"problem_{problem}" / f"trial_{trial_idx}"),
                "save_dir": str(SWEEP_CKPT_DIR / f"trial_{trial_idx}"),
            })

    print(
        f"Hyperparameter search: {len(problems)} problem(s), "
        f"up to {args.trials} trial(s) each. "
        f"{len(pending)} new trial(s) to run ({args.workers} worker(s))."
    )

    if args.workers == 1:
        for spec in pending:
            result = _run_trial_worker(spec)
            _handle_trial_result(result, args)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer)
        futures: Dict = {}
        _interrupted = False
        try:
            for spec in pending:
                futures[pool.submit(_run_trial_worker, spec)] = spec
            not_done: set = set(futures)
            with _tqdm(total=len(pending), desc="Search trials", unit="trial", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    not_done.discard(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        spec = futures[future]
                        result = {
                            "problem": spec["problem"], "trial_idx": spec["trial_idx"],
                            "hparams": spec["hparams"],
                            "metrics": {"hypervolume": 0.0, "normalized_hypervolume": 0.0,
                                        "percent_non_dominated": 0.0, "ordering_score": 0.0},
                            "error": str(exc),
                        }
                    _handle_trial_result(result, args)
                    pbar.update(1)
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
            _tqdm.write("Workers terminated. Results saved and resumable.")
            sys.exit(130)

    _update_best_configs(problems)
    print(f"\nSearch complete. Best configs → {BEST_CONFIGS_PATH}")


# ── Subcommand: report ────────────────────────────────────────────────────────


def cmd_report(args: Namespace) -> None:
    problems = args.problems or DEFAULT_PROBLEMS
    _update_best_configs(problems)
    best_configs = _load_best_configs()
    if not best_configs:
        print("No search results found. Run `sweep_envelope.py search` first.")
        return

    hparam_keys = ["lr", "gamma", "batch_size", "mem_size", "epsilon",
                   "weight_num", "beta", "homotopy", "update_freq", "optimizer", "total_steps"]
    header = f"{'Problem':<12} {'NormHV':>8} {'PNDS':>8} {'Order':>8}  " + "  ".join(hparam_keys)
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


# ── Subcommand: train ─────────────────────────────────────────────────────────


def cmd_train(args: Namespace) -> None:
    problems = args.problems or DEFAULT_PROBLEMS
    seeds = args.seeds
    best_configs = _load_best_configs()

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
        if raw_cfg is not None:
            hparams = {
                "lr":          float(raw_cfg["lr"]),
                "gamma":       float(raw_cfg.get("gamma", 0.99)),
                "batch_size":  int(float(raw_cfg["batch_size"])),
                "mem_size":    int(float(raw_cfg.get("mem_size", 10_000))),
                "epsilon":     float(raw_cfg.get("epsilon", 0.5)),
                "weight_num":  int(float(raw_cfg.get("weight_num", 32))),
                "beta":        float(raw_cfg.get("beta", 0.01)),
                "homotopy":    str(raw_cfg.get("homotopy", "True")).lower() == "true",
                "update_freq": int(float(raw_cfg.get("update_freq", 100))),
                "optimizer":   raw_cfg.get("optimizer", "Adam"),
                "total_steps": int(float(raw_cfg.get("total_steps", args.total_steps))),
            }
            source = "sweep-tuned"
        else:
            hparams = {
                "lr": 1e-3, "gamma": 0.99, "batch_size": 256, "mem_size": 10_000,
                "epsilon": 0.5, "weight_num": 32, "beta": 0.01, "homotopy": True,
                "update_freq": 100, "optimizer": "Adam", "total_steps": args.total_steps,
            }
            source = "defaults (no sweep results found)"

        print(f"  problem={problem} [{source}]: " + ", ".join(f"{k}={v}" for k, v in hparams.items()))

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
                eval_interval=train_steps,
                seed=seed,
                save_dir=str(CANONICAL_CKPT_DIR),
                log_dir=str(SCRIPT_DIR / "runs" / f"problem_{problem}"),
                envelope_root=args.envelope_root,
            )
            pending.append({
                "problem": problem,
                "problem_name": problem_name,
                "config_path": config_path,
                "hparams": hparams,
                "seed": seed,
                "train_args": train_args,
            })

    print(f"\nTraining {len(pending)} (problem × seed) combinations ({args.workers} worker(s)).")

    if args.workers == 1:
        for spec in pending:
            result = _run_seed_worker(spec)
            _handle_seed_result(result, args)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer)
        futures: Dict = {}
        _interrupted = False
        try:
            for spec in pending:
                futures[pool.submit(_run_seed_worker, spec)] = spec
            not_done: set = set(futures)
            with _tqdm(total=len(pending), desc="Training runs", unit="run", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    not_done.discard(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        spec = futures[future]
                        result = {
                            "problem": spec["problem"], "problem_name": spec["problem_name"],
                            "config_path": spec["config_path"], "hparams": spec["hparams"],
                            "seed": spec["seed"], "metrics": None, "error": str(exc),
                        }
                    _handle_seed_result(result, args)
                    pbar.update(1)
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
            _tqdm.write("Workers terminated. Results saved and resumable.")
            sys.exit(130)

    print(f"\nTraining complete. Results appended to {csv_out}")


# ── Subcommand: summarize ─────────────────────────────────────────────────────


def _load_csv(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _aggregate_by_problem(rows: List[Dict]) -> Dict[str, Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("problem", "")].append(row)
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


def _fmtval(mean: float, std: float, bold: bool = False) -> str:
    s = f"{mean:.3f} ± {std:.3f}"
    return f"**{s}**" if bold else s


def cmd_summarize(args: Namespace) -> None:
    envelope_rows = _load_csv(Path(args.envelope_csv))
    pcpl_rows = _load_csv(Path(args.pcpl_csv))

    if not envelope_rows:
        print(f"[WARN] No Envelope results found at {args.envelope_csv}.")
    if not pcpl_rows:
        print(f"[WARN] No PCPL results found at {args.pcpl_csv}.")

    env_agg = _aggregate_by_problem(envelope_rows)

    # Normalise PCPL column names to match.
    pcpl_norm = []
    for row in pcpl_rows:
        env_name = row.get("env_name", "")
        if "_" in env_name:
            problem = "problem_" + env_name.split("_")[1]
        else:
            problem = env_name
        pcpl_norm.append({
            "problem": problem,
            "normalized_hv": row.get("normalized_hv", ""),
            "percent_non_dominated": row.get("non_dominated", ""),
            "ordering_score": row.get("ordering", ""),
        })
    pcpl_agg = _aggregate_by_problem(pcpl_norm)

    problem_order = [
        "problem_0",
        "problem_1a", "problem_1b", "problem_1c",
        "problem_2a", "problem_2b", "problem_2c",
        "problem_3a", "problem_3b",
        "problem_4a", "problem_4b",
        "problem_5a", "problem_5b", "problem_5c", "problem_5d", "problem_5e",
    ]

    header = "| Problem | Method | Normalized HV | PNDS | Ordering Score |"
    sep    = "|---------|--------|--------------|------|----------------|"
    lines = [header, sep]

    for prob in problem_order:
        label = prob.replace("problem_", "Problem ")
        ed = env_agg.get(prob)
        pd = pcpl_agg.get(prob)
        if ed is None and pd is None:
            continue

        def _entry(data: Optional[Dict], bh: bool, bp: bool, bo: bool) -> str:
            if data is None:
                return "| — | — | — "
            nhv  = _fmtval(data["normalized_hv_mean"], data["normalized_hv_std"], bh)
            pnds = _fmtval(data["pnds_mean"], data["pnds_std"], bp)
            ord_ = _fmtval(data["ordering_mean"], data["ordering_std"], bo)
            return f"| {nhv} | {pnds} | {ord_} "

        e_nhv = ed["normalized_hv_mean"] if ed else float("-inf")
        p_nhv = pd["normalized_hv_mean"] if pd else float("-inf")
        e_bh, e_bp, e_bo = e_nhv >= p_nhv, (ed["pnds_mean"] if ed else float("-inf")) >= (pd["pnds_mean"] if pd else float("-inf")), (ed["ordering_mean"] if ed else float("-inf")) >= (pd["ordering_mean"] if pd else float("-inf"))

        n_e = ed["n_seeds"] if ed else 0
        n_p = pd["n_seeds"] if pd else 0
        lines.append(f"| {label} | Envelope (n={n_e}) " + _entry(ed, e_bh, e_bp, e_bo) + "|")
        lines.append(f"| | PCPL (n={n_p}) " + _entry(pd, not e_bh, not e_bp, not e_bo) + "|")

    table = "\n".join(lines)
    print("\n" + table + "\n")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print(f"Saved Markdown table to {out_md}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--envelope-root", default=str(DEFAULT_ENVELOPE_ROOT))
    parser.add_argument(
        "--problems", nargs="*", default=None, metavar="PROBLEM",
        help=f"Problem IDs (e.g. 0 1a 2b). Default: {DEFAULT_PROBLEMS}.",
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    sp = sub.add_parser("search", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(sp)
    sp.add_argument("--trials", type=int, default=10)
    sp.add_argument("--search-seed", type=int, default=0)
    sp.add_argument("--search-seed-trial", type=int, default=0)
    sp.add_argument("--workers", type=int, default=1)

    # report
    sp2 = sub.add_parser("report", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(sp2)

    # train
    sp3 = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(sp3)
    sp3.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    sp3.add_argument("--total-steps", type=int, default=1_000_000)
    sp3.add_argument("--workers", type=int, default=1)
    sp3.add_argument("--csv-output", default="")

    # summarize
    sp4 = sub.add_parser("summarize", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sp4.add_argument("--envelope-csv", default=str(CANONICAL_CSV))
    sp4.add_argument("--pcpl-csv", default=str(SCRIPT_DIR / "data" / "pcpl_stats.csv"))
    sp4.add_argument("--output-md", default=str(SWEEP_DATA_DIR / "comparison_table.md"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {"search": cmd_search, "report": cmd_report, "train": cmd_train, "summarize": cmd_summarize}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
