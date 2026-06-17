# PSL (Pareto Set Learning) baseline

A from-scratch implementation of **PSL-MORL** (Liu et al., 2025,
[arXiv:2501.06773](https://arxiv.org/abs/2501.06773)) adapted to GraphAllocBench.

> **Note:** PSL-MORL released no official code, so this is a clean reimplementation
> from the paper. It is independent of the PD-MORL / Envelope submodules.

## What PSL is

PSL-MORL learns a **hypernetwork** `φ(ω)` that maps a preference (decomposition
weight) vector `ω` to (part of) the parameters of a policy network. A single
trained hypernetwork therefore produces a preference-conditioned policy for *any*
continuous preference at inference time — exactly the PCPL paradigm that
GraphAllocBench targets. The paper is algorithm-agnostic and instantiates the
framework with **Double-DQN** for discrete action spaces (and TD3 for continuous).

### Why it fits GraphAllocBench

- GraphAllocBench exposes vector rewards + a discrete action space through the
  existing [`GraphAllocDiscreteEnv`](../pdmorl/common.py) wrapper (shared with
  PD-MORL and Envelope).
- Because PSL is a *single preference-conditioned model*, it plugs straight into
  the standard evaluation pipeline (Das–Dennis preference sampling → deterministic
  rollout → HV / PNDS / OS) via the existing `OrderingPolicyAdapter`.

## Design

`psl/common.py`:

- `PSLHyperQNet` — a scalar-Q network. A shared trunk (`θ₁`) encodes the flattened
  state; the hypernetwork `φ(ω)` (`θ₂`) generates the final linear **head**
  weights/bias from the preference, so different preferences yield different
  policies while sharing the state encoder (the partial-parameter scheme from the
  paper).
- `PSLAgent` — hypernetwork **Double-DQN** with a soft-updated target network. It
  is callable as `agent(flat_obs, pref, deterministic=True) → action_idx`, so it
  reuses `OrderingPolicyAdapter` for evaluation with no extra glue.
- `scalarize_psl` — the DDQN reward utility. `linear` (`u = ω·J`) reproduces the
  paper's linearized utility (default); `smooth_tchebycheff` reuses
  [`graphallocbench/city_env/scalarize.py`](../graphallocbench/city_env/scalarize.py)
  and matches PCPL-PPO for parity on non-convex fronts (uses a running ideal-point
  estimate; nadir is the origin since objectives are non-negative).

`psl/train_graphalloc_psl.py` — per-step DDQN training loop: sample `ω ~ Dirichlet`
per episode, act ε-greedy with the hypernet-conditioned Q-net, store
`(s, a, r_vec, s', done, ω)`, and minimise the Double-DQN loss on the **scalarized**
reward. The outer objective is `max_φ E_{ω~Λ}[u(J, ω)]`.

## Scope & caveats

- **Targeted at problems 0–5** — the same scope as the PD-MORL/Envelope comparison
  in the paper's Table 4. The 100×100 problem-6 set is HGNN territory and out of
  scope here (a flat MLP-DDQN hypernetwork on that scale is impractical).
- **Linear scalarization cannot recover non-convex Pareto fronts** (e.g. problems
  1c, 3b). This is an inherent, expected weakness of weighted-sum utilities — run
  `--scalarization smooth_tchebycheff` if you want a non-convex-capable variant.

## Running

### Single problem

```bash
# Default (linear scalarization, faithful to the paper)
uv run python psl/train_graphalloc_psl.py \
    --config graphallocbench/config/problems/problem_0.yml \
    --seeds 0 1 2 3 4 --total-steps 1000000

# Smooth Tchebycheff variant (parity with PCPL-PPO on non-convex fronts)
uv run python psl/train_graphalloc_psl.py \
    --config graphallocbench/config/problems/problem_1c.yml \
    --scalarization smooth_tchebycheff --seeds 0
```

### All problems with a progress bar (sweep)

`psl/sweep_psl.py` mirrors `pdmorl/sweep_pdmorl.py` /
`envelope/sweep_envelope.py` — a four-stage `search → report → train → summarize`
workflow. With `--workers > 1` it shows a single top-level progress bar across all
runs (per-step bars are suppressed); with `--workers 1` it runs sequentially with
the per-step bars visible.

```bash
# 1. Random hyperparameter search (one seed per trial, per problem)
uv run python psl/sweep_psl.py search --trials 10 --workers 4

# 2. Inspect the best config found per problem
uv run python psl/sweep_psl.py report

# 3. Train the best per-problem config across seeds (uses the search results)
uv run python psl/sweep_psl.py train --seeds 0 1 2 3 4 --workers 4

# 4. Aggregate per-seed results into a mean ± std table
uv run python psl/sweep_psl.py summarize
```

The search space (lr, gamma, batch size, tau, buffer size, ε-start, ε-decay,
hidden size, step budget) is defined in `SEARCH_SPACE` in `sweep_psl.py`. Useful
flags:

- `--problems 0 1c 3b` — restrict to a subset (default: all of 0–5).
- `--scalarization {linear,smooth_tchebycheff}` — applies to every trial/run;
  search results are stored separately per scalarization.
- `--search-steps N` — override each trial's sampled step budget for a faster
  search.
- `train --allow-defaults` — train with `PSLTrainingConfig` defaults if you want to
  skip the search phase entirely.

Both `search` and `train` are **resumable** — re-running skips trials / (problem,
seed) pairs already recorded in the CSVs.

### Outputs

- Checkpoints: `psl/checkpoints/<problem>/seed-<n>/psl_step_<step>.pt`
- Per-eval metrics: `psl/checkpoints/<problem>/seed-<n>/metrics_step_<step>.json`
  (HV, normalized HV, PNDS, ordering score)
- TensorBoard logs: `psl/runs/<problem>/seed-<n>/`
- Sweep (`sweep_psl.py`): per-trial CSVs and `best_configs_<scalarization>.json`
  under `psl/data/sweep/`; per-seed `train` rows in `psl/data/psl_stats.csv`;
  summary table in `psl/data/psl_summary.md`.

Evaluation reuses the official metric implementations
([`graphallocbench/evaluation/utils.py`](../graphallocbench/evaluation/utils.py)),
so PSL reports the same HV / PNDS / OS as the other baselines. **PSL never writes to
the shared result CSVs** under `graphallocbench/examples/data/` — all its output
stays under `psl/`.
```
