# Envelope Q-Learning integration for GraphAllocBench

Utilities in this folder let you train and evaluate the Envelope Q-Learning agent (Yang et al., NeurIPS 2019) on GraphAllocBench problems without touching the core package.

## Layout

- `common.py` – environment adapter reused from `pdmorl.common`, plus `GraphAllocEnvelopeCQN` (the fixed-width Q-network), `EnvelopePolicyAdapter`, and `EnvelopeTrainingConfig`.
- `train_graphalloc_envelope.py` – single-configuration training entrypoint.
- `evaluate_graphalloc_envelope.py` – checkpoint evaluator plus CSV/JSON exporters.
- `sweep_envelope.py` – **comprehensive hyperparameter search** (random search, per-problem best-config selection, full multi-seed training, paper-ready summary table).
- `external/MORL` – git submodule pointing to [RunzheYang/MORL](https://github.com/RunzheYang/MORL). Only `synthetic/crl/envelope/` is used.
- `checkpoints/`, `runs/`, `data/` – created at runtime.
- `checkpoints_sweep/`, `runs_sweep/` – used during the search phase.

## Implementation notes

### Network architecture (fair comparison with PCPL and PD-MORL)

The original `EnvelopeLinearCQN` from Yang et al. scales hidden-layer widths as multiples of `(state_size + reward_size)`, producing models with 672K–17M parameters depending on the problem.  PD-MORL's `MO_DDQN` and PCPL both use a fixed `hidden_size=256` with 3 hidden layers (~200K parameters for all problems).

To ensure a fair comparison, this integration replaces `EnvelopeLinearCQN` with **`GraphAllocEnvelopeCQN`** — a fixed-width MLP with the same interface and envelope H-operator, but with `hidden_size=256` and `n_layers=3` by default.  The number of trainable parameters is now ~150–230K across all 16 problems, matching the other baselines.

| Problem class | Original params | `GraphAllocEnvelopeCQN` params |
|---|---|---|
| 0, 1a–1c (state=10, obj=2) | 672 K | ~138 K |
| 2a–2c (state=22, obj=2) | 2.7 M | ~157 K |
| 3a–3b (state=22, obj=5) | 3.4 M | ~160 K |
| 4a–4b (state=44–55, obj=5) | 11–17 M | ~175 K |
| 5a–5e (state=55, obj=3–5) | 16–17 M | ~167–175 K |

Override with `--hidden-size` / `--n-layers` if needed.

### Episode-based vs. step-based training

Envelope Q-Learning uses episode-level scheduling for ε and β (both anneal once per terminal step). The training loop here is internally episode-based but the outer step counter aligns the evaluation / checkpointing protocol with PCPL and PD-MORL:

- `episode_num = total_steps // env.max_episode_steps` is passed to `MetaAgent` so ε-decay and the homotopy β-schedule complete correctly over the full step budget.
- Checkpoints and metrics are written every `--eval-interval` steps (default 100 k), matching the other baselines.

### Compatibility patches (applied automatically)

| Issue | Fix |
|---|---|
| `np.float` removed in NumPy ≥ 1.24 | `ensure_numpy_float_patch()` adds `np.float = np.float64` before first import |
| `masked_select` requires `BoolTensor` in PyTorch ≥ 2.0 | Built into `GraphAllocEnvelopeCQN.H` using `dtype=torch.bool` |
| Verbose `print(self.beta)` on every terminal step | `GraphAllocEnvelopeAgent` subclass overrides `memorize()` |

## 1. Environment setup

1. Install GraphAllocBench dependencies (`conda env create -f environment.yml && conda activate graphallocbench`).
2. Initialize the submodule (once after cloning):

   ```bash
   git submodule update --init --recursive
   ```

## 2. Comprehensive hyperparameter search (recommended)

**Search space:**

| Hyperparameter | Values |
|---|---|
| Learning rate | 1e-4, 3e-4, 6e-4, 1e-3 |
| Gamma | 0.95, 0.99 |
| Batch size | 32, 64, 128, 256 |
| Memory size | 5 000, 10 000, 20 000 |
| Epsilon | 0.3, 0.5, 0.8 |
| Weight num | 8, 16, 32 |
| Beta (initial) | 0.01, 0.1, 0.3 |
| Homotopy | True, False |
| Update freq | 50, 100, 200 |
| Optimizer | Adam, RMSprop |
| Training steps | 500 000, 1 000 000 |

**Step 1 – search** (one training seed per trial):

```bash
python envelope/sweep_envelope.py search \
    --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \
    --trials 10
```

Results go to `envelope/data/sweep/<problem>_search.csv` and `best_configs.json`. Resumable: re-running skips completed trials.

**Step 2 – inspect** the best config per problem:

```bash
python envelope/sweep_envelope.py report
```

**Step 3 – full training** with best hyperparameters across 5 seeds:

```bash
python envelope/sweep_envelope.py train \
    --problems 0 1a 1b 1c 2a 2b 2c 3a 3b 4a 4b 5a 5b 5c 5d 5e \
    --seeds 0 1 2 3 4
```

Metrics are appended to `envelope/data/envelope_stats.csv`.

**Step 4 – summary table** (Markdown, mirrors paper Table 4):

```bash
python envelope/sweep_envelope.py summarize \
    --pcpl-csv graphallocbench/examples/data/envelope_stats.csv
```

Pass `--workers N` to any subcommand to run N trials/seeds in parallel (uses `ProcessPoolExecutor`).

## 3. Single-configuration training

```bash
python envelope/train_graphalloc_envelope.py \
  --config graphallocbench/config/problems/problem_0.yml \
  --total-steps 1000000
```

Key CLI flags (see `--help` for full list):

| Flag | Default | Description |
|---|---|---|
| `--total-steps` | 1 000 000 | Total environment steps |
| `--eval-interval` | 100 000 | Steps between evaluations |
| `--mem-size` | 10 000 | Replay buffer size |
| `--batch-size` | 256 | Mini-batch size |
| `--weight-num` | 32 | Preference samples per learning step |
| `--beta` | 0.01 | Initial homotopy β |
| `--homotopy` / `--no-homotopy` | True | Enable homotopy annealing |
| `--epsilon` | 0.5 | Initial ε for ε-greedy |
| `--epsilon-decay` / `--no-epsilon-decay` | True | Linear ε decay |
| `--update-freq` | 100 | Target network hard-update frequency |
| `--optimizer` | Adam | Adam or RMSprop |
| `--hidden-size` | 256 | Hidden-layer width (matches PD-MORL) |
| `--n-layers` | 3 | Number of hidden layers (matches PD-MORL) |
| `--seeds` | 0 1 2 3 4 | Seeds to train |

## 4. Evaluation

```bash
python envelope/evaluate_graphalloc_envelope.py \
  --config graphallocbench/config/problems/problem_0.yml \
  --checkpoint "envelope/checkpoints/problem_0/seed-{seed}/envelope_step_1000000.pt"
```

Outputs:
- Per-seed metrics to stdout.
- Aggregated JSON under `envelope/data/evaluation/`.
- CSV rows appended to `envelope/data/envelope_stats.csv`.

## 5. Comparing with PCPL and PD-MORL

The `envelope_stats.csv` uses the same columns as `pdmorl_stats.csv`, so `pdmorl/plot.py` can consume both. For a three-way comparison table, run:

```bash
python envelope/sweep_envelope.py summarize \
    --envelope-csv envelope/data/envelope_stats.csv \
    --pcpl-csv graphallocbench/examples/data/pcpl_stats.csv
```
