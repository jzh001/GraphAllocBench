# PD-MORL integration for GraphAllocBench

Utilities in this folder let you train and evaluate PD-MORL’s MO-DDQN-HER agent on GraphAllocBench problems without touching the core package.

## Layout

- `common.py` – shared environment wrapper, observation flattening, preference adapters, and compatibility shims.
- `train_graphalloc_pdmorl.py` – training entrypoint.
- `evaluate_graphalloc_pdmorl.py` – checkpoint evaluator plus CSV/JSON exporters.
- `external/PDMORL-…` – git submodule pointing to the upstream PD-MORL repo.
- `checkpoints/`, `runs/`, `data/` – default output directories for artifacts produced by the scripts.

## 1. Environment setup

1. Install GraphAllocBench dependencies (e.g. `conda env create -f environment.yml && conda activate graphallocbench`).
2. Initialize the PD-MORL submodule (needed once after cloning this repo):

   ```bash
   git submodule update --init --recursive
   ```

   This clones `https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm` into `pdmorl/external/…`. If you keep a different checkout elsewhere, pass `--pdmorl-root /path/to/PDMORL` when running the scripts.

## 2. Training

```bash
python pdmorl/train_graphalloc_pdmorl.py \
  --config graphallocbench/config/problems/problem_0.yml \
  --device cuda
```

Key behaviours:

- Wraps `CityPlannerEnv` so MO-DDQN-HER receives flat observations and a discrete action space.
- Samples Dirichlet preference vectors each episode, applies them to the environment, and feeds them into the agent.
- Writes checkpoints plus per-evaluation metric JSON under `pdmorl/checkpoints/` and TensorBoard logs under `pdmorl/runs/`.
- Every `--eval-interval` steps, runs the official GraphAllocBench metrics (hypervolume, percent non dominated, ordering score) and prints them to stdout.

Override hyperparameters via CLI flags (see `python pdmorl/train_graphalloc_pdmorl.py --help`). Defaults target CUDA (`--device cuda`).

## 3. Evaluation

Evaluate any checkpoint across multiple seeds (defaults to 0–4) and export both JSON + CSV summaries:

```bash
python pdmorl/evaluate_graphalloc_pdmorl.py \
  --config graphallocbench/config/problems/problem_0.yml \
  --checkpoint pdmorl/checkpoints/ddqn_step_60000.pt \
  --device cuda
```

Outputs:

- Per-seed metrics printed to stdout.
- Aggregated JSON (`pdmorl/evaluation.json` by default) containing the full objective arrays for each seed.
- CSV rows appended to `pdmorl/data/pdmorl_stats_<config>.csv` with hypervolume, normalized HV (vs. reference stats copied into `pdmorl/data/GraphAllocBench-v2`), percent non dominated, ordering score, and eval settings.

Pass `--seeds` to control the exact seed list or `--csv-output` to write metrics elsewhere.

## 4. Tips

- To compare against official baselines, copy `graphallocbench/examples/data/GraphAllocBench-v2` into `pdmorl/data/` so normalized hypervolume uses the same reference.
- `OrderingPolicyAdapter` exposes the PD-MORL policy through a Stable-Baselines-style API, enabling `graphallocbench.evaluation` to call it without modifications.
- To run on another problem, point `--config` to the desired YAML in `graphallocbench/config/problems/` during both training and evaluation.
