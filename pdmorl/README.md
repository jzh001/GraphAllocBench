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
- Writes checkpoints plus per-evaluation metric JSON under `pdmorl/checkpoints/<problem>/seed-<seed>/` and TensorBoard logs under `pdmorl/runs/` so multiple problems/seeds can coexist without overwriting.
- Every `--eval-interval` steps (default 100k), runs the official GraphAllocBench metrics (hypervolume, normalized hypervolume vs. the analytical ideal when available, percent non dominated, ordering score) and prints them to stdout.
- Runs seeds 0–4 by default (set `--seeds` or `--seed` to override) so every run produces problem/seed-specific checkpoints and logs.

Override hyperparameters via CLI flags (see `python pdmorl/train_graphalloc_pdmorl.py --help`). The script defaults to 1M training steps, so pass `--total-steps` if you want shorter smoke tests. Defaults target CUDA (`--device cuda`).

## 3. Evaluation

Evaluate any checkpoint across multiple seeds (defaults to 0–4) and export both JSON + CSV summaries:

```bash
python pdmorl/evaluate_graphalloc_pdmorl.py \
  --config graphallocbench/config/problems/problem_0.yml \
  --checkpoint pdmorl/checkpoints/problem_0/seed-42/ddqn_step_1000000.pt \
  --device cuda
```

Outputs:

- Per-seed metrics printed to stdout.
- Aggregated JSON (e.g. `pdmorl/data/evaluation/problem_0_seed_42.json` or `..._seeds_0_1_2_3_4.json`) containing the full objective arrays for each seed.
- CSV rows appended to `pdmorl/data/pdmorl_stats.csv` for every problem. Each row includes the `problem` label along with hypervolume, normalized HV vs. reference stats copied into `pdmorl/data/GraphAllocBench-v2`, normalized HV vs. the analytical ideal (when demand count < 6), percent non dominated, ordering score, and eval settings, so you can sort/filter inside one sheet.
  Aggregated JSON files are named `{problem}_{seed_label}.json` under `pdmorl/data/evaluation/`, so rerunning with a different `--seeds` list keeps a separate JSON snapshot.

Pass `--seeds` to control the exact seed list or `--csv-output` to write metrics elsewhere. All comparisons should use the 1M-step checkpoint (`ddqn_step_1000000.pt`) so hypervolume numbers line up with the training defaults noted above.

## 4. Tips

- To compare against official baselines, copy `graphallocbench/examples/data/GraphAllocBench-v2` into `pdmorl/data/` so normalized hypervolume uses the same reference.
- `OrderingPolicyAdapter` exposes the PD-MORL policy through a Stable-Baselines-style API, enabling `graphallocbench.evaluation` to call it without modifications.
- To run on another problem, point `--config` to the desired YAML in `graphallocbench/config/problems/` during both training and evaluation.
