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
2. Initialize the PD-MORL submodule (CRITICAL - needed once after cloning this repo):

   ```bash
   git submodule update --init --recursive
   ```

   This clones `https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm` into `pdmorl/external/…`. The training script directly imports modules from this submodule to ensure exact architectural compliance.

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
  --checkpoint "pdmorl/checkpoints/problem_0/seed-{seed}/ddqn_step_1000000.pt" \
  --device cuda
```

Outputs:

- Per-seed metrics printed to stdout.
- Aggregated JSON (e.g. `pdmorl/data/evaluation/problem_0_seed_0.json` or `..._seeds_0_1_2_3_4.json`) containing the full objective arrays for each seed.
- CSV rows appended to `pdmorl/data/pdmorl_stats.csv` for every problem. Each row includes the `problem` label along with hypervolume, normalized HV vs. reference stats copied into `pdmorl/data/GraphAllocBench-v2`, normalized HV vs. the analytical ideal (when demand count < 6), percent non dominated, ordering score, and eval settings, so you can sort/filter inside one sheet.
  Aggregated JSON files are named `{problem}_{seed_label}.json` under `pdmorl/data/evaluation/`, so rerunning with a different `--seeds` list keeps a separate JSON snapshot.

Pass `--seeds` to control the exact seed list or `--csv-output` to write metrics elsewhere. The `--checkpoint` flag accepts `{seed}` as a placeholder, so the command above automatically evaluates `seed-0` through `seed-4` when run with the default `--seeds 0 1 2 3 4`. If you only want a single seed, pass an explicit checkpoint path without `{seed}` (or override `--seeds`).

## 4. Implementation Details

This implementation strictly follows the architecture defined in the [PD-MORL repository](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm):

1.  **Network Architecture**: Uses the `MO_DDQN` network directly from the submodule.
2.  **Inputs**: The state `s` (allocation + requirements) and preference `w` are fed into the network separately. The preference vector is **not** duplicated in the state vector.
3.  **Replay Buffer**: Uses `ExperienceReplayBuffer_HER_MO` from the submodule (Hindsight Experience Replay with uniform sampling). It does **not** use Prioritized Experience Replay (PER), matching the reference implementation.
4.  **Exploration**: Uses `EpsilonGreedyActionSelector` with linear decay.

## 5. Tips

- To compare against official baselines, copy `graphallocbench/examples/data/GraphAllocBench-v2` into `pdmorl/data/` so normalized hypervolume uses the same reference.
- `OrderingPolicyAdapter` exposes the PD-MORL policy through a Stable-Baselines-style API, enabling `graphallocbench.evaluation` to call it without modifications.
- To run on another problem, point `--config` to the desired YAML in `graphallocbench/config/problems/` during both training and evaluation.
