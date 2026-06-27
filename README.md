
# GraphAllocBench

**A Flexible Benchmark for Preference-Conditioned Multi-Objective Policy Learning**

GraphAllocBench is a benchmark and toolkit for Preference-Conditioned Policy Learning (PCPL) in Multi-Objective Reinforcement Learning (MORL). It is built on **CityPlannerEnv**, a novel graph-based resource-allocation sandbox inspired by city management, and ships a suite of **19 configurable problems** spanning six challenge categories together with a set of evaluation utilities, so researchers and practitioners can design scenarios that stress different trade-offs and Pareto-front geometries.

At its core, GraphAllocBench makes it easy to:

- Create customizable problems by varying numbers of demands, resources, objectives, and objective shapes, composed from elementary functions (linear, quadratic, logarithmic, logistic, Gaussian, ceiling, and floor, or sums thereof).
- Produce diverse Pareto fronts and objective landscapes — smooth convex, discontinuous multi-segment, and non-convex with oscillatory structure — to evaluate PCPL and scalarization strategies.
- Scale from small 2-objective tasks up to large bipartite dependency graphs (100 demands, 100 resources) to stress-test agent architectures.
- Run batch preference sweeps and standardized evaluations: Pareto extraction, **hypervolume (HV) ratio**, **Proportion of Non-Dominated Solutions (PNDS)**, and **Ordering Score (OS)**, plus inference helpers.

![Illustration of PCPL workflow with GraphAllocBench](images/flowchart.png)

We include an example PCPL setup, **PCPL-PPO**, using Stable Baselines3 PPO paired with a Smooth Tchebycheff scalarization to demonstrate how to train and evaluate agents with preference-conditioned rewards. PCPL-PPO supports both Multi-Layer Perceptron (MLP) and Heterogeneous Graph Neural Network (HGNN) feature extractors, the latter for scaling to large, complex dependency graphs.

## Quick Start

This package requires **Python 3.10 or higher** and uses [uv](https://docs.astral.sh/uv/)
for package management.

### Installation (uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't
have it, then from the repository root:

```bash
uv sync          # creates a managed .venv, resolves dependencies, writes uv.lock
```

This installs `graphallocbench` (editable) together with all dependencies into a
project-local virtual environment. Run any command inside that environment with
`uv run`:

```bash
uv run python -c "import graphallocbench; print('ok')"
```

> Prefer a classic editable install instead? `uv pip install -e .` (after
> `uv venv`) also works, but `uv sync` is recommended for a reproducible lockfile.

### Pulling baseline submodules

The PD-MORL and Envelope Q-Learning baselines depend on external repositories
vendored under `pdmorl/external/` and `envelope/external/`. Fetch them with:

```bash
git submodule update --init --recursive          # PD-MORL
git clone https://github.com/RunzheYang/MORL envelope/external/MORL   # Envelope
```

(The Envelope repository is referenced in `.gitmodules` but is not committed as a
gitlink, so it is cloned directly into the expected path.) These are **not** needed
to run PCPL-PPO or the PSL baseline.

### Example Usage

```python
from graphallocbench import CityPlannerEnv
from graphallocbench.evaluation import run_experiments

env = CityPlannerEnv("graphallocbench/config/problems/problem_0.yml") # Enter the path to your problem configuration YAML file
obs, info = env.reset()
print(env.action_space, env.observation_space)
```

More examples can be found in `graphallocbench/examples/*`.

### Training & Evaluating

Train and evaluate the PCPL-PPO agent on a single problem (run via `uv run`):

```bash
uv run python train.py    --problems problem_0 --seeds 0 --workers 1 --wandb-mode disabled
uv run python evaluate.py --problems problem_0 --seeds 0
```

Omit `--problems` to run the full benchmark, add `--gnn` to use the HGNN feature
extractor for the large 100×100 graph problems (6a–6c), and use
`--seeds 0 1 2 3 4` for multi-seed runs.

### Baselines

Three algorithmic baselines are provided for comparison:

- **PD-MORL** (`pdmorl/`) — `uv run python pdmorl/train_graphalloc_pdmorl.py --config graphallocbench/config/problems/problem_0.yml`
- **Envelope Q-Learning** (`envelope/`) — `uv run python envelope/train_graphalloc_envelope.py --config graphallocbench/config/problems/problem_0.yml`
- **PSL (Pareto Set Learning)** (`psl/`) — `uv run python psl/train_graphalloc_psl.py --config graphallocbench/config/problems/problem_0.yml`

See [psl/README.md](psl/README.md) for details on the PSL baseline.

## Benchmark Problems

GraphAllocBench defines **19 evaluation problems** (configs under
`graphallocbench/config/problems/`) across six challenge categories, each
targeting a distinct optimization difficulty for PCPL:

| Problems | \|P\| | N | Fully-Connected | Description |
| --- | --- | --- | --- | --- |
| `0` | 2 | 2 | Yes | Baseline with smooth, convex Pareto front and simple logarithmic objectives. |
| `1a`–`1c` | 2 | 2 | Yes | Difficult objective functions: oscillatory behavior, stationary rewards, and spikes. |
| `2a`–`2c` | 5 | 2 | Yes | More demands, expanding the action and observation spaces. |
| `3a`–`3b` | 5 | 5 | Yes | Five objectives with highly sparse reward signals. |
| `4a`–`4b` | 5 | 5 | No | Non-fully-connected dependency graphs with varied dependencies and resources. |
| `5a`–`5e` | 5 | 3–5 | No | Randomly sampled dependencies, resources, and objectives at extended planning horizons. |
| `6a`–`6c` | 100 | 5 | No | Large-scale 100×100 graphs with complex bipartite structure (ideal Pareto front not computed). |

## Components

The package exposes four main modules:

1. `graphallocbench.city_env` – Environment implementation (`CityPlannerEnv`) and neural architectures / feature extractors.
2. `graphallocbench.evaluation` – Utilities for evaluating trained PCPL agents (Pareto front extraction, hypervolume, ordering score, inference helpers, etc.).
3. `graphallocbench.train_utils` – Training helpers (single/parallel PPO training utilities).
4. `graphallocbench.constants` – Centralized global constants (e.g., RL model class, allowed devices, inference batch size).

## Environment implementation details

For full implementation details of the environment (observations, action space, requirements matrix, allocation matrix, productions, objective functions, and reward modes) see the companion document: [GraphAllocBench.md](GraphAllocBench.md).

## Config Files
Problem configuration YAMLs describe resource capacities, demands, objectives, and scalarization settings. You can create your own or adapt the examples shipped under `graphallocbench/config/problems/*`.

### Using Stable-Baselines3 PPO
```python
from stable_baselines3 import PPO
model = PPO("MultiInputPolicy", env)
model.learn(total_timesteps=10_000)
```

### Batch Inference / Preference Sweep
```python
from graphallocbench.evaluation import run_experiments
final_objectives, allocations = run_experiments(env, model=model, n_iter=32)
print(final_objectives.shape)
```

## Citing
If you use GraphAllocBench, please cite the accompanying research paper:

> *GraphAllocBench: A Flexible Benchmark for Preference-Conditioned Multi-Objective Policy Learning.*

The official code repository is hosted at
[github.com/jzh001/GraphAllocBench](https://github.com/jzh001/GraphAllocBench).
(BibTeX will be added when the publication record is available.)

## License
MIT License (see `LICENSE`).

## Disclaimer
The original research code is preserved under `legacy/*`.