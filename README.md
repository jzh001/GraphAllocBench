
# GraphAllocBench

GraphAllocBench is a benchmark and toolkit for Preference-Conditioned Policy Learning (PCPL) for multiple objectives. It provides a flexible resource-allocation environment, a set of configurable problem definitions, and a suite of evaluation utilities so researchers and practitioners can design scenarios that stress different trade-offs and Pareto fronts.

At its core, GraphAllocBench makes it easy to:

- Create customizable problems by varying numbers of demands, resources, objectives, and objective shapes (e.g. sinusoidal, concave, convex, bell-shaped, S-shaped).
- Produce diverse Pareto fronts and objective landscapes to evaluate PCPL and scalarization strategies.
- Run batch preference sweeps and standardized evaluations (Pareto extraction, hypervolume, proportion of non-dominated solutions, ordering score, inference helpers).

![Illustration of PCPL workflow with GraphAllocBench](images/flowchart.png)

We include an example PCPL setup using Stable Baselines3 PPO paired with a Smooth Tchebycheff scalarization to demonstrate how to train and evaluate agents with preference-conditioned rewards.

## Quick Start

This package requires **Python 3.10 or higher**.

### Installation (Local Editable)
```bash
pip install -e .
```

### Example Usage

```python
from graphallocbench import CityPlannerEnv
from graphallocbench.evaluation import run_experiments

env = CityPlannerEnv("graphallocbench/config/problems/problem_0.yml") # Enter the path to your problem configuration YAML file
obs, info = env.reset()
print(env.action_space, env.observation_space)
```

More examples can be found in `graphallocbench/examples/*`.

## Components

The package exposes four main modules:

1. `graphallocbench.city_env` – Environment implementation (`CityPlannerEnv`) and neural architectures / feature extractors.
2. `graphallocbench.evaluation` – Utilities for evaluating trained PCPL agents (Pareto front extraction, hypervolume, ordering score, inference helpers, etc.).
3. `graphallocbench.train_utils` – Training helpers (single/parallel PPO training utilities).
4. `graphallocbench.constants` – Centralized global constants (e.g., RL model class, allowed devices, inference batch size).

## Environment implementation details

For full implementation details of the environment (observations, action space, requirements matrix, allocation matrix, productions, objective functions, and reward modes) see the companion document: [GraphAllocBench.md](GraphAllocBench.md).

## Config Files
Problem configuration YAMLs describe resource capacities, demands, objectives, and scalarization settings. You can create your own or adapt the examples shipped under `graphallocbench/configs/problems/*`.

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
If you use GraphAllocBench, please cite the accompanying research paper. (BibTeX will be added when available.)

## License
MIT License (see `LICENSE`).

## Disclaimer
The original research code is preserved under `legacy/*`.