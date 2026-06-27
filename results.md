# GraphAllocBench — Evaluation Results

Per-problem comparison of **PD-MORL** (MO-DDQN-HER; Basaklar et al. 2023), **Envelope Q-Learning** (Yang et al. 2019), **PSL** (Pareto Set Learning; Liu et al. 2025), and **PCPL-PPO** (PPO + Smooth Tchebycheff Scalarization) across three metrics.
All values are mean ± sample std (ddof=1) over 5 independent seeds. These tables mirror Table 4 of the camera-ready paper.
**Bold** indicates the best result per metric per problem.

---

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Norm. HV** | Normalized hypervolume of the recovered Pareto front relative to the reference front | [0, 1] ↑ |
| **PNDS** | Proportion of Non-Dominated Solutions — fraction of evaluated preference outcomes that are non-dominated | [0, 1] ↑ |
| **OS** | Ordering Score — how well the policy respects the preference ordering across objectives | [0, 1] ↑ |

---

## Results Table

| Problem | Method | Norm. HV | PNDS | OS |
|---------|--------|----------|------|----|
| **0** | PD-MORL | 0.973 ± 0.012 | 0.923 ± 0.094 | 0.997 ± 0.003 |
|  | Envelope | 0.889 ± 0.020 | 0.723 ± 0.222 | 0.937 ± 0.054 |
|  | PSL | 0.990 ± 0.009 | 0.985 ± 0.034 | **1.000 ± 0.000** |
|  | PCPL-PPO | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **1a** | PD-MORL | 0.608 ± 0.078 | 0.846 ± 0.094 | 0.956 ± 0.023 |
|  | Envelope | 0.385 ± 0.159 | 0.477 ± 0.333 | 0.961 ± 0.041 |
|  | PSL | **0.943 ± 0.059** | **0.877 ± 0.117** | **0.995 ± 0.010** |
|  | PCPL-PPO | 0.696 ± 0.079 | 0.708 ± 0.100 | 0.954 ± 0.047 |
| **1b** | PD-MORL | 0.439 ± 0.126 | **0.908 ± 0.138** | 0.945 ± 0.052 |
|  | Envelope | 0.459 ± 0.246 | 0.400 ± 0.324 | 0.665 ± 0.080 |
|  | PSL | **0.887 ± 0.061** | 0.800 ± 0.069 | 0.876 ± 0.069 |
|  | PCPL-PPO | 0.696 ± 0.122 | 0.846 ± 0.077 | **0.982 ± 0.017** |
| **1c** | PD-MORL | 0.136 ± 0.118 | **0.877 ± 0.177** | 0.942 ± 0.075 |
|  | Envelope | 0.065 ± 0.000 | 0.769 ± 0.094 | 0.922 ± 0.015 |
|  | PSL | 0.211 ± 0.239 | 0.600 ± 0.148 | **0.980 ± 0.025** |
|  | PCPL-PPO | **0.282 ± 0.213** | 0.538 ± 0.094 | 0.955 ± 0.040 |
| **2a** | PD-MORL | 0.290 ± 0.093 | 0.862 ± 0.126 | 0.965 ± 0.074 |
|  | Envelope | 0.235 ± 0.205 | 0.477 ± 0.333 | 0.739 ± 0.071 |
|  | PSL | 0.873 ± 0.047 | 0.862 ± 0.084 | 0.999 ± 0.001 |
|  | PCPL-PPO | **0.992 ± 0.002** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **2b** | PD-MORL | 0.676 ± 0.191 | 0.692 ± 0.122 | 0.911 ± 0.056 |
|  | Envelope | 0.548 ± 0.126 | 0.523 ± 0.257 | 0.972 ± 0.010 |
|  | PSL | **0.912 ± 0.038** | **0.769 ± 0.188** | **0.983 ± 0.018** |
|  | PCPL-PPO | 0.599 ± 0.187 | 0.615 ± 0.224 | 0.902 ± 0.070 |
| **2c** | PD-MORL | 0.169 ± 0.070 | 0.800 ± 0.177 | 0.977 ± 0.030 |
|  | Envelope | 0.087 ± 0.088 | 0.415 ± 0.235 | 0.698 ± 0.141 |
|  | PSL | **0.748 ± 0.061** | 0.662 ± 0.088 | 0.950 ± 0.066 |
|  | PCPL-PPO | 0.614 ± 0.075 | **0.877 ± 0.088** | **0.999 ± 0.003** |
| **3a** | PD-MORL | 0.495 ± 0.094 | **0.395 ± 0.102** | 0.737 ± 0.083 |
|  | Envelope | 0.240 ± 0.221 | 0.216 ± 0.209 | 0.738 ± 0.071 |
|  | PSL | 0.578 ± 0.048 | 0.162 ± 0.059 | 0.809 ± 0.020 |
|  | PCPL-PPO | **0.807 ± 0.035** | 0.381 ± 0.043 | **0.934 ± 0.023** |
| **3b** | PD-MORL | 0.013 ± 0.012 | 0.627 ± 0.101 | **0.838 ± 0.088** |
|  | Envelope | 0.112 ± 0.060 | **0.680 ± 0.067** | 0.782 ± 0.047 |
|  | PSL | 0.197 ± 0.156 | 0.087 ± 0.044 | 0.607 ± 0.078 |
|  | PCPL-PPO | **0.295 ± 0.083** | 0.257 ± 0.087 | 0.833 ± 0.075 |
| **4a** | PD-MORL | 0.621 ± 0.049 | 0.069 ± 0.043 | 0.804 ± 0.117 |
|  | Envelope | 0.399 ± 0.292 | 0.048 ± 0.035 | 0.672 ± 0.059 |
|  | PSL | **0.994 ± 0.009** | 0.221 ± 0.080 | **0.982 ± 0.016** |
|  | PCPL-PPO | 0.835 ± 0.107 | **0.297 ± 0.107** | 0.862 ± 0.045 |
| **4b** | PD-MORL | 0.141 ± 0.061 | 0.386 ± 0.179 | 0.663 ± 0.026 |
|  | Envelope | 0.143 ± 0.105 | 0.255 ± 0.052 | 0.713 ± 0.102 |
|  | PSL | **0.863 ± 0.042** | 0.404 ± 0.098 | 0.780 ± 0.082 |
|  | PCPL-PPO | 0.764 ± 0.088 | **0.953 ± 0.023** | **0.938 ± 0.037** |
| **5a** | PD-MORL | 0.414 ± 0.117 | 0.442 ± 0.105 | 0.760 ± 0.075 |
|  | Envelope | 0.425 ± 0.009 | 0.415 ± 0.029 | 0.866 ± 0.012 |
|  | PSL | **0.834 ± 0.033** | 0.589 ± 0.129 | 0.907 ± 0.050 |
|  | PCPL-PPO | 0.593 ± 0.052 | **0.699 ± 0.100** | **0.952 ± 0.028** |
| **5b** | PD-MORL | 0.962 ± 0.005 | **0.558 ± 0.077** | 0.815 ± 0.132 |
|  | Envelope | 0.404 ± 0.199 | 0.145 ± 0.130 | 0.692 ± 0.068 |
|  | PSL | **0.976 ± 0.013** | 0.514 ± 0.100 | **0.934 ± 0.060** |
|  | PCPL-PPO | 0.966 ± 0.004 | 0.209 ± 0.075 | 0.922 ± 0.032 |
| **5c** | PD-MORL | 0.903 ± 0.062 | 0.516 ± 0.050 | 0.751 ± 0.132 |
|  | Envelope | 0.405 ± 0.161 | 0.132 ± 0.064 | 0.488 ± 0.117 |
|  | PSL | **0.924 ± 0.099** | 0.266 ± 0.109 | 0.647 ± 0.056 |
|  | PCPL-PPO | 0.898 ± 0.039 | **0.859 ± 0.202** | **0.916 ± 0.076** |
| **5d** | PD-MORL | 0.484 ± 0.196 | **0.312 ± 0.180** | 0.521 ± 0.098 |
|  | Envelope | **0.692 ± 0.047** | 0.188 ± 0.066 | 0.625 ± 0.047 |
|  | PSL | 0.658 ± 0.135 | 0.197 ± 0.060 | **0.800 ± 0.047** |
|  | PCPL-PPO | 0.516 ± 0.166 | 0.094 ± 0.034 | 0.686 ± 0.080 |
| **5e** | PD-MORL | 0.464 ± 0.164 | 0.625 ± 0.098 | 0.565 ± 0.099 |
|  | Envelope | 0.520 ± 0.105 | **0.703 ± 0.125** | 0.749 ± 0.044 |
|  | PSL | **0.900 ± 0.070** | 0.699 ± 0.060 | **0.864 ± 0.014** |
|  | PCPL-PPO | 0.723 ± 0.126 | 0.483 ± 0.171 | 0.754 ± 0.036 |

---

## Problem Set Description

| Problems | N | \|P\| | FC | Description |
|----------|---|-------|----|-------------|
| 0 | 2 | 2 | Yes | Baseline simple logarithmic objective functions with smooth convex Pareto Fronts. Fully-Connected (FC) dependency graph: every demand depends on every resource. |
| 1a–1c | 2 | 2 | Yes | Difficult objective functions, including oscillatory behavior, stationary rewards, and spikes. |
| 2a–2c | 5 | 2 | Yes | More demands, which increases the action and observation spaces. |
| 3a–3b | 5 | 5 | Yes | 5 objectives with sparse rewards, building on difficult objective functions from previous testcases. |
| 4a–4b | 5 | 5 | No | Varied dependencies and resources, instead of Fully-Connected (FC) dependency graphs. |
| 5a–5e | 5 | 3–5 | No | Testcases with dependencies, resources, and objectives sampled randomly. Extended horizon for allocating more available resources. |
| 6a–6c | 100 | 5 | No | Random testcases with simple convex functions, but with more complex graph structures (100 demands, 100 resources). The ideal Pareto Front is not computed due to computational complexity (see Table 1 / Problem 6 in the paper). |

*N = number of demands (\|D\|), \|P\| = number of objectives, FC = fully-connected dependency graph. For more detailed problem definitions, refer to Appendix E (Table 2) of the paper.*

---

## Experimental Setup

- Each baseline (PD-MORL, Envelope Q-Learning, PSL) used a per-problem random hyperparameter search, with the best configuration evaluated over 5 seeds at up to 1M steps.
- **PCPL-PPO**: per-problem random hyperparameter search, best config evaluated over 5 seeds.
- Hypervolume is normalized against the reference Pareto front for each problem.
- Sample standard deviation (ddof=1) reported for ± values, matching the paper.
- Source data: `graphallocbench/examples/data/{pdmorl,envelope,psl,pcpl}_stats.csv`.

