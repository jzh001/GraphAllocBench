# GraphAllocBench — Evaluation Results

Results for the 16-problem benchmark comparing **PD-MORL** (MO-DDQN-HER, Basaklar et al. 2023) and **PCPL** (PPO + Smooth Tchebycheff Scalarization) across three metrics.
All values are mean ± population std over 5 independent seeds.
**Bold** indicates the better result per metric per problem.

---

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **HV Ratio** | Normalized hypervolume of the recovered Pareto front relative to the reference front | [0, 1] ↑ |
| **PNDS** | Proportion of Non-Dominated Solutions — fraction of evaluated preference vectors whose outcome lies on the Pareto front | [0, 1] ↑ |
| **OS** | Ordering Score — how well the policy respects the preference ordering across objectives | [0, 1] ↑ |

---

## Results Table

| Problem | Method | HV Ratio | PNDS | OS |
|---------|--------|----------|------|----|
| **0** | PD-MORL | 0.986 ± 0.008 | 0.771 ± 0.196 | **1.000 ± 0.000** |
| | PCPL | **0.997 ± 0.007** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **1a** | PD-MORL | 0.646 ± 0.064 | 0.533 ± 0.174 | **0.952 ± 0.043** |
| | PCPL | **0.724 ± 0.049** | **0.615 ± 0.169** | 0.947 ± 0.008 |
| **1b** | PD-MORL | **0.645 ± 0.039** | **0.933 ± 0.071** | **0.995 ± 0.011** |
| | PCPL | 0.632 ± 0.067 | 0.738 ± 0.151 | 0.986 ± 0.008 |
| **1c** | PD-MORL | **0.312 ± 0.302** | **0.438 ± 0.285** | **0.975 ± 0.050** |
| | PCPL | 0.230 ± 0.188 | 0.215 ± 0.090 | 0.945 ± 0.089 |
| **2a** | PD-MORL | 0.238 ± 0.188 | 0.133 ± 0.070 | **1.000 ± 0.000** |
| | PCPL | **0.989 ± 0.004** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| **2b** | PD-MORL | 0.560 ± 0.015 | **0.648 ± 0.246** | **0.968 ± 0.026** |
| | PCPL | **0.605 ± 0.163** | 0.554 ± 0.277 | 0.896 ± 0.057 |
| **2c** | PD-MORL | 0.200 ± 0.042 | 0.514 ± 0.288 | 0.941 ± 0.059 |
| | PCPL | **0.680 ± 0.084** | **0.769 ± 0.138** | **0.987 ± 0.016** |
| **3a** | PD-MORL | 0.505 ± 0.104 | 0.014 ± 0.014 | 0.860 ± 0.029 |
| | PCPL | **0.827 ± 0.047** | **0.268 ± 0.020** | **0.917 ± 0.043** |
| **3b** | PD-MORL | 0.053 ± 0.098 | 0.006 ± 0.005 | 0.797 ± 0.068 |
| | PCPL | **0.301 ± 0.138** | **0.014 ± 0.010** | **0.866 ± 0.031** |
| **4a** | PD-MORL | 0.695 ± 0.157 | **0.026 ± 0.046** | 0.757 ± 0.084 |
| | PCPL | **0.829 ± 0.113** | 0.017 ± 0.004 | **0.854 ± 0.064** |
| **4b** | PD-MORL | 0.417 ± 0.238 | 0.093 ± 0.078 | 0.644 ± 0.143 |
| | PCPL | **0.802 ± 0.100** | **0.718 ± 0.132** | **0.963 ± 0.018** |
| **5a** | PD-MORL | **0.595 ± 0.006** | 0.153 ± 0.233 | 0.849 ± 0.104 |
| | PCPL | 0.588 ± 0.025 | **0.262 ± 0.071** | **0.979 ± 0.008** |
| **5b** | PD-MORL | **0.957 ± 0.003** | 0.095 ± 0.078 | 0.778 ± 0.109 |
| | PCPL | 0.940 ± 0.035 | **0.121 ± 0.053** | **0.943 ± 0.021** |
| **5c** | PD-MORL | 0.880 ± 0.075 | 0.059 ± 0.035 | 0.744 ± 0.101 |
| | PCPL | **0.905 ± 0.038** | **0.648 ± 0.399** | **0.916 ± 0.047** |
| **5d** | PD-MORL | 0.678 ± 0.136 | **0.033 ± 0.018** | **0.665 ± 0.081** |
| | PCPL | **0.699 ± 0.103** | 0.027 ± 0.016 | 0.621 ± 0.066 |
| **5e** | PD-MORL | 0.633 ± 0.022 | **0.352 ± 0.172** | 0.595 ± 0.071 |
| | PCPL | **0.721 ± 0.066** | 0.288 ± 0.076 | **0.768 ± 0.099** |

---

## Problem Set Description

| Problems | \|P\| | N | FC | Description |
|----------|-------|---|----|-------------|
| 0 | 2 | 2 | Yes | Baseline simple logarithmic objective functions with smooth convex Pareto Fronts. The baseline problem uses a Fully-Connected (FC) dependency graph, where every demand depends on every resource. |
| 1a–1c | 2 | 2 | Yes | Difficult objective functions, including oscillatory behavior, stationary rewards, and spikes. |
| 2a–2c | 5 | 2 | Yes | More demands, which increases the action and observation spaces. |
| 3a–3b | 5 | 5 | Yes | 5 objectives with sparse rewards, building on difficult objective functions from previous testcases. |
| 4a–4b | 5 | 5 | No | Varied dependencies and resources, instead of Fully-Connected (FC) dependency graphs. |
| 5a–5e | 5 | 3–5 | No | Testcases with dependencies, resources, and objectives sampled randomly. Extended horizon for allocating more available resources. |
| 6a–6c | 100 | 5 | No | Random testcases with simple convex functions similar to baseline, but with more complex graph structures with 100 demands. The ideal Pareto Front is not computed due to computational complexity. |

*\|P\| = number of objectives, N = number of demands, FC = fully-connected dependency graph. For more detailed problem definitions, refer to Appendix E (Table 2) of the paper.*

---

## Experimental Setup

- **PD-MORL**: Per-problem random hyperparameter search over 20 trials, best config evaluated over 5 seeds.
- **PCPL**: Per-problem random hyperparameter search, best config evaluated over 5 seeds.
- Hypervolume normalized against the reference Pareto front for each problem.
- Population standard deviation (ddof=0) reported for ± values.
