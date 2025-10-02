# GraphAllocBench Environment Implementation Details

## Problem Definition
Suppose we have a city which has a set of resources R, and a set of demands D. Each demand has a set of dependencies R'(d) which is a subset of R.

### Key Terms
- Resource: Constraints over actions e.g. {water, food, housing}
- Allocation: Action e.g. use water to provide drinking water
- Demand: This is a dynamic environment e.g. {drinking water, sanitation services, food bank}
- Objectives: reward functions e.g. {hunger satisfaction, sanitation access}

## Preferences Vector
Our preferences vector `w` (i.e. weights) can be defined as such:
```python
w = [w_1, w_2, ..., w_N]
``` 

There are two types of weights:

1. `weights_type='fixed'`
    
    This means that we randomly generatean array of weights of length N corresponding to each of the N objectives, from a fixed discrete set of options.

2. `weights_type='uniform'`
    
    This means that we randomly generate a `w` array of length N, from a Dirichlet distribution.

Note that for both cases,

$$\sum_n w_n = 1$$

## Action Vector

Our action is a dictionary comprising of 3 actions:

1. Choose to add / remove / do nothing
2. Choose a specific demand to act on

## Requirements Matrix
The `requirements_matrix` is a binary matrix of shape `(n_demands, n_resources)`, containing 1 if the a resource depends on a demand, and 0 otherwise.

## Allocation Matrix
The `allocation_matrix` has shape `(n_demands + 1, n_resources)`, and stores the number of of resource $r_i$ assigned to demand $d_i$.

All resources start off unallocated.

## Observation
The observation vector returned from the environment is a dictionary containing the flattened normalized `allocation_matrix` and the `requirements_matrix` which may be used for graph based implementations.

If a preferences vector `w` is defined (see [here](#preferences-vector)), we concatenate `w` to the final observation vector.

## Productions
Productions $P$ is a vector of shape `(n_demands,)` which represents the minimum amount of resources allocated across all resource dependencies for each demand. We enumerate the resource dependencies using the `requirements_matrix`, and calculate the minimum resources allocated across all resource dependencies using the `allocation_matrix`.

For each demand, its production is set to the minimum number of allocated units among all resources it requires. This models a system where a demand can only be fulfilled as much as its most limiting required resource allows.

### Production State
The production state is a vector of shape `(n_demands,)` which   to the total number of resources successfully produced for each demand over time.

## Objective Function and Vector Components
Our objective function `F` can be defined as follows:

```python
Phi = [J_1, J_2, ..., J_N]
```

Each $J_n$ component of $J$ is defined as follows ($Z_n'$ is an intermediate value):

$Z_n'(x_i) = a_{n,i} x_i^2 + b_{n,i} x_i + c_{n,i} + \left[ (d_{n,i} x_i + e_{n,i}) \cdot \log(f_{n,i} x_i + 0.0001 + g_{n,i}) \right] + \left[ \frac{h_{n,i}}{1 + \exp(-i_{n,i}(x_i - j_{n,i}))} \right] + (\alpha_{n,i} x + \beta_{n,i})\sin (\gamma_{n,i} x_i + \zeta_{n,i}) + \rho_{n,i} \exp (-\phi_{n,i} (x_i - \mu_{n,i})^2 ) + \sqrt{u_{n,i} x_i + v_{n,i}}$

```python
if mode == 'max':
    Z_n = max(Z_n_prime, k_ni)
elif mode == 'min':
    Z_n = min(Z_n_prime, k_ni)
else:
    Z_n = Z_n_prime
```

where the constants in the definition of $Z_n'$ are objective-specific variables, and $x$ is the relevant input defined for the objective (usually production, i.e. $x_i = p_i$). For each value of $n$, we sum up the $Z_n$ across all $x_i$ (e.g. sum across all productions) to obtain $J_n$, which is the component in the n-th dimension of the final objective vector $J$.

These objective-specific variables are expressed either in terms of demand, available resources or constant, determined by the type of objective (covered [here](#types-of-objectives)).


For example, if we are evaluating productions,

Let the objective vector be $J$, where

$$J = [J_1, J_2, ..., J_N]$$

For each component of the objective vector $J_n$,

$$J_n = \sum_i Z_n(p_i)$$

### Types of Objectives
Take $0 \leq i < |D|$ or $0 \leq i < |R|$. depending on `eval_production` or `eval_allocation` respectively. Each objective will take in $x$ and $y$ as inputs. 
- `eval_production`

    $x=p_i$
    
    $y=d_i$ ($i$ th demand)
- `eval_allocation`

    $x =$ `np.sum(allocation_matrix[:, p_i])` (total number of resources allocated to production $i$)

    $y = r_i$ ($i$ th resource)

During the calculation of each of the objective-specific variables (e.g. $a_{n,i}$), each variable will have their own $\omega_0, \omega_1, \omega_2$ such that

$$a_{n,i} = \omega_0 y + \omega_1 / y + \omega_2$$

The same applies for $b_{n,i}$, $c_{n,i}$, etc.

This allows us to scale our objective variables $a_{n,i}$, $b_{n,i}$, $c_{n,i}$, etc. with respect to $y$, which can be $d_i$ or $r_i$.

## Reward Function

There are two modes of rewards, scalar and vector.

### Scalar Rewards
Using $J$ and weights `w`, we can calculate our scalar reward function:

$$\text{Reward} = J \cdot w$$

### Vector Rewards
If we specify that we want vector rewards, we can just use the objective vector $J$.

## Metrics
These metrics below will be returned as part of the `info` dictionary in the `step()` function.
- `step_scalar_reward` = $J \cdot w$

- `correct_allocation_reward`

    This is determined based on the `requirements_matrix`. This is not used as actual reward in the default implementation, and it is just used as a visualized statistic to understand how frequently the agent is allocating resources to demands. For example, for some problems, the total allocation reward will increase as the agent explores more and adds more productions than necessary, before decreasing over many epochs as more efficient trajectories toward the best allocation are found.

- `current_step`

    This counter is incremented every time `step()` function is called.

- `step_objectives`

    This is the objectives vector $J$, converted to a Python List format.

- `step_productions`

    This is the productions vector $P$, converted to a Python List format.

- `episode`

    - `l` = `current_step`
    - `t` = 0 (time, possible future implementation)
    - `r` = `episode_objectives`
    - `dr` = `episode_objectives`
    - `final_r` = $J$

Note that episode objectives is the sum of all objectives obtained throughout the entire episode.

## End Conditions

Let $L_\text{max}$ be the maximum number of steps we allow in a single episode. The end condition, specified by the boolean flag `done`, is defined as

$$l >= L_\text{max}$$

where $l$ is the current step.