# Problem Set

## Design Philosophy
- Pareto Front should span a sufficient range so that we can have a significant number of discrete solutions for different preferences. We should avoid degenerate solutions as much as possible. This makes the multi-objective optimization problem more meaningful.
- Pareto Front should cover different types of shapes (i.e. convex and non-convex areas), which should test for the robustness of the scalarization used.
- Some of the tests will be designed to be adversarial, and aim to trap less optimal algorithms with local optimas and sparse rewards.
- We use the Das and Dennis method for sampling during inference, to ensure consistent spacing between objective vectors across different dimensions. We sample for 12 partitions, which may not be sufficient to cover entire analytical Pareto Front. However, we do this for computational efficiency, and also to enforce that our points are sufficiently spread out to cover as much Pareto Front as possible. Hence our analysis will penalize clustering of points on the Pareto Front indirectly due to a smaller number of samples.
- From a reinforcement learning perspective, we will also test for how RL algorithms deal with large and sparse action and observation spaces, as well as large number of objectives.
- We will split our testcases into two main areas for testing:
    1. Ideal Pareto Front can be computed. (We will compare hypervolume and other metrics with the ideal analytical solution)
    2. Ideal Pareto Front cannot be computed analytically, or within reasonable compute.


## Categories

1. Simple Problems with Known Pareto Front (3 testcases)

    These problems involve only 2 objectives, with fully connected dependencies, to test for the RL Agent's ability to approximate the entire Pareto Front given the ideal Pareto Front.
    - Non-Concave Pareto Front + Scaling
    - Sinusoidal
    - Stationary Rewards + Peak

2. Scaling up to more productions (2 testcases)
    
    These problems are designed to test the model's robustness when it comes to increasingly sparse observation spaces. We define two types of problems in this domain.

    - Increasing separable objective functions only (this clear tradeoff pattern allows us to evaluate the ideal Pareto Front)
    - General objective functions (we evaluate the hypervolume)

3. Scaling up to more objectives (2 testcases)

    - 5 separable objectives with increasing objective functions
    - 5 general objectives

1, 2 and 3 will involve fully connected dependencies, and will only work on evaluating production.

4. Dependencies (3 testcases)
    
    - Including non-separable objectives such as entropy, together with different dependencies, number of resources, and number of productions
    - Assymmetric number of resources
    - Evaluation of allocation in addition to evaluation of production?

5. General (5 testcases)

    - This will be a mix of 1, 2, 3 and 4.
    - Ideal pareto front may not be known

Total: 15 testcases

## Problem 1a

Stationary or Non-Monotonous Objectives.

## Problem 1b

Oscillating Objectives with multiple local minima in each objective function.

## Problem 1c
Complex Pareto Front shape with non-convex and gradual shapes.

## Problem 2a
2 resources, 5 demands. Individual objective functions are all monotonous, and clear tradeoff is imposed, where improving one objective must result in giving up some reward for another objective. Different scaling.

## Problem 2b
A more general case of Problem 2a, where individual objective functions need not be monotonous. In particular, we have two different convex segments of the Pareto Front, introducing a local minima for an algorithm to become trapped at one of the convex segments.

## Problem 2c
Highly concave Pareto Front with oscillations, sparse reward.

## Problem 3a
Larger number of objectives, sparse rewards with respect to productions.

## Problem 3b
Increasing functions, concave pareto front. Sparse rewards.

## Problem 4a
Sparse dependencies and assymmetric number of resources, with otherwise same setup as Problem 3b.

## Problem 4b
Sparse dependencies and assymmetric number of resources, with a mix of objectives from Problems 2b and 2c (i.e. convex with local pareto fronts in 2 dimensions, concave in other 2 dimensions, with entropy constraint)

## Problem 5
Generated, can have any number of productions and resources. At most 5 objectives.

## Additional Note
We enforce that objective vectors are always non-negative (if the function drops below 0, we take it as 0).
