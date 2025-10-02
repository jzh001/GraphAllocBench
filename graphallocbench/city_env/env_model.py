"""Environment model (copied from city-management/city_env/env_model.py with adjusted imports)."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
import yaml
from pathlib import Path
import math
from matplotlib import colors
import random

from .scalarize import scalarize as scalarize_helper


class CityPlannerEnv(gym.Env):
    def __init__(self, config_path):
        super(CityPlannerEnv, self).__init__()
        # Resolve config path robustly (accept absolute path, repo-relative, or basename)
        def _resolve_config_path(path_str: str) -> str:
            p = Path(path_str)
            # Direct file
            if p.is_file():
                return str(p)
            # If given a path like config/problems/problem_x.yml, try relative to repo root
            # Try package-relative locations and upward search
            name = p.stem
            # Search common candidate locations starting from CWD upwards
            start = Path.cwd()
            for parent in [start] + list(start.parents):
                cand1 = parent / 'graphallocbench' / 'config' / 'problems' / f"{name}.yml"
                if cand1.is_file():
                    return str(cand1)
                cand2 = parent / 'config' / 'problems' / f"{name}.yml"
                if cand2.is_file():
                    return str(cand2)
            # Fallback: use package location (two levels up from this file)
            repo_root = Path(__file__).resolve().parents[2]
            cand3 = repo_root / 'graphallocbench' / 'config' / 'problems' / f"{name}.yml"
            if cand3.is_file():
                return str(cand3)
            raise FileNotFoundError(f"Could not locate problem config for '{path_str}'. Searched common locations.")

        resolved_path = _resolve_config_path(config_path)
        # Load configuration from YAML file
        with open(resolved_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.config_path = resolved_path
        self.config_name = Path(resolved_path).name.replace('.yml', '')

        # Predefined Configurations
        self.max_steps = self.config.get("max_steps", 30)
        
        self.scalar_rewards = (
            True  # If True, reward is scalar (dot product), else vector
        )
        self.episode_reward = False  # If True, reward only at end of episode

        self.reset_weights_per_step = False

        self.demand_count = self.config["demand_count"]
        self.demands = self.config["demands"]
        self.resource_count = self.config["resource_count"]
        self.avail_resources = self.config["avail_resources"]
        self.n_objectives = len(self.config["objectives"])
        self.scalarization_method = self.config.get("scalarization_method", "smooth_tchebycheff")
        self.smoothness = self.config.get("smoothness", 0.01)

        assert self.demand_count == len(self.demands)
        assert self.resource_count == len(self.avail_resources)
        assert self.n_objectives > 0
        assert self.config["weights"]["type"] in ["fixed", "uniform"]
        if self.config["weights"]["type"] == "fixed":
            assert len(self.config["weights"]["values"]) == self.n_objectives
            self.isWeightsFixed = True
            self.weights = np.array(self.config["weights"]["values"])
        else:
            self.isWeightsFixed = False

        self.n_objectives = len(self.config["objectives"])

        self.action_space = spaces.MultiDiscrete([
            3, # add, remove or do nothing
            self.demand_count, # choose demand to add / remove 1 unit of production
        ])

        self.observation_space = spaces.Dict(
            {
                "allocation_matrix": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.demand_count + 1, self.resource_count),
                    dtype=np.float32,
                ),
                "requirements_matrix": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.demand_count, self.resource_count),
                    dtype=np.float32,
                ),
                "prefs": spaces.Box(
                    low=0, high=1, shape=(self.n_objectives,), dtype=np.float32
                ),
            }
        )

        # Set reward_space based on scalar_rewards
        if self.scalar_rewards:
            self.reward_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        else:
            self.reward_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_objectives,), dtype=np.float32
            )

        self.estimated_ideal = np.array(self.config.get("estimated_ideal", np.zeros(self.n_objectives)), dtype=np.float32)
        self.estimated_nadir = np.array(self.config.get("estimated_nadir", np.full(self.n_objectives, np.inf)), dtype=np.float32)

        self.ideal_objectives = self.config.get("ideal_objectives", None)
        if self.ideal_objectives is not None:
            assert len(self.ideal_objectives) == self.n_objectives, "Ideal objectives must match number of objectives in config"
            self.ideal_objectives = np.array(self.ideal_objectives, dtype=np.float32)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        self.cmap = colors.LinearSegmentedColormap.from_list('yellow_blue', ['yellow', 'blue'])
        self.norm = plt.Normalize(vmin=0, vmax=max(self.avail_resources))
        self.sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        self.sm.set_array([])
        plt.close(self.fig)  # Suppress axes display on creation
    
    def set_scalarization_method(self, method, smoothness = None, log_smoothness = None):
        self.scalarization_method = method
        if smoothness is not None:
            self.smoothness = smoothness
        elif log_smoothness is not None:
            self.smoothness = np.exp(log_smoothness)

    def reset(self, seed=None):
        self.current_step = 0

        # Parse configuration for requirements_matrix
        self.requirements_matrix = np.zeros((self.demand_count, self.resource_count))
        for edge in self.config["dependencies"]:
            assert len(edge) == 2
            u, v = edge
            self.requirements_matrix[u, v] = 1  # Resource v is required by Demand u

        # Now we initialize allocation_matrix (all resources are unallocated initially)
        self.allocation_matrix = np.zeros(
            (self.demand_count + 1, self.resource_count)
        )  # +1 for unallocated

        for i in range(self.resource_count):
            self.allocation_matrix[self.demand_count, i] = self.avail_resources[i]

        self.productions = np.zeros(self.demand_count)  # productions for each step
        self.production_state = np.zeros(
            self.demand_count
        )  # total productions over time

        if not self.isWeightsFixed:
            self.set_random_weights()

        self.episode_scalar_reward = 0
        self.episode_objectives = np.zeros(self.n_objectives)

        self.estimated_ideal = np.maximum(self.estimated_ideal, self.get_objectives(self.productions, self.allocation_matrix))
        self.estimated_nadir = np.minimum(self.estimated_nadir, self.get_objectives(self.productions, self.allocation_matrix))

        obs = self.get_observation()

        return (obs, {})
    
    def set_random_weights(self):
        alpha0 = self.config.get("weights", {}).get("alpha0", None)
        alpha1 = self.config.get("weights", {}).get("alpha1", None)

        alpha = self.config.get("weights", {}).get("alpha", 1)

        if alpha0 is not None and alpha1 is not None:
            if random.random() < 0.5:
                alpha = alpha0
            else:
                alpha = alpha1

        self.weights = np.random.dirichlet(alpha * np.ones(self.n_objectives))
    
    def fix_weights(self, weights: list):
        """
        Fix weights for inference/evaluation.
        """
        assert len(weights) == self.n_objectives
        self.weights = np.array(weights)
        self.isWeightsFixed = True

    def set_ideal_objectives(self, ideal_objectives):
        self.ideal_objectives = ideal_objectives

    def scalarize(self, objectives_components):

        return scalarize_helper(weights=self.weights,
                    objectives_components=objectives_components,
                    ideal_point=self.estimated_ideal,
                    nadir_point=self.estimated_nadir,
                    scalarization_method=self.scalarization_method,
                    normalize = self.config.get("normalize_objectives", True),
                    smoothness=self.smoothness
                    )

    def try_allocate(self, raw_action):
        addOrRemove = int(raw_action[0])
        demand_idx = int(raw_action[1])

        allocate_success = 1
        
        if addOrRemove == 0: # Add
            for r in range(self.resource_count):
                if (self.requirements_matrix[demand_idx, r] == 1
                    and self.allocation_matrix[self.demand_count, r] == 0):
                    allocate_success = 0
                    break
            
            if allocate_success == 1:
                for r in range(self.resource_count):
                    if self.requirements_matrix[demand_idx, r] == 1:
                        self.allocation_matrix[self.demand_count, r] -= 1
                        self.allocation_matrix[demand_idx, r] += 1

        elif addOrRemove == 1: # Remove
            for r in range(self.resource_count):
                if (self.requirements_matrix[demand_idx, r] == 1
                    and self.allocation_matrix[demand_idx, r] == 0):
                    allocate_success = 0
                    break
            
            if allocate_success == 1:
                for r in range(self.resource_count):
                    if self.requirements_matrix[demand_idx, r] == 1:
                        self.allocation_matrix[self.demand_count, r] += 1
                        self.allocation_matrix[demand_idx, r] -= 1
        else: # Do nothing
            allocate_success = 0

        return allocate_success
    
    def get_observation(self):
        normalized_allocation = self.allocation_matrix / self.allocation_matrix.sum(
            axis=0, keepdims=True
        )
        obs = {
            "allocation_matrix": normalized_allocation.astype(np.float32),
            "requirements_matrix": self.requirements_matrix.astype(np.int64),
            "prefs": self.weights.astype(np.float32),
        }
        
        return obs
    
    def check_productions_possible(self, productions):
        self.reset()
        alloc_copy = np.copy(self.allocation_matrix)
        for demand_idx in range(self.demand_count):
            required_resources = self.requirements_matrix[demand_idx] == 1
            for resource_idx, required in enumerate(required_resources):
                if required:
                    if alloc_copy[self.demand_count, resource_idx] < productions[demand_idx]:
                        return None # fail
                    alloc_copy[self.demand_count, resource_idx] -= productions[demand_idx]
                    alloc_copy[demand_idx, resource_idx] += productions[demand_idx]
        
        return alloc_copy

    def get_objectives(self, productions, allocation_matrix):
        objective_results = []
        productions_sum = np.sum(productions)
        allocations_sum = None
        if allocation_matrix is not None:
            allocations_sum = np.sum(allocation_matrix[:-1])  # Sum all rows except the last (unallocated)

        for objective in self.config["objectives"]:
            objective_returns = 0

            if objective["type"] == "eval_entropy":
                ceoff = objective["coeff"]
                mode = objective["mode"]

                if mode == "prod":
                    if productions_sum > 0:
                        probabilities = productions / productions_sum
                        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                    else:
                        entropy = 0
                    objective_returns += ceoff * entropy
                elif mode == "alloc":
                    resources_used = np.sum(allocation_matrix[:-1], axis=0)
                    if allocations_sum > 0:
                        probabilities = resources_used / allocations_sum
                        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                    else:
                        entropy = 0
                    objective_returns += ceoff * entropy
                else:
                    raise Exception("Mode not valid")
            else:
                for p_i in range(len(objective["params"])):
                    if objective["type"] == "eval_production":
                        x = productions[p_i]
                        y = self.config["demands"][p_i]
                    elif objective["type"] == "eval_allocation":
                        x = np.sum(allocation_matrix[:-1, p_i])
                        y = self.avail_resources[p_i]
                    
                    eval_config = objective["params"][p_i]
                    k = self.get_term(eval_config, y, "k")
                    if eval_config["outer_op"] == "max":
                        objective_returns += max(self.get_eval(eval_config, x, y), k)
                    elif eval_config["outer_op"] == "min":
                        objective_returns += min(self.get_eval(eval_config, x, y), k)
                    elif eval_config["outer_op"] == "none":
                        objective_returns += self.get_eval(eval_config, x, y)

            objective_results.append(max(objective_returns, 0))  # Ensure non-negative objectives
        
        return objective_results

    def calculate_productions(self, allocation_matrix = []):
        if len(allocation_matrix) == 0:
            allocation_matrix = self.allocation_matrix
        self.productions = np.zeros(self.demand_count, dtype=np.int32)
        for demand_idx, row in enumerate(allocation_matrix[:-1]):
            self.productions[demand_idx] = allocation_matrix[demand_idx][
                self.requirements_matrix[demand_idx] == 1
            ].min()
        self.production_state = self.production_state + self.productions

    def get_term(self, p_i, y, component):
        if component not in p_i:
            return 0
        return (
            p_i[component]["y"] * y
            + p_i[component]["y_inv"] * (1 / y)
            + p_i[component]["constant"]
        )

    def get_eval(self, p_i, x, y):
        a = self.get_term(p_i, y, "a")
        b = self.get_term(p_i, y, "b")
        c = self.get_term(p_i, y, "c")
        d = self.get_term(p_i, y, "d")
        e = self.get_term(p_i, y, "e")
        f = self.get_term(p_i, y, "f")
        g = self.get_term(p_i, y, "g")
        h = self.get_term(p_i, y, "h")
        i = self.get_term(p_i, y, "i")
        j = self.get_term(p_i, y, "j")
        u = self.get_term(p_i, y, "u")
        v = self.get_term(p_i, y, "v")

        alpha = self.get_term(p_i, y, "alpha")
        beta = self.get_term(p_i, y, "beta")
        gamma = self.get_term(p_i, y, "gamma")
        zeta = self.get_term(p_i, y, "zeta")
        rho = self.get_term(p_i, y, "rho")
        phi = self.get_term(p_i, y, "phi")
        mu = self.get_term(p_i, y, "mu")

        res = (
            (x**2) * a
            + (x) * b
            + c
            + ((d * x + e) * math.log((f * x + 0.0001) + g))
            + (h / (1 + math.exp(- i * (x - j))))
            + (alpha * x + beta)*math.sin(gamma * x + zeta)
            + (rho * math.exp(- phi * ((x - mu)**2)))
            + math.sqrt(u * x + v)
        )

        return res

    def step(self, raw_action):
        self.current_step += 1

        if self.reset_weights_per_step and not self.isWeightsFixed:
            self.set_random_weights()

        allocation_reward = self.try_allocate(raw_action)

        obs = self.get_observation()

        done = self.current_step >= self.max_steps

        self.calculate_productions()

        objectives_components = np.array(
            self.get_objectives(self.productions, self.allocation_matrix),
            dtype=np.float32,
        )

        self.estimated_ideal = np.maximum(self.estimated_ideal, objectives_components)
        self.estimated_nadir = np.minimum(self.estimated_nadir, objectives_components)

        self.episode_objectives = self.episode_objectives + objectives_components
        step_scalar_reward = self.scalarize(objectives_components)
        self.episode_scalar_reward += step_scalar_reward

        info = {
            "step_scalar_reward": step_scalar_reward,
            "correct_allocation_reward": allocation_reward,
            "current_step": self.current_step,
            "step_objectives": objectives_components.tolist(),
            "step_productions": self.productions.tolist(),
            "estimated_ideal_mean": np.mean(self.estimated_ideal).item(),
        }
        if done:
            info["episode"] = {
                "l": self.current_step,
                "t": 0,
                "r": self.episode_objectives,
                "dr": self.episode_objectives,
                "final_r": objectives_components.tolist(),
            }
            info["final_allocation"] = self.allocation_matrix.copy()

        if self.episode_reward and self.scalar_rewards:
            reward = self.episode_scalar_reward if done else 0
        elif self.episode_reward and not self.scalar_rewards:
            reward = self.episode_objectives if done else np.zeros(self.n_objectives)
        elif not self.episode_reward and self.scalar_rewards:
            reward = step_scalar_reward
        else:
            reward = objectives_components

        return (obs, reward, done, done, info)

    def render(self, mode='rgb_array'):
        rounded_weights = [round(w, 2) for w in self.weights.tolist()]
        rounded_objectives = [round(float(j), 2) for j in self.get_objectives(self.productions, self.allocation_matrix)]
        self.fig.suptitle(f"Step: {self.current_step}, Weights: {rounded_weights}, Objectives: {rounded_objectives}")

        for ax in self.axes:
            ax.clear()

        resource_positions = {f"R{i}": (i, 1) for i in range(self.resource_count)}
        demand_positions = {f"D{i}": (i, 0) for i in range(self.demand_count)}
        demand_positions[f"UA"] = (self.demand_count, 0)
        pos = {**resource_positions, **demand_positions}

        G_requirements = nx.DiGraph()
        for resource_idx in range(self.resource_count):
            G_requirements.add_node(f"R{resource_idx}")
        for demand_idx in range(self.demand_count):
            G_requirements.add_node(f"D{demand_idx}")
        for resource_idx in range(self.resource_count):
            for demand_idx in range(self.demand_count):
                if self.requirements_matrix[demand_idx][resource_idx] == 1:
                    G_requirements.add_edge(f"R{resource_idx}", f"D{demand_idx}")

        nx.draw(G_requirements, pos, ax=self.axes[0], with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
        self.axes[0].set_title("Requirements Graph")

        G_allocation = nx.DiGraph()
        edge_colors = []

        for resource_idx in range(self.resource_count):
            G_allocation.add_node(f"R{resource_idx}")
        for demand_idx in range(self.demand_count):
            G_allocation.add_node(f"D{demand_idx}")
        G_allocation.add_node("UA")
        for resource_idx in range(self.resource_count):
            for demand_idx in range(self.demand_count + 1):
                allocation_value = self.allocation_matrix[demand_idx][resource_idx]
                if allocation_value > 0:
                    if demand_idx == self.demand_count:
                        G_allocation.add_edge(f"R{resource_idx}", "UA")
                    else:
                        G_allocation.add_edge(f"R{resource_idx}", f"D{demand_idx}")
                    edge_colors.append(self.cmap(self.norm(allocation_value)))

        nx.draw(G_allocation, pos, ax=self.axes[1], with_labels=True, node_size=2000, node_color='lightgreen', font_size=10, font_weight='bold', edge_color=edge_colors)
        self.axes[1].set_title("Allocation Graph")

        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig.colorbar(self.sm, ax=self.axes[1], orientation='vertical', label='Allocation Value')

        self.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.asarray(self.canvas.buffer_rgba()).reshape(int(height), int(width), 4)[..., :3]

        return image
