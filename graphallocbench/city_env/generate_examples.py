import yaml
from copy import deepcopy
import os
import random
import numpy as np

# Copied utility functions for generating synthetic problem configs.


def expand_productions(n_configs=5,
                       config_file='concave',
                       input_directory='template',
                       output_directory='test',
                       even_only=True):
    config_path = f'config/{input_directory}/{config_file}.yml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    new_config_paths = []
    upper_bound = 2 + n_configs * 2 if even_only else 2 + n_configs
    step = 2 if even_only else 1
    for n_prod in range(2, upper_bound, step):
        new_config_path = f'config/{output_directory}/{config_file}-p{n_prod}.yml'
        new_config_paths.append(new_config_path)
        new_config = deepcopy(config)
        new_config['demand_count'] = n_prod
        new_config['demands'] = [config['demands'][0]] * n_prod
        for obj_idx in range(len(config['objectives'])):
            new_config['objectives'][obj_idx]['params'] = []
            params0 = deepcopy(config['objectives'][0]['params'][0])
            params1 = deepcopy(config['objectives'][0]['params'][1])
            for j in range(n_prod):
                new_config['objectives'][obj_idx]['params'].append(params0 if j % 2 == obj_idx % 2 else params1)
        new_config['dependencies'] = [[u, v] for u in range(n_prod) for v in range(config['resource_count'])]
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        with open(new_config_path, "w") as outfile:
            yaml.dump(new_config, outfile, default_flow_style=False, sort_keys=False)
    return new_config_paths


def expand_productions_and_objectives(n_configs=5,
                                      config_file='convex',
                                      input_directory='template',
                                      output_directory='test',
                                      even_only=True):
    config_path = f'config/{input_directory}/{config_file}.yml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    new_config_paths = []
    upper_bound = 2 + n_configs * 2 if even_only else 2 + n_configs
    step = 2 if even_only else 1
    for n_prod in range(2, upper_bound, step):
        new_config_path = f'config/{output_directory}/{config_file}-p{n_prod}-o{n_prod}.yml'
        new_config_paths.append(new_config_path)
        new_config = deepcopy(config)
        new_config['demand_count'] = n_prod
        new_config['demands'] = [config['demands'][0]] * n_prod
        new_config['objectives'] = []
        for obj_idx in range(n_prod):
            new_config['objectives'].append({'type': 'eval_production', 'params': []})
            params0 = deepcopy(config['objectives'][0]['params'][0])
            params1 = deepcopy(config['objectives'][0]['params'][1])
            for j in range(n_prod):
                new_config['objectives'][obj_idx]['params'].append(params0 if j == obj_idx else params1)
        new_config['dependencies'] = [[u, v] for u in range(n_prod) for v in range(config['resource_count'])]
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        with open(new_config_path, "w") as outfile:
            yaml.dump(new_config, outfile, default_flow_style=False, sort_keys=False)
    return new_config_paths


def generate_random_config(n_demands=5,
                           n_resources=5,
                           n_objectives=5,
                           edge_probability=0.3,
                           objective_density=0.3,
                           template_path='config/template/objectives_template.yml',
                           save_path='config/problems/problem_test.yml',
                           random_seed=42,
                           convex_only=False,
                           min_avail_resources=7,
                           max_avail_resources=15,
                           max_steps=None):
    random.seed(random_seed)
    np.random.seed(random_seed)
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)
    demands = np.random.randint(1, 10, size=n_demands).tolist()
    avail_resources = np.random.randint(min_avail_resources, max_avail_resources, size=n_resources).tolist()
    template['demand_count'] = n_demands
    template['resource_count'] = n_resources
    template['demands'] = demands
    template['avail_resources'] = avail_resources
    if max_steps is not None:
        template['max_steps'] = max_steps
    assert len(template['objectives']) >= n_objectives
    template['dependencies'] = generate_random_dependencies(n_demands, n_resources, edge_probability)
    objective_templates = deepcopy(template['objectives'])
    template['objectives'] = []
    demand_has_component = [False] * n_demands
    for i in range(n_objectives):
        objective = {'type': 'eval_production', 'params': []}
        non_empty_indices = []
        for j in range(n_demands):
            if random.random() < objective_density:
                if convex_only:
                    objective_component = {
                        "outer_op": "none",
                        "e": {"y": 0, "y_inv": 0, "constant": random.randint(1, 5)},
                        "f": {"y": 0, "y_inv": 0, "constant": 1},
                        "g": {"y": 0, "y_inv": 0, "constant": 1}
                    }
                else:
                    objective_component = random.choice(objective_templates)
                objective['params'].append(objective_component)
                demand_has_component[j] = True
                non_empty_indices.append(j)
            else:
                objective['params'].append({'outer_op': 'none'})
        if not non_empty_indices:
            j = random.randint(0, n_demands - 1)
            if convex_only:
                objective_component = {
                    "outer_op": "none",
                    "e": {"y": 0, "y_inv": 0, "constant": random.randint(1, 5)},
                    "f": {"y": 0, "y_inv": 0, "constant": 1},
                    "g": {"y": 0, "y_inv": 0, "constant": 1}
                }
            else:
                objective_component = random.choice(objective_templates)
            objective['params'][j] = objective_component
            demand_has_component[j] = True
        template['objectives'].append(objective)
    for j in range(n_demands):
        if not demand_has_component[j]:
            obj_idx = random.randint(0, n_objectives - 1)
            if convex_only:
                objective_component = {
                    "outer_op": "none",
                    "e": {"y": 0, "y_inv": 0, "constant": random.randint(1, 5)},
                    "f": {"y": 0, "y_inv": 0, "constant": 1},
                    "g": {"y": 0, "y_inv": 0, "constant": 1}
                }
            else:
                objective_component = random.choice(objective_templates)
            template['objectives'][obj_idx]['params'][j] = objective_component
            demand_has_component[j] = True
    with open(save_path, 'w') as f:
        yaml.dump(template, f, sort_keys=False)
    return template


def expand_single_config(n_demands=5,
                         n_objectives=5,
                         config_file='sqrt',
                         input_directory='template',
                         output_directory='test',
                         template_path='config/template/increasing_objectives_template.yml',
                         is_random_increasing=True,
                         config_name=None,
                         edge_probability=None,
                         seed=42):
    np.random.seed(seed)
    random.seed(seed)
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)
    objective_templates = deepcopy(template['objectives'])
    config_path = f'config/{input_directory}/{config_file}.yml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    new_config = deepcopy(config)
    if config_name is None:
        config_name = f'{config_file}-p{n_demands}-o{n_objectives}'
    new_config_path = f'config/{output_directory}/{config_name}.yml'
    new_config['demand_count'] = n_demands
    new_config['demands'] = [config['demands'][0]] * n_demands
    new_config['objectives'] = []
    template_params = config['objectives'][0].get('params', []) if 'objectives' in config and len(config['objectives']) > 0 else []
    params0 = deepcopy(template_params[0]) if len(template_params) > 0 else {'outer_op': 'none'}
    params1 = {'outer_op': 'none'}
    for obj_idx in range(n_objectives):
        obj = {'type': 'eval_production', 'params': []}
        for j in range(n_demands):
            if j == obj_idx:
                obj['params'].append(deepcopy(params0) if not is_random_increasing else random.choice(objective_templates))
            else:
                obj['params'].append(deepcopy(params1))
        new_config['objectives'].append(obj)
    if edge_probability is None:
        new_config['dependencies'] = [[u, v] for u in range(n_demands) for v in range(config['resource_count'])]
    else:
        new_config['dependencies'] = generate_random_dependencies(n_demands, config['resource_count'], edge_probability)
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    with open(new_config_path, 'w') as outfile:
        yaml.dump(new_config, outfile, default_flow_style=False, sort_keys=False)
    generate_default_train_config_from_env(config_name, project_name="MoreObjectives")
    return new_config_path


def generate_random_dependencies(n_demands, n_resources, edge_probability):
    dependencies = []
    for i in range(n_demands):
        connected = False
        local_probability = random.uniform(0, min(edge_probability * 2, 1))
        for j in range(n_resources):
            if random.random() < local_probability:
                dependencies.append([i, j])
                connected = True
        if not connected:
            j = random.randint(0, n_resources - 1)
            dependencies.append([i, j])
    for j in range(n_resources):
        if not any(dep[1] == j for dep in dependencies):
            i = random.randint(0, n_demands - 1)
            dependencies.append([i, j])
    return dependencies


def generate_default_train_config_from_env(env_name: str, train_config_path='train_config/train.yml', project_name='test'):
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    train_config['env_name'] = env_name
    new_config_path = f'train_config/{project_name}/train_{env_name}.yml'
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    with open(new_config_path, 'w') as outfile:
        yaml.dump(train_config, outfile, default_flow_style=False, sort_keys=False)

__all__ = [
    'expand_productions', 'expand_productions_and_objectives', 'generate_random_config',
    'expand_single_config', 'generate_random_dependencies', 'generate_default_train_config_from_env'
]
