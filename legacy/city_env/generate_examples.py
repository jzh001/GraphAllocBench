import yaml

from copy import deepcopy
import os
import random
import numpy as np

def expand_productions(n_configs = 5, 
                       config_file = 'concave', 
                       input_directory = 'template', 
                       output_directory = 'test', 
                       even_only = True):
    
    # Generate examples based on input config, to expand to more resources

    config_path = f'config/{input_directory}/{config_file}.yml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    new_config_paths = []
    upper_bound = 2 + n_configs * 2 if even_only else 2 + n_configs
    step = 2 if even_only else 1
    for n_prod in range(2, upper_bound, step): # n_prod =  number of productions
        new_config_path = f'config/{output_directory}/{config_file}-p{n_prod}.yml'
        new_config_paths.append(new_config_path)

        new_config = deepcopy(config)

        new_config['demand_count'] = n_prod
        new_config['demands'] = [config['demands'][0]] * n_prod
        # new_config['max_steps'] = n_prod * 15
        # new_config['max_steps'] = 25

        # new_config['avail_resources'] = [5 * i] * config['resource_count'] # for each resource, quantity = 5 * demand_count_i, for resource_count resources

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


def expand_productions_and_objectives(n_configs = 5, 
                       config_file = 'convex', 
                       input_directory = 'template', 
                       output_directory = 'test', 
                       even_only = True):
    
    # Generate examples based on input config, to expand to more resources

    config_path = f'config/{input_directory}/{config_file}.yml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    new_config_paths = []
    upper_bound = 2 + n_configs * 2 if even_only else 2 + n_configs
    step = 2 if even_only else 1
    for n_prod in range(2, upper_bound, step): # n_prod =  number of productions
        new_config_path = f'config/{output_directory}/{config_file}-p{n_prod}-o{n_prod}.yml'
        new_config_paths.append(new_config_path)

        new_config = deepcopy(config)

        new_config['demand_count'] = n_prod
        new_config['demands'] = [config['demands'][0]] * n_prod
        # new_config['max_steps'] = n_prod * 15
        # new_config['max_steps'] = 25

        # new_config['avail_resources'] = [5 * i] * config['resource_count'] # for each resource, quantity = 5 * demand_count_i, for resource_count resources
        new_config['objectives'] = []
        for obj_idx in range(n_prod):
            new_config['objectives'].append({
                'type': 'eval_production',
                'params': []
            })
            params0 = deepcopy(config['objectives'][0]['params'][0])
            params1 = deepcopy(config['objectives'][0]['params'][1]) # blank

            for j in range(n_prod):
                new_config['objectives'][obj_idx]['params'].append(params0 if j == obj_idx else params1)

        new_config['dependencies'] = [[u, v] for u in range(n_prod) for v in range(config['resource_count'])]

        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        
        with open(new_config_path, "w") as outfile:            
            yaml.dump(new_config, outfile, default_flow_style=False, sort_keys=False)


    return new_config_paths



def generate_random_config(
        n_demands = 5,
        n_resources = 5,
        n_objectives = 5,
        edge_probability = 0.3,
        objective_density = 0.3,
        template_path = 'config/template/objectives_template.yml',
        save_path = 'config/problems/problem_test.yml',
        random_seed = 42,
        convex_only = False,
        min_avail_resources = 7,
        max_avail_resources = 15,
        max_steps = None,
):
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
    # Randomly select objectives

    objective_templates = deepcopy(template['objectives'])
    template['objectives'] = []

    # Track for each demand if it has at least one objective_component assigned
    demand_has_component = [False] * n_demands

    for i in range(n_objectives):
        objective = {
            'type': 'eval_production',
            'params': []
        }
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
                    objective_component = np.random.choice(objective_templates)
                objective['params'].append(objective_component)
                demand_has_component[j] = True
                non_empty_indices.append(j)
            else:
                objective['params'].append({'outer_op': 'none',})
        # Ensure at least one non-empty component per objective
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
                objective_component = np.random.choice(objective_templates)
            objective['params'][j] = objective_component
            demand_has_component[j] = True
        template['objectives'].append(objective)

    # Ensure every demand has at least one objective_component
    for j in range(n_demands):
        if not demand_has_component[j]:
            # Randomly pick an objective to assign a component to this demand
            obj_idx = random.randint(0, n_objectives - 1)
            if convex_only:
                objective_component = {
                    "outer_op": "none",
                    "e": {"y": 0, "y_inv": 0, "constant": random.randint(1, 5)},
                    "f": {"y": 0, "y_inv": 0, "constant": 1},
                    "g": {"y": 0, "y_inv": 0, "constant": 1}
                }
            else:
                objective_component = random.sample(objective_templates, 1)[0]
            template['objectives'][obj_idx]['params'][j] = objective_component
            demand_has_component[j] = True

    with open(save_path, 'w') as f:
        yaml.dump(template, f, sort_keys=False)

    return template


def expand_single_config(n_demands = 5,
                         n_objectives = 5,
                         config_file = 'sqrt',
                         input_directory = 'template',
                         output_directory = 'test',
                         template_path = 'config/template/increasing_objectives_template.yml',
                         is_random_increasing = True,
                         config_name = None,
                         edge_probability = None,
                         seed=42,
                         ):
    """
    Generate a single configuration based on the template config file. The caller supplies
    the desired number of productions (demands) and the desired number of objectives.

    Returns the path to the created config file.
    """
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

    # Build objectives list with n_objectives items. Each objective contains n_prod params.
    new_config['objectives'] = []

    # Use the first objective in the template as a source of components when available
    template_params = []
    if 'objectives' in config and len(config['objectives']) > 0:
        template_params = config['objectives'][0].get('params', [])

    params0 = deepcopy(template_params[0]) if len(template_params) > 0 else {'outer_op': 'none'}
    params1 = {'outer_op': 'none'}

    for obj_idx in range(n_objectives):
        obj = {
            'type': 'eval_production',
            'params': []
        }
        for j in range(n_demands):
            # Prefer to assign a "meaningful" component to the objective's matching demand index
            if j == obj_idx:
                obj['params'].append(deepcopy(params0) if not is_random_increasing else random.choice(objective_templates))
            else:
                obj['params'].append(deepcopy(params1))
        new_config['objectives'].append(obj)

    # Fully connected dependencies between demands and resources (same pattern used elsewhere)
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
    # Generate a random bipartite graph for dependencies where every node is connected to at least one other node
    dependencies = []

    # Ensure every demand is connected to at least one resource
    for i in range(n_demands):
        connected = False
        local_probability = random.uniform(0, min(edge_probability * 2, 1))
        for j in range(n_resources):
            if random.random() < local_probability:
                dependencies.append([i, j])
                connected = True
        if not connected:
            # Randomly connect to one resource if none were chosen
            j = random.randint(0, n_resources - 1)
            dependencies.append([i, j])

    # Ensure every resource is connected to at least one demand
    for j in range(n_resources):
        if not any(dep[1] == j for dep in dependencies):
            i = random.randint(0, n_demands - 1)
            dependencies.append([i, j])
    
    return dependencies

def generate_default_train_config_from_env(env_name: str, train_config_path = 'train_config/train.yml', project_name = 'test'):
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    train_config['env_name'] = env_name
    new_config_path = f'train_config/{project_name}/train_{env_name}.yml'
    os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
    with open(new_config_path, 'w') as outfile:
        yaml.dump(train_config, outfile, default_flow_style=False, sort_keys=False)