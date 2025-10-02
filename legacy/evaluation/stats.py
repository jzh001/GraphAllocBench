from random import seed
from city_env.env_model import ResourceManagementEnv

from pymoo.util.ref_dirs import get_reference_directions
from evaluation.inference import run_experiments
from evaluation.utils import calculate_hypervolume, calculate_non_dominated, calculate_ordering_score, calculate_proportion_allocated, load_model
from evaluation.analytical import get_analytical_objectives
import os
import pandas as pd
from tqdm import tqdm

def calculate_stats(env: ResourceManagementEnv, 
                    model = None, 
                    model_path = None, # provide either model or model_path
                    n_partitions = 8, # we use a smaller resolution for hyperparameter search
                    ):
    preferences = get_reference_directions("das-dennis", env.n_objectives, n_partitions=n_partitions)

    objectives, _ = run_experiments(env=env, model=model, model_path=model_path, preferences=preferences)
    ideal_objectives = get_analytical_objectives(env=env) if env.demand_count < 6 else None
    pred_hv = calculate_hypervolume(objectives)
    ideal_hv = calculate_hypervolume(ideal_objectives) if env.demand_count < 6 else 0
    return {
        "hypervolume": pred_hv,
        "normalized_hv": pred_hv / ideal_hv if ideal_hv != 0 else 0,
        "non_dominated": calculate_non_dominated(objectives),
        "ordering": calculate_ordering_score(env=env, model=model, model_path=model_path),
    }

def calculate_all_stats(project_name: str, 
                        arch_idx: int, 
                        model_checkpoint: str = 'model_step_100000',
                        n_partitions = 12,
                        seeds = range(0, 5),
                        ):
    model_project_dir = f"models/{project_name}/arch-{arch_idx}"
    hyperparameters_csv_path = f"data/{project_name}/best_hyperparameters.csv"
    hyperparameters_df = pd.read_csv(hyperparameters_csv_path)
    csv_save_path = f"data/{project_name}/summary_{model_checkpoint}.csv"

    all_stats = []

    for problem in tqdm(os.listdir(model_project_dir)):
        hyperparams_row = hyperparameters_df[hyperparameters_df['env_name'] == problem]
        if hyperparams_row.empty:
            continue
        row_data = hyperparams_row.iloc[0].to_dict()
        env_config_path = f"config/problems/{problem}.yml"
        env = ResourceManagementEnv(env_config_path)
        for seed in seeds:
            model_path = os.path.join(model_project_dir, problem, str(seed), f"{model_checkpoint}.zip")
            stats = calculate_stats(env=env, model_path=model_path, n_partitions = n_partitions)
            stats_with_hyperparams = row_data.copy()
            stats_with_hyperparams['seed'] = int(seed)
            stats_with_hyperparams.update(stats)
            all_stats.append(stats_with_hyperparams)
        
    df = pd.DataFrame(all_stats)
    df = df.sort_values(by="env_name").reset_index(drop=True)
    df.to_csv(csv_save_path, index = False)
    return df

def calculate_stats_more_objectives(problem="logistic_long", 
                                    project_name = 'MoreObjectives', 
                                    seeds = range(0, 5),
                                    arch_idx = 0,
                                    total_steps = 1000000,
                                    step_size = 100000,
                                    ):
    
    stats = []

    for step in tqdm(range(step_size, total_steps+1, step_size)):
        for i in range(5, 21, 5):
            env_name = f"{problem}-p{i}-o{i}"
            env = ResourceManagementEnv(f"config/problems/{env_name}.yml")
            for seed in seeds:
                model_path = f"models/{project_name}/arch-{arch_idx}/{env_name}/{seed}/model_step_{step}.zip"
                model = load_model(model_path, env)

                stats.append({
                    "n_obj": i,
                    "steps": step,
                    "proportion_allocated": calculate_proportion_allocated(env=env, model=model),
                })

    df = pd.DataFrame(stats)
    csv_path = f"data/{project_name}/more_obj_{problem}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df 

def calculate_stats_across_arch(problems: list[str], 
                                project_name: str, 
                                arch_idx_all: list[int], 
                                is_finetuned: list[bool], 
                                pretrain_steps: int = 1000000,
                                finetune_steps: int = 1000000,
                                n_partitions = 12,
                                seeds = range(0, 5)):
    all_stats = []
    for problem in problems:
        for arch_idx, finetuned in tqdm(zip(arch_idx_all, is_finetuned)):
            env = ResourceManagementEnv(f"config/problems/{problem}.yml")
            for seed in seeds:
                steps = finetune_steps if finetuned else pretrain_steps + finetune_steps
                model_path = f"models/{project_name}/arch-{arch_idx}/{problem}/{seed}/model_step_{steps}"
                if finetuned:
                    model_path = f"models/{project_name}/arch-{arch_idx}-finetune/{problem}/{seed}/model_step_{steps}"
                
                stats = calculate_stats(env, model_path = model_path, n_partitions=n_partitions)
                stats['arch_idx'] = arch_idx
                all_stats.append(stats)

    df = pd.DataFrame(all_stats)

    csv_path = f"data/{project_name}/summary_across_arch_{'-'.join(map(str, arch_idx_all))}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    return df
