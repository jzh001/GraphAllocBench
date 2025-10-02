from train_utils.general import train_parallel

# PROJECT_NAME = 'GraphAllocBench-v2'
# PROJECT_NAME = 'MoreObjectives'
PROJECT_NAME = 'GraphAllocBench-GNN-v1-1'

if __name__ == "__main__":

    train_parallel(train_config_paths=[
        f"train_config/{PROJECT_NAME}/train_problem_6c_arch-0.yml",
        f"train_config/{PROJECT_NAME}/train_problem_6c_arch-2.yml",
        f"train_config/{PROJECT_NAME}/train_problem_6c_arch-3.yml",
    ], project_name=PROJECT_NAME)