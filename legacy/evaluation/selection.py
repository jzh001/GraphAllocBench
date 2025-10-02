import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml

def generate_problem_summary(project_name='GraphAllocBench-v2', save_path = None):
    if save_path is None:
        save_path = f"data/{project_name}/best_hyperparameters.csv"
    data = []
    data_dir = f'data/{project_name}'
    for csv_file in os.listdir(data_dir):
        csv_path = os.path.join(data_dir, csv_file)
        if csv_path != save_path and 'stats' in csv_path:
            df = pd.read_csv(csv_path)

            grouped = df.groupby('architecture_idx')
            for _, group in grouped:
                max_row = group.loc[group['hypervolume'].idxmax()]
                data.append(max_row.to_dict())

    final_df = pd.DataFrame(data)
    final_df.insert(0, 'env_name', final_df.pop('env_name'))
    final_df = final_df.sort_values(by='env_name').reset_index(drop=True)
    final_df.to_csv(save_path, index = False)


    for i, row in final_df.iterrows():
        row_dict = row.to_dict()
        for key in ['hypervolume', 'normalized_hv', 'non_dominated', 'ordering', 'wandb_run_name']:
            row_dict.pop(key, None)
        row_dict['total_timesteps'] = 2000000
        yaml_file_path = f"train_config/{project_name}/train_{row['env_name']}_arch-{row_dict['architecture_idx']}.yml"
        os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(row_dict, yaml_file)

    return final_df

def plot_problem_summary(project_name = 'GraphAllocBench-v2', summary_file = 'best_hyperparameters.csv'):
    df = pd.read_csv(f"data/{project_name}/{summary_file}")

    df.set_index('env_name', inplace=True)
    df[['normalized_hv', 'non_dominated', 'ordering']].plot(kind='bar', figsize=(12, 6))
    plt.axhline(y=1.0, color='black', linestyle='--')  # Dotted line at 1.0
    plt.title('Problem Summary')
    plt.xlabel('Environment Name')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics', loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)  # Legend at the bottom
    plt.tight_layout()
    plt.show()