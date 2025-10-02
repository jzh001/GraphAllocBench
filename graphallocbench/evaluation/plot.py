import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
import plotnine as p9

from pymoo.util.ref_dirs import get_reference_directions

from ..city_env.env_model import CityPlannerEnv
from .inference import run_experiments
from .analytical import get_analytical_objectives
from .utils import calculate_ordering_score
from .stats import (
    calculate_all_stats,
    calculate_stats_more_objectives,
    calculate_stats_across_arch,
)


def plot_summary_stats(project_name: str = 'GraphAllocBench-v2',
                       arch_idx: int = 0,
                       model_checkpoint: str = 'model_step_100000',
                       seeds=range(0, 5),
                       idx2name: dict | None = None,
                       legend2name: dict | None = None):
    # Always use resolver (typo with leading space previously prevented detection)
    data_dir = _resolve_data_dir(project_name)
    summary_csv_path = data_dir / f'summary_{model_checkpoint}.csv'
    if summary_csv_path.exists():
        df = pd.read_csv(summary_csv_path)
    else:
        df = calculate_all_stats(project_name=project_name,
                                 arch_idx=arch_idx,
                                 model_checkpoint=model_checkpoint,
                                 seeds=seeds)
    df['Problem'] = df['env_name'].apply(lambda s: s.split('_')[1])

    plot_df = ( 
        df[['Problem', 'normalized_hv','non_dominated','ordering','seed']]
        .set_index(['Problem','seed'])
        .rename_axis('Metric',axis=1)
        .stack()
    ).rename('Score').reset_index()
    metric_map = {
        'normalized_hv': 'HV Ratio',
        'non_dominated': '% Non-Dominated',
        'ordering': 'Ordering Score',
    }
    plot_df['Metric'] = pd.Categorical(
        values=plot_df['Metric'].map(metric_map),
        categories=['HV Ratio', '% Non-Dominated', 'Ordering Score'],
        ordered=True
    )

    p = (
        p9.ggplot(data=plot_df, mapping=p9.aes(x='Problem', y='Score', color='Metric', fill='Metric', group='Metric'))
        + p9.stat_summary(
            geom='point', 
            fun_y=np.mean, 
            size=6,
            shape='_',
            color='black',
        )
        + p9.geom_jitter(
            size=2.0,
            alpha=0.5,
            width=0.02,
            random_state=38383444,
        )
        + p9.facet_grid('Metric~.')
        + p9.theme_classic()
        + p9.theme(
            panel_grid_major_y=p9.element_line(color='lightgray', alpha=0.5),
            panel_grid_major_x=p9.element_line(color='lightgray', alpha=0.5),
            legend_position='none',        
        )
        + p9.scale_fill_brewer(type='qual', palette=2)
        + p9.scale_color_brewer(type='qual', palette=2)
    )

    # Save the plot as PDF and PNG
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    p.save(filename=str(output_dir / f"summary_stats_{model_checkpoint}.pdf"), format="pdf")
    p.save(filename=str(output_dir / f"summary_stats_{model_checkpoint}.png"), format="png")

    return p


def _resolve_data_dir(project_name: str) -> Path:
    """Return a best-effort Path for data/{project_name}.

    Search order:
    - cwd/data/{project_name}
    - repo root (two parents up from this file)/data/{project_name}
    - home/.graphallocbench/data/{project_name}
    If none exist, return cwd/data/{project_name} as the default location.
    """
    candidates = [
        Path.cwd() / 'data' / project_name,
        Path(__file__).resolve().parents[2] / 'data' / project_name,
        Path.home() / '.graphallocbench' / 'data' / project_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd() / 'data' / project_name


def plot_objectives_vs_production(env: CityPlannerEnv):
    import numpy as np
    fig, axes = plt.subplots(env.n_objectives, env.demand_count,
                             figsize=(6 * env.demand_count, 4 * env.n_objectives), squeeze=False)
    for i in range(env.n_objectives):
        for j in range(env.demand_count):
            productions = np.zeros(env.demand_count)
            prod_range = np.arange(0, int(max(env.avail_resources)) + 1)
            obj_values = []
            for val in prod_range:
                productions[j] = val
                obj_vector = env.get_objectives(productions, None)
                obj_values.append(obj_vector[i])
            ax = axes[i, j]
            ax.plot(prod_range, obj_values, marker='o')
            ax.set_title(f'Objective {i} vs Production {j}')
            ax.set_xlabel(f'Production {j}')
            ax.set_ylabel(f'Objective {i}')
            ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_2D_pareto_front(env: CityPlannerEnv,
                         model=None,
                         model_path: str | None = None,
                         n_partitions: int = 12):
    plt.clf()
    if env.n_objectives != 2:
        raise ValueError("2D Pareto front can only be plotted for 2 objectives.")
    preferences = get_reference_directions("das-dennis", env.n_objectives, n_partitions=n_partitions)
    objectives, _ = run_experiments(env=env, model=model, model_path=model_path, preferences=preferences)
    ideal_objectives = get_analytical_objectives(env=env) if env.demand_count < 6 else None
    plt.plot(objectives[:, 0], objectives[:, 1], 'o', markersize=5, label='Predicted Pareto Front')
    if ideal_objectives is not None:
        plt.plot(ideal_objectives[:, 0], ideal_objectives[:, 1], 'x', markersize=5, label='Ideal Pareto Front')
    plt.xlabel('Objective 0')
    plt.ylabel('Objective 1')
    title_name = env.config_name.replace("_", " ")
    title_name = title_name[0].upper() + title_name[1:] if title_name else title_name
    plt.title(f'2D Pareto Front for {title_name}')
    plt.legend()
    plt.grid()
    plt.show()


def plot_2D_pareto_fronts_all(problems: list[str],
                              checkpoint: str = 'model_step_500000',
                              project_name: str = 'GraphAllocBench-v2',
                              arch_idx: int = 0,
                              seeds=(1, 2),
                              n_partitions: int = 12,
                              ncols: int | None = None,
                              figsize_per_plot: tuple = (4, 4),
                              save: bool = True,
                              save_dir: str | None = None):
    if len(problems) == 0:
        raise ValueError("No problems provided to plot.")
    n_problems = len(problems)
    if ncols is None:
        ncols = min(n_problems, 6)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_problems / ncols))
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    global_handles = {}
    for idx, problem in enumerate(problems):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        env = CityPlannerEnv(f"config/problems/{problem}.yml")
        if env.n_objectives != 2:
            ax.text(0.5, 0.5, f"{env.n_objectives} objectives\n(skipped)", ha='center', va='center', fontsize=10)
            ax.set_title(problem.replace('_', ' '))
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        preferences = get_reference_directions("das-dennis", env.n_objectives, n_partitions=n_partitions)
        ideal_objectives = get_analytical_objectives(env=env) if env.demand_count < 6 else None
        if ideal_objectives is not None and len(ideal_objectives) > 0:
            h_ideal, = ax.plot(ideal_objectives[:, 0], ideal_objectives[:, 1], 's', markersize=7,
                               label='Ideal', markeredgecolor='black', markerfacecolor='none')
            global_handles['Ideal'] = h_ideal
        markers = ['x', 'o', '^', 'd', '*', 'v', '<', '>']
        for seed_idx, seed in enumerate(seeds):
            model_path = f"models/{project_name}/arch-{arch_idx}/{problem}/{seed}/{checkpoint}.zip"
            objectives, _ = run_experiments(env=env, model=None, model_path=model_path, preferences=preferences)
            marker = markers[seed_idx % len(markers)]
            h_pred, = ax.plot(objectives[:, 0], objectives[:, 1], marker, markersize=5,
                              label=f'Predicted (Seed {seed})', markerfacecolor='none')
            global_handles[f'Predicted (Seed {seed})'] = h_pred
        ax.set_xlabel('Objective 0')
        ax.set_ylabel('Objective 1')
        title_name = env.config_name.replace('_', ' ')
        title_name = title_name[0].upper() + title_name[1:] if title_name else problem
        ax.set_title(title_name, fontsize=9)
        ax.grid(True)
    for i in range(n_problems, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].axis('off')
    if global_handles:
        handles = [global_handles[k] for k in sorted(global_handles.keys())]
        labels = sorted(global_handles.keys())
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    if save:
        if save_dir is None:
            save_dir = _resolve_data_dir(project_name)
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            plt.savefig(str(save_dir / f'pareto_fronts_grid_{checkpoint}.{ext}'), bbox_inches='tight')
    plt.show()


def plot_more_objectives(problem: str = "logistic_long",
                         project_name: str = 'MoreObjectives',
                         seeds=range(0, 5),
                         arch_idx: int = 0,
                         total_steps: int = 1000000,
                         step_size: int = 100000,
                         problem_display_name: str = "Logistic Objective Function",
                         objectives_categories: list[int] = None):
    if objectives_categories is None:
        objectives_categories = [5, 10, 15, 20]  # Default values

    data_dir = _resolve_data_dir(project_name)
    csv_path = data_dir / f"more_obj_{problem}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = calculate_stats_more_objectives(problem=problem,
                                             project_name=project_name,
                                             seeds=seeds,
                                             arch_idx=arch_idx,
                                             total_steps=total_steps,
                                             step_size=step_size)
    if df is None or df.empty:
        raise ValueError(f"No data available for problem={problem} at {csv_path}")
    df = df.copy()
    df['n_obj'] = df['n_obj'].astype(int)
    df['Training Steps'] = df['steps'].astype(int)
    df['proportion_allocated'] = df['proportion_allocated'].astype(float)
    df['objectives'] = pd.Categorical(df['n_obj'], categories=objectives_categories, ordered=True)
    df['Proportion of Resources Allocated'] = df['proportion_allocated']

    p = (
    p9.ggplot(data=df, mapping=p9.aes(x='Training Steps',y='Proportion of Resources Allocated', color='objectives', fill='objectives'))
    + p9.stat_summary(geom='line', fun_y=np.mean, size=1.0)
    + p9.stat_summary(geom='point', fun_y=np.mean, size=1.5)
    + p9.stat_summary(geom='ribbon', fun_ymax=np.max, fun_ymin=np.min, alpha=0.3, color='none')
    + p9.scale_fill_brewer(type='qual', palette=2)
    + p9.scale_color_brewer(type='qual', palette=2)
    + p9.facet_grid('objectives~.')
    + p9.theme_classic()
    + p9.theme(panel_grid_major=p9.element_line(color='lightgray', alpha=0.5), legend_position='none',
              figure_size=(4,4))
    )

    # Save the plot as PDF and PNG
    output_dir = data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    p.save(filename=str(output_dir / f"more_obj_{problem}.pdf"), format="pdf")
    p.save(filename=str(output_dir / f"more_obj_{problem}.png"), format="png")
    return p

def plot_stats_across_arch(problems: list[str],
                           project_name: str,
                           arch_idx_all: list[int],
                           arch_names: list[str],
                           train_steps: int = 1000000,
                           n_partitions: int = 12,
                           seeds=range(0, 5)):
    assert len(arch_idx_all) == len(arch_names), (
        "arch_idx_all and arch_names must have the same length.")
    data_dir = _resolve_data_dir(project_name)
    csv_path = data_dir / f"summary_across_arch_{'-'.join(map(str, arch_idx_all))}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = calculate_stats_across_arch(problems=problems,
                                          project_name=project_name,
                                          arch_idx_all=arch_idx_all,
                                          train_steps=train_steps,
                                          n_partitions=n_partitions,
                                          seeds=seeds)
    if df is None or df.empty:
        raise ValueError(f"No stats available at {csv_path} and generation returned no data.")
    expected_rows = len(problems) * len(arch_idx_all) * len(list(seeds))
    if not {'problem', 'arch_idx', 'seed'}.issubset(df.columns):
        if len(df) == expected_rows:
            # attempt recovery not needed; the generated version already includes these columns
            pass
        else:
            print("Warning: unexpected row count; proceeding anyway.")
    metrics = ['hypervolume', 'non_dominated', 'ordering']
    metric_names = {'hypervolume': 'Hypervolume', 'non_dominated': 'Non-dominated', 'ordering': 'Ordering Score'}
    agg = df.groupby(['problem', 'arch_idx'])[metrics].agg(['mean', 'std'])
    arch_to_name = dict(zip(arch_idx_all, arch_names))
    n_problems = len(problems)
    n_arch = len(arch_idx_all)
    means = {m: np.zeros((n_problems, n_arch)) for m in metrics}
    stds = {m: np.zeros((n_problems, n_arch)) for m in metrics}
    for i, problem in enumerate(problems):
        for j, arch in enumerate(arch_idx_all):
            if (problem, arch) in agg.index:
                for m in metrics:
                    means[m][i, j] = agg.loc[(problem, arch)][(m, 'mean')]
                    stds[m][i, j] = agg.loc[(problem, arch)][(m, 'std')]
            else:
                for m in metrics:
                    means[m][i, j] = np.nan
                    stds[m][i, j] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 5), squeeze=False)
    axes = axes[0]
    x = np.arange(n_problems)
    width = 0.8 / max(1, n_arch)
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for j, arch in enumerate(arch_idx_all):
            pos = x + (j - (n_arch - 1) / 2) * width
            vals = means[metric][:, j]
            errs = stds[metric][:, j]
            mask = ~np.isnan(vals)
            if mask.any():
                arch_label = arch_to_name.get(arch, f'arch-{arch}')
                ax.bar(pos[mask], vals[mask], width, yerr=errs[mask], capsize=4, label=f'{arch_label}', alpha=0.9)
        for sep in range(n_problems - 1):
            ax.axvline(sep + 0.5, color='grey', linestyle='--', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ') for p in problems], rotation=45, ha='right', fontsize=9)
        ax.set_title(metric_names.get(metric, metric))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        desired_labels = [arch_to_name.get(a, f'arch-{a}') for a in arch_idx_all]
        label_to_handle = {lab: h for h, lab in zip(handles, labels)}
        ordered_handles = [label_to_handle.get(l) for l in desired_labels if l in label_to_handle]
        ordered_labels = [l for l in desired_labels if l in label_to_handle]
        if ordered_handles:
            fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(ordered_handles), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_dir = data_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"summary_across_arch_{'-'.join(map(str, arch_idx_all))}.png"
    plt.savefig(str(save_dir / out_name), bbox_inches='tight')
    plt.show()

__all__ = [
    'plot_summary_stats',
    'plot_objectives_vs_production',
    'plot_2D_pareto_front',
    'plot_2D_pareto_fronts_all',
    'plot_more_objectives',
    'plot_stats_across_arch'
]
