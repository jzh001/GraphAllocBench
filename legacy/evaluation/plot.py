from evaluation.stats import calculate_all_stats, calculate_stats_more_objectives, calculate_stats_across_arch
from evaluation.utils import calculate_ordering_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
import os

from city_env.env_model import ResourceManagementEnv
from evaluation.inference import run_experiments
from evaluation.analytical import get_analytical_objectives
from pymoo.util.ref_dirs import get_reference_directions

def plot_summary_stats(project_name = 'GraphAllocBench-v2', 
                       arch_idx = 0, 
                       model_checkpoint = 'model_step_100000',
                       seeds = range(0, 5),
                       idx2name: dict = None,
                       legend2name: dict = None):
    summary_csv_path = f'data/{project_name}/summary_{model_checkpoint}.csv'
    if os.path.exists(summary_csv_path):
        df = pd.read_csv(summary_csv_path)
    else:
        df = calculate_all_stats(project_name=project_name, 
                                 arch_idx=arch_idx, 
                                 model_checkpoint=model_checkpoint,
                                 seeds=seeds
                                 )

    # Group by env_name and calculate mean and std for the metrics
    metrics = ['normalized_hv', 'non_dominated', 'ordering']
    grouped = df.groupby('env_name')[metrics]
    means = grouped.mean()
    stds = grouped.std()

    # Plot means with error bars (std) on the top subplot; descriptions in a separate text area below
    # Create a two-row layout: bar plot above, description text below
    # compress the figure vertically: slightly shorter figure and smaller top subplot
    fig = plt.figure(figsize=(12, 3.5))
    # allocate less height to the bar plot so the overall figure is more compact
    # increase vertical space between chart and description area via hspace
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.5, 1.0], hspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[1, 0])

    x = range(len(means))
    width = 0.25
    colors = ['#A3C9A8',  # pastel green
              '#F6D186',  # pastel yellow
              '#F7A072',  # pastel orange
              '#B5BAD0',  # pastel purple
              '#8EC0E4',  # pastel blue
              '#F9DBBD']  # pastel peach
    for i, metric in enumerate(metrics):
        ax.bar([xi + i*width for xi in x], means[metric], width, yerr=stds[metric], label=metric, capsize=5, color=colors[i % len(colors)], edgecolor='grey', alpha=0.85)

    ax.set_xticks([xi + width for xi in x])
    # Use only the short name on x-axis to keep bars readable
    if idx2name is None:
        xt_names = list(means.index)
        desc_list = ["" for _ in means.index]
    else:
        xt_names = [idx2name.get(idx, {}).get('name', idx) for idx in means.index]
        desc_list = [idx2name.get(idx, {}).get('description', '') for idx in means.index]
    ax.set_xticklabels(xt_names, fontsize=9)
    ax.set_ylabel('Score')
    # Put the main title on the figure (so it's above the legend) and nudge positions for compact layout
    # fig.suptitle('GraphAllocBench Metrics', fontsize=14, y=0.98)
    # Move legend to the figure level and place it just below the suptitle
    handles, legend_labels = ax.get_legend_handles_labels()
    # remap legend labels if a mapping is provided
    if legend2name is not None:
        legend_labels = [legend2name.get(l, l) for l in legend_labels]
    fig.legend(handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.95),  # place legend in the top margin below the suptitle
               ncol=len(metrics),
               frameon=False,
               fontsize=9,
               handletextpad=0.6,
               columnspacing=1.0,
               borderaxespad=0.5)

    # Prepare the description area below the plot
    ax_table.axis('off')
    # Render descriptions in multiple columns to avoid a long vertical list
    n_items = len(xt_names)
    if n_items > 12:
        n_cols = 3
    elif n_items > 6:
        n_cols = 2
    else:
        n_cols = 1
    # split indices into roughly equal groups so each column has a similar number of rows
    idx_groups = np.array_split(np.arange(n_items), n_cols)
    for col_idx, idxs in enumerate(idx_groups):
        if len(idxs) == 0:
            continue
        names_col = [xt_names[i] for i in idxs]
        descs_col = [desc_list[i] for i in idxs]
        lines = []
        for name, desc in zip(names_col, descs_col):
            if desc:
                # narrower wrap per column
                desc_wrapped = textwrap.fill(desc, width=28)
                desc_wrapped = desc_wrapped.replace('\n', '\n' + ' ' * 4)
                lines.append(f"{name}: {desc_wrapped}")
            else:
                lines.append(f"{name}:")
        col_text = '\n'.join(lines)
        # distribute columns horizontally across the axis
        x_pos = 0.01 + col_idx * (0.98 / n_cols)
        ax_table.text(x_pos, 0.98, col_text, va='top', ha='left', fontsize=9, family='monospace')

    # Make room for the suptitle + legend placed in the figure's top margin
    fig.subplots_adjust(top=0.85)
    # Final layout and save
    plt.tight_layout()
    plt.savefig(f'data/{project_name}/summary_plot_{model_checkpoint}.png', bbox_inches='tight')
    plt.savefig(f'data/{project_name}/summary_plot_{model_checkpoint}.pdf', bbox_inches='tight')
    plt.show()

def plot_objectives_vs_production(env: ResourceManagementEnv):
    plt.clf()
    n_obj = env.n_objectives
    n_prod = env.demand_count

    fig, axes = plt.subplots(n_obj, n_prod, figsize=(6 * n_prod, 4 * n_obj), squeeze=False)
    for i in range(n_obj):
        for j in range(n_prod):
            productions = np.zeros(n_prod)
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

def plot_2D_pareto_front(env: ResourceManagementEnv, 
                         model = None, 
                         model_path = None, # either use model or model_path
                         n_partitions = 12,
                         ):
    plt.clf()
    n_obj = env.n_objectives

    if n_obj != 2:
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
                              checkpoint = 'model_step_500000',
                              project_name = 'GraphAllocBench-v2',
                              arch_idx = 0,
                              seeds = [1, 2],
                              n_partitions: int = 12,
                              ncols: int | None = None,
                              figsize_per_plot: tuple = (4, 4),
                              save: bool = True,
                              save_dir: str | None = None,
                              ):
    """Plot 2D Pareto fronts for multiple problems in a single horizontal grid figure.

    Each problem is plotted in its own subplot arranged left-to-right. By default the
    grid will try to stay horizontal (few rows, more columns). Common options:

    - n_partitions: reference direction partitions passed to the experiment runner.
    - ncols: number of columns in the grid; if None it will default to the number
      of problems but capped at 6 to keep the figure reasonably wide.
    - figsize_per_plot: size of each subplot (width, height) in inches.
    - save: whether to save the combined figure to disk.
    - save_dir: directory to save into; defaults to data/{project_name}.
    """
    if len(problems) == 0:
        raise ValueError("No problems provided to plot.")

    n_problems = len(problems)
    # default to a single row (fully horizontal) but cap columns so layout stays readable
    if ncols is None:
        ncols = min(n_problems, 6)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_problems / ncols))

    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    # keep track of a single legend handles for the whole figure
    global_handles = {}

    for idx, problem in enumerate(problems):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        env = ResourceManagementEnv(f"config/problems/{problem}.yml")

        if env.n_objectives != 2:
            ax.text(0.5, 0.5, f"{env.n_objectives} objectives\n(skipped)", ha='center', va='center', fontsize=10)
            ax.set_title(problem.replace('_', ' '))
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        preferences = get_reference_directions("das-dennis", env.n_objectives, n_partitions=n_partitions)

        ideal_objectives = get_analytical_objectives(env=env) if env.demand_count < 6 else None
        # Plot ideal if present
        if ideal_objectives is not None and len(ideal_objectives) > 0:
            h_ideal, = ax.plot(ideal_objectives[:, 0], ideal_objectives[:, 1], 's', markersize=7, label='Ideal', markeredgecolor='black', markerfacecolor='none')
            global_handles['Ideal'] = h_ideal

        markers = ['x', 'o', '^', 'd', '*', 'v', '<', '>']  # Define a list of distinct markers
        for seed_idx, seed in enumerate(seeds):
            model_path = f"models/{project_name}/arch-{arch_idx}/{problem}/{seed}/{checkpoint}.zip"
            objectives, _ = run_experiments(env=env, model=None, model_path=model_path, preferences=preferences)
            
            # Use a different marker for each seed
            marker = markers[seed_idx % len(markers)]  # Cycle through markers if seeds > len(markers)
            h_pred, = ax.plot(objectives[:, 0], objectives[:, 1], marker, markersize=5, label=f'Predicted (Seed {seed})', markerfacecolor='none')
            global_handles[f'Predicted (Seed {seed})'] = h_pred

            

        ax.set_xlabel('Objective 0')
        ax.set_ylabel('Objective 1')
        title_name = env.config_name.replace("_", " ")
        title_name = title_name[0].upper() + title_name[1:] if title_name else problem
        ax.set_title(title_name, fontsize=9)
        ax.grid(True)

    # hide any empty subplots
    for i in range(n_problems, nrows * ncols):
        r = i // ncols
        c = i % ncols
        axes[r][c].axis('off')

    # Place a single legend for the figure (if we have handles) and anchor it in the top margin
    if global_handles:
        handles = [global_handles[k] for k in sorted(global_handles.keys())]
        labels = sorted(global_handles.keys())
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)

    # Reserve a bit more top margin so legend has breathing room when no suptitle is present
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save:
        if save_dir is None:
            save_dir = f"data/{project_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'pareto_fronts_grid_{checkpoint}.png')
        plt.savefig(save_path, bbox_inches='tight')
        save_path = os.path.join(save_dir, f'pareto_fronts_grid_{checkpoint}.pdf')
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_more_objectives(problem="logistic_long", 
                                    project_name = 'MoreObjectives', 
                                    seeds = range(0, 5),
                                    arch_idx = 0,
                                    total_steps = 1000000,
                                    step_size = 100000,
                                    problem_display_name = "Logistic Objective Function"
                                    ):
    csv_path = f"data/{project_name}/more_obj_{problem}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # If the aggregated CSV doesn't exist, compute it using the more-specific stats generator
        df = calculate_stats_more_objectives(problem=problem,
                                             project_name=project_name,
                                             seeds=seeds,
                                             arch_idx=arch_idx,
                                             total_steps=total_steps,
                                             step_size=step_size)
    # Prepare data: group by number of objectives and steps, compute mean and std across seeds
    # Expect columns: n_obj, steps, proportion_allocated
    if df is None or df.empty:
        raise ValueError(f"No data available for problem={problem} at {csv_path}")

    # ensure correct dtypes
    df = df.copy()
    df['n_obj'] = df['n_obj'].astype(int)
    df['steps'] = df['steps'].astype(int)
    df['proportion_allocated'] = df['proportion_allocated'].astype(float)

    grouped = df.groupby(['n_obj', 'steps'])['proportion_allocated']
    stats = grouped.agg(['mean', 'std']).reset_index()

    # Plot one curve per n_obj across steps, with shaded std band
    unique_n_objs = sorted(stats['n_obj'].unique())
    n_curves = len(unique_n_objs)

    # Choose colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_curves)]

    plt.clf()
    plt.figure(figsize=(8, 4 + max(0, n_curves // 3)))

    for idx, n_obj in enumerate(unique_n_objs):
        sub = stats[stats['n_obj'] == n_obj].sort_values('steps')
        steps = sub['steps'].values
        mean = sub['mean'].values
        std = sub['std'].fillna(0).values

        label = f'{n_obj} objectives'
        plt.plot(steps, mean, label=label, color=colors[idx], marker='o')
        # shaded region mean +/- std
        plt.fill_between(steps, mean - std, mean + std, color=colors[idx], alpha=0.2)

    plt.xlabel('Training steps')
    # plt.ylabel('Proportion allocated (mean ± std)')
    plt.title(f'Proportion of Resources Allocated (mean ± std)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    save_dir = f'data/{project_name}'
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    # Calculate ordering score using the final checkpoint at `total_steps` for each seed
    try:
        env = ResourceManagementEnv(f"config/problems/{problem}-p20-o20.yml")
    except Exception:
        env = None

    ordering_scores = []
    if env is not None:
        checkpoint_name = f"model_step_{total_steps}.zip"
        for seed in seeds:
            model_path = f"models/{project_name}/arch-{arch_idx}/{problem}-p20-o20/{seed}/{checkpoint_name}"
            if os.path.exists(model_path):
                try:
                    score = calculate_ordering_score(env=env, model=None, model_path=model_path)
                    ordering_scores.append(score)
                except Exception:
                    # if ordering calculation fails for one seed, skip it
                    continue

    ordering_text = None
    if len(ordering_scores) == 0:
        ordering_text = "20-Obj Ordering score: N/A (no valid model checkpoints found)"
    else:
        mean_score = float(np.mean(ordering_scores))
        std_score = float(np.std(ordering_scores))
        ordering_text = f"Final 20-Obj Ordering Score: {mean_score:.3f} ± {std_score:.3f}"

    # Render ordering score text on the plot (top-right corner)
    ax = plt.gca()
    if ordering_text is not None:
        ax.text(0.98, 0.98, ordering_text, transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
    
    save_path = os.path.join(save_dir, f'more_obj_{problem}.png')
    plt.savefig(save_path, bbox_inches='tight')
    save_path = os.path.join(save_dir, f'more_obj_{problem}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_stats_across_arch(problems: list[str], 
                                project_name: str, 
                                arch_idx_all: list[int],
                                arch_names: list[str],
                                is_finetuned: list[bool],
                                pretrain_steps: int = 1000000,
                                finetune_steps: int = 1000000,
                                n_partitions = 12,
                                seeds = range(0, 5)):
    """Plot summary bar charts (mean ± std) across architectures for multiple problems.

    The function tries to load an aggregated CSV produced by
    `calculate_stats_across_arch`. If it doesn't exist, it will call that
    function to generate the data. The resulting figure contains three
    subplots (hypervolume, non_dominated, ordering). For each subplot we
    display grouped bars per problem, one bar per architecture, and draw
    vertical separators between problems.
    """
    assert(len(arch_idx_all) == len(arch_names) == len(is_finetuned)), "arch_idx_all, arch_names, and is_finetuned must have the same length."
    # determine csv path consistent with calculate_stats_across_arch
    csv_path = f"data/{project_name}/summary_across_arch_{'-'.join(map(str, arch_idx_all))}.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # generate the CSV via the stats helper (it will save the file)
        df = calculate_stats_across_arch(problems=problems,
                                        project_name=project_name,
                                        arch_idx_all=arch_idx_all,
                                        is_finetuned=is_finetuned,
                                        pretrain_steps=pretrain_steps,
                                        finetune_steps=finetune_steps,
                                        n_partitions=n_partitions,
                                        seeds=seeds)

    if df is None or df.empty:
        raise ValueError(f"No stats available at {csv_path} and generation returned no data.")

    # If the dataframe lacks explicit problem/arch/seed columns, reconstruct them
    expected_rows = len(problems) * len(arch_idx_all) * len(list(seeds))
    if not {'problem', 'arch', 'seed'}.issubset(set(df.columns)):
        if len(df) == expected_rows:
            problem_col = []
            arch_col = []
            seed_col = []
            for problem in problems:
                for arch_idx, _ in zip(arch_idx_all, is_finetuned):
                    for seed in seeds:
                        problem_col.append(problem)
                        arch_col.append(int(arch_idx))
                        seed_col.append(int(seed))

            df = df.reset_index(drop=True).copy()
            df['problem'] = problem_col
            df['arch'] = arch_col
            df['seed'] = seed_col
        else:
            # Can't reliably reconstruct mapping; try best-effort but warn
            # Fall back to returning a simple plot grouped only by index
            print(f"Warning: expected {expected_rows} rows to reconstruct problem/arch mapping but got {len(df)}. Proceeding with available data.")

    metrics = ['hypervolume', 'non_dominated', 'ordering']
    metric_names = {'hypervolume': 'Hypervolume', 'non_dominated': 'Non-dominated', 'ordering': 'Ordering Score'}

    # Prepare aggregation: mean and std per (problem, arch)
    agg = df.groupby(['problem', 'arch'])[metrics].agg(['mean', 'std'])

    # Map architecture indices to provided names
    arch_to_name = dict(zip(arch_idx_all, arch_names))

    # Build and save a tidy DataFrame containing mean/std per metric for each problem-arch
    stats_rows = []
    for (problem_key, arch_key), values in agg.iterrows():
        row = {'problem': problem_key, 'arch': arch_key, 'arch_name': arch_to_name.get(arch_key, f'arch-{arch_key}')}
        for m in metrics:
            # safe access in case some aggregates are missing
            mean_val = values.get((m, 'mean'), np.nan) if hasattr(values, 'get') else values[(m, 'mean')]
            std_val = values.get((m, 'std'), np.nan) if hasattr(values, 'get') else values[(m, 'std')]
            row[f'{m}_mean'] = mean_val
            row[f'{m}_std'] = std_val
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_csv_path = os.path.join(f'data/{project_name}', f'summary_across_arch_{"-".join(map(str, arch_idx_all))}_stats.csv')
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
    stats_df.to_csv(stats_csv_path, index=False)

    # Reorder to ensure consistent problem ordering
    problems_order = list(problems)
    arch_order = list(arch_idx_all)

    n_problems = len(problems_order)
    n_arch = len(arch_order)

    # Build matrices for plotting
    means = {m: np.zeros((n_problems, n_arch)) for m in metrics}
    stds = {m: np.zeros((n_problems, n_arch)) for m in metrics}

    for i, problem in enumerate(problems_order):
        for j, arch in enumerate(arch_order):
            if (problem, arch) in agg.index:
                for m in metrics:
                    means[m][i, j] = agg.loc[(problem, arch)][(m, 'mean')]
                    stds[m][i, j] = agg.loc[(problem, arch)][(m, 'std')]
            else:
                # missing entry -> NaN
                for m in metrics:
                    means[m][i, j] = np.nan
                    stds[m][i, j] = np.nan

    # Create figure with three subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 5), squeeze=False)
    axes = axes[0]

    x = np.arange(n_problems)
    width = 0.8 / max(1, n_arch)

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for j, arch in enumerate(arch_order):
            pos = x + (j - (n_arch - 1) / 2) * width
            vals = means[metric][:, j]
            errs = stds[metric][:, j]
            # handle nan by replacing with zeros and hiding bars via mask
            mask = ~np.isnan(vals)
            if mask.any():
                arch_label = arch_to_name.get(arch, f'arch-{arch}')
                ax.bar(pos[mask], vals[mask], width, yerr=errs[mask], capsize=4, label=f'{arch_label}', alpha=0.9)

        # vertical separators between problems
        for sep in range(n_problems - 1):
            ax.axvline(sep + 0.5, color='grey', linestyle='--', linewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ') for p in problems_order], rotation=45, ha='right', fontsize=9)
        ax.set_title(metric_names.get(metric, metric))
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    # shared legend for architectures placed on top (use arch_names ordering)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # reorder legend entries to match arch_order / arch_names
        # labels come from plotting order; create desired label list and pick matching handles
        desired_labels = [arch_to_name.get(a, f'arch-{a}') for a in arch_order]
        # build mapping from label->handle (keep first occurrence)
        label_to_handle = {lab: h for h, lab in zip(handles, labels)}
        ordered_handles = [label_to_handle.get(l) for l in desired_labels if l in label_to_handle]
        ordered_labels = [l for l in desired_labels if l in label_to_handle]
        if ordered_handles:
            fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(ordered_handles), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_dir = f'data/{project_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'summary_across_arch_{"-".join(map(str, arch_idx_all))}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()