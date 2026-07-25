"""Reward-landscape figure: shows how GraphAllocBench objective functions are
deliberately designed to be challenging (smooth baseline -> spikes / oscillation /
non-convex steps). Renders J_i over the (P0, P1) production grid as heatmaps."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from graphallocbench.city_env import CityPlannerEnv

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.8,
})

# (config, short label, challenge tag)
PROBLEMS = [
    ('problem_0',  'Problem 0',  'smooth  (baseline)'),
    ('problem_1a', 'Problem 1a', 'spikes'),
    ('problem_1b', 'Problem 1b', 'oscillatory'),
    ('problem_1c', 'Problem 1c', 'non-convex steps'),
]
GRID = 240
CMAP = 'magma'

def objective_grid(env, obj_i, pmax, n=GRID):
    xs = np.linspace(0, pmax, n)
    Z = np.empty((n, n))
    for a, p0 in enumerate(xs):
        for b, p1 in enumerate(xs):
            Z[a, b] = env.get_objectives(np.array([p0, p1]), None)[obj_i]
    return xs, Z  # Z[i,j] = J for P0=xs[i], P1=xs[j]

fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)

for col, (cfg, label, tag) in enumerate(PROBLEMS):
    env = CityPlannerEnv(cfg)
    pmax = float(min(env.avail_resources))
    for row in range(2):  # J0 top, J1 bottom
        ax = axes[row, col]
        xs, Z = objective_grid(env, row, pmax)
        # normalise each panel to its own range -> reveals shape, not magnitude
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
        im = ax.imshow(Zn.T, origin='lower', extent=[0, pmax, 0, pmax],
                       cmap=CMAP, aspect='equal', interpolation='bilinear', vmin=0, vmax=1)
        # contours emphasise ripples / step boundaries
        ax.contour(xs, xs, Zn.T, levels=7, colors='white', linewidths=0.45, alpha=0.35)
        ax.set_xticks([0, pmax]); ax.set_yticks([0, pmax])
        ax.set_xticklabels(['0', f'{int(pmax)}'], fontsize=9)
        ax.set_yticklabels(['0', f'{int(pmax)}'], fontsize=9)
        ax.tick_params(length=2, pad=1)
        if row == 0:
            ax.set_title(f'{label}\n{tag}', fontsize=12, pad=8,
                         fontweight='bold' if col == 0 else 'normal')
        if col == 0:
            ax.set_ylabel(f'$J_{row}$', rotation=0, fontsize=15, labelpad=14, va='center')
        if row == 1:
            ax.set_xlabel('$P_0$', fontsize=11)
        if col == 0:
            ax.text(-0.36, 0.5, '$P_1$', transform=ax.transAxes,
                    rotation=90, va='center', ha='center', fontsize=11)

# one shared colorbar for the "low -> high reward" magnitude encoding
cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.015, aspect=28, ticks=[0, 1])
cbar.ax.set_yticklabels(['low', 'high'])
cbar.set_label('objective reward  (per-panel normalized)', fontsize=10)

fig.suptitle('Designed objective landscapes:  each $J_i$ over the production grid $(P_0, P_1)$',
             fontsize=13.5, y=1.06, x=0.44)

for ext in ('png', 'pdf'):
    fig.savefig(f'images/reward_landscapes.{ext}', bbox_inches='tight', dpi=200)
print('saved images/reward_landscapes.png / .pdf')
