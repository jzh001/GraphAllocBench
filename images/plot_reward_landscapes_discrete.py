"""Discrete-aware companion to ``plot_reward_landscapes.py``.

Productions in GraphAllocBench are integers (production = min over required
resources), so the objective functions J_i are only ever evaluated on the
integer lattice (P0, P1) in {0..pmax}^2 -- never on the continuous plane.  The
continuous figure smooths J_i with a 240x240 grid, bilinear interpolation, and
smooth contours, which invents reward values at production states the
environment can never reach.  This version evaluates J_i at the integer
production states only and renders each objective as a lattice of discrete
cells, so the spikes / oscillations / non-convex steps are read off the states
that actually exist.

    uv run python images/plot_reward_landscapes_discrete.py
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
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
CMAP = 'magma'


def objective_grid(env, obj_i, pmax):
    """J_i at every integer production state (P0, P1) in {0..pmax}^2."""
    xs = np.arange(0, int(pmax) + 1)
    Z = np.empty((len(xs), len(xs)))
    for a, p0 in enumerate(xs):
        for b, p1 in enumerate(xs):
            Z[a, b] = env.get_objectives(np.array([float(p0), float(p1)]), None)[obj_i]
    return xs, Z  # Z[i,j] = J for P0=xs[i], P1=xs[j]


fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)

for col, (cfg, label, tag) in enumerate(PROBLEMS):
    env = CityPlannerEnv(cfg)
    pmax = int(min(env.avail_resources))
    edges = np.arange(-0.5, pmax + 1.5)        # cell borders around each integer
    for row in range(2):  # J0 top, J1 bottom
        ax = axes[row, col]
        xs, Z = objective_grid(env, row, pmax)
        # normalise each panel to its own range -> reveals shape, not magnitude
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
        # discrete cells only: one square per attainable integer state, no interp.
        # hairline seams keep the lattice legible without glaring on a slide
        im = ax.pcolormesh(edges, edges, Zn.T, cmap=CMAP, vmin=0, vmax=1,
                           edgecolors=(1, 1, 1, 0.12), linewidth=0.35)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, pmax + 0.5); ax.set_ylim(-0.5, pmax + 0.5)
        ax.set_xticks([0, pmax]); ax.set_yticks([0, pmax])
        ax.set_xticklabels(['0', f'{pmax}'], fontsize=9)
        ax.set_yticklabels(['0', f'{pmax}'], fontsize=9)
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

fig.suptitle('Designed objective landscapes on the integer production lattice:  '
             'each $J_i$ at every attainable state $(P_0, P_1)$',
             fontsize=13.5, y=1.06, x=0.44)

for ext in ('png', 'pdf'):
    fig.savefig(f'images/reward_landscapes_discrete.{ext}', bbox_inches='tight', dpi=200)
print('saved images/reward_landscapes_discrete.png / .pdf')
