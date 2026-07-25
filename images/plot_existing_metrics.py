"""Illustration of existing MORL evaluation methods (Hypervolume, Sparsity,
Generational Distance) on GraphAllocBench problems. Show-not-tell, minimalist.
  (a) Hypervolume        - Problem 0  : dominated area vs a reference point
  (b) Sparsity           - Problem 2b : spacing between adjacent solutions (gaps)
  (c) Generational Dist. - Problem 2b : distance from predictions to the true front
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from graphallocbench.city_env import CityPlannerEnv
from graphallocbench.evaluation import run_experiments, get_analytical_objectives

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_TEAL, C_ORANGE, C_GREY, C_DARK = '#2a9d8f', '#e76f51', '#9aa0a6', '#333333'
mp = lambda p, s=0: f'models/GraphAllocBench-v2/arch-0/{p}/{s}/model_step_1000000.zip'

def nd_mask(o):
    idx = NonDominatedSorting().do(-o, only_non_dominated_front=True)
    m = np.zeros(len(o), bool); m[idx] = True; return m

def get_front(prob, npart=12):
    env = CityPlannerEnv(prob)
    prefs = get_reference_directions('das-dennis', 2, n_partitions=npart)
    pred, _ = run_experiments(env=env, model_path=mp(prob), preferences=prefs, random_seed=42)
    ndp = np.unique(pred[nd_mask(pred)], axis=0)
    ndp = ndp[ndp[:, 0].argsort()]
    ideal = get_analytical_objectives(env=env)
    ideal = ideal[ideal[:, 0].argsort()]
    return ndp, ideal

def staircase(pts):
    """dominated-region boundary (max problem) from origin, for fill."""
    pts = pts[pts[:, 0].argsort()]
    vx, vy = [0.0, 0.0], [0.0, pts[0, 1]]
    for i, (px, py) in enumerate(pts):
        vx.append(px); vy.append(py)
        if i < len(pts) - 1:
            vx.append(px); vy.append(pts[i + 1, 1])
    vx.append(pts[-1, 0]); vy.append(0.0)
    return np.array(vx), np.array(vy)

# ── data ──
p0_pred, _ = get_front('problem_0')
p2_pred, p2_ideal = get_front('problem_2b', npart=24)  # denser front for spacing/GD

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
TITLE_KW = dict(loc='left', fontsize=13.5, fontweight='bold', pad=10)
def ptag(ax, name):
    ax.text(1.0, 1.015, name, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color=C_GREY)

# ═══ (a) Hypervolume — Problem 0 ════════════════════════════════════════════
ax = axes[0]
vx, vy = staircase(p0_pred)
ax.fill(vx, vy, color=C_ORANGE, alpha=0.22, zorder=1)
ax.plot(vx, vy, color=C_ORANGE, lw=1.7, zorder=3)
ax.scatter(p0_pred[:, 0], p0_pred[:, 1], s=42, color=C_ORANGE, edgecolors=C_DARK,
           lw=0.6, zorder=4, label='Solutions')
ax.scatter([0], [0], marker='*', s=200, color=C_DARK, zorder=5)
ax.annotate('reference point $r$', xy=(0, 0), xytext=(3.2, 2.6), fontsize=10, color=C_DARK,
            va='center')
xl0 = p0_pred[:, 0].max() * 1.1; yl0 = p0_pred[:, 1].max() * 1.1
ax.text(xl0*0.46, yl0*0.42, 'dominated\nvolume', ha='center', va='center',
        fontsize=11, color=C_DARK, fontweight='bold')
ax.set_title('(a) Hypervolume', **TITLE_KW)
ax.set_xlabel('Objective 0'); ax.set_ylabel('Objective 1')
ax.set_xlim(0, xl0); ax.set_ylim(0, yl0)
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
ptag(ax, 'Problem 0')

# ═══ (b) Sparsity — Problem 2b ══════════════════════════════════════════════
ax = axes[1]
pts = p0_pred
ax.plot(pts[:, 0], pts[:, 1], '-', color=C_ORANGE, lw=1.2, alpha=0.5, zorder=2)
# mark every adjacent spacing; highlight the widest one
gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
big = int(np.argmax(gaps))
for i in range(len(pts) - 1):
    a, b = pts[i], pts[i + 1]
    is_big = (i == big)
    ax.annotate('', xy=b, xytext=a, zorder=3,
                arrowprops=dict(arrowstyle='<->', color=(C_DARK if is_big else C_GREY),
                                lw=(1.9 if is_big else 1.1), shrinkA=5, shrinkB=5,
                                alpha=(1.0 if is_big else 0.7)))
ax.scatter(pts[:, 0], pts[:, 1], s=52, color=C_ORANGE, edgecolors=C_DARK, lw=0.7,
           zorder=4, label='Solutions')
a, b = pts[big], pts[big + 1]; mid = (a + b) / 2
ax.annotate('adjacent gap', xy=mid, xytext=(mid[0] + 0.3, mid[1] - 3.2),
            ha='center', va='top', fontsize=10, color=C_DARK, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=C_DARK, lw=0.8))
ax.set_title('(b) Sparsity', **TITLE_KW)
ax.set_xlabel('Objective 0'); ax.set_ylabel('Objective 1')
xl1 = pts[:, 0].max() * 1.1; yl1 = pts[:, 1].max() * 1.1
ax.set_xlim(0, xl1); ax.set_ylim(0, yl1)
ax.text(0.05, 0.12, 'average spacing between\nadjacent solutions\n(evenness of coverage)',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5, color=C_DARK)
ptag(ax, 'Problem 0')

# ═══ (c) Generational Distance — Problem 2b ═════════════════════════════════
ax = axes[2]
ax.plot(p2_ideal[:, 0], p2_ideal[:, 1], '-', color=C_TEAL, lw=1.8, zorder=2, label='True front')
ax.scatter(p2_ideal[:, 0], p2_ideal[:, 1], marker='s', s=30, facecolors='none',
           edgecolors=C_TEAL, lw=1.3, zorder=3)
# distance from each predicted point to nearest true-front point
for q in p2_pred:
    d = np.linalg.norm(p2_ideal - q, axis=1)
    nn = p2_ideal[d.argmin()]
    ax.plot([q[0], nn[0]], [q[1], nn[1]], ls='--', color=C_GREY, lw=1.2, zorder=3)
ax.scatter(p2_pred[:, 0], p2_pred[:, 1], s=55, color=C_ORANGE, edgecolors=C_DARK, lw=0.7,
           zorder=5, label='Predictions')
ax.set_title('(c) Generational distance', **TITLE_KW)
ax.set_xlabel('Objective 0'); ax.set_ylabel('Objective 1')
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.text(0.04, 0.10, 'avg. distance from predictions\nto the true Pareto front', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=9.5, color=C_DARK)
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
ptag(ax, 'Problem 2b')

plt.tight_layout(w_pad=2.5)
for ext in ('png', 'pdf'):
    fig.savefig(f'images/existing_metrics.{ext}', bbox_inches='tight', dpi=200)
print('saved images/existing_metrics.png / .pdf')
