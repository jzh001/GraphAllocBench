"""Three-panel figure justifying the three GraphAllocBench metrics, each on the
problem where its story is clearest:
  (a) HV Ratio   - Problem 2b : predicted front under-covers the ideal (gap = loss)
  (b) PNDS       - Problem 1c : many predictions are dominated (unreliable)
  (c) Ordering   - Problem 5d : objective ignores its own rising preference weight
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import spearmanr
from graphallocbench.city_env import CityPlannerEnv
from graphallocbench.evaluation import run_experiments, get_analytical_objectives

# ── house style (matches paper / existing notebooks) ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_TEAL, C_ORANGE, C_GREY, C_DARK = '#2a9d8f', '#e76f51', '#9aa0a6', '#333333'
MODEL_ROOT = 'models/GraphAllocBench-v2/arch-0'
mp = lambda p, s=0: f'{MODEL_ROOT}/{p}/{s}/model_step_1000000.zip'

def nd_mask(o):
    idx = NonDominatedSorting().do(-o, only_non_dominated_front=True)
    m = np.zeros(len(o), bool); m[idx] = True; return m

def hv_polygon(pts):
    pts = pts[pts[:, 0].argsort()]
    vx, vy = [0.0, 0.0], [0.0, pts[0, 1]]
    for i, (px, py) in enumerate(pts):
        vx.append(px); vy.append(py)
        if i < len(pts) - 1:
            vx.append(px); vy.append(pts[i + 1, 1])
    vx.append(pts[-1, 0]); vy.append(0.0)
    return np.array(vx), np.array(vy)

# ══ data ══════════════════════════════════════════════════════════════════
PROB_A, PROB_B = 'problem_2b', 'problem_2b'   # (a) HV ratio, (b) PNDS  (same predictions -> HV alone is insufficient)

# (a) HV ratio
env_a = CityPlannerEnv(PROB_A)
prefs_a = get_reference_directions('das-dennis', 2, n_partitions=12)
pred_a, _ = run_experiments(env=env_a, model_path=mp(PROB_A), preferences=prefs_a, random_seed=42)
ideal_a = get_analytical_objectives(env=env_a)
nd_a = nd_mask(pred_a); ndp_a = pred_a[nd_a]
from pymoo.indicators.hv import HV
hv = lambda o: HV(ref_point=np.full(o.shape[1], 1e-3))(-o)
hv_ratio = hv(pred_a) / hv(ideal_a)

# (b) PNDS
env_b = CityPlannerEnv(PROB_B)
prefs_b = get_reference_directions('das-dennis', 2, n_partitions=12)
pred_b, _ = run_experiments(env=env_b, model_path=mp(PROB_B), preferences=prefs_b, random_seed=42)
nd_b = nd_mask(pred_b); pnds = nd_b.mean()
ndp_b, domp_b = pred_b[nd_b], pred_b[~nd_b]

# (c) Ordering — Problem 5d, two selected objectives
OBJ_GOOD, OBJ_BAD = __import__('os').environ.get('OBJS', '1,2').split(',')
OBJ_GOOD, OBJ_BAD = int(OBJ_GOOD), int(OBJ_BAD)
env_c = CityPlannerEnv('problem_5d'); N = env_c.n_objectives
nstep, nsamp = 21, 10
wv = np.linspace(0, 1, nstep)
rng = np.random.default_rng(0)
def sweep(obj_i):
    curves = []
    for _ in range(nsamp):
        prefs = []
        for w in wv:
            rest = rng.dirichlet(np.ones(N - 1))
            p = np.empty(N); p[obj_i] = w
            idx = [k for k in range(N) if k != obj_i]; p[idx] = (1 - w) * rest
            prefs.append(p)
        objs, _ = run_experiments(env=env_c, model_path=mp('problem_5d'), preferences=np.array(prefs), random_seed=42)
        curves.append(objs[:, obj_i])
    curves = np.array(curves)
    # Ordering Score per Algorithm 1: average the per-sample Spearman(w, J_i),
    # mapped to [0,1]. (A constant sequence counts as perfectly ordered -> 1.)
    scores = []
    for c in curves:
        s = 1.0 if np.allclose(c, c[0]) else (spearmanr(wv, c).statistic + 1) / 2
        scores.append(s)
    return curves.mean(0), curves.std(0), float(np.mean(scores))
mg, sg, osg = sweep(OBJ_GOOD)
mb, sb, osb = sweep(OBJ_BAD)
# normalise each curve to [0,1] for shared axis (OS is rank-based / magnitude-agnostic)
def norm_pair(m, s):
    rng_ = m.max() - m.min() + 1e-9
    return (m - m.min()) / rng_, s / rng_
mgn, sgn = norm_pair(mg, sg)
mbn, sbn = norm_pair(mb, sb)

print(f'HV ratio={hv_ratio:.3f}  PNDS={pnds:.3f}  OS_good(obj{OBJ_GOOD})={osg:.3f}  OS_bad(obj{OBJ_BAD})={osb:.3f}')

# ══ figure ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
TITLE_KW = dict(loc='left', fontsize=13.5, fontweight='bold', pad=10)

# ── (a) HV ratio ──
ax = axes[0]
xl = ideal_a[:, 0].max() * 1.08; yl = ideal_a[:, 1].max() * 1.08
ix, iy = hv_polygon(ideal_a)
ax.fill(ix, iy, color=C_TEAL, alpha=0.15, zorder=1)
ax.plot(ix, iy, color=C_TEAL, lw=1.6, zorder=2, label='Ideal front')
px, py = hv_polygon(ndp_a)
ax.fill(px, py, color=C_ORANGE, alpha=0.30, zorder=3)
ax.plot(px, py, color=C_ORANGE, lw=1.6, zorder=4, label='Predicted front')
ax.scatter(ideal_a[:, 0], ideal_a[:, 1], marker='s', s=34, facecolors='none',
           edgecolors=C_TEAL, lw=1.4, zorder=5)
ax.scatter(pred_a[:, 0], pred_a[:, 1], marker='o', s=26, color=C_ORANGE, zorder=6)
ax.set_title('(a) Hypervolume ratio', **TITLE_KW)
ax.set_xlabel('Objective 0'); ax.set_ylabel('Objective 1')
ax.set_xlim(0, xl); ax.set_ylim(0, yl)
ax.annotate('coverage\ngap', xy=(xl*0.52, yl*0.62), fontsize=10.5, color=C_TEAL, ha='center', fontweight='bold')
ax.text(0.97, 0.95, f'HV ratio = {hv_ratio:.2f}', transform=ax.transAxes, ha='right', va='top',
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_ORANGE, alpha=0.9))
ax.text(1.0, 1.015, 'Problem ' + PROB_A.split('_')[1], transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, style='italic', color=C_GREY)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.9)

# ── (b) PNDS ──
ax = axes[1]
pnx, pny = hv_polygon(ndp_b)
ax.plot(pnx, pny, color=C_ORANGE, lw=1.3, ls='--', alpha=0.6, zorder=2)
ax.scatter(domp_b[:, 0], domp_b[:, 1], marker='x', s=70, color=C_GREY, lw=2.2,
           label=f'Dominated ({(~nd_b).sum()})', zorder=3)
ax.scatter(ndp_b[:, 0], ndp_b[:, 1], marker='o', s=60, color=C_ORANGE, edgecolors=C_DARK,
           lw=0.7, label=f'Non-dominated ({nd_b.sum()})', zorder=4)
ax.set_title('(b) Non-dominated %', **TITLE_KW)
ax.set_xlabel('Objective 0'); ax.set_ylabel('Objective 1')
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.text(0.97, 0.95, f'PNDS = {pnds:.2f}', transform=ax.transAxes, ha='right', va='top',
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_ORANGE, alpha=0.9))
ax.text(0.04, 0.14, 'nearly half the predictions\nare dominated (unreliable)', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=9.5, color=C_DARK)
ax.text(1.0, 1.015, 'Problem ' + PROB_B.split('_')[1], transform=ax.transAxes, ha='right', va='bottom',
        fontsize=10, style='italic', color=C_GREY)
ax.legend(loc='center right', fontsize=9.5, framealpha=0.9)

# ── (c) Ordering ──
ax = axes[2]
ax.plot([0, 1], [0, 1], color=C_GREY, lw=1.3, ls=':', zorder=1, label='ideal (monotone)')
ax.plot(wv, mgn, 'o-', color=C_TEAL, ms=4.5, lw=1.9, zorder=4,
        label=f'obeys pref  (obj {OBJ_GOOD}, OS={osg:.2f})')
ax.plot(wv, mbn, 's-', color=C_ORANGE, ms=4.5, lw=1.9, zorder=4,
        label=f'ignores pref  (obj {OBJ_BAD}, OS={osb:.2f})')
ax.set_title('(c) Ordering score', **TITLE_KW)
ax.set_xlabel('Preference weight on the objective  $w_i$')
ax.set_ylabel('Objective value  (normalized)')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.12)
ax.text(1.0, 1.015, 'Problem 5d', transform=ax.transAxes, ha='right', va='bottom',
        fontsize=10, style='italic', color=C_GREY)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95).set_zorder(10)

plt.tight_layout(w_pad=2.5)
for ext in ('png', 'pdf'):
    fig.savefig(f'images/metrics_justification.{ext}', bbox_inches='tight', dpi=200)
print('saved images/metrics_justification.png / .pdf')
