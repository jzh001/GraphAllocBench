"""Table 1 (GraphAllocBench, GNN sweep) as a visual: MLP vs HGNN+MeanMax pooling
vs HGNN+Attention pooling on the three large problems (6a-c), over 5 seeds.

Three metrics, all "higher = better":
  (a) Hypervolume  — shown relative to the MLP baseline (fold), scales vary by problem
  (b) % non-dominated (PNDS)
  (c) Ordering score
Story: graph structure + attention pooling expand the achievable objective volume
several-fold, at some cost to preference-ordering fidelity; PNDS stays comparable.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_MLP, C_MEAN, C_ATTN, C_DARK, C_GREY = '#6c757d', '#2a9d8f', '#e76f51', '#333333', '#9aa0a6'
ARCH = {0: ('MLP', C_MLP), 2: ('HGNN · MeanMax pool', C_MEAN), 3: ('HGNN · Attention pool', C_ATTN)}
ORDER = [0, 2, 3]

df = pd.read_csv('graphallocbench/examples/data/GraphAllocBench-GNN-v1-1/summary_across_arch_0-2-3.csv')
g = (df.groupby(['problem', 'arch_idx'])
       .agg(hv=('hypervolume', 'mean'), hv_sd=('hypervolume', 'std'),
            pnds=('non_dominated', 'mean'), pnds_sd=('non_dominated', 'std'),
            order=('ordering', 'mean'), order_sd=('ordering', 'std'))
       .reset_index())
probs = sorted(df['problem'].unique())
plabels = [f"Problem {p.split('_')[1]}" for p in probs]

def series(prob, col):
    sub = g[g.problem == prob].set_index('arch_idx')
    return np.array([sub.loc[a, col] for a in ORDER])

# hypervolume relative to MLP within each problem
hv = np.array([series(p, 'hv') for p in probs])
hv_sd = np.array([series(p, 'hv_sd') for p in probs])
hv_rel = hv / hv[:, [0]]
hv_rel_sd = hv_sd / hv[:, [0]]
pnds = np.array([series(p, 'pnds') for p in probs]); pnds_sd = np.array([series(p, 'pnds_sd') for p in probs])
order = np.array([series(p, 'order') for p in probs]); order_sd = np.array([series(p, 'order_sd') for p in probs])

fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
x = np.arange(len(probs)); w = 0.26
colors = [ARCH[a][1] for a in ORDER]
TITLE_KW = dict(loc='left', fontsize=13.5, fontweight='bold', pad=10)

def grouped(ax, vals, sds, fold=None):
    for k, a in enumerate(ORDER):
        xs = x + (k - 1) * w
        ax.bar(xs, vals[:, k], w, yerr=sds[:, k], color=ARCH[a][1],
               edgecolor='white', linewidth=0.6, capsize=3,
               error_kw=dict(lw=1.0, ecolor=C_DARK, alpha=0.55), zorder=3)
        if fold is not None:
            for xi, v, s, f in zip(xs, vals[:, k], sds[:, k], fold[:, k]):
                if k > 0:
                    ax.text(xi, v + s + vals.max() * 0.02, f'{f:.1f}×', ha='center',
                            va='bottom', fontsize=9, color=ARCH[a][1], fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(plabels)
    ax.margins(x=0.04)

# (a) Hypervolume (native ×10^6, matching the paper table), fold over MLP annotated
ax = axes[0]
grouped(ax, hv / 1e6, hv_sd / 1e6, fold=hv_rel)
ax.set_title('(a) Hypervolume', **TITLE_KW)
ax.set_ylabel('hypervolume  ($\\times10^{6}$)')
ax.set_ylim(0, (hv / 1e6).max() * 1.22)
ax.text(0.02, 0.97, 'higher = better', transform=ax.transAxes, fontsize=9,
        color=C_GREY, va='top')

# (b) PNDS
ax = axes[1]
grouped(ax, pnds, pnds_sd)
ax.set_title('(b) Non-dominated %', **TITLE_KW)
ax.set_ylabel('fraction of solutions non-dominated')
ax.set_ylim(0, max(0.26, (pnds + pnds_sd).max() * 1.18))
ax.text(0.02, 0.97, 'higher = better', transform=ax.transAxes, fontsize=9,
        color=C_GREY, va='top')

# (c) Ordering
ax = axes[2]
grouped(ax, order, order_sd)
ax.set_title('(c) Ordering score', **TITLE_KW)
ax.set_ylabel('preference-ordering fidelity')
ax.set_ylim(0, 1.05)
ax.text(0.02, 0.97, 'higher = better', transform=ax.transAxes, fontsize=9,
        color=C_GREY, va='top')

# shared legend
handles = [plt.Rectangle((0, 0), 1, 1, color=ARCH[a][1]) for a in ORDER]
labels = [ARCH[a][0] for a in ORDER]
fig.legend(handles, labels, ncol=3, loc='lower center', frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=11, handlelength=1.3)

fig.suptitle('Graph structure + attention pooling trade preference-ordering for far larger objective coverage',
             fontsize=13.5, y=1.005, color=C_DARK)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
for ext in ('png', 'pdf'):
    fig.savefig(f'images/arch_comparison_table1.{ext}', bbox_inches='tight', dpi=200)
print('saved images/arch_comparison_table1.png / .pdf')
