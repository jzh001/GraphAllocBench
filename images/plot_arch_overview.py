"""One-slide overview: architecture schematics (top) + Table 1 results (bottom).

Top band  — three compact pipelines. MLP flattens the observation; both HGNNs
share a GAT x2 message-passing backbone on the demand-resource graph and differ
only in POOLING (mean+max vs preference-conditioned attention) — the accent box.
Bottom band — the three Table 1 metrics (5 seeds, problems 6a-c), grouped by arch.
Landscape, space-efficient, colours link schematic -> bars.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_MLP, C_MEAN, C_ATTN = '#6c757d', '#2a9d8f', '#e76f51'
C_BONE, C_DARK, C_GREY = '#495057', '#333333', '#9aa0a6'
C_PREF = '#e0a028'   # preference-injection colour (amber)
ARCH = {0: ('MLP', C_MLP), 2: ('HGNN + MeanMax pool', C_MEAN), 3: ('HGNN + Attention pool', C_ATTN)}
ORDER = [0, 2, 3]

# ── data (canonical CSV, aggregated over seeds; matches paper Table 1) ──
df = pd.read_csv('graphallocbench/examples/data/GraphAllocBench-GNN-v1-1/summary_across_arch_0-2-3.csv')
g = (df.groupby(['problem', 'arch_idx'])
       .agg(hv=('hypervolume', 'mean'), hv_sd=('hypervolume', 'std'),
            pnds=('non_dominated', 'mean'), pnds_sd=('non_dominated', 'std'),
            order=('ordering', 'mean'), order_sd=('ordering', 'std')).reset_index())
probs = sorted(df['problem'].unique())
plabels = [f"6{p.split('_')[1][-1]}" if False else f"Problem {p.split('_')[1]}" for p in probs]
def series(prob, col):
    sub = g[g.problem == prob].set_index('arch_idx')
    return np.array([sub.loc[a, col] for a in ORDER])
hv = np.array([series(p, 'hv') for p in probs]); hv_sd = np.array([series(p, 'hv_sd') for p in probs])
hv_rel = hv / hv[:, [0]]
pnds = np.array([series(p, 'pnds') for p in probs]); pnds_sd = np.array([series(p, 'pnds_sd') for p in probs])
order = np.array([series(p, 'order') for p in probs]); order_sd = np.array([series(p, 'order_sd') for p in probs])

# ══ figure ══
fig = plt.figure(figsize=(15.5, 7.1))
gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.3], hspace=0.34, wspace=0.2,
                      left=0.055, right=0.985, top=0.9, bottom=0.13)
ax_sch = [fig.add_subplot(gs[0, j]) for j in range(3)]
ax_res = [fig.add_subplot(gs[1, j]) for j in range(3)]

# ── schematic primitives ──
def box(ax, cx, cy, w, h, text, fc, ec, tc='white', fs=9.5, lw=1.3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle='round,pad=0.006,rounding_size=0.04', fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold', zorder=4, linespacing=1.05)

def arrow(ax, x0, x1, y=0.52, color=C_DARK):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5, shrinkA=0, shrinkB=0), zorder=2)

def graph_icon(ax, cx, y=0.52):
    dem = [(cx - 0.045, y + 0.15), (cx + 0.045, y + 0.15)]
    res = [(cx - 0.06, y - 0.15), (cx, y - 0.15), (cx + 0.06, y - 0.15)]
    for a in dem:
        for b in res:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C_GREY, lw=0.7, alpha=0.7, zorder=2)
    ax.scatter([p[0] for p in dem], [p[1] for p in dem], s=90, color=C_MEAN, edgecolors='white', lw=1, zorder=3)
    ax.scatter([p[0] for p in res], [p[1] for p in res], s=90, color=C_BONE, edgecolors='white', lw=1, zorder=3)
    ax.text(cx, y + 0.30, 'demands', ha='center', fontsize=7.5, color=C_GREY)
    ax.text(cx, y - 0.34, 'resources', ha='center', fontsize=7.5, color=C_GREY)

def vector_icon(ax, cx, y=0.52):
    n = 5; s = 0.05
    for i in range(n):
        ax.add_patch(Rectangle((cx - s / 2, y - n * s / 2 + i * s), s, s * 0.9,
                     fc='#dfe3e6', ec=C_GREY, lw=0.8, zorder=3))
    ax.text(cx, y + n * s / 2 + 0.06, 'flattened\nobs', ha='center', va='bottom',
            fontsize=7.5, color=C_GREY, linespacing=1.0)

for ax in ax_sch:
    ax.set_xlim(0, 1.04); ax.set_ylim(0, 1); ax.axis('off')

HB = 0.36  # box height
BB = 0.52 - HB / 2  # box bottom
BUSY = 0.14         # preference-bus height

def pref_bus(ax, taps):
    """Amber bus tapping up into each box that consumes the preference vector."""
    x0 = 0.02; xe = max(taps)
    ax.plot([x0, xe], [BUSY, BUSY], color=C_PREF, lw=2.2, zorder=2, solid_capstyle='round')
    for tx in taps:
        ax.annotate('', xy=(tx, BB - 0.005), xytext=(tx, BUSY),
                    arrowprops=dict(arrowstyle='-|>', color=C_PREF, lw=1.7), zorder=2)
    ax.text(x0, BUSY - 0.10, 'preferences $w$', ha='left', va='top', fontsize=8.5,
            color=C_PREF, fontweight='bold')

# (0) MLP  — prefs are part of the flattened observation -> into the encoder
ax = ax_sch[0]
ax.set_title('MLP', loc='left', fontsize=13, fontweight='bold', color=C_MLP, pad=6)
vector_icon(ax, 0.10)
arrow(ax, 0.19, 0.31)
box(ax, 0.47, 0.52, 0.28, HB, 'MLP encoder\n[256, 256]', C_MLP, C_MLP)
arrow(ax, 0.62, 0.74)
box(ax, 0.87, 0.52, 0.20, HB, 'π · V', 'white', C_DARK, tc=C_DARK)
pref_bus(ax, [0.47])

# (2) HGNN + MeanMax  — prefs in node features (GAT) and final head; NOT in pooling
ax = ax_sch[1]
ax.set_title('HGNN + MeanMax pool', loc='left', fontsize=13, fontweight='bold', color=C_MEAN, pad=6)
graph_icon(ax, 0.10)
arrow(ax, 0.19, 0.27)
box(ax, 0.40, 0.52, 0.22, HB, 'GAT × 2\nmsg-passing', C_BONE, C_BONE)
arrow(ax, 0.52, 0.57)
box(ax, 0.70, 0.52, 0.22, HB, 'mean ⊕ max\npool', C_MEAN, C_MEAN)
arrow(ax, 0.82, 0.875)
box(ax, 0.95, 0.52, 0.10, HB, 'π·V', 'white', C_DARK, tc=C_DARK, fs=8.5)
pref_bus(ax, [0.40, 0.95])

# (3) HGNN + Attention  — prefs in node features, POOLING (conditions attention), final head
ax = ax_sch[2]
ax.set_title('HGNN + Attention pool', loc='left', fontsize=13, fontweight='bold', color=C_ATTN, pad=6)
graph_icon(ax, 0.10)
arrow(ax, 0.19, 0.27)
box(ax, 0.40, 0.52, 0.22, HB, 'GAT × 2\nmsg-passing', C_BONE, C_BONE)
arrow(ax, 0.52, 0.575)
box(ax, 0.70, 0.52, 0.21, HB, 'attention\npool', C_ATTN, C_ATTN)
arrow(ax, 0.815, 0.875)
box(ax, 0.95, 0.52, 0.10, HB, 'π·V', 'white', C_DARK, tc=C_DARK, fs=8.5)
pref_bus(ax, [0.40, 0.70, 0.95])

# ── results ──
x = np.arange(len(probs)); w = 0.26
def grouped(ax, vals, sds, fold=None):
    for k, a in enumerate(ORDER):
        xs = x + (k - 1) * w
        ax.bar(xs, vals[:, k], w, yerr=sds[:, k], color=ARCH[a][1], edgecolor='white',
               linewidth=0.6, capsize=2.5, error_kw=dict(lw=1.0, ecolor=C_DARK, alpha=0.5), zorder=3)
        if fold is not None:
            for xi, v, s, f in zip(xs, vals[:, k], sds[:, k], fold[:, k]):
                if k > 0:
                    ax.text(xi, v + s + vals.max() * 0.02, f'{f:.1f}×', ha='center', va='bottom',
                            fontsize=8.5, color=ARCH[a][1], fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(plabels, fontsize=10); ax.margins(x=0.04)
    ax.tick_params(labelsize=9)

TK = dict(loc='left', fontsize=12.5, fontweight='bold', pad=6)
ax = ax_res[0]; grouped(ax, hv / 1e6, hv_sd / 1e6, fold=hv_rel)
ax.set_title('Hypervolume  ↑', **TK); ax.set_ylabel('$\\times10^{6}$', fontsize=10)
ax.set_ylim(0, (hv / 1e6).max() * 1.2)
ax = ax_res[1]; grouped(ax, pnds, pnds_sd)
ax.set_title('Non-dominated %  ↑', **TK); ax.set_ylim(0, max(0.26, (pnds + pnds_sd).max() * 1.15))
ax = ax_res[2]; grouped(ax, order, order_sd)
ax.set_title('Ordering score  ↑', **TK); ax.set_ylim(0, 1.05)

handles = [plt.Rectangle((0, 0), 1, 1, color=ARCH[a][1]) for a in ORDER]
fig.legend(handles, [ARCH[a][0] for a in ORDER], ncol=3, loc='lower center',
           frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=11, handlelength=1.3)
fig.text(0.985, 0.006, 'mean ± s.d. over 5 seeds', ha='right', va='bottom',
         fontsize=8.5, color=C_GREY)
fig.text(0.055, 0.006, 'amber = where the preference vector $w$ is injected', ha='left',
         va='bottom', fontsize=8.5, color=C_PREF, fontweight='bold')
fig.suptitle('Graph structure + attention pooling trade preference-ordering for far larger objective coverage',
             fontsize=13.5, y=0.98, color=C_DARK)

for ext in ('png', 'pdf'):
    fig.savefig(f'images/arch_overview.{ext}', bbox_inches='tight', dpi=200)
print('saved images/arch_overview.png / .pdf')
