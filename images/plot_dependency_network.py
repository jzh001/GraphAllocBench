"""Scalability of GraphAllocBench: the dependency network of a 100-demand problem.

Unimodal projection of the demand<->resource bipartite graph: two demands are
linked when they compete for the same resources (edge weight = #shared resources).
Node colour = how much that demand is produced for (summed over the preference
simplex); node size = its dependency footprint (#resources it needs). The dense
web + few brightly-served nodes shows how coupling limits what can be satisfied.

    python images/plot_dependency_network.py            # problem_6a (default)
    python images/plot_dependency_network.py problem_6b
"""
import sys, numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from net_common import (layout, PROD_CMAP, C_IDLE, C_ORANGE, C_DARK, C_GREY, ARCH_LABEL)

PROB = sys.argv[1] if len(sys.argv) > 1 else 'problem_6a'
CACHE = f'cache/scalability/alloc_{PROB}.npz'   # build with images/gen_dependency_network_data.py
d = np.load(CACHE)
S, req, demands, prod = d['S'], d['req'], d['demands'], d['prod_agg'].astype(float)
D = S.shape[0]
footprint = req.sum(1)              # #resources each demand needs (dependency burden)
n_links = int((S > 0).sum() // 2)
n_dep = int(req.sum())
served = prod > 0

G, pos, comm_of, comms = layout(S, k=3, seed=7)
P = np.array([pos[i] for i in range(D)])

# node sizing by dependency footprint
smin, smax = 45, 620
fn = (footprint - footprint.min()) / (np.ptp(footprint) + 1e-9)
sizes = smin + (smax - smin) * fn

# node colour by production intensity (idle = pale grey)
norm = Normalize(vmin=0, vmax=np.quantile(prod[served], 0.97) if served.any() else 1)

fig, ax = plt.subplots(figsize=(14.5, 8.2))

# ── faint per-cluster tints (cluster legibility) ──
from matplotlib.patches import Circle as MplCircle
for ci, c in enumerate(comms):
    idx = np.array(sorted(c))
    ctr = P[idx].mean(0)
    rad = np.linalg.norm(P[idx] - ctr, axis=1).max() + 0.28
    ax.add_patch(MplCircle(ctr, rad, facecolor=C_GREY, alpha=0.06,
                           edgecolor='none', zorder=0))

# ── edges: intra-cluster solid-ish, inter-cluster faint (shows coupling) ──
from matplotlib.collections import LineCollection
ws = np.array([G[u][v]['weight'] for u, v in G.edges()])
wn = ws / ws.max()
same = np.array([comm_of.get(u) == comm_of.get(v) for u, v in G.edges()])
segs = [(P[u], P[v]) for u, v in G.edges()]
cols = [(0.42, 0.45, 0.48, 0.18 + 0.30 * wn[i]) if same[i]
        else (0.60, 0.63, 0.66, 0.05 + 0.10 * wn[i]) for i in range(len(segs))]
lws = [0.3 + 1.3 * wn[i] if same[i] else 0.25 + 0.5 * wn[i] for i in range(len(segs))]
ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1))

# ── nodes ──
# idle first (recede), then served on top
ax.scatter(P[~served, 0], P[~served, 1], s=sizes[~served], c=C_IDLE,
           edgecolors='white', linewidths=0.5, zorder=2)
ax.scatter(P[served, 0], P[served, 1], s=sizes[served], c=prod[served],
           cmap=PROD_CMAP, norm=norm, edgecolors=C_DARK, linewidths=0.5, zorder=3)

# highlight + label the few most-served demands (single-episode scale)
top = np.argsort(prod)[::-1][:6]
ax.scatter(P[top, 0], P[top, 1], s=sizes[top] + 120, facecolors='none',
           edgecolors=C_ORANGE, linewidths=1.8, zorder=4)
for i in top:
    ax.annotate(f'D{i}', P[i], textcoords='offset points', xytext=(0, 11),
                ha='center', fontsize=8.5, color=C_DARK, fontweight='bold', zorder=5)

ax.set_axis_off()
ax.set_aspect('equal')
ax.margins(0.04)

# title / framing
prob_tag = 'Problem ' + PROB.split('_')[1]
ax.set_title(f'When dependencies couple everything: {prob_tag}, a 100-demand allocation problem',
             loc='left', fontsize=15.5, fontweight='bold', pad=22)
ax.text(0.0, 1.012, ARCH_LABEL, transform=ax.transAxes, ha='left', va='bottom',
        fontsize=10.5, color=C_GREY)

# scale callout box
txt = (f'{D} demands  ·  {req.shape[1]} resources\n'
       f'{n_dep:,} demand→resource dependencies\n'
       f'{n_links:,} competing-demand links\n'
       f'each demand needs {int(footprint.mean())} resources on average')
ax.text(0.008, 0.99, txt, transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
        color=C_DARK, linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=C_GREY, alpha=0.92))

# colourbar for production intensity
sm = ScalarMappable(norm=norm, cmap=PROD_CMAP); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.024, pad=0.01)
cb.set_label('production intensity\n(summed over preferences)', fontsize=9.5)
cb.ax.tick_params(labelsize=8)

# size / idle legend (proxy handles)
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', ls='', mfc=C_IDLE, mec='white', ms=9, label='never served'),
    Line2D([0], [0], marker='o', ls='', mfc='#2a9d8f', mec=C_DARK, ms=9, label='served (colour = amount)'),
    Line2D([0], [0], marker='o', ls='', mfc='none', mec=C_ORANGE, mew=1.8, ms=13, label='most-served demands'),
    Line2D([0], [0], marker='o', ls='', mfc=C_GREY, mec='white', ms=6, label='small ● = few deps'),
    Line2D([0], [0], marker='o', ls='', mfc=C_GREY, mec='white', ms=13, label='large ● = many deps'),
]
ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.92,
          borderpad=0.6, handletextpad=0.5)

plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'images/dependency_network_{PROB}.{ext}', bbox_inches='tight', dpi=200)
print(f'saved images/dependency_network_{PROB}.png / .pdf  '
      f'(served {int(served.sum())}/{D})')
