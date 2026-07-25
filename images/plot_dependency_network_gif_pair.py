"""Side-by-side allocation animation on the same dependency network under two
one-hot preferences. Each panel shows (top) which demands get served and (bottom)
the resulting objective vector J — the emphasised objective towers over the rest,
showing the HGNN policy adheres to the preference.

    python images/gen_dependency_network_pair.py problem_6a arch-3 0 3 4   # build cache
    python images/plot_dependency_network_gif_pair.py problem_6a           # STRIDE 2
    python images/plot_dependency_network_gif_pair.py problem_6a 1         # every step
"""
import sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as MplCircle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from net_common import layout, PROD_CMAP, C_IDLE, C_ORANGE, C_DARK, C_GREY, C_TEAL, ARCH_LABEL

PROB = sys.argv[1] if len(sys.argv) > 1 else 'problem_6a'
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CACHE = f'cache/scalability/pair_{PROB}.npz'   # build with gen_dependency_network_pair.py
OUT = f'images/allocation_pair_{PROB}.gif'
PNUM = PROB.split('_')[1]

d = np.load(CACHE)
S, req, T = d['S'], d['req'], int(d['max_steps'])
footprint = req.sum(1)
D = S.shape[0]
NOBJ = int(d['n_objectives'])
PANELS = [
    dict(prod=d['step_prod_A'], act=d['step_action_A'], ok=d['step_success_A'],
         pref=d['pref_A'], obj=int(d['obj_A']), jvec=d['step_obj_A'], tag='A'),
    dict(prod=d['step_prod_B'], act=d['step_action_B'], ok=d['step_success_B'],
         pref=d['pref_B'], obj=int(d['obj_B']), jvec=d['step_obj_B'], tag='B'),
]

G, pos, comm_of, comms = layout(S, k=4, seed=7)
P = np.array([pos[i] for i in range(D)])
smin, smax = 30, 460
fn = (footprint - footprint.min()) / (np.ptp(footprint) + 1e-9)
sizes = smin + (smax - smin) * fn
vmax = max(1, max(p['prod'][-1].max() for p in PANELS))
norm = Normalize(vmin=0, vmax=vmax)
JMAX = max(p['jvec'].max() for p in PANELS) * 1.18   # shared y-scale for objective bars

edges = list(G.edges())
segs = [(P[u], P[v]) for u, v in edges]
ewn = np.array([G[u][v]['weight'] for u, v in edges]); ewn = ewn / ewn.max()
same = np.array([comm_of.get(u) == comm_of.get(v) for u, v in edges])
nbrs = {i: list(G.neighbors(i)) for i in range(D)}
# saturation step per panel
def sat_step(prod):
    fin = prod[-1]
    for t in range(len(prod)):
        if np.array_equal(prod[t], fin):
            return t
    return len(prod) - 1
for pnl in PANELS:
    pnl['sat'] = sat_step(pnl['prod'])
# cluster centroids + sizes (sorted big->small already)
clusters = [(np.array(sorted(c)), len(c)) for c in comms]


def panel_net(ax, pnl, t):
    prod = pnl['prod'][t]; served = prod > 0
    ak, ad = int(pnl['act'][t, 0]), int(pnl['act'][t, 1])
    active = ad if (ak == 0 and pnl['ok'][t]) else -1

    for idx, n in clusters:
        ctr = P[idx].mean(0); rad = np.linalg.norm(P[idx] - ctr, axis=1).max() + 0.28
        ax.add_patch(MplCircle(ctr, rad, facecolor=C_GREY, alpha=0.06, ec='none', zorder=0))
        ax.text(ctr[0], ctr[1] - rad - 0.14, f'{n} demands', ha='center', va='top',
                fontsize=9, color=C_GREY, fontweight='bold', zorder=2)

    cols = [(0.45, 0.48, 0.5, 0.09 + 0.16 * ewn[i]) if same[i]
            else (0.6, 0.63, 0.66, 0.035 + 0.05 * ewn[i]) for i in range(len(edges))]
    lws = [0.25 + 0.8 * ewn[i] if same[i] else 0.2 + 0.3 * ewn[i] for i in range(len(edges))]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=1))

    if active >= 0 and nbrs[active]:
        hi = [(P[active], P[j]) for j in nbrs[active]]
        ax.add_collection(LineCollection(hi, colors=[C_ORANGE], linewidths=1.1, alpha=0.55, zorder=2))

    ax.scatter(P[~served, 0], P[~served, 1], s=sizes[~served], c=C_IDLE,
               edgecolors='white', linewidths=0.4, zorder=3)
    if served.any():
        ax.scatter(P[served, 0], P[served, 1], s=sizes[served], c=prod[served],
                   cmap=PROD_CMAP, norm=norm, edgecolors=C_DARK, linewidths=0.5, zorder=4)
    if active >= 0:
        ax.scatter([P[active, 0]], [P[active, 1]], s=sizes[active] + 200, facecolors='none',
                   edgecolors=C_ORANGE, linewidths=2.4, zorder=6)
        ax.annotate(f'D{active} +1', P[active], textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=10, color=C_ORANGE, fontweight='bold', zorder=7)

    ax.set_axis_off(); ax.set_aspect('equal'); ax.margins(0.04)
    y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))   # top headroom
    ax.set_title(f'Preference {pnl["tag"]}:  one-hot on Objective {pnl["obj"]}',
                 loc='left', fontsize=13, fontweight='bold', pad=8)
    hud = (f'step {t + 1:>2}/{T}\n'
           f'units: {int(prod.sum()):>2}   served: {int(served.sum()):>2}/{D}')
    ax.text(0.01, 0.99, hud, transform=ax.transAxes, ha='left', va='top', fontsize=10.5,
            family='monospace', color=C_DARK, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_GREY, alpha=0.92))
    if t >= pnl['sat'] and active < 0:
        ax.text(0.98, 0.99, 'resources exhausted', transform=ax.transAxes, ha='right',
                va='top', fontsize=10.5, color=C_ORANGE, fontweight='bold')


def panel_bars(ax, pnl, t):
    """Objective vector J at step t — emphasised objective highlighted."""
    J = pnl['jvec'][t]
    x = np.arange(NOBJ)
    colors = [C_TEAL if i == pnl['obj'] else C_IDLE for i in range(NOBJ)]
    ax.bar(x, J, color=colors, edgecolor=C_DARK, linewidth=0.5, width=0.68, zorder=3)
    for i in range(NOBJ):
        ax.text(i, J[i] + JMAX * 0.02, f'{J[i]:.0f}', ha='center', va='bottom',
                fontsize=8.5, color=(C_DARK if i == pnl['obj'] else C_GREY),
                fontweight=('bold' if i == pnl['obj'] else 'normal'))
    ax.set_xticks(x); ax.set_xticklabels([f'J{i}' for i in range(NOBJ)], fontsize=9)
    ax.set_ylim(0, JMAX); ax.set_yticks([])
    ax.set_xlim(-0.6, NOBJ - 0.4)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(f'objective vector J  —  emphasis lifts $J_{pnl["obj"]}$',
                 loc='left', fontsize=10.5, color=C_DARK, pad=4)


DPI, FIGW, FIGH = 92, 16.6, 8.7
def draw(t):
    fig = plt.figure(figsize=(FIGW, FIGH), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1.45], hspace=0.24, wspace=0.05,
                          left=0.02, right=0.9, top=0.885, bottom=0.075)
    axnet = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axbar = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for an, ab, pnl in zip(axnet, axbar, PANELS):
        panel_net(an, pnl, t)
        panel_bars(ab, pnl, t)
    fig.suptitle(f'Problem {PNUM}: allocating the same 100-demand network under two preferences',
                 fontsize=16, fontweight='bold', x=0.5, y=0.98)
    fig.text(0.5, 0.935, ARCH_LABEL, ha='center', va='top', fontsize=10.5, color=C_GREY)
    sm = ScalarMappable(norm=norm, cmap=PROD_CMAP); sm.set_array([])
    cax = fig.add_axes([0.915, 0.42, 0.011, 0.4])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('units produced (this episode)', fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.text(0.5, 0.02, 'each +1 unit consumes one unit from every resource the demand needs — '
             'the preference decides which few of 100 demands are served, and which objective is maximised',
             ha='center', fontsize=10, color=C_GREY)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


idx = list(range(0, T, STRIDE))
if idx[-1] != T - 1:
    idx.append(T - 1)
frames_idx = [0] * 3 + idx + [T - 1] * 6
n_head = 3 + len(idx)
sat_max = max(p['sat'] for p in PANELS)
def frame_ms(t, last_hold):
    if last_hold:
        return 1600
    return 220 if t >= sat_max else 560

print(f'rendering {len(frames_idx)} frames ...', flush=True)
imgs = [draw(t) for t in frames_idx]
durations = [frame_ms(frames_idx[i], i >= n_head) for i in range(len(frames_idx))]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=durations, loop=0,
             optimize=True, disposal=2)
print(f'saved {OUT}  ({len(imgs)} frames)  '
      f"A: {int(PANELS[0]['prod'][-1].sum())}u/{int((PANELS[0]['prod'][-1]>0).sum())} "
      f"B: {int(PANELS[1]['prod'][-1].sum())}u/{int((PANELS[1]['prod'][-1]>0).sum())}")
