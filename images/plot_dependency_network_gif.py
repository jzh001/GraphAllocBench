"""Animated allocation process on the GraphAllocBench dependency network.

Same clustered layout as the static figure. Over a 30-step episode the trained
policy adds production one demand at a time; each unit consumes a resource from
*every* demand it depends on, so only a handful of demands can ever be served.
Node colour grows with cumulative production; the demand acted on each step
pulses and its competing links light up (the resources it just drew down).

    python images/plot_dependency_network_gif.py            # problem_6a
    python images/plot_dependency_network_gif.py problem_6b
"""
import sys, numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle as MplCircle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from net_common import layout, PROD_CMAP, C_IDLE, C_ORANGE, C_DARK, C_GREY, ARCH_LABEL

PROB = sys.argv[1] if len(sys.argv) > 1 else 'problem_6a'
CACHE = f'cache/scalability/alloc_{PROB}.npz'   # build with images/gen_dependency_network_data.py
OUT = f'images/allocation_process_{PROB}.gif'

d = np.load(CACHE)
S, req = d['S'], d['req']
step_prod = d['step_prod']              # (T, D) cumulative
step_action = d['step_action']          # (T, 2)  [addOrRemove, demand]
step_success = d['step_success']        # (T,)
D = S.shape[0]
footprint = req.sum(1)
T = len(step_prod)

G, pos, comm_of, comms = layout(S, k=4, seed=7)
P = np.array([pos[i] for i in range(D)])

smin, smax = 45, 620
fn = (footprint - footprint.min()) / (np.ptp(footprint) + 1e-9)
sizes = smin + (smax - smin) * fn
vmax = max(1, step_prod.max())
norm = Normalize(vmin=0, vmax=vmax)

# static edge geometry
edges = list(G.edges())
segs = np.array([(P[u], P[v]) for u, v in edges])
ews = np.array([G[u][v]['weight'] for u, v in edges]); ewn = ews / ews.max()
same = np.array([comm_of.get(u) == comm_of.get(v) for u, v in edges])
# neighbour lookup for the active-demand highlight
nbrs = {i: list(G.neighbors(i)) for i in range(D)}

# run the full episode to completion. STRIDE=1 shows every step; if that makes
# the GIF too heavy, set STRIDE=2 (increments of 2).
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
# first step from which production no longer changes (resources exhausted)
final = step_prod[-1]
sat = T - 1
for t in range(T):
    if np.array_equal(step_prod[t], final):
        sat = t
        break
idx = list(range(0, T, STRIDE))
if idx[-1] != T - 1:
    idx.append(T - 1)
# frame plan: brief hold on first, every (STRIDE-th) step to the end, hold on last
frames_idx = [0] * 3 + idx + [T - 1] * 6
DPI, FIGW, FIGH = 92, 12.8, 7.4
# per-frame timing: linger on the active allocation phase, zip through the
# saturated tail, rest on the final state
def frame_ms(t, is_last_hold):
    if is_last_hold:
        return 1600
    return 200 if t >= sat else 520


def draw(t):
    prod = step_prod[t]
    served = prod > 0
    act_kind, act_d = int(step_action[t, 0]), int(step_action[t, 1])
    active = act_d if (act_kind == 0 and step_success[t]) else -1

    fig, ax = plt.subplots(figsize=(FIGW, FIGH), dpi=DPI)
    # cluster tints
    for c in comms:
        idx = np.array(sorted(c)); ctr = P[idx].mean(0)
        rad = np.linalg.norm(P[idx] - ctr, axis=1).max() + 0.28
        ax.add_patch(MplCircle(ctr, rad, facecolor=C_GREY, alpha=0.06, ec='none', zorder=0))

    # base edges (dim)
    cols = [(0.45, 0.48, 0.5, 0.10 + 0.18 * ewn[i]) if same[i]
            else (0.6, 0.63, 0.66, 0.04 + 0.06 * ewn[i]) for i in range(len(edges))]
    lws = [0.3 + 0.9 * ewn[i] if same[i] else 0.25 + 0.35 * ewn[i] for i in range(len(edges))]
    ax.add_collection(LineCollection(list(segs), colors=cols, linewidths=lws, zorder=1))

    # active demand's competing links light up (resources it just drew on)
    if active >= 0 and nbrs[active]:
        hi = np.array([(P[active], P[j]) for j in nbrs[active]])
        ax.add_collection(LineCollection(list(hi), colors=[C_ORANGE], linewidths=1.3,
                                         alpha=0.6, zorder=2))

    # nodes
    ax.scatter(P[~served, 0], P[~served, 1], s=sizes[~served], c=C_IDLE,
               edgecolors='white', linewidths=0.5, zorder=3)
    if served.any():
        ax.scatter(P[served, 0], P[served, 1], s=sizes[served], c=prod[served],
                   cmap=PROD_CMAP, norm=norm, edgecolors=C_DARK, linewidths=0.5, zorder=4)
    if active >= 0:
        ax.scatter([P[active, 0]], [P[active, 1]], s=sizes[active] + 260, facecolors='none',
                   edgecolors=C_ORANGE, linewidths=2.6, zorder=6)
        ax.annotate(f'D{active} +1', P[active], textcoords='offset points', xytext=(0, 14),
                    ha='center', fontsize=11, color=C_ORANGE, fontweight='bold', zorder=7)

    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.margins(0.04)

    ax.set_title(f'Allocating on a 100-demand dependency network  ·  Problem {PROB.split("_")[1]}',
                 loc='left', fontsize=15, fontweight='bold', pad=20)
    ax.text(0.0, 1.012, ARCH_LABEL, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, color=C_GREY)
    # HUD
    hud = (f'step {t + 1:>2}/{T}\n'
           f'units produced: {int(prod.sum()):>3}\n'
           f'demands served: {int(served.sum()):>2}/{D}')
    ax.text(0.008, 0.99, hud, transform=ax.transAxes, ha='left', va='top', fontsize=12,
            family='monospace', color=C_DARK, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=C_GREY, alpha=0.92))
    ax.text(0.008, 0.02,
            'each +1 unit consumes one unit from every resource the demand needs',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=10, color=C_GREY)
    if t >= sat and active < 0:
        ax.text(0.5, 0.985, 'resources exhausted — no further allocation possible',
                transform=ax.transAxes, ha='center', va='top', fontsize=11.5,
                color=C_ORANGE, fontweight='bold')

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


print(f'rendering {len(frames_idx)} frames ...', flush=True)
imgs = [draw(t) for t in frames_idx]
n_head = 3 + len(idx)                       # frames before the trailing hold
durations = [frame_ms(frames_idx[i], i >= n_head) for i in range(len(frames_idx))]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=durations, loop=0,
             optimize=True, disposal=2)
print(f'saved {OUT}  ({len(imgs)} frames, {step_prod[-1].sum()} units, '
      f'{int((step_prod[-1] > 0).sum())} demands served)')
