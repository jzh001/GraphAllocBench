"""Shared helpers for the dependency-network scalability figures
(static plot + animation). Keeps layout / palette identical across both."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── house style (matches the other images/ figures) ──
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_TEAL, C_ORANGE, C_GREY, C_DARK = '#2a9d8f', '#e76f51', '#9aa0a6', '#333333'
C_IDLE = '#dfe3e6'          # unserved demand
# policy that produced the allocations (arch-3: HGNN + multi-head, preference-
# conditioned attention pooling). Human-readable label for slides.
ARCH_LABEL = 'HGNN policy · preference-conditioned attention pooling'
# sequential ramp for "production intensity": pale -> deep teal -> dark
PROD_CMAP = LinearSegmentedColormap.from_list(
    'prod', ['#bfe3dd', '#2a9d8f', '#1d6f66', '#0f3b36'])


def backbone(S, k=3):
    """Top-k strongest shared-resource links per node -> sparse, readable skeleton."""
    D = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(D))
    for i in range(D):
        order = np.argsort(S[i])[::-1]
        kept = [j for j in order if S[i, j] > 0][:k]
        for j in kept:
            w = S[i, j]
            if G.has_edge(i, j):
                G[i][j]['weight'] = max(G[i][j]['weight'], w)
            else:
                G.add_edge(i, j, weight=float(w))
    return G


def layout(S, k=4, seed=7):
    """Deterministic grouped layout: each community is drawn as its own separated
    clump (centroids on a circle; intra-community spring inside each clump). This
    makes cluster structure legible even for a near-complete projection graph.
    """
    G = backbone(S, k=k)
    try:
        comms = list(nx.community.greedy_modularity_communities(G, weight='weight'))
    except Exception:
        comms = [set(G.nodes())]
    comms = sorted(comms, key=len, reverse=True)
    comm_of = {n: ci for ci, c in enumerate(comms) for n in c}
    nc = len(comms)

    pos = {}
    RX, RY = 5.0, 3.0                         # elliptical ring of cluster centres
    a0 = np.pi / 2                            # start at top, go clockwise
    for ci, c in enumerate(comms):
        idx = sorted(c)
        ang = a0 - 2 * np.pi * ci / max(1, nc)
        cx, cy = (RX * np.cos(ang), RY * np.sin(ang)) if nc > 1 else (0.0, 0.0)
        r = 0.55 + 0.9 * np.sqrt(len(idx) / 100.0)   # clump radius scales with size
        sub = G.subgraph(idx)
        if len(idx) == 1:
            local = {idx[0]: np.array([0.0, 0.0])}
        else:
            local = nx.spring_layout(sub, weight='weight', seed=seed,
                                     k=1.4 / np.sqrt(len(idx)), iterations=250)
            pts = np.array([local[n] for n in idx])
            span = np.abs(pts).max() + 1e-9
            local = {n: local[n] / span * r for n in idx}
        for n in idx:
            pos[n] = np.array([cx, cy]) + local[n]
    return G, pos, comm_of, comms
