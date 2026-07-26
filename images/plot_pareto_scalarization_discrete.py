"""Discrete-aware companion to ``plot_pareto_scalarization.py``.

In GraphAllocBench every reward is a *count of units produced* (production =
min over required resources, summed), so it is an integer.  The attainable
reward set is therefore a **lattice of discrete points**, and the Pareto front
is a set of dots — not the smooth curve drawn in the continuous figure.

Discreteness is not cosmetic: it changes the conclusion.  Rounding to the
integer grid pushes some Pareto-optimal points strictly *inside* the convex
hull of the attainable set, and linear scalarization can never select a point
interior to that hull, for any positive weights.  So even on a "convex" front,
linear scalarization misses discrete Pareto points; on the non-convex front it
collapses to the two endpoints.  Reference-point (Smooth Tchebycheff)
scalarization recovers every discrete Pareto point in both cases.

Reachability is computed empirically by sweeping positive weight vectors and
collecting every arg-max — no hand-waving about which points are "supported".

Run from the repository root:

    uv run python images/plot_pareto_scalarization_discrete.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).parent
PNG_PATH = OUTPUT_DIR / "pareto_scalarization_discrete.png"
PDF_PATH = OUTPUT_DIR / "pareto_scalarization_discrete.pdf"

GRID = 10                      # integer axis range 0..GRID (units produced)
IDEAL_PAD = 1.0                # reference point sits just past the best reward
SMOOTHNESS = 0.35             # Smooth Tchebycheff temperature (integer scale)
SWEEP = np.linspace(0.02, np.pi / 2 - 0.02, 400)   # positive-weight directions

C_LIN = "#0072B2"   # linear-reachable (supported)
C_TCH = "#D55E00"   # Tchebycheff-reachable
C_MISS = "#9aa0a6"  # linearly unreachable
C_HULL = "#7a828a"  # convex hull of the attainable set
C_CURVE = "#c7ccd1"  # continuous relaxation (reference only)
C_DARK = "#202020"


# ── discrete fronts ────────────────────────────────────────────────────────
def continuous(kind: str, x: np.ndarray) -> np.ndarray:
    if kind == "convex":            # concave, fully supported in the continuum
        return GRID * (1.0 - (x / GRID) ** 2)
    return GRID * (1.0 - np.sqrt(x / GRID))   # non-convex


def discrete_front(kind: str) -> np.ndarray:
    xs = np.arange(0, GRID + 1)
    ys = np.round(continuous(kind, xs)).astype(int)
    P = np.column_stack((xs, ys)).astype(float)
    return P[nondominated(P)]


def nondominated(P: np.ndarray) -> np.ndarray:
    keep = [i for i, p in enumerate(P)
            if not np.any(np.all(P >= p, axis=1) & np.any(P > p, axis=1))]
    return np.array(sorted(keep))


# ── scalarizers & empirical reachability ────────────────────────────────────
def smooth_tchebycheff(rewards: np.ndarray, weights: np.ndarray,
                       ideal: np.ndarray, u: float) -> np.ndarray:
    """Reward-maximization Smooth Tchebycheff (higher is better)."""
    shortfall = weights * np.maximum(ideal - rewards, 0.0)
    scaled = shortfall / u
    m = scaled.max(axis=1, keepdims=True)
    return -u * (m[:, 0] + np.log(np.exp(scaled - m).sum(axis=1)))


def reachable(P: np.ndarray, method: str, ideal: np.ndarray) -> np.ndarray:
    hit: set[int] = set()
    for theta in SWEEP:
        w = np.array([np.cos(theta), np.sin(theta)])
        w = w / w.sum()
        if method == "linear":
            score = P @ w
        else:
            score = smooth_tchebycheff(P, w, ideal, SMOOTHNESS)
        hit.update(np.flatnonzero(score >= score.max() - 1e-9))
    return np.array(sorted(hit))


# ── drawing ─────────────────────────────────────────────────────────────────
def draw_panel(ax: plt.Axes, kind: str, method: str) -> tuple[int, int]:
    P = discrete_front(kind)
    ideal = P.max(0) + IDEAL_PAD
    hull = reachable(P, "linear", ideal)          # linear-reachable == hull vertices
    reach = reachable(P, method, ideal)
    reach_set = set(reach.tolist())
    accent = C_LIN if method == "linear" else C_TCH

    # integer lattice
    ax.set_xticks(range(0, GRID + 1)); ax.set_yticks(range(0, GRID + 1))
    ax.grid(color="#E4E7EA", lw=0.7, zorder=0)

    # continuous relaxation, for reference only
    xf = np.linspace(0, GRID, 300)
    ax.plot(xf, continuous(kind, xf), color=C_CURVE, lw=1.4, ls=(0, (1, 1.6)),
            zorder=1)

    # convex hull of the attainable set (concave majorant through hull vertices)
    Ph = P[hull]
    ax.plot(Ph[:, 0], Ph[:, 1], color=C_HULL, lw=1.3, ls="--", alpha=0.9, zorder=2)

    # level sets illustrating each scalarizer's geometry
    if method == "linear":
        for i in (hull[1], hull[-2]) if len(hull) >= 3 else hull[:1]:
            p = P[i]; w = np.array([1.0, 1.0])  # slope -1 supporting line, drawn short
            xl = np.array([max(0, p[0] - 2.4), min(GRID, p[0] + 2.4)])
            ax.plot(xl, p[1] + -1.0 * (xl - p[0]), color=accent, lw=1.1,
                    ls="--", alpha=0.75, zorder=2)
    else:
        # L-shaped reference-point contours anchored at interior points linear misses
        interior = [i for i in reach.tolist() if i not in set(hull.tolist())]
        picks = interior[len(interior) // 2: len(interior) // 2 + 2] or reach.tolist()[:2]
        for i in picks:
            p = P[i]
            ax.plot([p[0], p[0], ideal[0]], [ideal[1], p[1], p[1]],
                    color=accent, lw=1.1, ls="--", alpha=0.7, zorder=2)

    # points: reachable (filled accent) vs linearly unreachable (hollow grey)
    for i in range(len(P)):
        x, y = P[i]
        if i in reach_set:
            ax.scatter(x, y, s=95, color=accent, edgecolor="white",
                       linewidth=1.2, zorder=5)
        else:
            ax.scatter(x, y, s=80, facecolor="white", edgecolor=C_MISS,
                       linewidth=1.6, zorder=4)
            ax.plot(x, y, marker="x", ms=6, color=C_MISS, mew=1.6, zorder=5)

    ax.scatter(*ideal, marker="*", s=150, color=C_DARK, edgecolor="white",
               linewidth=0.6, zorder=6)
    ax.annotate("ideal", ideal, textcoords="offset points", xytext=(4, 4),
                fontsize=8, color=C_DARK)

    ax.set_xlim(-0.6, GRID + IDEAL_PAD + 0.6); ax.set_ylim(-0.6, GRID + IDEAL_PAD + 0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"reaches {len(reach)} / {len(P)} Pareto points",
                 fontsize=10.5, color=accent, weight="bold", pad=8)
    return len(reach), len(P)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.2, top=0.72, wspace=0.14)

    draw_panel(axes[0], "convex", "linear")
    draw_panel(axes[1], "non-convex", "linear")
    draw_panel(axes[2], "convex", "smooth_tchebycheff")
    draw_panel(axes[3], "non-convex", "smooth_tchebycheff")

    for ax, sub in zip(axes, ("Convex", "Non-convex", "Convex", "Non-convex")):
        ax.set_xlabel(r"Reward $r_1$  (units produced)", fontsize=10, labelpad=6)
        ax.text(0.5, 1.11, sub, transform=ax.transAxes, ha="center",
                fontsize=11, weight="bold")
    axes[0].set_ylabel(r"Reward $r_2$  (units produced)", fontsize=10.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_LIN,
               markeredgecolor="white", markersize=9, label="Reachable optimum"),
        Line2D([0], [0], marker="x", color=C_MISS, lw=0, markersize=8,
               markeredgewidth=1.6, label="Linearly unreachable Pareto point"),
        Line2D([0], [0], color=C_HULL, lw=1.3, ls="--",
               label="Convex hull of attainable set"),
        Line2D([0], [0], color=C_CURVE, lw=1.4, ls=(0, (1, 1.6)),
               label="Continuous relaxation (reference)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_DARK,
               markeredgecolor="white", markersize=12, label="Reference / ideal point"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.suptitle("Discrete Reward Landscapes: Linear Scalarization Misses "
                 "Integer Pareto Points that Reference-Point Scalarization Recovers",
                 y=0.965, fontsize=15, weight="bold")
    fig.text(0.275, 0.82, "Linear scalarization", ha="center", fontsize=13,
             weight="bold", color=C_LIN)
    fig.text(0.735, 0.82, "Smooth Tchebycheff", ha="center", fontsize=13,
             weight="bold", color=C_TCH)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
