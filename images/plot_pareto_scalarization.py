"""Plot scalarization on supported and unsupported Pareto fronts.

This is a presentation figure for *reward maximization*.  It deliberately uses
two monotone Pareto fronts on [0, 1]^2:

* ``1 - r_1**2`` bounds a convex attainable reward set (a supported, concave
  Pareto front): positive linear weights can recover its interior points.
* ``1 - sqrt(r_1)`` is an unsupported (non-convex) front: linear scalarization
  skips its interior points, whereas reference-point scalarization can select
  them.

The Smooth Tchebycheff expression matches ``graphallocbench.city_env.scalarize``
for normalized rewards and an ideal point larger than every attainable reward.
Run from the repository root:

    uv run python images/plot_pareto_scalarization.py

The script writes a high-resolution PNG and a vector PDF beside this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).parent
PNG_PATH = OUTPUT_DIR / "pareto_scalarization_reward_maximization.png"
PDF_PATH = OUTPUT_DIR / "pareto_scalarization_reward_maximization.pdf"

# Two contrasting preferences keep the endpoint failure on the non-convex
# front unmistakable, without overlapping annotations.
WEIGHTS = np.array([[0.25, 0.75], [0.65, 0.35]])
COLORS = ("#0072B2", "#D55E00")  # colour-blind-safe palette
# A dense preference sweep visualizes each scalarizer's coverage.  The two
# WEIGHTS above remain the only ones with labelled indifference contours.
COVERAGE_WEIGHTS = np.column_stack((np.linspace(0.05, 0.95, 19), np.linspace(0.95, 0.05, 19)))
IDEAL = np.array([1.05, 1.05])
SMOOTHNESS = 0.035


def supported_front(r1: np.ndarray) -> np.ndarray:
    """A concave, fully supported Pareto front for reward maximization."""
    return 1.0 - r1**2


def unsupported_front(r1: np.ndarray) -> np.ndarray:
    """A convex-upward Pareto front whose interior is unsupported linearly."""
    return 1.0 - np.sqrt(r1)


def smooth_tchebycheff_rewards(
    rewards: np.ndarray, weights: np.ndarray, ideal: np.ndarray, smoothness: float
) -> np.ndarray:
    """Return the reward-maximization Smooth Tchebycheff value.

    This is ``-u log(sum(exp(w_i * max(z_i - r_i, 0) / u)))``, evaluated with
    the usual log-sum-exp stabilization.  Higher values are preferred.
    """
    weighted_shortfalls = weights * np.maximum(ideal - rewards, 0.0)
    scaled = weighted_shortfalls / smoothness
    maximum = scaled.max(axis=1, keepdims=True)
    return -smoothness * (maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1)))


def select_points(front: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Globally maximize each scalarized value over the densely sampled front."""
    values = []
    indices = []
    for weight in WEIGHTS:
        if method == "linear":
            score = front @ weight
        elif method == "smooth_tchebycheff":
            score = smooth_tchebycheff_rewards(front, weight, IDEAL, SMOOTHNESS)
        else:
            raise ValueError(f"Unknown scalarization method: {method}")
        values.append(score)
        indices.append(int(np.argmax(score)))
    return np.asarray(indices), np.asarray(values)


def coverage_indices(front: np.ndarray, method: str) -> np.ndarray:
    """Return all optima found across the additional preference sweep.

    Keeping all ties is important on the non-convex front: equal linear
    weights support both endpoints, rather than a single arbitrary endpoint.
    """
    indices: list[int] = []
    for weight in COVERAGE_WEIGHTS:
        if method == "linear":
            score = front @ weight
        elif method == "smooth_tchebycheff":
            score = smooth_tchebycheff_rewards(front, weight, IDEAL, SMOOTHNESS)
        else:
            raise ValueError(f"Unknown scalarization method: {method}")
        indices.extend(np.flatnonzero(score >= score.max() - 1e-12))
    return np.unique(indices)


def validate_geometry(fronts: dict[str, np.ndarray]) -> None:
    """Guard against accidentally drawing a misleading workshop figure."""
    for name, front in fronts.items():
        # Every sampled point is Pareto-optimal: increasing r1 strictly lowers r2.
        assert np.all(np.diff(front[:, 0]) > 0), f"{name}: r1 is not increasing"
        assert np.all(np.diff(front[:, 1]) < 0), f"{name}: r2 is not decreasing"

    linear_supported, linear_supported_scores = select_points(fronts["supported"], "linear")
    linear_unsupported, linear_unsupported_scores = select_points(fronts["unsupported"], "linear")
    smooth_supported, smooth_supported_scores = select_points(fronts["supported"], "smooth_tchebycheff")
    smooth_unsupported, smooth_unsupported_scores = select_points(fronts["unsupported"], "smooth_tchebycheff")

    n = len(fronts["supported"])
    interior = lambda indices: np.all((indices > n // 100) & (indices < n - n // 100))
    endpoints = lambda indices: np.all((indices < n // 100) | (indices > n - n // 100))

    # Positive linear weights recover interior supported points but not the
    # interior of the unsupported front. Smooth Tchebycheff does for both.
    assert interior(linear_supported), "Linear scalarization should recover the supported front."
    assert endpoints(linear_unsupported), "Linear scalarization should skip unsupported interior points."
    assert interior(smooth_supported), "Smooth Tchebycheff should select supported interiors."
    assert interior(smooth_unsupported), "Smooth Tchebycheff should select unsupported interiors."

    # Coverage markers make the same point visually: linear scalarization
    # reaches only the two endpoints on the non-convex front, whereas the
    # other displayed combinations produce a broad spread of optima.
    assert len(coverage_indices(fronts["unsupported"], "linear")) == 2
    assert len(coverage_indices(fronts["supported"], "linear")) >= 14
    assert len(coverage_indices(fronts["supported"], "smooth_tchebycheff")) >= 15
    assert len(coverage_indices(fronts["unsupported"], "smooth_tchebycheff")) >= 15

    # Selection must be a global maximum on the plotted candidate set.
    for scores, indices in (
        (linear_supported_scores, linear_supported),
        (linear_unsupported_scores, linear_unsupported),
        (smooth_supported_scores, smooth_supported),
        (smooth_unsupported_scores, smooth_unsupported),
    ):
        assert all(score[index] >= score.max() - 1e-12 for score, index in zip(scores, indices))


def draw_panel(
    ax: plt.Axes, front: np.ndarray, method: str, title: str
) -> None:
    indices, _ = select_points(front, method)
    sweep_indices = coverage_indices(front, method)
    ax.plot(front[:, 0], front[:, 1], color="#202020", lw=2.5, zorder=2)
    ax.fill_between(front[:, 0], front[:, 1], 0, color="#B8C4CE", alpha=0.24, zorder=0)
    ax.scatter(
        front[sweep_indices, 0],
        front[sweep_indices, 1],
        s=38,
        color="#CC79A7",
        edgecolor="white",
        linewidth=0.8,
        alpha=1.0,
        zorder=3,
    )

    for weight, color, index in zip(WEIGHTS, COLORS, indices):
        point = front[index]
        ax.scatter(*point, s=70, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(
            rf"$w=({weight[0]:.2f}, {weight[1]:.2f})$",
            xy=point,
            xytext=(7, 8),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            weight="medium",
        )

    if method == "linear":
        # A level set has normal w; showing one per preference makes the
        # supporting-hyperplane interpretation concrete for maximization.
        for weight, color, index in zip(WEIGHTS, COLORS, indices):
            point = front[index]
            slope = -weight[0] / weight[1]
            x_line = np.array([max(0, point[0] - 0.18), min(1, point[0] + 0.18)])
            y_line = point[1] + slope * (x_line - point[0])
            ax.plot(x_line, y_line, color=color, lw=1.2, ls="--", alpha=0.8, zorder=1)

    if method == "smooth_tchebycheff":
        # Draw exact indifference contours of the smooth shortfall objective.
        # Their rounded, ideal-point-oriented shape contrasts with linear lines.
        coordinate = np.linspace(0.0, 1.05, 240)
        r1_grid, r2_grid = np.meshgrid(coordinate, coordinate)
        reward_grid = np.column_stack((r1_grid.ravel(), r2_grid.ravel()))
        for weight, color, index in zip(WEIGHTS, COLORS, indices):
            level = smooth_tchebycheff_rewards(front[index][None, :], weight, IDEAL, SMOOTHNESS)[0]
            score_grid = smooth_tchebycheff_rewards(reward_grid, weight, IDEAL, SMOOTHNESS).reshape(r1_grid.shape)
            ax.contour(
                r1_grid,
                r2_grid,
                score_grid,
                levels=[level],
                colors=[color],
                linewidths=1.2,
                linestyles="--",
                alpha=0.8,
                zorder=1,
            )

    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.03, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(color="#D9DDE1", lw=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    r1 = np.linspace(0.0, 1.0, 20_001)
    fronts = {
        "supported": np.column_stack((r1, supported_front(r1))),
        "unsupported": np.column_stack((r1, unsupported_front(r1))),
    }
    validate_geometry(fronts)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    # A four-column composition uses the horizontal space of a slide while
    # retaining equal axes, so the Pareto-front geometry is not distorted.
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.73, wspace=0.27)
    draw_panel(axes[0], fronts["supported"], "linear", "Convex")
    draw_panel(axes[1], fronts["unsupported"], "linear", "Non-convex")
    draw_panel(axes[2], fronts["supported"], "smooth_tchebycheff", "Convex")
    draw_panel(axes[3], fronts["unsupported"], "smooth_tchebycheff", "Non-convex")

    axes[0].set_ylabel(r"Reward $r_2$  (maximize)", fontsize=11)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    for ax in axes:
        ax.set_xlabel(r"Reward $r_1$", fontsize=11, labelpad=8)

    legend_handles = [
        Line2D([0], [0], color="#202020", lw=2.5, label="Pareto front"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CC79A7", markeredgecolor="white", markersize=7, label="Optima across 19 extra preferences"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2", markeredgecolor="white", markersize=8, label="Selected optimum"),
        Line2D([0], [0], color="#0072B2", lw=1.2, ls="--", label="Scalarization level set"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Scalarization on Pareto Fronts for Reward Maximization", y=0.965, fontsize=17, weight="bold")
    fig.text(0.275, 0.83, "Linear scalarization", ha="center", fontsize=14, weight="bold")
    fig.text(0.735, 0.83, "Smooth Tchebycheff", ha="center", fontsize=14, weight="bold")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
