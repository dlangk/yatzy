"""Probability density plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..config import MAX_SCORE
from .spec import PLOT_SPECS, PlotSpec
from .style import CMAP, FONT_AXIS_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE, apply_theta_legend, fmt_theta, make_norm, setup_theme, theta_color


def plot_density(
    thetas: list[float],
    kde_df: pl.DataFrame,
    out_dir: Path,
    *,
    norm: mcolors.Normalize | None = None,
    ax=None,
    spec: PlotSpec | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    if norm is None:
        norm = make_norm(thetas)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 6))

    for t in thetas:
        subset = kde_df.filter(pl.col("theta") == t).sort("score").to_pandas()
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.4
        alpha = 0.9 if t == 0 else 0.7
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=lw, alpha=alpha,
        )

    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score Distribution by Risk Parameter θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(50, MAX_SCORE)
    apply_theta_legend(ax, norm, spec or PLOT_SPECS["density"])

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"density.{fmt}", dpi=dpi)
        plt.close(fig)


_ADAPTIVE_COLORS = {
    "bonus-adaptive": "tab:green",
    "phase-based": "tab:orange",
    "combined": "tab:purple",
}
_ADAPTIVE_DASHES = {
    "bonus-adaptive": (4, 2),
    "phase-based": (6, 2, 2, 2),
    "combined": (2, 2),
}


def plot_density_with_adaptive(
    thetas: list[float],
    kde_df: pl.DataFrame,
    adaptive_kde_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Density plot showing both fixed-θ sweeps and adaptive strategy densities."""
    setup_theme()
    norm = make_norm(thetas)
    fig, ax = plt.subplots(figsize=(16, 7))

    # Fixed-θ densities (thin, semi-transparent, no label)
    for t in thetas:
        subset = kde_df.filter(pl.col("theta") == t).sort("score").to_pandas()
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.0
        alpha = 0.9 if t == 0 else 0.45
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=lw, alpha=alpha,
        )

    # Adaptive strategy densities (thick, dashed, prominent)
    for policy in sorted(adaptive_kde_df["policy"].unique().to_list()):
        subset = adaptive_kde_df.filter(pl.col("policy") == policy).sort("score").to_pandas()
        color = _ADAPTIVE_COLORS.get(policy, "tab:brown")
        dashes = _ADAPTIVE_DASHES.get(policy, (4, 2))
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=2.8, alpha=0.95,
            dashes=dashes, label=policy,
        )

    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Score Distribution: Fixed-θ Sweep + Adaptive Strategies",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.set_xlim(50, MAX_SCORE)
    # Colorbar for fixed-θ lines
    apply_theta_legend(ax, norm, PLOT_SPECS["density_adaptive"])
    # Small legend for adaptive policies only
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / f"density_adaptive.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_density_3d(
    thetas: list[float],
    kde_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """3D surface plot of score density as a function of θ and score.

    X = score, Y = θ, Z = density.  Surface colored by θ (coolwarm).
    Viewing angle chosen so the θ axis recedes into the plot, giving
    a waterfall-like perspective where each θ slice is clearly visible.
    """
    setup_theme()

    norm = make_norm(thetas)

    # Build grid: rows = thetas (sorted), cols = score values
    sorted_thetas = sorted(thetas)
    # Use score range from data, interpolated to a common grid
    score_min, score_max = 50, MAX_SCORE
    score_grid = np.linspace(score_min, score_max, 400)

    Z = np.zeros((len(sorted_thetas), len(score_grid)))
    for i, t in enumerate(sorted_thetas):
        subset = kde_df.filter(pl.col("theta") == t).sort("score").to_pandas()
        if len(subset) > 2:
            Z[i, :] = np.interp(score_grid, subset["score"].values, subset["density"].values)

    X, Y = np.meshgrid(score_grid, sorted_thetas)

    # Color surface by θ value
    theta_colors = CMAP(norm(Y))

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        facecolors=theta_colors,
        rstride=1, cstride=4,
        shade=True,
        alpha=0.92,
        linewidth=0,
        antialiased=True,
    )

    # Add a few highlighted wireframe slices for key thetas
    key_thetas = [t for t in [-1.0, -0.1, 0.0, 0.1, 1.0] if t in sorted_thetas]
    for t in key_thetas:
        idx = sorted_thetas.index(t)
        color = CMAP(norm(t))
        lw = 2.0 if t == 0 else 1.2
        ax.plot(score_grid, [t] * len(score_grid), Z[idx, :],
                color=color, linewidth=lw, zorder=5)

    ax.set_xlabel("Score", fontsize=FONT_AXIS_LABEL, labelpad=10)
    ax.set_ylabel("θ", fontsize=FONT_AXIS_LABEL, labelpad=10)
    ax.set_zlabel("Density", fontsize=FONT_AXIS_LABEL, labelpad=8)
    ax.set_title("Score Density Surface across θ", fontsize=FONT_TITLE, fontweight="bold")

    # Viewing angle: elevated, looking along θ toward the peak
    ax.view_init(elev=32, azim=-55)

    # Reduce tick clutter
    ax.set_xlim(score_min, score_max)
    ax.tick_params(axis="both", labelsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, aspect=25)
    cbar.set_label("θ  (blue = risk-averse, red = risk-seeking)", fontsize=FONT_TICK)

    out_path = out_dir / f"density_3d.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_density_3d_gif(
    thetas: list[float],
    kde_df: pl.DataFrame,
    out_dir: Path,
    *,
    elev: int = 45,
    n_frames: int = 180,
    fps: int = 24,
    dpi: int = 120,
) -> Path:
    """Rotating 360-degree GIF of the 3D density surface.

    elev=45 gives a slightly-above perspective that shows the surface
    shape while keeping all θ slices visible during the full rotation.
    """
    from matplotlib.animation import FuncAnimation

    setup_theme()
    norm = make_norm(thetas)
    sorted_thetas = sorted(thetas)
    score_min, score_max = 50, MAX_SCORE
    score_grid = np.linspace(score_min, score_max, 400)

    Z = np.zeros((len(sorted_thetas), len(score_grid)))
    for i, t in enumerate(sorted_thetas):
        subset = kde_df.filter(pl.col("theta") == t).sort("score").to_pandas()
        if len(subset) > 2:
            Z[i, :] = np.interp(score_grid, subset["score"].values, subset["density"].values)

    X, Y = np.meshgrid(score_grid, sorted_thetas)
    theta_colors = CMAP(norm(Y))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        facecolors=theta_colors,
        rstride=1, cstride=4,
        shade=True, alpha=1.0,
        linewidth=0, antialiased=True,
    )

    key_thetas = [t for t in [-1.0, -0.1, 0.0, 0.1, 1.0] if t in sorted_thetas]
    for t in key_thetas:
        idx = sorted_thetas.index(t)
        color = CMAP(norm(t))
        lw = 2.5 if t == 0 else 1.5
        ax.plot(score_grid, [t] * len(score_grid), Z[idx, :],
                color=color, linewidth=lw, zorder=5)

    ax.set_xlabel("Score", fontsize=FONT_AXIS_LABEL, labelpad=8)
    ax.set_ylabel("θ", fontsize=FONT_AXIS_LABEL, labelpad=8)
    ax.set_zlabel("Density", fontsize=FONT_AXIS_LABEL, labelpad=6)
    ax.set_xlim(score_min, score_max)
    ax.tick_params(axis="both", labelsize=8)
    ax.view_init(elev=elev, azim=0)

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)

    def _update(frame: int):
        ax.view_init(elev=elev, azim=frame * (360 / n_frames))
        return []

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    out_path = out_dir / "density_3d_rotate.gif"
    anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_density_ridge(
    thetas: list[float],
    kde_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Ridge (joy division) plot: stacked density curves, one per θ.

    Each row is one θ value.  Curves are filled with a white-to-color
    gradient and overlap slightly so the shape evolution is clear without
    any occlusion ambiguity.  θ sorted bottom-to-top (risk-averse → risk-seeking).
    """
    setup_theme()

    norm = make_norm(thetas)
    sorted_thetas = sorted(thetas)

    # Spacing: each row gets 1 unit; overlap controls how much curves
    # can poke into the row above (relative to max density).
    overlap = 0.7
    max_density = 0.0
    curves: list[tuple[float, np.ndarray, np.ndarray]] = []

    for t in sorted_thetas:
        subset = kde_df.filter(pl.col("theta") == t).sort("score").to_pandas()
        if len(subset) < 2:
            continue
        x = subset["score"].values
        y = subset["density"].values
        max_density = max(max_density, y.max())
        curves.append((t, x, y))

    if not curves or max_density == 0:
        raise ValueError("No density data to plot")

    # Scale factor: how tall one curve can be in row-spacing units
    scale = (1.0 + overlap) / max_density

    n_rows = len(curves)
    fig_height = max(8, n_rows * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    yticks = []
    ytick_labels = []

    for i, (t, x, y) in enumerate(curves):
        baseline = i  # vertical offset for this row
        scaled_y = y * scale + baseline
        color = CMAP(norm(t))

        # Fill from baseline to curve
        ax.fill_between(x, baseline, scaled_y, color=color, alpha=0.35, linewidth=0)
        # White strip at bottom edge to separate from row below
        ax.fill_between(x, baseline, baseline + 0.02, color="white", linewidth=0)
        # Outline
        lw = 1.8 if t == 0.0 else 0.9
        ax.plot(x, scaled_y, color=color, linewidth=lw)

        yticks.append(baseline)
        ytick_labels.append(fmt_theta(t))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_ylabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_xlabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score Density by θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(50, MAX_SCORE)
    ax.set_ylim(-0.3, n_rows + overlap)

    # Remove horizontal gridlines (they clash with the ridges)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.15)

    # Spine cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / f"density_ridge.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
