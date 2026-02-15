"""Percentile frontiers and risk tradeoff plots, parameterized by θ."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .style import FONT_AXIS_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE, GRID_ALPHA, setup_theme

# Statistics ordered from lowest (blue) to highest (red).
FRONTIER_STATS = [
    ("min", "min"),
    ("bot5_avg", "bot5"),
    ("p1", "p1"),
    ("p5", "p5"),
    ("p10", "p10"),
    ("p25", "p25"),
    ("p50", "p50"),
    ("mean", "mean"),
    ("p75", "p75"),
    ("p90", "p90"),
    ("p95", "p95"),
    ("p99", "p99"),
    ("p995", "p99.5"),
    ("p999", "p99.9"),
    ("p9999", "p99.99"),
    ("top5_avg", "top5"),
    ("max", "max"),
]


def plot_percentile_frontiers(
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Plot score vs std frontiers for all percentiles, colored blue->red.

    Each curve traces one statistic (e.g. p95) through (std, score) space
    as θ varies. Color: coolwarm from min (blue) to max (red). Opacity
    fades for points far from the peak (maximum score on that curve).
    """
    setup_theme()

    df = stats_df.sort_values("theta").copy()
    df = df[df["n"] >= 100_000].copy()

    cmap = plt.cm.coolwarm
    n_stats = len(FRONTIER_STATS)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Track peak y-values for right-margin label placement
    peak_info: list[tuple[float, str, tuple]] = []  # (y_peak, label, rgb)

    for i, (col, label) in enumerate(FRONTIER_STATS):
        if col not in df.columns:
            continue

        color_val = i / (n_stats - 1)
        base_rgb = cmap(color_val)[:3]

        x = df["std"].values.copy()
        y = df[col].values.astype(float).copy()

        # Peak: the point where this statistic is maximized
        peak_idx = int(np.argmax(y))
        peak_val = y[peak_idx]
        val_range = peak_val - y.min()

        # Alpha: 1.0 at peak, fading toward the minimum
        if val_range > 0:
            alpha_raw = (y - y.min()) / val_range
        else:
            alpha_raw = np.ones_like(y)
        alpha_vals = np.clip(alpha_raw * 0.85 + 0.15, 0.15, 1.0)

        # Build line segments with per-segment alpha
        points = np.column_stack([x, y])
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        seg_alpha = (alpha_vals[:-1] + alpha_vals[1:]) / 2.0
        colors = np.array([(*base_rgb, a) for a in seg_alpha])

        lc = LineCollection(segments, colors=colors, linewidths=2.0, zorder=2)
        ax.add_collection(lc)

        # Dot at peak
        x_peak, y_peak = x[peak_idx], y[peak_idx]
        ax.plot(x_peak, y_peak, "o", color=base_rgb, markersize=3.5, zorder=4, alpha=0.9)

        peak_info.append((y_peak, label, base_rgb))

    # Place labels on the right margin, aligned to peak y-values
    x_right = 50.5  # just past the data range
    _place_right_labels(ax, peak_info, x_right)

    ax.set_xlim(33, 51)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Standard Deviation", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Percentile Frontiers (each curve parameterized by θ)", fontsize=FONT_TITLE)

    # Light annotation explaining the loops
    ax.text(
        0.02, 0.97,
        "Curves loop because std is non-monotonic in θ:\n"
        "risk-averse (left) → EV-optimal (center) → risk-seeking (right)",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        color="0.4", style="italic",
    )

    fig.tight_layout()
    out_path = out_dir / f"percentile_frontiers_full.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _place_right_labels(
    ax: plt.Axes,
    items: list[tuple[float, str, tuple]],
    x: float,
) -> None:
    """Place labels at fixed x, nudging y to avoid overlap."""
    if not items:
        return

    sorted_items = sorted(items, key=lambda p: p[0])
    min_gap = 10.0

    placed: list[tuple[float, str, tuple]] = []
    for y, label, rgb in sorted_items:
        ny = y
        for py, _, _ in placed:
            if ny - py < min_gap:
                ny = py + min_gap
        placed.append((ny, label, rgb))

    for y, label, rgb in placed:
        ax.annotate(
            label,
            (x, y),
            fontsize=7.5,
            fontweight="bold",
            color=(*rgb, 0.95),
            ha="left",
            va="center",
            annotation_clip=False,
            zorder=5,
        )


# ---------------------------------------------------------------------------
# Shared helper: θ-colored line with labeled scatter points
# ---------------------------------------------------------------------------

def _theta_colored_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    thetas: np.ndarray,
    norm: mcolors.Normalize,
    *,
    linewidth: float = 2.5,
    marker_size: float = 40,
    label_thetas: list[float] | None = None,
) -> None:
    """Draw a θ-parameterized curve with per-segment coolwarm coloring.

    Also scatters marker dots colored by θ.  If *label_thetas* is given,
    annotate those specific θ values on the curve.
    """
    cmap = plt.cm.coolwarm

    # Line segments colored by midpoint θ
    points = np.column_stack([x, y])
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    seg_theta = (thetas[:-1] + thetas[1:]) / 2.0
    seg_colors = cmap(norm(seg_theta))

    lc = LineCollection(segments, colors=seg_colors, linewidths=linewidth, zorder=2)
    ax.add_collection(lc)

    # Scatter dots
    ax.scatter(x, y, c=thetas, cmap=cmap, norm=norm, s=marker_size,
               edgecolors="white", linewidths=0.4, zorder=3)

    # Label selected thetas
    if label_thetas is None:
        label_thetas = [
            t for t in [-3, -1, -0.5, -0.1, -0.03, 0, 0.03, 0.1, 0.5, 1, 3]
            if t >= thetas.min() - 0.01 and t <= thetas.max() + 0.01
        ]

    for lt in label_thetas:
        idx = int(np.argmin(np.abs(thetas - lt)))
        lbl = f"{thetas[idx]:g}"
        ax.annotate(
            lbl,
            (x[idx], y[idx]),
            fontsize=7,
            fontweight="bold",
            color=cmap(norm(thetas[idx])),
            xytext=(6, 4),
            textcoords="offset points",
            zorder=5,
        )


def _add_theta_colorbar(ax: plt.Axes, norm: mcolors.Normalize) -> None:
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("θ  (blue = risk-averse, red = risk-seeking)", fontsize=FONT_TICK)


# ---------------------------------------------------------------------------
# 1. Mean vs CVaR_5  —  the classic risk-return frontier
# ---------------------------------------------------------------------------

def plot_mean_vs_cvar(
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Mean vs CVaR_5 frontier, colored by θ.

    X = mean score, Y = CVaR_5 (average of worst 5% games).
    Moving along the curve shows the cost of downside protection.
    """
    setup_theme()

    df = stats_df.sort_values("theta").copy()
    df = df[df["n"] >= 100_000].copy()

    x = df["mean"].values
    y = df["cvar_5"].values
    thetas = df["theta"].values
    norm = mcolors.SymLogNorm(linthresh=0.05, linscale=1.0,
                              vmin=-max(abs(thetas)), vmax=max(abs(thetas)))

    fig, ax = plt.subplots(figsize=(10, 8))
    _theta_colored_curve(ax, x, y, thetas, norm)

    # Mark θ=0 with a star
    idx0 = int(np.argmin(np.abs(thetas)))
    ax.plot(x[idx0], y[idx0], "*", color="black", markersize=14, zorder=6)
    ax.annotate("θ=0\n(EV-optimal)", (x[idx0], y[idx0]),
                xytext=(-12, -18), textcoords="offset points",
                fontsize=9, ha="center", fontweight="bold", zorder=6)

    # Reference line: y = x (CVaR = mean means no tail risk)
    lim_lo = min(x.min(), y.min()) - 5
    lim_hi = max(x.max(), y.max()) + 5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="0.7",
            linewidth=1, zorder=1, label="CVaR5 = mean")

    ax.set_xlabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("CVaR5 (average of worst 5%)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Risk-Return Frontier: Mean vs Downside Protection", fontsize=FONT_TITLE)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND)
    _add_theta_colorbar(ax, norm)

    fig.tight_layout()
    out_path = out_dir / f"frontier_mean_vs_cvar.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. P5 vs P95  —  pure tail-vs-tail tradeoff
# ---------------------------------------------------------------------------

def plot_p5_vs_p95(
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """P5 vs P95 frontier, colored by θ.

    X = 5th percentile (floor), Y = 95th percentile (ceiling).
    Shows the pure downside-vs-upside tradeoff with no middle involved.
    """
    setup_theme()

    df = stats_df.sort_values("theta").copy()
    df = df[df["n"] >= 100_000].copy()

    x = df["p5"].values.astype(float)
    y = df["p95"].values.astype(float)
    thetas = df["theta"].values
    norm = mcolors.SymLogNorm(linthresh=0.05, linscale=1.0,
                              vmin=-max(abs(thetas)), vmax=max(abs(thetas)))

    fig, ax = plt.subplots(figsize=(10, 8))
    _theta_colored_curve(ax, x, y, thetas, norm)

    # Mark θ=0
    idx0 = int(np.argmin(np.abs(thetas)))
    ax.plot(x[idx0], y[idx0], "*", color="black", markersize=14, zorder=6)
    ax.annotate("θ=0", (x[idx0], y[idx0]),
                xytext=(-10, -14), textcoords="offset points",
                fontsize=9, ha="center", fontweight="bold", zorder=6)

    # Ideal direction arrow (upper-right = both improve)
    ax.annotate(
        "", xy=(0.95, 0.95), xytext=(0.85, 0.85),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="0.6", lw=1.5),
    )
    ax.text(0.96, 0.92, "ideal\n(both improve)", transform=ax.transAxes,
            fontsize=8, color="0.5", ha="right", va="top", style="italic")

    ax.set_xlabel("P5 (5th percentile — floor)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("P95 (95th percentile — ceiling)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Tail Tradeoff: Downside Floor vs Upside Ceiling", fontsize=FONT_TITLE)
    _add_theta_colorbar(ax, norm)

    fig.tight_layout()
    out_path = out_dir / f"frontier_p5_vs_p95.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Skewness vs Kurtosis  —  distribution shape journey
# ---------------------------------------------------------------------------

def plot_skewness_vs_kurtosis(
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Skewness vs excess kurtosis trajectory, colored by θ.

    Traces how the distribution shape morphs as θ changes. Normal
    distribution reference at (0, 0).
    """
    setup_theme()

    df = stats_df.sort_values("theta").copy()
    df = df[df["n"] >= 100_000].copy()

    if "skewness" not in df.columns or "kurtosis" not in df.columns:
        raise ValueError("skewness/kurtosis columns missing — run `compute --csv` first")

    x = df["skewness"].values
    y = df["kurtosis"].values
    thetas = df["theta"].values
    norm = mcolors.SymLogNorm(linthresh=0.05, linscale=1.0,
                              vmin=-max(abs(thetas)), vmax=max(abs(thetas)))

    fig, ax = plt.subplots(figsize=(10, 8))
    _theta_colored_curve(ax, x, y, thetas, norm, marker_size=50)

    # Mark θ=0
    idx0 = int(np.argmin(np.abs(thetas)))
    ax.plot(x[idx0], y[idx0], "*", color="black", markersize=14, zorder=6)
    ax.annotate("θ=0", (x[idx0], y[idx0]),
                xytext=(8, -12), textcoords="offset points",
                fontsize=9, ha="left", fontweight="bold", zorder=6)

    # Normal reference point
    ax.plot(0, 0, "D", color="0.4", markersize=8, zorder=5, alpha=0.6)
    ax.annotate("Normal\n(0, 0)", (0, 0), xytext=(8, 8),
                textcoords="offset points", fontsize=8, color="0.4",
                style="italic", zorder=5)

    # Axis lines through origin
    ax.axhline(0, color="0.8", linewidth=0.8, zorder=1)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=1)

    # Quadrant labels
    pad = 0.02
    ax.text(pad, 1 - pad, "left-skewed\nheavy-tailed",
            transform=ax.transAxes, fontsize=8, color="0.6",
            ha="left", va="top", style="italic")
    ax.text(1 - pad, 1 - pad, "right-skewed\nheavy-tailed",
            transform=ax.transAxes, fontsize=8, color="0.6",
            ha="right", va="top", style="italic")
    ax.text(pad, pad, "left-skewed\nlight-tailed",
            transform=ax.transAxes, fontsize=8, color="0.6",
            ha="left", va="bottom", style="italic")
    ax.text(1 - pad, pad, "right-skewed\nlight-tailed",
            transform=ax.transAxes, fontsize=8, color="0.6",
            ha="right", va="bottom", style="italic")

    ax.set_xlabel("Skewness", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Excess Kurtosis", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Distribution Shape Journey as θ Varies", fontsize=FONT_TITLE)
    _add_theta_colorbar(ax, norm)

    fig.tight_layout()
    out_path = out_dir / f"frontier_skewness_vs_kurtosis.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
