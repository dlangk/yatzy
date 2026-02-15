"""Efficiency analysis: MER, frontier, CDF difference, CVaR deficit."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import FONT_AXIS_LABEL, FONT_LEGEND, FONT_SUPTITLE, FONT_TITLE, GRID_ALPHA, make_norm, setup_theme, theta_color, theta_colorbar

# Shared axis ranges for all adaptive frontier plots (individual + combined).
# Data ranges: Mean 186–248, p95 266–314, p99 300–329, p999 328–341.
_ADAPTIVE_XLIM = (180, 255)
_ADAPTIVE_YLIM = (258, 348)


def plot_efficiency(
    thetas: list[float],
    summary_df: pd.DataFrame,
    kde_df: pd.DataFrame,
    mer_df: pd.DataFrame,
    sdva_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Four-panel efficiency figure."""
    setup_theme()
    norm = make_norm(thetas)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "The Cost of Risk-Seeking in Yatzy",
        fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98,
    )

    # Filter to available thetas and positive-theta only for MER
    mer = mer_df[mer_df["theta"].isin(thetas) & (mer_df["theta"] > 0)].copy()
    stats = summary_df[summary_df["theta"].isin(thetas)].copy()

    # Panel 1: MER curves for multiple quantiles
    ax = axes[0, 0]
    mer_cols = {
        "mer_p75": ("p75", "tab:green"),
        "mer_p90": ("p90", "tab:blue"),
        "mer_p95": ("p95", "tab:orange"),
        "mer_p99": ("p99", "tab:red"),
        "mer_max": ("max", "tab:purple"),
    }
    for col, (label, color) in mer_cols.items():
        valid = mer[mer[col].between(-50, 50)]  # filter out inf/dominated
        if not valid.empty:
            ax.plot(valid["theta"], valid[col], marker="o", markersize=4,
                    linewidth=1.8, color=color, label=label)

    ax.set_xlabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean points lost per point gained", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Marginal Exchange Rate (MER)", fontsize=FONT_TITLE, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(-5, 30)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Efficient frontier in (mean, p95) space
    ax = axes[0, 1]
    for _, row in stats.iterrows():
        t = row["theta"]
        color = theta_color(t, norm)
        ax.scatter(row["mean"], row["p95"], color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)
    # Connect with line
    ax.plot(stats["mean"], stats["p95"], color="gray", linewidth=0.8, alpha=0.5, zorder=1)
    # Mark baseline
    base = stats[stats["theta"] == 0.0]
    if not base.empty:
        ax.scatter(base["mean"], base["p95"], color="black", s=120, marker="*",
                   zorder=10, label="θ=0 (EV-optimal)")
    # Mark peak p95
    peak_idx = stats["p95"].idxmax()
    peak = stats.loc[peak_idx]
    ax.scatter(peak["mean"], peak["p95"], color="red", s=120, marker="D",
               zorder=10, label=f"peak p95 (θ={peak['theta']:.2f})")
    # Shade dominated region
    if not base.empty:
        b = base.iloc[0]
        ax.axvspan(ax.get_xlim()[0], b["mean"], alpha=0.03, color="red")
        ax.axhspan(ax.get_ylim()[0], b["p95"], alpha=0.03, color="red")

    ax.set_xlabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("p95 Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Efficient Frontier: Mean vs p95", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)
    theta_colorbar(ax, norm, label="θ")

    # Panel 3: CDF difference D(x) for representative thetas
    ax = axes[1, 0]
    representative = [t for t in [0.05, 0.08, 0.10, 0.20] if t in thetas]
    base_kde = kde_df[kde_df["theta"] == 0.0]
    if not base_kde.empty:
        scores = base_kde["score"].values
        for t in representative:
            t_kde = kde_df[kde_df["theta"] == t]
            if t_kde.empty:
                continue
            d = t_kde["cdf"].values - base_kde["cdf"].values
            color = theta_color(t, norm)
            ax.plot(scores, d, color=color, linewidth=1.8, label=f"θ={t:.2f}")
            ax.fill_between(scores, d, 0, where=d > 0, alpha=0.15, color="red")
            ax.fill_between(scores, d, 0, where=d < 0, alpha=0.15, color="green")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("F_θ(x) - F_0(x)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("CDF Difference (red=worse, green=better)", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add crossing points from SDVA
    for t in representative:
        row = sdva_df[sdva_df["theta"] == t]
        if not row.empty:
            xc = row.iloc[0]["x_cross"]
            ax.axvline(xc, color=theta_color(t, norm), linewidth=0.8,
                       linestyle=":", alpha=0.7)

    # Panel 4: CVaR deficit overlaid with mean
    ax = axes[1, 1]
    pos = stats[stats["theta"] >= 0].copy()
    ax.plot(pos["theta"], pos["mean"], color="black", linewidth=2, label="Mean", marker="o",
            markersize=4)
    ax.plot(pos["theta"], pos["cvar_5"], color="tab:red", linewidth=2, label="CVaR 5%",
            marker="s", markersize=4)
    ax.plot(pos["theta"], pos["cvar_10"], color="tab:orange", linewidth=1.5, label="CVaR 10%",
            marker="^", markersize=4, alpha=0.8)
    ax.plot(pos["theta"], pos["cvar_1"], color="darkred", linewidth=1.5, label="CVaR 1%",
            marker="v", markersize=4, alpha=0.8)

    ax.set_xlabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Mean and CVaR vs θ (downside severity)", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"efficiency.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_adaptive_frontier(
    thetas: list[float],
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    out_dir: Path,
    percentile_col: str,
    percentile_label: str,
    filename: str,
    *,
    ax: plt.Axes | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Efficient frontier with adaptive policy overlay for a given percentile.

    Shows fixed-θ points, their convex hull, and adaptive policy points.
    Key question: do any adaptive points lie above the convex hull?

    When *ax* is provided, plots onto that axes (panel mode) — no figure
    creation or save.  When *ax* is None, creates a standalone figure.
    """
    from scipy.spatial import ConvexHull

    standalone = ax is None
    if standalone:
        setup_theme()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(
            f"Adaptive θ Policies vs Fixed-θ Frontier (Mean vs {percentile_label})",
            fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98,
        )

    norm = make_norm(thetas)
    stats = summary_df[summary_df["theta"].isin(thetas)].copy()

    # Plot fixed-θ points
    for _, row in stats.iterrows():
        t = row["theta"]
        color = theta_color(t, norm)
        ax.scatter(row["mean"], row[percentile_col], color=color, s=50, zorder=5,
                   edgecolors="white", linewidths=0.5, alpha=0.7)

    # Connect fixed-θ points
    ax.plot(stats["mean"], stats[percentile_col], color="gray", linewidth=0.8, alpha=0.4, zorder=1)

    # Compute and shade convex hull of fixed-θ points
    points = stats[["mean", percentile_col]].values
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close the polygon
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.08, color="blue",
                    label="Fixed-θ convex hull")
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="blue", linewidth=1.0,
                    alpha=0.4, linestyle="--")
        except Exception:
            pass  # degenerate hull

    # Mark θ=0 baseline
    base = stats[stats["theta"] == 0.0]
    if not base.empty:
        ax.scatter(base["mean"], base[percentile_col], color="black", s=150, marker="*",
                   zorder=10, label="θ=0 (EV-optimal)")

    # Mark best fixed-θ for this percentile
    peak_idx = stats[percentile_col].idxmax()
    peak = stats.loc[peak_idx]
    ax.scatter(peak["mean"], peak[percentile_col], color="red", s=120, marker="D",
               zorder=10, label=f"Best fixed {percentile_label} (θ={peak['theta']:.2f})")

    # Plot adaptive policy points
    markers = {"bonus-adaptive": "^", "phase-based": "s", "combined": "P", "always-ev": "o"}
    colors = {"bonus-adaptive": "tab:green", "phase-based": "tab:orange",
              "combined": "tab:purple", "always-ev": "gray"}

    for _, row in adaptive_df.iterrows():
        name = row["policy"]
        m = markers.get(name, "X")
        c = colors.get(name, "tab:brown")
        ax.scatter(row["mean"], row[percentile_col], color=c, s=200, marker=m,
                   zorder=15, edgecolors="black", linewidths=1.0,
                   label=f"{name}")

    ax.set_xlabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(f"{percentile_label} Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title(f"Mean vs {percentile_label}", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_LEGEND, loc="lower left")
    ax.grid(True, alpha=GRID_ALPHA)
    theta_colorbar(ax, norm, label="θ (fixed policies)")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if standalone:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{filename}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_efficiency_with_adaptive(
    thetas: list[float],
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Efficient frontier with adaptive overlay for p95 (default view)."""
    _plot_adaptive_frontier(
        thetas, summary_df, adaptive_df, out_dir,
        percentile_col="p95", percentile_label="p95",
        filename="efficiency_adaptive",
        xlim=_ADAPTIVE_XLIM, ylim=_ADAPTIVE_YLIM,
        dpi=dpi, fmt=fmt,
    )


def plot_efficiency_adaptive_p99(
    thetas: list[float],
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Efficient frontier with adaptive overlay for p99."""
    _plot_adaptive_frontier(
        thetas, summary_df, adaptive_df, out_dir,
        percentile_col="p99", percentile_label="p99",
        filename="efficiency_adaptive_p99",
        xlim=_ADAPTIVE_XLIM, ylim=_ADAPTIVE_YLIM,
        dpi=dpi, fmt=fmt,
    )


def plot_efficiency_adaptive_p999(
    thetas: list[float],
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Efficient frontier with adaptive overlay for p99.9."""
    _plot_adaptive_frontier(
        thetas, summary_df, adaptive_df, out_dir,
        percentile_col="p999", percentile_label="p99.9",
        filename="efficiency_adaptive_p999",
        xlim=_ADAPTIVE_XLIM, ylim=_ADAPTIVE_YLIM,
        dpi=dpi, fmt=fmt,
    )


def plot_efficiency_adaptive_combined(
    thetas: list[float],
    summary_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Three-panel combined adaptive frontier: p95, p99, p99.9 side-by-side."""
    setup_theme()
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    for ax_i, (pcol, plabel) in zip(axes, [
        ("p95", "p95"), ("p99", "p99"), ("p999", "p99.9"),
    ]):
        _plot_adaptive_frontier(
            thetas, summary_df, adaptive_df, out_dir,
            percentile_col=pcol, percentile_label=plabel,
            filename="",
            ax=ax_i, xlim=_ADAPTIVE_XLIM, ylim=_ADAPTIVE_YLIM,
        )
    fig.suptitle(
        "Adaptive θ Policies vs Fixed-θ Frontier",
        fontsize=FONT_SUPTITLE, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"efficiency_adaptive_combined.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
