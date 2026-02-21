"""Policy compression analysis plots: gap CDFs, heatmaps, visit coverage."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .style import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    FIG_WIDE,
    FONT_AXIS_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    save_fig,
    setup_theme,
)

DECISION_COLORS = {
    "category": COLOR_BLUE,
    "reroll1": COLOR_ORANGE,
    "reroll2": COLOR_RED,
}
DECISION_LABELS = {
    "category": "Category choice",
    "reroll1": "Reroll 1 (2 rerolls left)",
    "reroll2": "Reroll 2 (1 reroll left)",
}


def plot_gap_cdf(
    gap_cdf_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """CDF of decision gaps, one line per decision type. Log x-axis."""
    setup_theme()
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for dt in ["category", "reroll1", "reroll2"]:
        sub = gap_cdf_df.filter(pl.col("decision_type") == dt)
        if sub.is_empty():
            continue
        sub = sub.sort("gap_threshold")
        ax.plot(
            sub["gap_threshold"].to_numpy(),
            sub["fraction_below"].to_numpy(),
            color=DECISION_COLORS[dt],
            label=DECISION_LABELS[dt],
            linewidth=2,
        )

    # Vertical threshold lines
    for thresh, ls in [(0.1, ":"), (0.5, "--"), (1.0, "-"), (5.0, "-.")]:
        ax.axvline(thresh, color="gray", linestyle=ls, alpha=0.4, linewidth=0.8)
        ax.text(
            thresh, 0.02, f"{thresh}", ha="center", fontsize=8, color="gray",
        )

    ax.set_xscale("log")
    ax.set_xlim(0.005, 50)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision gap (EV points)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Fraction of decisions below gap", fontsize=FONT_AXIS_LABEL)
    ax.set_title("CDF of Decision Gaps by Type", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)

    return save_fig(fig, out_dir, "gap_cdf", dpi=dpi, fmt=fmt)


def plot_gap_by_turn(
    gap_summary_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Heatmap: turn (y) x gap threshold (x) -> fraction below."""
    setup_theme()

    thresholds = ["frac_below_0.1", "frac_below_0.5", "frac_below_1.0", "frac_below_5.0"]
    threshold_labels = ["<0.1", "<0.5", "<1.0", "<5.0"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    fig.suptitle("Fraction of Decisions Below Gap Threshold by Turn", fontsize=FONT_TITLE + 2)

    for idx, dt in enumerate(["category", "reroll1", "reroll2"]):
        ax = axes[idx]
        sub = gap_summary_df.filter(pl.col("decision_type") == dt).sort("turn")
        if sub.is_empty():
            continue

        # Build matrix: turns × thresholds
        matrix = np.zeros((15, len(thresholds)))
        for row in sub.iter_rows(named=True):
            t = int(row["turn"])
            for j, col in enumerate(thresholds):
                matrix[t, j] = row[col]

        im = ax.imshow(
            matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
            origin="lower",
        )
        ax.set_xticks(range(len(threshold_labels)))
        ax.set_xticklabels(threshold_labels)
        ax.set_xlabel("Gap threshold", fontsize=FONT_AXIS_LABEL)
        if idx == 0:
            ax.set_ylabel("Turn", fontsize=FONT_AXIS_LABEL)
        ax.set_yticks(range(15))
        ax.set_title(DECISION_LABELS[dt], fontsize=FONT_TITLE)

        # Annotate cells
        for ti in range(15):
            for tj in range(len(thresholds)):
                val = matrix[ti, tj]
                color = "white" if val > 0.6 else "black"
                ax.text(tj, ti, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=axes, label="Fraction below threshold", shrink=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_fig(fig, out_dir, "gap_by_turn", dpi=dpi, fmt=fmt)


def plot_visit_coverage(
    visit_gaps_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Line: gap threshold (x) -> decisions per game above threshold (y)."""
    setup_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)

    gap_thresh = visit_gaps_df["gap_threshold"].to_numpy()
    decisions_above = visit_gaps_df["decisions_per_game_above"].to_numpy()
    visit_frac = visit_gaps_df["visit_fraction_above"].to_numpy()

    # Left: decisions per game above threshold
    ax1.plot(
        gap_thresh,
        decisions_above,
        "o-",
        color=COLOR_BLUE,
        linewidth=2,
        markersize=6,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Gap threshold (EV points)", fontsize=FONT_AXIS_LABEL)
    ax1.set_ylabel("Decisions per game above threshold", fontsize=FONT_AXIS_LABEL)
    ax1.set_title("How Many Decisions Actually Matter?", fontsize=FONT_TITLE)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.axhline(45, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.text(0.02, 45.5, "45 decisions/game", fontsize=8, color="gray")

    # Right: fraction above
    ax2.plot(
        gap_thresh,
        visit_frac * 100,
        "o-",
        color=COLOR_RED,
        linewidth=2,
        markersize=6,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Gap threshold (EV points)", fontsize=FONT_AXIS_LABEL)
    ax2.set_ylabel("% of visited decisions above threshold", fontsize=FONT_AXIS_LABEL)
    ax2.set_title("Visit-Weighted Gap Distribution", fontsize=FONT_TITLE)
    ax2.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    return save_fig(fig, out_dir, "visit_coverage", dpi=dpi, fmt=fmt)


def plot_policy_distinct(
    distinct_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Bar chart: total states vs policy-distinct states per turn."""
    setup_theme()
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    turns = distinct_df["turn"].to_numpy()
    total = distinct_df["total_states"].to_numpy()
    width = 0.2

    x = np.arange(len(turns))
    ax.bar(x - 1.5 * width, total, width, label="Total reachable states", color="lightgray", edgecolor="gray")
    ax.bar(x - 0.5 * width, distinct_df["distinct_category"].to_numpy(), width,
           label="Distinct category policies", color=COLOR_BLUE, alpha=0.8)
    ax.bar(x + 0.5 * width, distinct_df["distinct_reroll1"].to_numpy(), width,
           label="Distinct reroll1 policies", color=COLOR_ORANGE, alpha=0.8)
    ax.bar(x + 1.5 * width, distinct_df["distinct_reroll2"].to_numpy(), width,
           label="Distinct reroll2 policies", color=COLOR_RED, alpha=0.8)

    ax.set_xlabel("Turn", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Policy-Distinct States vs Total Reachable States", fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(turns)
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    fig.tight_layout()
    return save_fig(fig, out_dir, "policy_distinct", dpi=dpi, fmt=fmt)


def plot_all_compression(
    out_dir: Path,
    data_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Load all CSVs from data_dir, generate all plots."""
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gap CDF
    cdf_path = data_dir / "gap_cdf.csv"
    if cdf_path.exists():
        cdf_df = pl.read_csv(cdf_path)
        paths.append(plot_gap_cdf(cdf_df, out_dir, dpi=dpi, fmt=fmt))

    # Gap by turn heatmap
    summary_path = data_dir / "gap_summary.csv"
    if summary_path.exists():
        summary_df = pl.read_csv(summary_path)
        paths.append(plot_gap_by_turn(summary_df, out_dir, dpi=dpi, fmt=fmt))

    # Visit coverage
    visit_path = data_dir / "visit_gaps.csv"
    if visit_path.exists():
        visit_df = pl.read_csv(visit_path)
        paths.append(plot_visit_coverage(visit_df, out_dir, dpi=dpi, fmt=fmt))

    # Policy distinct
    distinct_path = data_dir / "policy_distinct.csv"
    if distinct_path.exists():
        distinct_df = pl.read_csv(distinct_path)
        paths.append(plot_policy_distinct(distinct_df, out_dir, dpi=dpi, fmt=fmt))

    return paths
