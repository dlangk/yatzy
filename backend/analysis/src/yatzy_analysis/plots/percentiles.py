"""Percentile curves vs theta."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import setup_theme


def plot_percentiles(
    thetas: list[float],
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    ax=None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 7))

    percentiles = ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    pct_colors = sns.color_palette("viridis", len(percentiles))

    for i, p in enumerate(percentiles):
        ax.plot(
            stats_df["theta"], stats_df[p],
            marker="o", markersize=4, linewidth=1.8, color=pct_colors[i], label=p, zorder=3,
        )

    ax.plot(
        stats_df["theta"], stats_df["min"],
        linestyle="--", linewidth=1.2, color="gray", alpha=0.7, label="min", zorder=2,
    )
    ax.plot(
        stats_df["theta"], stats_df["max"],
        linestyle="--", linewidth=1.2, color="black", alpha=0.7, label="max", zorder=2,
    )
    ax.plot(
        stats_df["theta"], stats_df["bot5_avg"],
        linestyle=":", linewidth=1.4, color="gray", alpha=0.7, label="bot5 avg", zorder=2,
    )
    ax.plot(
        stats_df["theta"], stats_df["top5_avg"],
        linestyle=":", linewidth=1.4, color="black", alpha=0.7, label="top5 avg", zorder=2,
    )

    ax.set_xlabel("θ", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Score Percentiles vs Risk Parameter θ", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"percentiles_vs_theta.{fmt}", dpi=dpi)
        plt.close(fig)
