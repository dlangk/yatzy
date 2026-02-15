"""Percentile curves vs theta."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import setup_theme

# All percentiles to plot, ordered low→high.
# "extra" ones (p1, p999, p9999) are plotted with dashed style if available.
_CORE = ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
_EXTRA = ["p1", "p999", "p9999"]


def _plot_percentile_curves(
    df: pd.DataFrame,
    ax,
    *,
    include_extra: bool = True,
) -> None:
    """Draw percentile curves on the given axes."""
    pct_colors = sns.color_palette("rocket", len(_CORE))

    for i, p in enumerate(_CORE):
        if p in df.columns:
            ax.plot(
                df["theta"], df[p],
                marker="o", markersize=4, linewidth=1.8, color=pct_colors[i],
                label=p, zorder=3,
            )

    if include_extra:
        extra_styles = {"p1": ("--", "tab:blue"), "p999": ("--", "tab:orange"), "p9999": ("--", "tab:red")}
        for p, (ls, color) in extra_styles.items():
            if p in df.columns:
                ax.plot(
                    df["theta"], df[p],
                    linestyle=ls, marker="s", markersize=3, linewidth=1.4,
                    color=color, alpha=0.8, label=p, zorder=3,
                )

    ax.plot(df["theta"], df["min"], linestyle="--", linewidth=1.2, color="gray", alpha=0.7, label="min", zorder=2)
    ax.plot(df["theta"], df["max"], linestyle="--", linewidth=1.2, color="black", alpha=0.7, label="max", zorder=2)
    ax.plot(df["theta"], df["bot5_avg"], linestyle=":", linewidth=1.4, color="gray", alpha=0.7, label="bot5 avg", zorder=2)
    ax.plot(df["theta"], df["top5_avg"], linestyle=":", linewidth=1.4, color="black", alpha=0.7, label="top5 avg", zorder=2)


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

    _plot_percentile_curves(stats_df, ax)

    ax.set_xlabel("θ", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Score Percentiles vs Risk Parameter θ", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"percentiles_vs_theta.{fmt}", dpi=dpi)
        plt.close(fig)

        # Zoomed version: θ ∈ [-0.10, +0.45] — covers all percentile peaks with margin
        zoomed_df = stats_df[(stats_df["theta"] >= -0.10) & (stats_df["theta"] <= 0.45)]
        if len(zoomed_df) > 0:
            fig_z, ax_z = plt.subplots(figsize=(14, 8))
            _plot_percentile_curves(zoomed_df, ax_z)
            ax_z.set_xlabel("θ", fontsize=13)
            ax_z.set_ylabel("Score", fontsize=13)
            ax_z.set_title(
                "Score Percentiles vs θ (zoomed: peaks region)",
                fontsize=15, fontweight="bold",
            )
            ax_z.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
            ax_z.grid(True, alpha=0.3)
            fig_z.tight_layout()
            fig_z.savefig(out_dir / f"percentiles_vs_theta_zoomed.{fmt}", dpi=dpi)
            plt.close(fig_z)
