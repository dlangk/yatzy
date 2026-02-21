"""Mean vs standard deviation scatter plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import polars as pl

from .spec import PLOT_SPECS, PlotSpec
from .style import FONT_AXIS_LABEL, FONT_TITLE, GRID_ALPHA, apply_theta_legend, make_norm, setup_theme, theta_color


def plot_mean_vs_std(
    thetas: list[float],
    stats_df: pl.DataFrame,
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
        fig, ax = plt.subplots(figsize=(10, 7))

    pdf = stats_df.to_pandas()
    for _, row in pdf.iterrows():
        t = row["theta"]
        color = theta_color(t, norm)
        ax.scatter(
            row["std"], row["mean"],
            color=color, s=80, zorder=5, edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("Standard Deviation", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Mean–Variance Tradeoff by θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
    apply_theta_legend(ax, norm, spec or PLOT_SPECS["mean_vs_std"])

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"mean_vs_std.{fmt}", dpi=dpi)
        plt.close(fig)
