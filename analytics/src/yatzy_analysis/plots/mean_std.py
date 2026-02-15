"""Mean vs standard deviation scatter plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from .style import FONT_AXIS_LABEL, FONT_TITLE, GRID_ALPHA, fmt_theta, make_norm, setup_theme, theta_color, theta_colorbar


def plot_mean_vs_std(
    thetas: list[float],
    stats_df: pd.DataFrame,
    out_dir: Path,
    *,
    norm: mcolors.Normalize | None = None,
    ax=None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    if norm is None:
        norm = make_norm(thetas)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in stats_df.iterrows():
        t = row["theta"]
        color = theta_color(t, norm)
        ax.scatter(
            row["std"], row["mean"],
            color=color, s=80, zorder=5, edgecolors="white", linewidths=0.5,
        )
        offset = (6, 4) if t >= 0 else (-8, 4)
        ax.annotate(
            f"θ={fmt_theta(t)}", (row["std"], row["mean"]),
            textcoords="offset points", xytext=offset, fontsize=8, alpha=0.8,
        )

    ax.set_xlabel("Standard Deviation", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Mean–Variance Tradeoff by θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)
    theta_colorbar(ax, norm, label="θ")

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"mean_vs_std.{fmt}", dpi=dpi)
        plt.close(fig)
