"""Quantile (inverse CDF) plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from ..config import MAX_SCORE
from .style import FONT_AXIS_LABEL, FONT_LEGEND, FONT_TITLE, GRID_ALPHA, fmt_theta, make_norm, setup_theme, theta_color


def plot_quantile(
    thetas: list[float],
    cdf_df: pd.DataFrame,
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
        fig, ax = plt.subplots(figsize=(12, 6))

    for t in thetas:
        subset = cdf_df[cdf_df["theta"] == t]
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.4
        ax.plot(
            subset["cdf"], subset["score"],
            color=color, linewidth=lw, alpha=0.85, label=f"θ={fmt_theta(t)}",
        )

    ax.set_xlabel("Cumulative Probability", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Quantile Function (Inverse CDF) by θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(50, MAX_SCORE)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=GRID_ALPHA)

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"quantile.{fmt}", dpi=dpi)
        plt.close(fig)
