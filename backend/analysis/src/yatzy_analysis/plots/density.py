"""Probability density plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import MAX_SCORE
from .style import fmt_theta, setup_theme, theta_color


def plot_density(
    thetas: list[float],
    kde_df: pd.DataFrame,
    out_dir: Path,
    *,
    ax=None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 6))

    for t in thetas:
        subset = kde_df[kde_df["theta"] == t].sort_values("score")
        color = theta_color(t)
        lw = 2.5 if t == 0 else 1.4
        alpha = 0.9 if t == 0 else 0.7
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

    ax.set_xlabel("Total Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Score Distribution by Risk Parameter θ", fontsize=15, fontweight="bold")
    ax.set_xlim(50, MAX_SCORE)
    ax.legend(fontsize=9, framealpha=0.9, ncol=2)

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"density.{fmt}", dpi=dpi)
        plt.close(fig)
