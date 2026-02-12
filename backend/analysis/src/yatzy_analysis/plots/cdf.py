"""CDF and tail plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import MAX_SCORE
from .style import fmt_theta, setup_theme, theta_color, theta_colorbar


def plot_cdf(
    thetas: list[float],
    cdf_df: pd.DataFrame,
    out_dir: Path,
    *,
    ax=None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 7))

    for t in thetas:
        subset = cdf_df[cdf_df["theta"] == t]
        color = theta_color(t)
        lw = 2.5 if t == 0 else 1.4
        alpha = 1.0 if t == 0 else 0.85
        ax.plot(
            subset["score"], subset["cdf"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

    ax.set_xlabel("Total Score", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.set_title("Score CDF by Risk Parameter θ", fontsize=15, fontweight="bold")
    ax.set_xlim(50, MAX_SCORE)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    theta_colorbar(ax)

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"cdf_full.{fmt}", dpi=dpi)
        plt.close(fig)


def plot_tails(
    thetas: list[float],
    cdf_df: pd.DataFrame,
    out_dir: Path,
    *,
    axes=None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    standalone = axes is None
    if standalone:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        ax_left, ax_right = axes

    for t in thetas:
        subset = cdf_df[cdf_df["theta"] == t]
        color = theta_color(t)
        lw = 2.5 if t == 0 else 1.4
        alpha = 1.0 if t == 0 else 0.85

        left = subset[subset["cdf"] <= 0.10]
        ax_left.plot(
            left["score"], left["cdf"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

        right = subset[subset["cdf"] >= 0.90]
        ax_right.plot(
            right["score"], right["survival"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

    ax_left.set_xlabel("Total Score", fontsize=12)
    ax_left.set_ylabel("Cumulative Probability", fontsize=12)
    ax_left.set_title("Left Tail (bottom 10%)", fontsize=13, fontweight="bold")
    ax_left.set_ylim(0, 0.10)
    ax_left.set_xlim(80, 220)
    ax_left.legend(fontsize=7, loc="upper left", framealpha=0.9)

    ax_right.set_xlabel("Total Score", fontsize=12)
    ax_right.set_ylabel("P(Score > x)  [survival]", fontsize=12)
    ax_right.set_title("Right Tail (top 10%)", fontsize=13, fontweight="bold")
    ax_right.set_ylim(0, 0.10)
    ax_right.set_xlim(270, MAX_SCORE)
    ax_right.legend(fontsize=7, loc="upper right", framealpha=0.9)

    if standalone:
        fig.suptitle(
            "Tail Behavior by Risk Parameter θ", fontsize=15, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"tails_zoomed.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
