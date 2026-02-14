"""Combined 3x2 dashboard figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import MAX_SCORE
from .cdf import plot_cdf
from .density import plot_density
from .mean_std import plot_mean_vs_std
from .percentiles import plot_percentiles
from .style import fmt_theta, make_norm, setup_theme, theta_color


def plot_combined(
    thetas: list[float],
    cdf_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    kde_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    norm = make_norm(thetas)
    fig = plt.figure(figsize=(28, 24))

    gs = fig.add_gridspec(
        3, 2, hspace=0.35, wspace=0.3, left=0.05, right=0.95, top=0.93, bottom=0.05,
    )

    fig.suptitle(
        "Risk-Sensitive Yatzy: Score Distributions by θ",
        fontsize=22, fontweight="bold", y=0.97,
    )

    # Row 0: CDF + Percentiles
    plot_cdf(thetas, cdf_df, out_dir, norm=norm, ax=fig.add_subplot(gs[0, 0]))
    plot_percentiles(thetas, stats_df, out_dir, ax=fig.add_subplot(gs[0, 1]))

    # Row 1: Tails (left + right share the row)
    gs_tails = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax_ltail = fig.add_subplot(gs_tails[0, 0])
    ax_rtail = fig.add_subplot(gs_tails[0, 1])

    for t in thetas:
        subset = cdf_df[cdf_df["theta"] == t]
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.4
        alpha = 1.0 if t == 0 else 0.85

        left = subset[subset["cdf"] <= 0.05]
        ax_ltail.plot(
            left["score"], left["cdf"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

        right = subset[subset["cdf"] >= 0.97]
        ax_rtail.plot(
            right["score"], right["survival"],
            color=color, linewidth=lw, alpha=alpha,
        )

    ax_ltail.set_xlabel("Total Score", fontsize=12)
    ax_ltail.set_ylabel("Cumulative Probability", fontsize=12)
    ax_ltail.set_title("Left Tail (bottom 5%)", fontsize=13, fontweight="bold")
    ax_ltail.set_ylim(0, 0.05)
    ax_ltail.set_xlim(0, 200)

    ax_rtail.set_xlabel("Total Score", fontsize=12)
    ax_rtail.set_ylabel("P(Score > x)  [survival]", fontsize=12)
    ax_rtail.set_title("Right Tail (top 3%)", fontsize=13, fontweight="bold")
    ax_rtail.set_ylim(0, 0.03)
    ax_rtail.set_xlim(300, MAX_SCORE)

    handles, labels = ax_ltail.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.635), ncol=8, fontsize=8, framealpha=0.9,
    )

    # Row 2: Mean-Std + Density
    plot_mean_vs_std(thetas, stats_df, out_dir, norm=norm, ax=fig.add_subplot(gs[2, 0]))
    plot_density(thetas, kde_df, out_dir, norm=norm, ax=fig.add_subplot(gs[2, 1]))

    fig.savefig(out_dir / f"combined.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
