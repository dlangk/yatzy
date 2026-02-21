"""CDF and tail plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import polars as pl

from ..config import MAX_SCORE
from .spec import PLOT_SPECS, PlotSpec
from .style import (
    FONT_AXIS_LABEL,
    FONT_SUPTITLE,
    FONT_TITLE,
    apply_theta_legend,
    make_norm,
    setup_theme,
    theta_color,
)


def plot_cdf(
    thetas: list[float],
    cdf_df: pl.DataFrame,
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
        fig, ax = plt.subplots(figsize=(14, 7))

    for t in thetas:
        subset = cdf_df.filter(pl.col("theta") == t).to_pandas()
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.4
        alpha = 1.0 if t == 0 else 0.85
        ax.plot(
            subset["score"], subset["cdf"],
            color=color, linewidth=lw, alpha=alpha,
        )

    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Cumulative Probability", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score CDF by Risk Parameter θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(50, MAX_SCORE)
    ax.set_ylim(0, 1)
    apply_theta_legend(ax, norm, spec or PLOT_SPECS["cdf_full"])

    if standalone:
        fig.tight_layout()
        fig.savefig(out_dir / f"cdf_full.{fmt}", dpi=dpi)
        plt.close(fig)


def plot_tails(
    thetas: list[float],
    cdf_df: pl.DataFrame,
    out_dir: Path,
    *,
    norm: mcolors.Normalize | None = None,
    axes=None,
    spec: PlotSpec | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    setup_theme()
    if norm is None:
        norm = make_norm(thetas)
    standalone = axes is None
    if standalone:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        ax_left, ax_right = axes

    for t in thetas:
        subset = cdf_df.filter(pl.col("theta") == t).to_pandas()
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.4
        alpha = 1.0 if t == 0 else 0.85

        left = subset[subset["cdf"] <= 0.05]
        ax_left.plot(
            left["score"], left["cdf"],
            color=color, linewidth=lw, alpha=alpha,
        )

        right = subset[subset["cdf"] >= 0.97]
        ax_right.plot(
            right["score"], right["survival"],
            color=color, linewidth=lw, alpha=alpha,
        )

    _spec = spec or PLOT_SPECS["tails_zoomed"]
    ax_left.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax_left.set_ylabel("Cumulative Probability", fontsize=FONT_AXIS_LABEL)
    ax_left.set_title("Left Tail (bottom 5%)", fontsize=FONT_TITLE, fontweight="bold")
    ax_left.set_ylim(0, 0.05)
    ax_left.set_xlim(0, 200)

    ax_right.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax_right.set_ylabel("P(Score > x)  [survival]", fontsize=FONT_AXIS_LABEL)
    ax_right.set_title("Right Tail (top 3%)", fontsize=FONT_TITLE, fontweight="bold")
    ax_right.set_ylim(0, 0.03)
    ax_right.set_xlim(300, MAX_SCORE)
    apply_theta_legend(ax_right, norm, _spec)

    if standalone:
        fig.suptitle(
            "Tail Behavior by Risk Parameter θ", fontsize=FONT_SUPTITLE, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"tails_zoomed.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
