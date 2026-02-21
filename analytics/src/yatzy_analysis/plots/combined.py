"""Combined 3x2 dashboard figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from .cdf import plot_cdf, plot_tails
from .density import plot_density
from .mean_std import plot_mean_vs_std
from .percentiles import plot_percentiles
from .spec import NO_LEGEND
from .style import CMAP, FONT_AXIS_LABEL, make_norm, setup_theme


def plot_combined(
    thetas: list[float],
    cdf_df: pl.DataFrame,
    stats_df: pl.DataFrame,
    kde_df: pl.DataFrame,
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

    # Suppress per-panel colorbars; figure gets one shared colorbar below.
    no = NO_LEGEND

    # Row 0: CDF + Percentiles
    plot_cdf(thetas, cdf_df, out_dir, norm=norm, ax=fig.add_subplot(gs[0, 0]), spec=no)
    plot_percentiles(thetas, stats_df, out_dir, ax=fig.add_subplot(gs[0, 1]))

    # Row 1: Tails (left + right share the row)
    gs_tails = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax_ltail = fig.add_subplot(gs_tails[0, 0])
    ax_rtail = fig.add_subplot(gs_tails[0, 1])

    plot_tails(thetas, cdf_df, out_dir, norm=norm, axes=(ax_ltail, ax_rtail), spec=no)

    # Row 2: Mean-Std + Density
    plot_mean_vs_std(thetas, stats_df, out_dir, norm=norm, ax=fig.add_subplot(gs[2, 0]), spec=no)
    plot_density(thetas, kde_df, out_dir, norm=norm, ax=fig.add_subplot(gs[2, 1]), spec=no)

    # Single shared colorbar for the full figure (right margin)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.97, 0.08, 0.012, 0.83])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("θ  (blue=risk-averse, red=risk-seeking)", fontsize=FONT_AXIS_LABEL)

    fig.savefig(out_dir / f"combined.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
