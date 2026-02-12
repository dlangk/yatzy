"""Probability density plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from ..config import MAX_SCORE
from .style import fmt_theta, make_norm, setup_theme, theta_color


def plot_density(
    thetas: list[float],
    kde_df: pd.DataFrame,
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
        fig, ax = plt.subplots(figsize=(14, 6))

    for t in thetas:
        subset = kde_df[kde_df["theta"] == t].sort_values("score")
        color = theta_color(t, norm)
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


_ADAPTIVE_COLORS = {
    "bonus-adaptive": "tab:green",
    "phase-based": "tab:orange",
    "combined": "tab:purple",
}
_ADAPTIVE_DASHES = {
    "bonus-adaptive": (4, 2),
    "phase-based": (6, 2, 2, 2),
    "combined": (2, 2),
}


def plot_density_with_adaptive(
    thetas: list[float],
    kde_df: pd.DataFrame,
    adaptive_kde_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Density plot showing both fixed-θ sweeps and adaptive strategy densities."""
    setup_theme()
    norm = make_norm(thetas)
    fig, ax = plt.subplots(figsize=(16, 7))

    # Fixed-θ densities (thin, semi-transparent)
    for t in thetas:
        subset = kde_df[kde_df["theta"] == t].sort_values("score")
        color = theta_color(t, norm)
        lw = 2.5 if t == 0 else 1.0
        alpha = 0.9 if t == 0 else 0.45
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=lw, alpha=alpha, label=f"θ={fmt_theta(t)}",
        )

    # Adaptive strategy densities (thick, dashed, prominent)
    for policy in sorted(adaptive_kde_df["policy"].unique()):
        subset = adaptive_kde_df[adaptive_kde_df["policy"] == policy].sort_values("score")
        color = _ADAPTIVE_COLORS.get(policy, "tab:brown")
        dashes = _ADAPTIVE_DASHES.get(policy, (4, 2))
        ax.plot(
            subset["score"], subset["density"],
            color=color, linewidth=2.8, alpha=0.95,
            dashes=dashes, label=policy,
        )

    ax.set_xlabel("Total Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        "Score Distribution: Fixed-θ Sweep + Adaptive Strategies",
        fontsize=15, fontweight="bold",
    )
    ax.set_xlim(50, MAX_SCORE)
    ax.legend(fontsize=9, framealpha=0.9, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / f"density_adaptive.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
