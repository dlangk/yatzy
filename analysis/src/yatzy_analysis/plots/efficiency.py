"""Efficiency analysis: MER, frontier, CDF difference, CVaR deficit."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import make_norm, setup_theme, theta_color, theta_colorbar


def plot_efficiency(
    thetas: list[float],
    summary_df: pd.DataFrame,
    kde_df: pd.DataFrame,
    mer_df: pd.DataFrame,
    sdva_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Four-panel efficiency figure."""
    setup_theme()
    norm = make_norm(thetas)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "The Cost of Risk-Seeking in Yatzy",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # Filter to available thetas and positive-theta only for MER
    mer = mer_df[mer_df["theta"].isin(thetas) & (mer_df["theta"] > 0)].copy()
    stats = summary_df[summary_df["theta"].isin(thetas)].copy()

    # Panel 1: MER curves for multiple quantiles
    ax = axes[0, 0]
    mer_cols = {
        "mer_p75": ("p75", "tab:green"),
        "mer_p90": ("p90", "tab:blue"),
        "mer_p95": ("p95", "tab:orange"),
        "mer_p99": ("p99", "tab:red"),
        "mer_max": ("max", "tab:purple"),
    }
    for col, (label, color) in mer_cols.items():
        valid = mer[mer[col].between(-50, 50)]  # filter out inf/dominated
        if not valid.empty:
            ax.plot(valid["theta"], valid[col], marker="o", markersize=4,
                    linewidth=1.8, color=color, label=label)

    ax.set_xlabel("θ", fontsize=12)
    ax.set_ylabel("Mean points lost per point gained", fontsize=12)
    ax.set_title("Marginal Exchange Rate (MER)", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(-5, 30)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: Efficient frontier in (mean, p95) space
    ax = axes[0, 1]
    for _, row in stats.iterrows():
        t = row["theta"]
        color = theta_color(t, norm)
        ax.scatter(row["mean"], row["p95"], color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)
    # Connect with line
    ax.plot(stats["mean"], stats["p95"], color="gray", linewidth=0.8, alpha=0.5, zorder=1)
    # Mark baseline
    base = stats[stats["theta"] == 0.0]
    if not base.empty:
        ax.scatter(base["mean"], base["p95"], color="black", s=120, marker="*",
                   zorder=10, label="θ=0 (EV-optimal)")
    # Mark peak p95
    peak_idx = stats["p95"].idxmax()
    peak = stats.loc[peak_idx]
    ax.scatter(peak["mean"], peak["p95"], color="red", s=120, marker="D",
               zorder=10, label=f"peak p95 (θ={peak['theta']:.2f})")
    # Shade dominated region
    if not base.empty:
        b = base.iloc[0]
        ax.axvspan(ax.get_xlim()[0], b["mean"], alpha=0.03, color="red")
        ax.axhspan(ax.get_ylim()[0], b["p95"], alpha=0.03, color="red")

    ax.set_xlabel("Mean Score", fontsize=12)
    ax.set_ylabel("p95 Score", fontsize=12)
    ax.set_title("Efficient Frontier: Mean vs p95", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    theta_colorbar(ax, norm, label="θ")

    # Panel 3: CDF difference D(x) for representative thetas
    ax = axes[1, 0]
    representative = [t for t in [0.05, 0.08, 0.10, 0.20] if t in thetas]
    base_kde = kde_df[kde_df["theta"] == 0.0]
    if not base_kde.empty:
        scores = base_kde["score"].values
        for t in representative:
            t_kde = kde_df[kde_df["theta"] == t]
            if t_kde.empty:
                continue
            d = t_kde["cdf"].values - base_kde["cdf"].values
            color = theta_color(t, norm)
            ax.plot(scores, d, color=color, linewidth=1.8, label=f"θ={t:.2f}")
            ax.fill_between(scores, d, 0, where=d > 0, alpha=0.15, color="red")
            ax.fill_between(scores, d, 0, where=d < 0, alpha=0.15, color="green")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total Score", fontsize=12)
    ax.set_ylabel("F_θ(x) - F_0(x)", fontsize=12)
    ax.set_title("CDF Difference (red=worse, green=better)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add crossing points from SDVA
    for t in representative:
        row = sdva_df[sdva_df["theta"] == t]
        if not row.empty:
            xc = row.iloc[0]["x_cross"]
            ax.axvline(xc, color=theta_color(t, norm), linewidth=0.8,
                       linestyle=":", alpha=0.7)

    # Panel 4: CVaR deficit overlaid with mean
    ax = axes[1, 1]
    pos = stats[stats["theta"] >= 0].copy()
    ax.plot(pos["theta"], pos["mean"], color="black", linewidth=2, label="Mean", marker="o",
            markersize=4)
    ax.plot(pos["theta"], pos["cvar_5"], color="tab:red", linewidth=2, label="CVaR 5%",
            marker="s", markersize=4)
    ax.plot(pos["theta"], pos["cvar_10"], color="tab:orange", linewidth=1.5, label="CVaR 10%",
            marker="^", markersize=4, alpha=0.8)
    ax.plot(pos["theta"], pos["cvar_1"], color="darkred", linewidth=1.5, label="CVaR 1%",
            marker="v", markersize=4, alpha=0.8)

    ax.set_xlabel("θ", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Mean and CVaR vs θ (downside severity)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"efficiency.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
