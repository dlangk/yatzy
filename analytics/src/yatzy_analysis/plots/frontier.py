"""Frontier test plots: adaptive θ(s) vs constant-θ Pareto frontier."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from .style import ADAPTIVE_COLORS as _STYLE_ADAPTIVE_COLORS, FONT_AXIS_LABEL, FONT_LEGEND, FONT_TITLE, GRID_ALPHA, setup_theme


# Consistent colors for adaptive policies
ADAPTIVE_COLORS = {**_STYLE_ADAPTIVE_COLORS, "upper-deficit": "#e67e22"}

BASELINE_COLOR = "#3498db"


def load_frontier_data(
    frontier_dir: Path,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Load frontier_results.csv and per-policy score arrays."""
    results = pl.read_csv(frontier_dir / "frontier_results.csv")

    scores: dict[str, np.ndarray] = {}
    for row in results.iter_rows(named=True):
        name = row["policy"]
        safe = name.replace(":", "_").replace(" ", "_")
        path = frontier_dir / f"frontier_scores_{safe}.csv"
        if path.exists():
            scores[name] = pl.read_csv(path)["score"].to_numpy()

    return results, scores


def plot_pareto_frontier(
    results: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Mean vs σ scatter: baseline curve + adaptive policy points."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 7))

    baselines = results.filter(pl.col("kind") == "baseline").sort("std")
    adaptive = results.filter(pl.col("kind") == "adaptive")

    # Baseline frontier line
    ax.plot(
        baselines["std"].to_numpy(),
        baselines["mean"].to_numpy(),
        color=BASELINE_COLOR,
        linewidth=2.5,
        zorder=3,
        label="Constant-θ frontier",
    )
    # Baseline points
    for row in baselines.iter_rows(named=True):
        ax.scatter(
            row["std"],
            row["mean"],
            color=BASELINE_COLOR,
            s=80,
            zorder=4,
            edgecolors="white",
            linewidths=0.8,
        )
        theta_str = f"θ={row['theta']:.2f}" if row["theta"] > 0 else "θ=0 (EV)"
        ax.annotate(
            theta_str,
            (row["std"], row["mean"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            alpha=0.7,
        )

    # Adaptive policy points
    for row in adaptive.iter_rows(named=True):
        name = row["policy"]
        color = ADAPTIVE_COLORS.get(name, "#95a5a6")
        ax.scatter(
            row["std"],
            row["mean"],
            color=color,
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=1.0,
            marker="D",
            label=name,
        )
        # Draw vertical line to frontier
        frontier_mean = row.get("frontier_mean")
        if frontier_mean is not None and frontier_mean == frontier_mean:  # not NaN
            ax.plot(
                [row["std"], row["std"]],
                [row["mean"], frontier_mean],
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
            )

    ax.set_xlabel("Standard Deviation (σ)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Adaptive θ(s) Policies vs Constant-θ Pareto Frontier",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_dir / f"frontier_pareto.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_frontier_cdf(
    results: pl.DataFrame,
    scores: dict[str, np.ndarray],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Overlaid CDFs for baselines (thin) and adaptive (thick)."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(14, 7))

    baselines = results.filter(pl.col("kind") == "baseline").sort("theta")
    adaptive = results.filter(pl.col("kind") == "adaptive")

    # Baselines: thin gray lines
    for row in baselines.iter_rows(named=True):
        name = row["policy"]
        if name not in scores:
            continue
        s = np.sort(scores[name])
        cdf = np.arange(1, len(s) + 1) / len(s)
        theta_str = f"θ={row['theta']:.2f}" if row["theta"] > 0 else "EV (θ=0)"
        lw = 2.0 if row["theta"] == 0 else 1.0
        alpha = 0.9 if row["theta"] == 0 else 0.4
        ax.plot(s, cdf, color=BASELINE_COLOR, linewidth=lw, alpha=alpha, label=theta_str)

    # Adaptive: colored thick lines
    for row in adaptive.iter_rows(named=True):
        name = row["policy"]
        if name not in scores:
            continue
        color = ADAPTIVE_COLORS.get(name, "#95a5a6")
        s = np.sort(scores[name])
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, color=color, linewidth=2.2, label=name)

    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Cumulative Probability", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Score CDF: Adaptive Policies vs Constant-θ Baselines",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.set_xlim(80, 370)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_dir / f"frontier_cdf.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_frontier_delta(
    results: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Bar chart of Δμ (mean - frontier) for each adaptive policy."""
    setup_theme()

    adaptive = results.filter(pl.col("kind") == "adaptive")
    adaptive = adaptive.drop_nulls(subset=["delta_mu"])
    adaptive = adaptive.sort("delta_mu")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [ADAPTIVE_COLORS.get(n, "#95a5a6") for n in adaptive["policy"].to_list()]
    bars = ax.barh(
        adaptive["policy"].to_numpy(),
        adaptive["delta_mu"].to_numpy(),
        color=colors,
        edgecolor="white",
    )

    # Add value labels
    delta_vals = adaptive["delta_mu"].to_list()
    for bar, delta in zip(bars, delta_vals):
        ax.text(
            bar.get_width() - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.2f}",
            va="center",
            ha="right",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=1.0, color="red", linewidth=1.2, linestyle="--", alpha=0.6, label="H1 threshold")
    ax.set_xlabel("Δμ (Mean − Frontier Mean at matched σ)", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Distance from Constant-θ Pareto Frontier",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.legend(fontsize=FONT_LEGEND)
    ax.set_xlim(min(adaptive["delta_mu"].min() - 0.3, -1.5), 1.5)
    ax.grid(True, alpha=GRID_ALPHA, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / f"frontier_delta.{fmt}", dpi=dpi)
    plt.close(fig)


def generate_frontier_plots(base_path: str = ".", fmt: str = "png") -> None:
    """Generate all frontier plots from outputs/frontier/ data."""
    base = Path(base_path)
    frontier_dir = base / "outputs" / "frontier"
    out_dir = base / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (frontier_dir / "frontier_results.csv").exists():
        print(f"No frontier data found at {frontier_dir}/frontier_results.csv")
        print("Run: just frontier-test")
        return

    print("Loading frontier data...")
    results, scores = load_frontier_data(frontier_dir)
    print(f"  {len(results)} policies, {sum(len(v) for v in scores.values()):,} total scores")

    print("Plotting Pareto frontier...")
    plot_pareto_frontier(results, out_dir, fmt=fmt)

    print("Plotting CDFs...")
    plot_frontier_cdf(results, scores, out_dir, fmt=fmt)

    print("Plotting Δμ bar chart...")
    plot_frontier_delta(results, out_dir, fmt=fmt)

    print(f"Done. Plots saved to {out_dir}/frontier_*.{fmt}")
