"""Frontier test plots: adaptive θ(s) vs constant-θ Pareto frontier."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import setup_theme


# Consistent colors for adaptive policies
ADAPTIVE_COLORS = {
    "bonus-adaptive": "#e74c3c",
    "phase-based": "#2ecc71",
    "combined": "#9b59b6",
    "upper-deficit": "#e67e22",
}

BASELINE_COLOR = "#3498db"


def load_frontier_data(
    frontier_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load frontier_results.csv and per-policy score arrays."""
    results = pd.read_csv(frontier_dir / "frontier_results.csv")

    scores: dict[str, np.ndarray] = {}
    for _, row in results.iterrows():
        name = row["policy"]
        safe = name.replace(":", "_").replace(" ", "_")
        path = frontier_dir / f"frontier_scores_{safe}.csv"
        if path.exists():
            scores[name] = pd.read_csv(path)["score"].values

    return results, scores


def plot_pareto_frontier(
    results: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Mean vs σ scatter: baseline curve + adaptive policy points."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 7))

    baselines = results[results["kind"] == "baseline"].sort_values("std")
    adaptive = results[results["kind"] == "adaptive"]

    # Baseline frontier line
    ax.plot(
        baselines["std"],
        baselines["mean"],
        color=BASELINE_COLOR,
        linewidth=2.5,
        zorder=3,
        label="Constant-θ frontier",
    )
    # Baseline points
    for _, row in baselines.iterrows():
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
    for _, row in adaptive.iterrows():
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
        if pd.notna(row.get("frontier_mean")):
            ax.plot(
                [row["std"], row["std"]],
                [row["mean"], row["frontier_mean"]],
                color=color,
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
            )

    ax.set_xlabel("Standard Deviation (σ)", fontsize=13)
    ax.set_ylabel("Mean Score", fontsize=13)
    ax.set_title(
        "Adaptive θ(s) Policies vs Constant-θ Pareto Frontier",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"frontier_pareto.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_frontier_cdf(
    results: pd.DataFrame,
    scores: dict[str, np.ndarray],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Overlaid CDFs for baselines (thin) and adaptive (thick)."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(14, 7))

    baselines = results[results["kind"] == "baseline"].sort_values("theta")
    adaptive = results[results["kind"] == "adaptive"]

    # Baselines: thin gray lines
    for _, row in baselines.iterrows():
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
    for _, row in adaptive.iterrows():
        name = row["policy"]
        if name not in scores:
            continue
        color = ADAPTIVE_COLORS.get(name, "#95a5a6")
        s = np.sort(scores[name])
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, cdf, color=color, linewidth=2.2, label=name)

    ax.set_xlabel("Total Score", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.set_title(
        "Score CDF: Adaptive Policies vs Constant-θ Baselines",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(80, 370)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"frontier_cdf.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_frontier_delta(
    results: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Bar chart of Δμ (mean - frontier) for each adaptive policy."""
    setup_theme()

    adaptive = results[results["kind"] == "adaptive"].copy()
    adaptive = adaptive.dropna(subset=["delta_mu"])
    adaptive = adaptive.sort_values("delta_mu", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [ADAPTIVE_COLORS.get(n, "#95a5a6") for n in adaptive["policy"]]
    bars = ax.barh(adaptive["policy"], adaptive["delta_mu"], color=colors, edgecolor="white")

    # Add value labels
    for bar, delta in zip(bars, adaptive["delta_mu"]):
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
    ax.set_xlabel("Δμ (Mean − Frontier Mean at matched σ)", fontsize=12)
    ax.set_title(
        "Distance from Constant-θ Pareto Frontier",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(min(adaptive["delta_mu"].min() - 0.3, -1.5), 1.5)
    ax.grid(True, alpha=0.3, axis="x")

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
