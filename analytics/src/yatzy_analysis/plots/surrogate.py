"""Surrogate model Pareto frontier and accuracy plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    FONT_AXIS_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    GRID_ALPHA,
    FIG_WIDE,
    FIG_SQUARE,
    save_fig,
    setup_theme,
    CATEGORY_SHORT,
)

DECISION_COLORS = {
    "category": "#2ca02c",
    "reroll1": "#1f77b4",
    "reroll2": "#ff7f0e",
}

MODEL_MARKERS = {
    "dt": "o",
    "mlp": "s",
    "baseline": "x",
}


def plot_all_surrogate(
    output_dir: Path,
    results_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Generate all surrogate plots from results CSVs."""
    setup_theme()
    paths: list[Path] = []

    # Load results
    all_results: dict[str, pd.DataFrame] = {}
    for dtype in ["category", "reroll1", "reroll2"]:
        p = results_dir / f"results_{dtype}.csv"
        if p.exists():
            all_results[dtype] = pd.read_csv(p)

    if not all_results:
        return paths

    # 1. Pareto frontier
    paths.append(_plot_pareto(all_results, output_dir, dpi=dpi, fmt=fmt))

    # 2. DT vs MLP scatter
    paths.append(_plot_dt_vs_mlp(all_results, output_dir, dpi=dpi, fmt=fmt))

    # 3. Feature importance
    for dtype in all_results:
        fi_path = results_dir / f"feature_importance_{dtype}.npz"
        if fi_path.exists():
            paths.append(
                _plot_feature_importance(fi_path, dtype, output_dir, dpi=dpi, fmt=fmt)
            )
            break  # One importance plot is sufficient (category is most interesting)

    # 4. Accuracy by turn heatmap
    paths.append(_plot_accuracy_heatmap(all_results, results_dir, output_dir, dpi=dpi, fmt=fmt))

    return paths


def _plot_pareto(
    all_results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for dtype, df in all_results.items():
        color = DECISION_COLORS[dtype]
        for mtype, marker in MODEL_MARKERS.items():
            subset = df[df["model_type"] == mtype]
            if subset.empty:
                continue
            ax.scatter(
                subset["n_params"],
                subset["ev_loss"],
                c=color,
                marker=marker,
                s=60,
                alpha=0.7,
                label=f"{dtype} ({mtype})" if mtype != "baseline" else None,
                zorder=3,
            )

    # Pareto frontier line
    pareto_path = output_dir.parent / "surrogate" / "pareto_frontier.csv"
    if pareto_path.exists():
        pareto = pd.read_csv(pareto_path)
        pareto_sorted = pareto.sort_values("n_params")
        ax.step(
            pareto_sorted["n_params"],
            pareto_sorted["ev_loss"],
            where="post",
            color="black",
            linewidth=2,
            alpha=0.5,
            label="Pareto frontier",
            zorder=2,
        )

    # Reference lines
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="EV loss = 0.1")
    ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="EV loss = 1.0")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("EV loss per game (points)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Surrogate Model Pareto Frontier", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA)

    return save_fig(fig, output_dir, "surrogate_pareto", dpi=dpi, fmt=fmt)


def _plot_dt_vs_mlp(
    all_results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, (dtype, df) in enumerate(all_results.items()):
        ax = axes[i]
        for mtype, marker in [("dt", "o"), ("mlp", "s")]:
            subset = df[(df["model_type"] == mtype)]
            if subset.empty:
                continue
            ax.scatter(
                subset["n_params"],
                subset["ev_loss"],
                marker=marker,
                s=80,
                alpha=0.7,
                label=mtype.upper(),
                zorder=3,
            )
            # Connect with line
            sorted_sub = subset.sort_values("n_params")
            ax.plot(
                sorted_sub["n_params"],
                sorted_sub["ev_loss"],
                alpha=0.3,
                linewidth=1,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters", fontsize=FONT_AXIS_LABEL)
        ax.set_title(dtype, fontsize=FONT_TITLE)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(fontsize=FONT_LEGEND)

    axes[0].set_ylabel("EV loss per game", fontsize=FONT_AXIS_LABEL)
    fig.suptitle("Decision Trees vs MLPs by Decision Type", fontsize=FONT_TITLE + 2)
    fig.tight_layout()

    return save_fig(fig, output_dir, "surrogate_dt_vs_mlp", dpi=dpi, fmt=fmt)


def _plot_feature_importance(
    fi_path: Path,
    dtype: str,
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    data = np.load(fi_path, allow_pickle=True)
    importances = data["importances"]
    names = data["names"]

    # Sort by importance
    idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(idx))
    ax.barh(y_pos, importances[idx], color=DECISION_COLORS.get(dtype, "#1f77b4"), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Random Forest)", fontsize=FONT_AXIS_LABEL)
    ax.set_title(f"Top Features — {dtype} decisions", fontsize=FONT_TITLE)
    ax.grid(True, axis="x", alpha=GRID_ALPHA)

    return save_fig(fig, output_dir, "surrogate_feature_importance", dpi=dpi, fmt=fmt)


def _plot_accuracy_heatmap(
    all_results: dict[str, pd.DataFrame],
    results_dir: Path,
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    """Heatmap of accuracy by turn for selected models."""
    # Select a few representative models per type
    selected = ["dt_d3", "dt_d10", "dt_full", "mlp_64_32", "mlp_128_64_32"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, dtype in enumerate(["category", "reroll1", "reroll2"]):
        ax = axes[i]

        # For the heatmap, we need per-turn accuracy which isn't in the CSV.
        # Use a simple accuracy-bar comparison instead.
        df = all_results.get(dtype)
        if df is None:
            continue

        models = df[df["name"].isin(selected)].sort_values("n_params")
        if models.empty:
            # Fall back to showing all non-baseline models
            models = df[df["model_type"] != "baseline"].sort_values("n_params")

        y_pos = np.arange(len(models))
        colors = [
            "#1f77b4" if row["model_type"] == "dt" else "#ff7f0e"
            for _, row in models.iterrows()
        ]

        ax.barh(y_pos, models["accuracy"], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models["name"])
        ax.set_xlabel("Accuracy", fontsize=FONT_AXIS_LABEL)
        ax.set_title(dtype, fontsize=FONT_TITLE)
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", alpha=GRID_ALPHA)

    fig.suptitle("Model Accuracy Comparison", fontsize=FONT_TITLE + 2)
    fig.tight_layout()

    return save_fig(fig, output_dir, "surrogate_accuracy_by_turn", dpi=dpi, fmt=fmt)


# ── Game-level evaluation plots ──────────────────────────────────────────


def plot_game_level_results(
    csv_path: Path,
    output_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Generate game-level surrogate evaluation plots."""
    setup_theme()
    df = pd.read_csv(csv_path)
    paths: list[Path] = []

    paths.append(_plot_params_vs_mean(df, output_dir, dpi=dpi, fmt=fmt))
    paths.append(_plot_score_summary(df, output_dir, dpi=dpi, fmt=fmt))

    return paths


def _plot_params_vs_mean(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    """Log10(total_params) vs mean score with reference lines."""
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    # Separate DT and MLP models
    dt_mask = df["name"].str.startswith("dt_")
    mlp_mask = df["name"].str.startswith("mlp_")
    heur_mask = df["name"] == "heuristic"

    # Plot DTs
    dt_df = df[dt_mask].sort_values("total_params")
    if not dt_df.empty:
        ax.scatter(
            dt_df["total_params"], dt_df["mean"],
            marker="o", s=100, c="#1f77b4", zorder=4, label="Decision Tree",
        )
        ax.plot(
            dt_df["total_params"], dt_df["mean"],
            color="#1f77b4", alpha=0.4, linewidth=1.5, zorder=3,
        )
        # Label each point
        for _, row in dt_df.iterrows():
            ax.annotate(
                row["name"], (row["total_params"], row["mean"]),
                textcoords="offset points", xytext=(8, -4),
                fontsize=7, alpha=0.7,
            )

    # Plot MLPs
    mlp_df = df[mlp_mask].sort_values("total_params")
    if not mlp_df.empty:
        ax.scatter(
            mlp_df["total_params"], mlp_df["mean"],
            marker="s", s=100, c="#ff7f0e", zorder=4, label="MLP",
        )
        ax.plot(
            mlp_df["total_params"], mlp_df["mean"],
            color="#ff7f0e", alpha=0.4, linewidth=1.5, zorder=3,
        )

    # Plot heuristic
    heur_df = df[heur_mask]
    if not heur_df.empty:
        heur_mean = heur_df.iloc[0]["mean"]
        ax.axhline(
            y=heur_mean, color="#2ca02c", linestyle=":", alpha=0.7,
            label=f"Heuristic ({heur_mean:.0f})", zorder=2,
        )

    # Reference lines
    ax.axhline(y=248.4, color="black", linestyle="--", alpha=0.5, label="Optimal (248.4)")
    ax.axhspan(220, 230, color="gold", alpha=0.12, label="Human range (220-230)")

    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (3 models combined)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean game score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Surrogate Model Size vs Game Performance", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)

    return save_fig(fig, output_dir, "surrogate_game_scores", dpi=dpi, fmt=fmt)


def _plot_score_summary(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int,
    fmt: str,
) -> Path:
    """Box-plot style summary of score distributions for selected models."""
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    # Sort by total_params for display
    df_sorted = df.sort_values("total_params")

    y_pos = np.arange(len(df_sorted))
    names = df_sorted["name"].values

    # Draw whisker-style range bars
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        color = "#1f77b4" if row["name"].startswith("dt_") else (
            "#ff7f0e" if row["name"].startswith("mlp_") else "#2ca02c"
        )
        # p5-p95 range
        ax.barh(i, row["p95"] - row["p5"], left=row["p5"], height=0.5,
                color=color, alpha=0.3, zorder=2)
        # p25-p75 range
        ax.barh(i, row["p75"] - row["p25"], left=row["p25"], height=0.5,
                color=color, alpha=0.6, zorder=3)
        # Median line
        ax.plot([row["p50"], row["p50"]], [i - 0.25, i + 0.25],
                color=color, linewidth=2, zorder=4)
        # Mean marker
        ax.scatter([row["mean"]], [i], color=color, marker="D", s=40, zorder=5)
        # Params annotation
        params = int(row["total_params"])
        ax.annotate(
            f"{params:,d}", (row["p95"] + 2, i),
            fontsize=7, alpha=0.6, va="center",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Game Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score Distributions by Model", fontsize=FONT_TITLE)
    ax.axvline(x=248.4, color="black", linestyle="--", alpha=0.4, label="Optimal mean")
    ax.axvspan(220, 230, color="gold", alpha=0.1, label="Human range")
    ax.legend(fontsize=FONT_LEGEND, loc="lower right")
    ax.grid(True, axis="x", alpha=GRID_ALPHA)

    fig.tight_layout()
    return save_fig(fig, output_dir, "surrogate_score_distributions", dpi=dpi, fmt=fmt)


# ── Diagnostic plots ────────────────────────────────────────────────────


def plot_data_scaling_curves(
    scaling_data: dict[str, list[dict]],
    output_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Plot EV loss vs training set size for each decision type."""
    setup_theme()
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for dtype, results in scaling_data.items():
        color = DECISION_COLORS.get(dtype, "#333333")
        games = [r["n_games"] for r in results]
        losses = [r["ev_loss"] for r in results]
        ax.plot(games, losses, "o-", color=color, label=dtype, linewidth=2, markersize=8)

    ax.set_xlabel("Training games", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("EV loss per game (dt_full)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Data Scaling: EV Loss vs Training Set Size", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    return save_fig(fig, output_dir, "surrogate_data_scaling", dpi=dpi, fmt=fmt)


def plot_feature_ablation_bars(
    ablation_data: dict[str, list[dict]],
    output_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Bar chart of delta EV loss when removing each feature group."""
    setup_theme()
    n_types = len(ablation_data)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6), sharey=True)
    if n_types == 1:
        axes = [axes]

    for i, (dtype, results) in enumerate(ablation_data.items()):
        ax = axes[i]
        # Skip the "all_features" baseline
        ablated = [r for r in results if r["group"] != "all_features"]
        groups = [r["group"] for r in ablated]
        deltas = [r["delta_ev_loss"] for r in ablated]

        colors = ["#d62728" if d > 0.01 else "#2ca02c" if d < -0.01 else "#7f7f7f" for d in deltas]
        y_pos = np.arange(len(groups))
        ax.barh(y_pos, deltas, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel("Δ EV loss (higher = more important)", fontsize=FONT_AXIS_LABEL)
        ax.set_title(dtype, fontsize=FONT_TITLE)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=GRID_ALPHA)

    fig.suptitle("Feature Ablation: Impact of Removing Feature Groups", fontsize=FONT_TITLE + 2)
    fig.tight_layout()
    return save_fig(fig, output_dir, "surrogate_feature_ablation", dpi=dpi, fmt=fmt)


def plot_forward_selection_elbow(
    selection_data: dict[str, list[dict]],
    output_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Plot EV loss vs number of features (forward selection elbow)."""
    setup_theme()
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for dtype, results in selection_data.items():
        color = DECISION_COLORS.get(dtype, "#333333")
        steps = [r["n_features"] for r in results]
        losses = [r["ev_loss"] for r in results]
        ax.plot(steps, losses, "o-", color=color, label=dtype, linewidth=2, markersize=6)

        # Annotate first few features
        for r in results[:5]:
            ax.annotate(
                r["feature_name"], (r["n_features"], r["ev_loss"]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=7, alpha=0.7, color=color,
            )

    ax.set_xlabel("Number of features", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("EV loss per game (dt_d15)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Forward Feature Selection: Diminishing Returns", fontsize=FONT_TITLE)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(True, alpha=GRID_ALPHA)

    return save_fig(fig, output_dir, "surrogate_forward_selection", dpi=dpi, fmt=fmt)


def plot_error_analysis(
    error_data: dict[str, dict],
    output_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Plot error analysis: gap distribution + turn error rates."""
    setup_theme()
    n_types = len(error_data)
    fig, axes = plt.subplots(2, n_types, figsize=(6 * n_types, 10))
    if n_types == 1:
        axes = axes.reshape(-1, 1)

    for i, (dtype, data) in enumerate(error_data.items()):
        color = DECISION_COLORS.get(dtype, "#333333")

        if data["n_wrong"] == 0:
            axes[0, i].text(0.5, 0.5, "No errors", ha="center", va="center")
            axes[1, i].text(0.5, 0.5, "No errors", ha="center", va="center")
            continue

        # Top: turn error rates
        ax_turn = axes[0, i]
        turns = sorted(data["turn_error_rates"].keys())
        rates = [data["turn_error_rates"][t]["rate"] * 100 for t in turns]
        ax_turn.bar(turns, rates, color=color, alpha=0.7)
        ax_turn.set_xlabel("Turn", fontsize=FONT_AXIS_LABEL)
        ax_turn.set_ylabel("Error rate (%)", fontsize=FONT_AXIS_LABEL)
        ax_turn.set_title(f"{dtype}: Error Rate by Turn", fontsize=FONT_TITLE)
        ax_turn.grid(True, axis="y", alpha=GRID_ALPHA)

        # Bottom: gap distribution annotation
        ax_gap = axes[1, i]
        gap_stats = data["gap_stats"]
        stats_text = (
            f"Total errors: {data['n_wrong']:,d} / {data['n_total']:,d}\n"
            f"Error rate: {data['error_rate']:.2%}\n"
            f"EV loss: {data['ev_loss']:.4f}\n\n"
            f"Gap of errors:\n"
            f"  Mean: {gap_stats['mean']:.3f}\n"
            f"  Median: {gap_stats['median']:.3f}\n"
            f"  P90: {gap_stats['p90']:.3f}\n"
            f"  P99: {gap_stats['p99']:.3f}\n"
            f"  Max: {gap_stats['max']:.3f}\n"
            f"  Near-zero (<0.1): {gap_stats['near_zero_frac']:.1%}\n\n"
            f"Near bonus threshold: {data['near_bonus_fraction']:.1%}"
        )
        ax_gap.text(0.1, 0.9, stats_text, transform=ax_gap.transAxes,
                    fontsize=10, verticalalignment="top", fontfamily="monospace")
        ax_gap.set_title(f"{dtype}: Error Statistics", fontsize=FONT_TITLE)
        ax_gap.axis("off")

    fig.tight_layout()
    return save_fig(fig, output_dir, "surrogate_error_analysis", dpi=dpi, fmt=fmt)
