"""Score distribution modality analysis: why Yatzy scores aren't normal."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import gaussian_kde, norm

from .style import (
    CATEGORY_NAMES,
    CATEGORY_SHORT,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    FIG_QUAD,
    FIG_WIDE,
    FONT_ANNOTATION,
    FONT_AXIS_LABEL,
    FONT_SUPTITLE,
    FONT_TICK,
    FONT_TITLE,
    GRID_ALPHA,
    save_fig,
    setup_theme,
)


def plot_histogram_vs_kde(
    scores: NDArray[np.int32],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Histogram (integer bins) vs KDE at multiple bandwidths.

    Left: histogram + KDE(bw=0.04) showing artifact spikes.
    Right: histogram + KDE at bw=1, 3, scott showing smoothed reality.
    """
    setup_theme()
    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE, sharey=True)

    score_min, score_max = int(scores.min()), int(scores.max())
    bins = np.arange(score_min - 0.5, score_max + 1.5, 1)
    x_grid = np.linspace(score_min, score_max, 2000)

    # Subsample for KDE
    rng = np.random.default_rng(42)
    sub = rng.choice(scores.astype(float), size=min(100_000, len(scores)), replace=False)

    # Left panel: narrow bandwidth
    ax = axes[0]
    ax.hist(scores, bins=bins, density=True, color=COLOR_BLUE, alpha=0.3, label="Histogram (1-pt bins)")
    kde_narrow = gaussian_kde(sub, bw_method=0.04)
    ax.plot(x_grid, kde_narrow(x_grid), color=COLOR_RED, lw=1.5, label="KDE bw=0.04")
    ax.set_title("Narrow KDE (bw=0.04) — artifact spikes", fontsize=FONT_TITLE)
    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_AXIS_LABEL)
    ax.legend(fontsize=FONT_ANNOTATION)
    ax.set_xlim(score_min, score_max)

    # Right panel: reasonable bandwidths
    ax = axes[1]
    ax.hist(scores, bins=bins, density=True, color=COLOR_BLUE, alpha=0.3, label="Histogram (1-pt bins)")
    for bw, color, label in [
        (1.0, COLOR_ORANGE, "KDE bw=1"),
        (3.0, COLOR_GREEN, "KDE bw=3"),
        ("scott", COLOR_RED, "KDE scott"),
    ]:
        kde = gaussian_kde(sub, bw_method=bw)
        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=label)
    ax.set_title("Reasonable bandwidths — true shape", fontsize=FONT_TITLE)
    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.legend(fontsize=FONT_ANNOTATION)
    ax.set_xlim(score_min, score_max)

    fig.suptitle(
        "Score Distribution: Histogram vs KDE Bandwidth Choice",
        fontsize=FONT_SUPTITLE, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return save_fig(fig, out_dir, "modality_histogram_vs_kde", dpi=dpi, fmt=fmt)


def plot_bonus_yatzy_decomposition(
    game_data: dict,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """2x2 grid: score distributions for the 4 (bonus, yatzy) sub-populations."""
    setup_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIG_QUAD, sharex=True, sharey=True)

    total = game_data["total_scores"].astype(np.int32)
    cat_scores = game_data["category_scores"]
    got_bonus = game_data["got_bonus"]
    got_yatzy = cat_scores[:, 14] > 0
    n = game_data["num_games"]

    combos = [
        (False, False, "No Bonus, No Yatzy"),
        (False, True, "No Bonus, Yatzy"),
        (True, False, "Bonus, No Yatzy"),
        (True, True, "Bonus + Yatzy"),
    ]

    score_min, score_max = int(total.min()), int(total.max())
    bins = np.arange(score_min - 0.5, score_max + 1.5, 1)
    x_grid = np.linspace(score_min, score_max, 500)

    colors = [COLOR_RED, COLOR_ORANGE, COLOR_BLUE, COLOR_GREEN]

    for idx, (b_val, y_val, label) in enumerate(combos):
        ax = axes[idx // 2][idx % 2]
        mask = (got_bonus == b_val) & (got_yatzy == y_val)
        count = int(mask.sum())
        subset = total[mask]

        if count < 10:
            ax.set_title(f"{label}\n(n={count}, {100*count/n:.1f}%)", fontsize=FONT_TITLE)
            ax.text(0.5, 0.5, "Too few games", ha="center", va="center", transform=ax.transAxes)
            continue

        frac = count / n
        mu = float(subset.mean())
        sigma = float(subset.std())

        ax.hist(subset, bins=bins, density=True, color=colors[idx], alpha=0.4, label="Histogram")

        # Normal fit overlay
        ax.plot(x_grid, norm.pdf(x_grid, mu, sigma), color="black", lw=1.5, ls="--", label="Normal fit")

        ax.set_title(f"{label}\n(n={count:,}, {100*frac:.1f}%)", fontsize=FONT_TITLE)
        ax.annotate(
            f"mean={mu:.1f}\nstd={sigma:.1f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=FONT_ANNOTATION,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.legend(fontsize=FONT_ANNOTATION, loc="upper left")

    for ax in axes[1]:
        ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=FONT_AXIS_LABEL)

    fig.suptitle(
        "Score Distribution Decomposed by Bonus x Yatzy",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, out_dir, "modality_bonus_yatzy", dpi=dpi, fmt=fmt)


def plot_category_pmf_grid(
    game_data: dict,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """3x5 small multiples: per-category score PMF."""
    setup_theme()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))

    cat_scores = game_data["category_scores"]  # (N, 15)
    n = game_data["num_games"]

    # Binary categories get highlighted
    binary_cats = {10, 11, 14}  # Small Straight, Large Straight, Yatzy

    for i in range(15):
        ax = axes[i // 5][i % 5]
        col = cat_scores[:, i]
        unique_vals, counts = np.unique(col, return_counts=True)
        probs = counts / n

        color = COLOR_ORANGE if i in binary_cats else COLOR_BLUE
        ax.bar(unique_vals, probs, width=0.8, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)

        ax.set_title(CATEGORY_SHORT[i], fontsize=FONT_TICK, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, min(1.05, probs.max() * 1.3))

        # Annotate zero-score fraction for binary cats
        zero_frac = float((col == 0).sum()) / n
        if zero_frac > 0.01:
            ax.annotate(
                f"P(0)={zero_frac:.0%}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    # Shared labels
    for ax in axes[2]:
        ax.set_xlabel("Score", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(score=k)", fontsize=8)

    fig.suptitle(
        "Per-Category Score Distributions (orange = binary categories)",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    fig.tight_layout()
    return save_fig(fig, out_dir, "modality_category_pmf", dpi=dpi, fmt=fmt)


def plot_variance_decomposition(
    cov_matrix: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Horizontal bar chart: variance contribution per category + covariance."""
    setup_theme()
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    n_vars = len(labels)
    variances = np.diag(cov_matrix)
    total_var = cov_matrix.sum()  # Var(sum) = sum of all cov entries
    sum_var = variances.sum()
    total_cov = total_var - sum_var  # 2 * sum of off-diagonal covariances

    # Sort by variance descending
    order = np.argsort(variances)[::-1]
    sorted_labels = [labels[i] for i in order]
    sorted_vars = variances[order]
    sorted_pct = 100 * sorted_vars / total_var

    # Per-variable total covariance contribution: sum of row i (off-diagonal)
    cov_contributions = np.array([
        cov_matrix[i, :].sum() - cov_matrix[i, i] for i in range(n_vars)
    ])
    sorted_cov = cov_contributions[order]
    sorted_cov_pct = 100 * sorted_cov / total_var

    y = np.arange(n_vars)
    bar_height = 0.35

    bars1 = ax.barh(y + bar_height / 2, sorted_pct, bar_height, color=COLOR_BLUE, alpha=0.7, label="Var(X_i)")
    bars2 = ax.barh(y - bar_height / 2, sorted_cov_pct, bar_height, color=COLOR_ORANGE, alpha=0.7, label="Cov contribution")

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, fontsize=FONT_TICK)
    ax.set_xlabel("% of Total Score Variance", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        f"Variance Decomposition: Var(Total)={total_var:.1f}  "
        f"(ΣVar={sum_var:.1f} + ΣCov={total_cov:.1f})",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.legend(fontsize=FONT_ANNOTATION, loc="lower right")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=GRID_ALPHA)

    # Annotate raw variance values
    for i, (pct, var_val) in enumerate(zip(sorted_pct, sorted_vars)):
        if pct > 1:
            ax.text(pct + 0.3, i + bar_height / 2, f"{var_val:.0f}", va="center", fontsize=7)

    fig.tight_layout()
    return save_fig(fig, out_dir, "modality_variance_decomposition", dpi=dpi, fmt=fmt)


def plot_mixture_waterfall(
    mixture_df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Waterfall chart: mean score contribution of binary events."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate by each binary factor
    factors = [
        ("bonus", "Upper Bonus (+50)"),
        ("yatzy", "Yatzy (+50)"),
        ("small_straight", "Small Straight (+15)"),
        ("large_straight", "Large Straight (+20)"),
    ]

    items = []
    for col, label in factors:
        yes_rows = mixture_df.filter(pl.col(col) == "yes")
        no_rows = mixture_df.filter(pl.col(col) == "no")

        hit_rate = float(yes_rows["fraction"].sum())
        # Weighted mean of sub-populations that hit vs miss
        if not yes_rows.is_empty() and not no_rows.is_empty():
            mean_yes = float(
                (yes_rows["mean"] * yes_rows["fraction"]).sum() / yes_rows["fraction"].sum()
            )
            mean_no = float(
                (no_rows["mean"] * no_rows["fraction"]).sum() / no_rows["fraction"].sum()
            )
            delta = mean_yes - mean_no
        else:
            delta = 0.0
        items.append((label, hit_rate, delta))

    labels = [it[0] for it in items]
    hit_rates = [it[1] for it in items]
    deltas = [it[2] for it in items]

    x = np.arange(len(labels))
    colors = [COLOR_GREEN if d > 0 else COLOR_RED for d in deltas]

    bars = ax.bar(x, deltas, color=colors, alpha=0.7, edgecolor="white", width=0.6)

    # Annotate
    for i, (bar, rate, delta) in enumerate(zip(bars, hit_rates, deltas)):
        y_pos = delta + (1 if delta > 0 else -1)
        ax.text(
            i, y_pos, f"hit={rate:.0%}\n\u0394={delta:+.1f}",
            ha="center", va="bottom" if delta > 0 else "top",
            fontsize=FONT_ANNOTATION, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Mean Score Difference (hit - miss)", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Impact of Binary Categories on Total Score",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.tight_layout()
    return save_fig(fig, out_dir, "modality_mixture_waterfall", dpi=dpi, fmt=fmt)


def plot_all_modality(
    game_data: dict,
    scores: NDArray[np.int32],
    mixture_df: pl.DataFrame,
    cov_matrix: np.ndarray,
    means: np.ndarray,
    labels: list[str],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Generate all modality analysis plots."""
    paths = []
    paths.append(plot_histogram_vs_kde(scores, out_dir, dpi=dpi, fmt=fmt))
    paths.append(plot_bonus_yatzy_decomposition(game_data, out_dir, dpi=dpi, fmt=fmt))
    paths.append(plot_category_pmf_grid(game_data, out_dir, dpi=dpi, fmt=fmt))
    paths.append(plot_variance_decomposition(cov_matrix, means, labels, out_dir, dpi=dpi, fmt=fmt))
    paths.append(plot_mixture_waterfall(mixture_df, out_dir, dpi=dpi, fmt=fmt))
    return paths
