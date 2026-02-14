"""Per-category statistics visualizations across the θ sweep."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .style import CMAP, setup_theme

# Ordered category names matching category_id 0-14
CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

UPPER_CATS = set(range(6))
LOWER_CATS = set(range(6, 15))

# Short names for compact displays
SHORT_NAMES = [
    "1s", "2s", "3s", "4s", "5s", "6s",
    "Pair", "2Pair", "3Kind", "4Kind",
    "SmStr", "LgStr", "FHouse", "Chance", "Yatzy",
]


def load_category_stats(csv_path: Path) -> pd.DataFrame:
    """Load category_stats.csv and ensure proper types."""
    df = pd.read_csv(csv_path)
    df["theta"] = df["theta"].round(3)
    return df


def _section_color(cat_id: int) -> str:
    """Return a muted color by section."""
    if cat_id < 6:
        return "#4878CF"  # blue for upper
    return "#D65F5F"  # red for lower


# ── Heatmap helper ───────────────────────────────────────────────────────

def _heatmap_panel(
    df: pd.DataFrame,
    ax: plt.Axes,
    col: str,
    title: str,
    cmap: str,
    annot_fmt: str = ".1f",
    *,
    n_thetas: int,
) -> None:
    """Draw a single heatmap panel on the given axes."""
    pivot = df.pivot(index="category_id", columns="theta", values=col)
    pivot.index = [CATEGORY_NAMES[i] for i in pivot.index]
    theta_labels = [f"{t:.2f}" if t < 1 else f"{t:.1f}" for t in pivot.columns]
    annot = n_thetas <= 21

    sns.heatmap(
        pivot, ax=ax, cmap=cmap,
        xticklabels=theta_labels, yticklabels=True,
        linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"label": title, "shrink": 0.8},
        annot=annot, annot_kws={"size": 7}, fmt=annot_fmt,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("θ", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.axhline(y=6, color="black", linewidth=1.5)


# ── Plot 1a: Mean Score + % of Score Ceiling ────────────────────────────

def plot_category_score_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Heatmap: Mean Score and Score % of Ceiling."""
    setup_theme()
    n_thetas = df["theta"].nunique()

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_thetas * 0.55), 9))
    fig.suptitle(
        "Category Scoring Across θ Sweep",
        fontsize=16, fontweight="bold", y=0.995,
    )

    _heatmap_panel(df, axes[0], "mean_score", "Mean Score", CMAP, n_thetas=n_thetas)
    _heatmap_panel(df, axes[1], "score_pct_ceiling", "Score % of Ceiling", CMAP, n_thetas=n_thetas)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = out_dir / f"category_score_heatmaps{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 1b: Zero Rate + Hit Rate ───────────────────────────────────────

def plot_category_rate_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Heatmap: Zero Rate and Hit Rate."""
    setup_theme()
    n_thetas = df["theta"].nunique()

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_thetas * 0.55), 9))
    fig.suptitle(
        "Category Rates Across θ Sweep",
        fontsize=16, fontweight="bold", y=0.995,
    )

    _heatmap_panel(df, axes[0], "zero_rate", "Zero Rate", CMAP, ".2f", n_thetas=n_thetas)
    _heatmap_panel(df, axes[1], "hit_rate", "Hit Rate", CMAP, n_thetas=n_thetas)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = out_dir / f"category_rate_heatmaps{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 1c: Mean Fill Turn ─────────────────────────────────────────────

def plot_category_fill_turn_heatmaps(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Heatmap: Mean Fill Turn."""
    setup_theme()
    n_thetas = df["theta"].nunique()

    fig, ax = plt.subplots(1, 1, figsize=(max(14, n_thetas * 0.55), 5.5))

    _heatmap_panel(df, ax, "mean_fill_turn", "Mean Fill Turn (1-indexed)", "coolwarm_r", n_thetas=n_thetas)
    fig.suptitle(
        "Category Fill Turn Across θ Sweep",
        fontsize=16, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    path = out_dir / f"category_fill_turn_heatmaps{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Small multiples sparklines ─────────────────────────────────────

def plot_category_sparklines(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    stat: str = "mean_score",
    stat_label: str = "Mean Score",
    suffix: str = "",
    theta_range: tuple[float, float] = (-1.0, 1.0),
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """15 mini line charts in a 3×5 grid, one per category."""
    setup_theme()

    # Limit to theta_range
    df = df[(df["theta"] >= theta_range[0]) & (df["theta"] <= theta_range[1])].copy()

    fig, axes = plt.subplots(3, 5, figsize=(22, 10), sharex=True)
    fig.suptitle(
        f"Per-Category {stat_label} vs θ",
        fontsize=16, fontweight="bold", y=0.98,
    )

    thetas = sorted(df["theta"].unique())

    for cat_id in range(15):
        row, col = divmod(cat_id, 5)
        ax = axes[row, col]

        cat_data = df[df["category_id"] == cat_id].sort_values("theta")
        color = _section_color(cat_id)

        ax.plot(cat_data["theta"], cat_data[stat], color=color, linewidth=2)
        ax.fill_between(
            cat_data["theta"], cat_data[stat],
            alpha=0.15, color=color,
        )

        # Mark θ=0 with a dot
        t0_val = cat_data[cat_data["theta"] == 0.0][stat].values
        if len(t0_val) > 0:
            ax.plot(0.0, t0_val[0], "o", color=color, markersize=5, zorder=5)

        ax.set_title(CATEGORY_NAMES[cat_id], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

        if row == 2:
            ax.set_xlabel("θ", fontsize=9)
        if col == 0:
            ax.set_ylabel(stat_label, fontsize=9)

        # Clean up grid
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(thetas), max(thetas))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"category_sparklines_{stat}{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Faceted bar charts ─────────────────────────────────────────────

def plot_category_bars(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    selected_thetas: list[float] | None = None,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Bar charts comparing mean_score per category for selected θ values."""
    setup_theme()

    if selected_thetas is None:
        selected_thetas = [-1.0, -0.07, 0.0, 0.07, 1.0]

    sub = df[df["theta"].isin(selected_thetas)].copy()
    sub["category_name"] = sub["category_id"].map(lambda i: SHORT_NAMES[i])

    # Use coolwarm palette sampled at the selected theta positions
    norm = plt.Normalize(vmin=min(selected_thetas), vmax=max(selected_thetas))
    colors = [CMAP(norm(t)) for t in selected_thetas]
    palette = {f"θ={t:.2f}": c for t, c in zip(selected_thetas, colors)}
    sub["θ"] = sub["theta"].map(lambda t: f"θ={t:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Mean score bars
    ax = axes[0]
    sns.barplot(
        data=sub, x="category_name", y="mean_score", hue="θ",
        palette=palette, ax=ax, edgecolor="white", linewidth=0.5,
    )
    ax.set_title("Mean Score by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.axvline(x=5.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(2.5, ax.get_ylim()[1] * 0.95, "UPPER", ha="center", fontsize=10, alpha=0.5)
    ax.text(10, ax.get_ylim()[1] * 0.95, "LOWER", ha="center", fontsize=10, alpha=0.5)
    ax.legend(title="", loc="upper right", fontsize=9)

    # Zero rate bars
    ax = axes[1]
    sns.barplot(
        data=sub, x="category_name", y="zero_rate", hue="θ",
        palette=palette, ax=ax, edgecolor="white", linewidth=0.5,
    )
    ax.set_title("Zero Rate by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Zero Rate", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.axvline(x=5.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.legend(title="", loc="upper left", fontsize=9)

    plt.tight_layout()
    path = out_dir / f"category_bars{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: Slope / bump chart for fill turn reordering ────────────────────

def plot_category_slope(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    t_left: float = 0.0,
    t_right: float = 3.0,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Slope chart showing how category fill order changes between two θ values."""
    setup_theme()

    left = df[df["theta"] == t_left][["category_id", "mean_fill_turn"]].set_index("category_id")
    right = df[df["theta"] == t_right][["category_id", "mean_fill_turn"]].set_index("category_id")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use coolwarm based on the direction of change
    max_shift = 0
    shifts = {}
    for cat_id in range(15):
        l = left.loc[cat_id, "mean_fill_turn"]
        r = right.loc[cat_id, "mean_fill_turn"]
        shifts[cat_id] = r - l
        max_shift = max(max_shift, abs(r - l))

    shift_norm = plt.Normalize(vmin=-max_shift, vmax=max_shift)

    for cat_id in range(15):
        l = left.loc[cat_id, "mean_fill_turn"]
        r = right.loc[cat_id, "mean_fill_turn"]
        shift = shifts[cat_id]
        color = CMAP(shift_norm(shift))
        lw = 2.0 + abs(shift) * 0.3  # thicker for bigger shifts

        ax.plot([0, 1], [l, r], color=color, linewidth=lw, alpha=0.8, zorder=2)

        # Labels on left and right
        ax.text(
            -0.02, l, f"{CATEGORY_NAMES[cat_id]} ({l:.1f})",
            ha="right", va="center", fontsize=9, fontweight="bold",
            color=color,
        )
        ax.text(
            1.02, r, f"({r:.1f}) {CATEGORY_NAMES[cat_id]}",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=color,
        )

        # Dots at endpoints
        ax.plot(0, l, "o", color=color, markersize=7, zorder=3)
        ax.plot(1, r, "o", color=color, markersize=7, zorder=3)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.5, 15.5)
    ax.invert_yaxis()  # Turn 1 at top

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"θ = {t_left}", f"θ = {t_right}"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Fill Turn (earlier = top)", fontsize=12)
    ax.set_title(
        "How θ Reorders Category Filling\n"
        "Blue = filled earlier at high θ, Red = filled later",
        fontsize=14, fontweight="bold",
    )

    # Light grid for turn reference
    for turn in range(1, 16):
        ax.axhline(y=turn, color="#e0e0e0", linewidth=0.5, zorder=1)

    ax.set_yticks(range(1, 16))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    path = out_dir / f"category_slope{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 5: Multi-column slope chart ──────────────────────────────────────

def plot_category_slope_dense(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    selected_thetas: list[float] | None = None,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Slope chart with multiple θ columns showing full fill-order trajectory."""
    setup_theme()

    if selected_thetas is None:
        selected_thetas = [-3.0, -1.0, -0.10, 0.0, 0.10, 1.00, 3.0]

    # Get fill turns per theta
    cols = {}
    for t in selected_thetas:
        sub = df[df["theta"] == t][["category_id", "mean_fill_turn"]].set_index("category_id")
        if len(sub) == 0:
            continue
        cols[t] = sub["mean_fill_turn"]

    available_thetas = [t for t in selected_thetas if t in cols]
    n_cols = len(available_thetas)
    if n_cols < 2:
        print("  Skipping dense slope: need at least 2 thetas with data")
        return

    fig, ax = plt.subplots(figsize=(4 + 3 * n_cols, 11))

    # X positions: evenly spaced
    x_pos = np.linspace(0, 1, n_cols)

    # Color by total shift (first → last)
    first_t = available_thetas[0]
    last_t = available_thetas[-1]
    max_shift = 0
    shifts = {}
    for cat_id in range(15):
        s = cols[last_t].loc[cat_id] - cols[first_t].loc[cat_id]
        shifts[cat_id] = s
        max_shift = max(max_shift, abs(s))

    shift_norm = plt.Normalize(vmin=-max_shift, vmax=max_shift)

    for cat_id in range(15):
        color = CMAP(shift_norm(shifts[cat_id]))
        lw = 1.5 + abs(shifts[cat_id]) * 0.3

        ys = [cols[t].loc[cat_id] for t in available_thetas]
        ax.plot(x_pos, ys, color=color, linewidth=lw, alpha=0.8, zorder=2)

        # Dots at each column
        for xi, yi in zip(x_pos, ys):
            ax.plot(xi, yi, "o", color=color, markersize=5, zorder=3)

        # Labels on left and right
        ax.text(
            x_pos[0] - 0.02, ys[0], f"{CATEGORY_NAMES[cat_id]} ({ys[0]:.1f})",
            ha="right", va="center", fontsize=8, fontweight="bold", color=color,
        )
        ax.text(
            x_pos[-1] + 0.02, ys[-1], f"({ys[-1]:.1f}) {CATEGORY_NAMES[cat_id]}",
            ha="left", va="center", fontsize=8, fontweight="bold", color=color,
        )

    ax.set_xlim(x_pos[0] - 0.25, x_pos[-1] + 0.25)
    ax.set_ylim(0.5, 15.5)
    ax.invert_yaxis()

    # Theta labels as x-axis
    theta_labels = []
    for t in available_thetas:
        if t == 0.0:
            theta_labels.append("θ=0\n(EV)")
        elif t == int(t) and abs(t) >= 1:
            theta_labels.append(f"θ={int(t)}")
        else:
            theta_labels.append(f"θ={t:g}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(theta_labels, fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Fill Turn (earlier = top)", fontsize=12)
    ax.set_title(
        "Category Fill Order Across θ Spectrum\n"
        "Blue = filled earlier at high θ, Red = filled later",
        fontsize=14, fontweight="bold",
    )

    for turn in range(1, 16):
        ax.axhline(y=turn, color="#e0e0e0", linewidth=0.5, zorder=1)

    ax.set_yticks(range(1, 16))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    path = out_dir / f"category_slope_dense{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 6: Fill-turn heatmap ─────────────────────────────────────────────

def plot_fill_turn_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str = "",
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Heatmap: categories (rows) × all θ values, colored by mean fill turn."""
    setup_theme()

    pivot = df.pivot(index="category_id", columns="theta", values="mean_fill_turn")
    pivot.index = [CATEGORY_NAMES[i] for i in pivot.index]

    # Theta labels
    theta_labels = []
    for t in pivot.columns:
        if t == 0.0:
            theta_labels.append("0")
        elif t == int(t) and abs(t) >= 1:
            theta_labels.append(f"{int(t)}")
        else:
            theta_labels.append(f"{t:.2f}")

    fig, ax = plt.subplots(figsize=(max(16, len(pivot.columns) * 0.55), 8))

    sns.heatmap(
        pivot, ax=ax, cmap="coolwarm_r",
        xticklabels=theta_labels, yticklabels=True,
        linewidths=0.3, linecolor="#e0e0e0",
        cbar_kws={"label": "Mean Fill Turn (lower = earlier)", "shrink": 0.8},
        annot=True, annot_kws={"size": 7}, fmt=".1f",
    )

    ax.set_title(
        "Category Fill Order Across All θ Values",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("θ (risk parameter)", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=10)

    # Section divider between upper and lower
    ax.axhline(y=6, color="black", linewidth=2)

    plt.tight_layout()
    path = out_dir / f"category_fill_turn_heatmap{suffix}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Orchestrator ───────────────────────────────────────────────────────────

def plot_all_category_stats(
    csv_path: Path,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Generate all category visualizations."""
    df = load_category_stats(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating category stats plots...")
    plot_category_score_heatmaps(df, out_dir, dpi=dpi, fmt=fmt)
    plot_category_rate_heatmaps(df, out_dir, dpi=dpi, fmt=fmt)
    plot_category_fill_turn_heatmaps(df, out_dir, dpi=dpi, fmt=fmt)

    # Sparklines for each key stat
    for stat, label in [
        ("mean_score", "Mean Score"),
        ("zero_rate", "Zero Rate"),
        ("mean_fill_turn", "Mean Fill Turn"),
    ]:
        plot_category_sparklines(df, out_dir, stat=stat, stat_label=label, dpi=dpi, fmt=fmt)

    plot_category_bars(df, out_dir, dpi=dpi, fmt=fmt)
    plot_category_slope(df, out_dir, dpi=dpi, fmt=fmt)
    plot_category_slope_dense(df, out_dir, dpi=dpi, fmt=fmt)
    plot_fill_turn_heatmap(df, out_dir, dpi=dpi, fmt=fmt)

    # Zoomed versions: θ ∈ [-0.2, 0.2] — the practical range
    print("Generating zoomed category stats plots (|θ| ≤ 0.2)...")
    zdf = df[df["theta"].abs() <= 0.2].copy()
    sfx = "_zoomed"
    plot_category_score_heatmaps(zdf, out_dir, suffix=sfx, dpi=dpi, fmt=fmt)
    plot_category_rate_heatmaps(zdf, out_dir, suffix=sfx, dpi=dpi, fmt=fmt)
    plot_category_fill_turn_heatmaps(zdf, out_dir, suffix=sfx, dpi=dpi, fmt=fmt)
    for stat, label in [
        ("mean_score", "Mean Score"),
        ("zero_rate", "Zero Rate"),
        ("mean_fill_turn", "Mean Fill Turn"),
    ]:
        plot_category_sparklines(
            zdf, out_dir, stat=stat, stat_label=label, suffix=sfx, dpi=dpi, fmt=fmt,
        )
    plot_category_bars(
        zdf, out_dir,
        selected_thetas=[-0.20, -0.05, 0.0, 0.05, 0.20],
        suffix=sfx, dpi=dpi, fmt=fmt,
    )
    plot_category_slope(
        zdf, out_dir, t_left=-0.2, t_right=0.2, suffix=sfx, dpi=dpi, fmt=fmt,
    )
    plot_category_slope_dense(
        zdf, out_dir,
        selected_thetas=[-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
        suffix=sfx, dpi=dpi, fmt=fmt,
    )
    plot_fill_turn_heatmap(zdf, out_dir, suffix=sfx, dpi=dpi, fmt=fmt)
    print("Done.")
