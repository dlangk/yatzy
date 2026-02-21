"""Decision sensitivity plots: flip rates, θ distributions, gap analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from .style import COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, FONT_AXIS_LABEL, FONT_LEGEND, FONT_TITLE, GRID_ALPHA, setup_theme


def load_sensitivity_data(
    base_path: str = ".",
) -> tuple[pl.DataFrame, dict, dict]:
    """Load CSV + JSON outputs from decision_sensitivity binary."""
    scenario_dir = Path(base_path) / "outputs" / "scenarios"

    csv_path = scenario_dir / "decision_sensitivity.csv"
    flips_path = scenario_dir / "decision_sensitivity_flips.json"
    summary_path = scenario_dir / "decision_sensitivity_summary.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run yatzy-decision-sensitivity first."
        )

    df = pl.read_csv(csv_path)

    flips_data: list[dict] = []
    if flips_path.exists():
        with open(flips_path) as f:
            flips_data = json.load(f)

    summary_data: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary_data = json.load(f)

    return df, flips_data, summary_data


def plot_flip_rates(df: pl.DataFrame, out_dir: Path, dpi: int = 200, fmt: str = "png") -> None:
    """Plot 1: Grouped bar chart of flip rate by decision_type and game_phase."""
    setup_theme()

    # Compute flip rates
    grouped = (
        df.group_by(["decision_type", "game_phase"])
        .agg(
            pl.col("has_flip").count().alias("total"),
            pl.col("has_flip").sum().alias("flips"),
        )
    )
    grouped = grouped.with_columns(
        (pl.col("flips") / pl.col("total")).alias("flip_rate")
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    phases = ["early", "mid", "late"]
    dtypes = ["reroll1", "reroll2", "category"]
    x = np.arange(len(phases))
    width = 0.25
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]

    for i, dt in enumerate(dtypes):
        rates = []
        for phase in phases:
            row = grouped.filter(
                (pl.col("decision_type") == dt) & (pl.col("game_phase") == phase)
            )
            if len(row) > 0:
                rates.append(row["flip_rate"][0] * 100)
            else:
                rates.append(0)
        ax.bar(x + i * width, rates, width, label=dt, color=colors[i], alpha=0.85)

    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Flip Rate (%)")
    ax.set_title("Decision Flip Rates by Type and Game Phase")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{p}\n(turns {r})" for p, r in
                        zip(phases, ["1-5", "6-10", "11-15"])])
    ax.legend(title="Decision Type")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_flip_rates.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_flip_theta_distribution(
    df: pl.DataFrame, out_dir: Path, dpi: int = 200, fmt: str = "png"
) -> None:
    """Plot 2: Histogram of flip_theta values, stacked by decision_type."""
    setup_theme()

    flips = df.filter(pl.col("has_flip"))
    if flips.is_empty():
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    dtypes = ["reroll1", "reroll2", "category"]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]

    # Get unique theta values for bin edges
    thetas = sorted(flips["flip_theta"].unique().to_list())
    if len(thetas) < 2:
        return

    # Create bin edges between theta values
    edges = []
    for i in range(len(thetas)):
        if i == 0:
            edges.append(thetas[i] - (thetas[1] - thetas[0]) / 2)
        else:
            edges.append((thetas[i - 1] + thetas[i]) / 2)
    edges.append(thetas[-1] + (thetas[-1] - thetas[-2]) / 2)

    bottom = np.zeros(len(thetas))
    for dt, color in zip(dtypes, colors):
        subset = flips.filter(pl.col("decision_type") == dt)
        counts = []
        for t in thetas:
            counts.append(len(subset.filter(pl.col("flip_theta") == t)))
        counts_arr = np.array(counts, dtype=float)
        ax.bar(thetas, counts_arr, width=np.diff(edges).min() * 0.8,
               bottom=bottom, label=dt, color=color, alpha=0.85)
        bottom += counts_arr

    ax.set_xlabel("θ at First Flip")
    ax.set_ylabel("Number of Decisions")
    ax.set_title("Distribution of θ Where Decisions First Change")
    ax.legend(title="Decision Type")

    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_flip_theta.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_gap_distributions(
    df: pl.DataFrame, out_dir: Path, dpi: int = 200, fmt: str = "png"
) -> None:
    """Plot 3: Box/violin plots of gap_at_flip by decision_type."""
    setup_theme()

    flips = df.filter(pl.col("has_flip"))
    if flips.is_empty():
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gap at flip point
    dtypes = ["reroll1", "reroll2", "category"]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]
    palette = dict(zip(dtypes, colors))

    present_dtypes = [dt for dt in dtypes if dt in flips["decision_type"].unique().to_list()]

    if present_dtypes:
        # Convert to pandas for seaborn boxplot
        subset_df = flips.filter(pl.col("decision_type").is_in(present_dtypes)).to_pandas()
        sns.boxplot(
            data=subset_df,
            x="decision_type",
            y="gap_at_flip",
            hue="decision_type",
            order=present_dtypes,
            hue_order=present_dtypes,
            palette=palette,
            legend=False,
            ax=ax1,
        )
        ax1.set_xlabel("Decision Type")
        ax1.set_ylabel("Gap at Flip Point")
        ax1.set_title("Gap at θ Where Decision Flips")

        # Gap at θ=0
        sns.boxplot(
            data=subset_df,
            x="decision_type",
            y="gap_at_theta0",
            hue="decision_type",
            order=present_dtypes,
            hue_order=present_dtypes,
            palette=palette,
            legend=False,
            ax=ax2,
        )
        ax2.set_xlabel("Decision Type")
        ax2.set_ylabel("Gap at θ=0")
        ax2.set_title("θ=0 Gap for Decisions That Eventually Flip")

    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_gaps.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_turn_heatmap(
    df: pl.DataFrame, out_dir: Path, dpi: int = 200, fmt: str = "png"
) -> None:
    """Plot 4: Turn × decision type heatmap of flip rates."""
    setup_theme()

    dtypes = ["reroll1", "reroll2", "category"]

    # Build pivot: 15 turns × 3 decision types as numpy array
    import numpy as np
    pivot = np.zeros((15, len(dtypes)))

    for turn in range(15):
        for j, dt in enumerate(dtypes):
            subset = df.filter(
                (pl.col("turn") == turn) & (pl.col("decision_type") == dt)
            )
            if len(subset) > 0:
                pivot[turn, j] = float(subset["has_flip"].mean()) * 100

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Flip Rate (%)"},
        ax=ax,
        linewidths=0.5,
        xticklabels=dtypes,
        yticklabels=[f"Turn {i+1}" for i in range(15)],
    )
    ax.set_xlabel("Decision Type")
    ax.set_ylabel("")
    ax.set_title("Decision Sensitivity by Turn and Type\n(% of decisions that flip for theta in [0, 0.2])")

    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_heatmap.{fmt}", dpi=dpi)
    plt.close(fig)


def print_top_sensitive(df: pl.DataFrame, n: int = 20) -> None:
    """Console output: top N most sensitive decisions (smallest θ=0 gap among flips)."""
    flips = df.filter(pl.col("has_flip"))
    if flips.is_empty():
        print("No flips found.")
        return

    flips = flips.with_columns(
        pl.col("gap_at_theta0").abs().alias("abs_gap")
    )
    top = flips.sort("abs_gap").head(n)

    print(f"\n{'='*95}")
    print(f"  Top {n} Most Sensitive Decisions (smallest θ=0 gap among flips)")
    print(f"{'='*95}")
    print(
        f"{'Turn':>5} {'Dice':<16} {'Type':>10} {'θ=0 Action':>20} "
        f"{'Flip Action':>20} {'Flip θ':>8} {'θ=0 Gap':>10}"
    )
    print("-" * 100)
    for row in top.iter_rows(named=True):
        print(
            f"{int(row['turn'])+1:>5} {str(row['dice']):<16} {row['decision_type']:>10} "
            f"{row['theta_0_action']:>20} {row['flip_action']:>20} "
            f"{row['flip_theta']:>8.3f} {row['gap_at_theta0']:>10.3f}"
        )


def plot_all_sensitivity(
    base_path: str = ".",
    out_dir: Path | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Generate all sensitivity plots and print console summary."""
    df, _flips_data, summary_data = load_sensitivity_data(base_path)

    if out_dir is None:
        out_dir = Path(base_path) / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print summary stats
    if summary_data:
        total = summary_data.get("analyzed_decisions", 0)
        flips = summary_data.get("flip_count", 0)
        rate = summary_data.get("flip_rate", 0)
        print(f"\nDecision sensitivity analysis:")
        print(f"  Analyzed: {total:,d} unique decision points")
        print(f"  Flips:    {flips:,d} ({rate*100:.1f}%)")
        print()

        by_type = summary_data.get("by_decision_type", [])
        if by_type:
            print(f"  {'Type':>10} {'Total':>8} {'Flips':>8} {'Rate':>8}")
            print(f"  {'-'*38}")
            for t in by_type:
                print(
                    f"  {t['decision_type']:>10} {t['total']:>8,d} "
                    f"{t['flips']:>8,d} {t['flip_rate']*100:>7.1f}%"
                )
            print()

    plot_flip_rates(df, out_dir, dpi=dpi, fmt=fmt)
    print(f"  sensitivity_flip_rates.{fmt}")

    plot_flip_theta_distribution(df, out_dir, dpi=dpi, fmt=fmt)
    print(f"  sensitivity_flip_theta.{fmt}")

    plot_gap_distributions(df, out_dir, dpi=dpi, fmt=fmt)
    print(f"  sensitivity_gaps.{fmt}")

    plot_turn_heatmap(df, out_dir, dpi=dpi, fmt=fmt)
    print(f"  sensitivity_heatmap.{fmt}")

    print_top_sensitive(df)
