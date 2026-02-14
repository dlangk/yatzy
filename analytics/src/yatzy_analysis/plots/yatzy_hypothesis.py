"""Yatzy hypothesis testing: conditional hit rate analysis across θ values.

Three plots testing whether risk-seeking (high θ) sacrifices Yatzy universally
or only dumps it in poor games. Includes max-policy as a distinct strategy.

1. Conditional hit rate by score band (grouped bars)
2. Unconditional vs top-5% tail hit rate (line plot)
3. Dump gap: mean score when Yatzy hit vs missed (paired bars)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .style import CMAP, setup_theme

# Max-policy gets a distinct color (green)
MAX_POLICY_COLOR = "#2CA02C"


def load_conditional_csv(csv_path: Path) -> pd.DataFrame:
    """Load yatzy_conditional.csv. theta column is string (may contain 'max_policy')."""
    df = pd.read_csv(csv_path, dtype={"theta": str})
    return df


def _parse_strategies(df: pd.DataFrame) -> tuple[list[float], bool]:
    """Extract sorted numeric thetas and whether max_policy is present."""
    raw = df["theta"].unique()
    thetas = sorted(float(t) for t in raw if t != "max_policy")
    has_max = "max_policy" in raw
    return thetas, has_max


def _strategy_label(s: str) -> str:
    if s == "max_policy":
        return "max-policy"
    t = float(s)
    if t == 0.0:
        return "θ=0 (EV)"
    if t == int(t) and abs(t) >= 1:
        return f"θ={int(t)}"
    return f"θ={t:g}"


def _strategy_colors(thetas: list[float], has_max: bool) -> dict[str, tuple]:
    """Map strategy label → color. Thetas use coolwarm, max-policy gets green."""
    norm = plt.Normalize(vmin=0, vmax=max(thetas) if thetas else 1.0)
    colors = {}
    for t in thetas:
        key = f"{t:.3f}"
        colors[key] = CMAP(norm(t))
    if has_max:
        colors["max_policy"] = MAX_POLICY_COLOR
    return colors


def _ordered_strategies(thetas: list[float], has_max: bool) -> list[str]:
    """Return strategy keys in display order: thetas ascending, then max_policy."""
    keys = [f"{t:.3f}" for t in thetas]
    if has_max:
        keys.append("max_policy")
    return keys


# ── Plot 1: Conditional hit rate by score band ────────────────────────────

def plot_conditional_hit_rate(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Grouped bar chart: X=score bands, groups=strategy, Y=Yatzy hit rate."""
    setup_theme()

    score_bands = ["<200", "200-220", "220-240", "240-260", "260-280", "280-300", "300+", "top5pct"]
    band_df = df[df["band"].isin(score_bands)].copy()

    thetas, has_max = _parse_strategies(df)
    colors = _strategy_colors(thetas, has_max)
    strategies = _ordered_strategies(thetas, has_max)

    fig, ax = plt.subplots(figsize=(16, 8))

    n_bands = len(score_bands)
    n_strats = len(strategies)
    bar_width = 0.8 / n_strats
    x = np.arange(n_bands)

    for j, strat in enumerate(strategies):
        strat_data = band_df[band_df["theta"] == strat]
        rates = []
        for band in score_bands:
            row = strat_data[strat_data["band"] == band]
            rates.append(row["yatzy_hit_rate"].values[0] * 100 if len(row) > 0 else 0)

        offset = (j - n_strats / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, rates, bar_width,
            label=_strategy_label(strat), color=colors[strat],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(score_bands, fontsize=11)
    ax.set_xlabel("Score Band", fontsize=12)
    ax.set_ylabel("Yatzy Hit Rate (%)", fontsize=12)
    ax.set_title(
        "Conditional Yatzy Hit Rate by Score Band",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="", loc="upper left", fontsize=9, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    # Vertical separator before top5pct
    ax.axvline(x=n_bands - 1.5, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.text(
        n_bands - 1, ax.get_ylim()[1] * 0.95, "dynamic\nthreshold",
        ha="center", fontsize=9, alpha=0.5, style="italic",
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"yatzy_conditional_bars.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Unconditional vs tail hit rate ────────────────────────────────

def plot_unconditional_vs_tail(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Line plot: X=θ, two lines (unconditional + top-5% tail), max-policy as markers."""
    setup_theme()

    thetas, has_max = _parse_strategies(df)

    # Numeric θ data
    theta_strs = [f"{t:.3f}" for t in thetas]
    all_df = df[(df["band"] == "all") & (df["theta"].isin(theta_strs))].copy()
    all_df["theta_f"] = all_df["theta"].astype(float)
    all_df = all_df.sort_values("theta_f")

    top5_df = df[(df["band"] == "top5pct") & (df["theta"].isin(theta_strs))].copy()
    top5_df["theta_f"] = top5_df["theta"].astype(float)
    top5_df = top5_df.sort_values("theta_f")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        all_df["theta_f"], all_df["yatzy_hit_rate"] * 100,
        "o-", color="#2166AC", linewidth=2.5, markersize=8,
        label="Unconditional (all games)",
    )
    ax.plot(
        top5_df["theta_f"], top5_df["yatzy_hit_rate"] * 100,
        "s--", color="#B2182B", linewidth=2.5, markersize=8,
        label="Top 5% tail only",
    )

    # Max-policy as horizontal markers
    if has_max:
        mp_all = df[(df["band"] == "all") & (df["theta"] == "max_policy")]
        mp_top5 = df[(df["band"] == "top5pct") & (df["theta"] == "max_policy")]
        x_max = max(thetas) * 1.15 if thetas else 1.0

        if len(mp_all) > 0:
            rate = mp_all["yatzy_hit_rate"].values[0] * 100
            ax.plot(x_max, rate, "D", color=MAX_POLICY_COLOR, markersize=12, zorder=5)
            ax.annotate(
                f"max-policy\n{rate:.1f}%", (x_max, rate),
                textcoords="offset points", xytext=(12, 0), fontsize=9,
                color=MAX_POLICY_COLOR, fontweight="bold",
            )
        if len(mp_top5) > 0:
            rate = mp_top5["yatzy_hit_rate"].values[0] * 100
            ax.plot(x_max, rate, "D", color=MAX_POLICY_COLOR, markersize=12, zorder=5)
            ax.annotate(
                f"max-policy top5%\n{rate:.1f}%", (x_max, rate),
                textcoords="offset points", xytext=(12, 0), fontsize=9,
                color=MAX_POLICY_COLOR, fontweight="bold",
            )

    ax.set_xlabel("θ (risk parameter)", fontsize=12)
    ax.set_ylabel("Yatzy Hit Rate (%)", fontsize=12)
    ax.set_title(
        "Unconditional vs Top-5% Tail Yatzy Hit Rate",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(True, alpha=0.3)

    # Annotate θ endpoints
    if len(all_df) >= 2:
        t0_all = all_df.iloc[0]["yatzy_hit_rate"] * 100
        tmax_all = all_df.iloc[-1]["yatzy_hit_rate"] * 100
        ax.annotate(
            f"{t0_all:.1f}%", (all_df.iloc[0]["theta_f"], t0_all),
            textcoords="offset points", xytext=(10, 10), fontsize=9, color="#2166AC",
        )
        ax.annotate(
            f"{tmax_all:.1f}%", (all_df.iloc[-1]["theta_f"], tmax_all),
            textcoords="offset points", xytext=(10, -15), fontsize=9, color="#2166AC",
        )

    if len(top5_df) >= 2:
        t0_top5 = top5_df.iloc[0]["yatzy_hit_rate"] * 100
        tmax_top5 = top5_df.iloc[-1]["yatzy_hit_rate"] * 100
        ax.annotate(
            f"{t0_top5:.1f}%", (top5_df.iloc[0]["theta_f"], t0_top5),
            textcoords="offset points", xytext=(10, -15), fontsize=9, color="#B2182B",
        )
        ax.annotate(
            f"{tmax_top5:.1f}%", (top5_df.iloc[-1]["theta_f"], tmax_top5),
            textcoords="offset points", xytext=(10, 10), fontsize=9, color="#B2182B",
        )

    plt.tight_layout()
    path = out_dir / f"yatzy_unconditional_vs_tail.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: The dump gap ─────────────────────────────────────────────────

def plot_dump_gap(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Paired bars: mean total score when Yatzy hit vs missed, per strategy."""
    setup_theme()

    thetas, has_max = _parse_strategies(df)
    strategies = _ordered_strategies(thetas, has_max)
    colors = _strategy_colors(thetas, has_max)

    all_df = df[df["band"] == "all"].copy()
    # Order by strategy list
    all_df["_order"] = all_df["theta"].map({s: i for i, s in enumerate(strategies)})
    all_df = all_df.sort_values("_order")

    labels = [_strategy_label(s) for s in strategies]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(strategies))
    width = 0.35

    hit_scores = all_df["mean_score_hit"].values
    miss_scores = all_df["mean_score_miss"].values

    ax.bar(
        x - width / 2, hit_scores, width,
        label="Yatzy HIT", color="#2166AC", edgecolor="white", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, miss_scores, width,
        label="Yatzy MISSED", color="#B2182B", edgecolor="white", linewidth=0.5,
    )

    # Annotate gaps
    for i, (h, m) in enumerate(zip(hit_scores, miss_scores)):
        gap = h - m
        ax.annotate(
            f"+{gap:.0f}", (x[i], max(h, m) + 1),
            ha="center", fontsize=9, fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Mean Total Score", fontsize=12)
    ax.set_title(
        "The Dump Gap: Mean Score When Yatzy Hit vs Missed",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to start from a reasonable minimum
    all_vals = np.concatenate([hit_scores, miss_scores])
    ymin = max(0, min(all_vals) - 20)
    ax.set_ylim(ymin, max(all_vals) + 15)

    plt.tight_layout()
    path = out_dir / f"yatzy_dump_gap.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Orchestrator ─────────────────────────────────────────────────────────

def plot_all_yatzy_hypothesis(
    csv_path: Path,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Generate all three hypothesis test visualizations."""
    df = load_conditional_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Yatzy hypothesis plots...")
    plot_conditional_hit_rate(df, out_dir, dpi=dpi, fmt=fmt)
    plot_unconditional_vs_tail(df, out_dir, dpi=dpi, fmt=fmt)
    plot_dump_gap(df, out_dir, dpi=dpi, fmt=fmt)
    print("Done.")
