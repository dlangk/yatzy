"""Win rate analysis plots: θ vs win rate, conditional breakdown, PMF overlay."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import FONT_AXIS_LABEL, FONT_LEGEND, FONT_TITLE, GRID_ALPHA, setup_theme


def load_winrate_data(
    winrate_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load winrate_results.csv and winrate_conditional.csv."""
    results = pd.read_csv(winrate_dir / "winrate_results.csv")
    cond = pd.read_csv(winrate_dir / "winrate_conditional.csv")
    return results, cond


def plot_winrate_curve(
    results: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Win rate (%) vs θ with 50% parity and 51% threshold lines."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate baseline and challengers
    challengers = results[results["theta"] != 0.0].copy()
    challengers = challengers.sort_values("theta")

    # Color by win/loss relative to 50%
    colors = []
    for _, row in challengers.iterrows():
        wr = row["win_rate"]
        if wr > 0.51:
            colors.append("#27ae60")  # strong green
        elif wr > 0.50:
            colors.append("#2ecc71")  # light green
        elif wr > 0.49:
            colors.append("#e67e22")  # orange
        else:
            colors.append("#e74c3c")  # red

    # Plot curve
    ax.plot(
        challengers["theta"],
        challengers["win_rate"] * 100,
        color="#3498db",
        linewidth=2.0,
        zorder=3,
        alpha=0.7,
    )

    # Plot points
    ax.scatter(
        challengers["theta"],
        challengers["win_rate"] * 100,
        c=colors,
        s=80,
        zorder=4,
        edgecolors="white",
        linewidths=0.8,
    )

    # Annotate best
    best_idx = challengers["win_rate"].idxmax()
    best = challengers.loc[best_idx]
    ax.annotate(
        f"θ={best['theta']:+.3f}\n{best['win_rate']*100:.2f}%",
        (best["theta"], best["win_rate"] * 100),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
    )

    # Reference lines
    ax.axhline(y=50, color="gray", linewidth=1.5, linestyle="--", alpha=0.7, label="50% (parity)")
    ax.axhline(y=51, color="#e74c3c", linewidth=1.2, linestyle=":", alpha=0.7, label="51% (H₁ threshold)")
    ax.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("θ (risk parameter)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Win Rate vs θ=0 (%)", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Head-to-Head Win Rate: Constant-θ vs EV-Optimal (θ=0)",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_dir / f"winrate_vs_theta.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_conditional_winrate(
    results: pd.DataFrame,
    cond: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Grouped bars: win rate by opponent score band for top θ values + θ=0."""
    setup_theme()

    challengers = results[results["theta"] != 0.0].copy()
    if challengers.empty:
        return

    # Pick top 3 by win rate + the worst for contrast
    top3 = challengers.nlargest(3, "win_rate")["theta"].tolist()
    worst = challengers.nsmallest(1, "win_rate")["theta"].tolist()
    selected = sorted(set(top3 + worst))

    cond_sel = cond[cond["theta"].isin(selected)].copy()
    if cond_sel.empty:
        return

    # Create band labels
    cond_sel["band"] = cond_sel.apply(
        lambda r: f"{int(r['band_lo'])}-{int(r['band_hi'])}", axis=1
    )

    bands = cond_sel["band"].unique()
    n_bands = len(bands)
    n_thetas = len(selected)

    fig, ax = plt.subplots(figsize=(14, 7))

    bar_width = 0.8 / n_thetas
    x = np.arange(n_bands)

    colors = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6", "#e67e22"]

    for i, theta in enumerate(selected):
        subset = cond_sel[cond_sel["theta"] == theta].set_index("band")
        vals = [subset.loc[b, "win_rate"] * 100 if b in subset.index else 0 for b in bands]
        offset = (i - n_thetas / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width,
            label=f"θ={theta:+.3f}",
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(y=50, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Opponent Score Band", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Win Rate (%)", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Conditional Win Rate by Opponent Score Band",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / f"winrate_conditional.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_pmf_comparison(
    results: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Overlaid PMFs for best win-rate θ and θ=0, shading advantage region.

    Uses the mean/std from the CSV to reconstruct approximate Gaussian densities,
    since we don't store raw PMFs. For exact PMFs, this would need the score data.
    This is a useful approximation given the CLT nature of Yatzy scores.
    """
    setup_theme()

    baseline = results[results["theta"] == 0.0]
    challengers = results[results["theta"] != 0.0]
    if baseline.empty or challengers.empty:
        return

    best_idx = challengers["win_rate"].idxmax()
    best = challengers.loc[best_idx]
    bl = baseline.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(80, 380)

    # Approximate Gaussian density
    def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    y_bl = gauss(x, bl["mean"], bl["std"])
    y_best = gauss(x, best["mean"], best["std"])

    ax.fill_between(x, y_bl, alpha=0.25, color="#3498db", label=f"θ=0 (μ={bl['mean']:.1f}, σ={bl['std']:.1f})")
    ax.plot(x, y_bl, color="#3498db", linewidth=2.0)

    ax.fill_between(x, y_best, alpha=0.25, color="#e74c3c",
                     label=f"θ={best['theta']:+.3f} (μ={best['mean']:.1f}, σ={best['std']:.1f})")
    ax.plot(x, y_best, color="#e74c3c", linewidth=2.0)

    # Shade region where challenger density exceeds baseline
    higher = y_best > y_bl
    ax.fill_between(x, y_bl, y_best, where=higher, alpha=0.3, color="#27ae60",
                     label="Challenger > Baseline density")

    ax.set_xlabel("Total Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Probability Density", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        f"Score Distribution: θ=0 vs Best Win-Rate θ={best['theta']:+.3f}",
        fontsize=FONT_TITLE,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out_dir / f"winrate_pmf_overlay.{fmt}", dpi=dpi)
    plt.close(fig)


def generate_winrate_plots(base_path: str = ".", fmt: str = "png", dpi: int = 200) -> None:
    """Generate all win rate plots from outputs/winrate/ data."""
    base = Path(base_path)
    winrate_dir = base / "outputs" / "winrate"
    out_dir = base / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (winrate_dir / "winrate_results.csv").exists():
        print(f"No win rate data found at {winrate_dir}/winrate_results.csv")
        print("Run: just winrate")
        return

    print("Loading win rate data...")
    results, cond = load_winrate_data(winrate_dir)
    n_challengers = len(results) - 1  # exclude baseline
    print(f"  {n_challengers} challengers + 1 baseline")

    print("Plotting win rate curve...")
    plot_winrate_curve(results, out_dir, dpi=dpi, fmt=fmt)

    print("Plotting conditional win rates...")
    plot_conditional_winrate(results, cond, out_dir, dpi=dpi, fmt=fmt)

    print("Plotting PMF comparison...")
    plot_pmf_comparison(results, out_dir, dpi=dpi, fmt=fmt)

    print(f"Done. Plots saved to {out_dir}/winrate_*.{fmt}")
