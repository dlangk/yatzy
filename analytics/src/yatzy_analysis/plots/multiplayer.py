"""Multiplayer simulation plots (2-player 1v1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .style import COLOR_BLUE, COLOR_ORANGE, COLOR_RED, FONT_AXIS_LABEL, FONT_TITLE, GRID_ALPHA, setup_theme


def plot_score_difference_per_turn(
    turn_totals: NDArray[np.int16],
    scores: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Line plot of P1-P2 score difference per turn with confidence bands."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    # turn_totals: (N, 2, 15) — running totals excl. bonus
    diff = turn_totals[:, 0, :].astype(np.float64) - turn_totals[:, 1, :].astype(np.float64)
    turns = np.arange(1, 16)

    mean = diff.mean(axis=0)
    p5 = np.percentile(diff, 5, axis=0)
    p95 = np.percentile(diff, 95, axis=0)
    p25 = np.percentile(diff, 25, axis=0)
    p75 = np.percentile(diff, 75, axis=0)

    ax.fill_between(turns, p5, p95, alpha=0.15, color=COLOR_BLUE, label="5th–95th pctl")
    ax.fill_between(turns, p25, p75, alpha=0.3, color=COLOR_BLUE, label="25th–75th pctl")
    ax.plot(turns, mean, color=COLOR_BLUE, linewidth=2, label="Mean diff")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Score Difference (P1 − P2)")
    ax.set_title("Score Difference Per Turn")
    ax.legend(loc="upper left")
    ax.set_xticks(turns)

    fig.tight_layout()
    fig.savefig(out_dir / f"mp_score_diff_per_turn.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_score_scatter(
    scores: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Scatter plot of P1 vs P2 final scores colored by winner."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(8, 8))

    s1 = scores[:, 0].astype(np.int32)
    s2 = scores[:, 1].astype(np.int32)

    # Color by winner
    colors = np.where(s1 > s2, COLOR_BLUE, np.where(s2 > s1, COLOR_RED, "#888888"))

    ax.scatter(s1, s2, c=colors, s=0.3, alpha=0.15, rasterized=True, edgecolors="none")

    # Diagonal line
    lo = min(s1.min(), s2.min()) - 5
    hi = max(s1.max(), s2.max()) + 5
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Player 1 Score")
    ax.set_ylabel("Player 2 Score")
    ax.set_title("Final Score Scatter (blue=P1 wins, red=P2 wins)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_dir / f"mp_score_scatter.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_score_difference_histogram(
    scores: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Histogram of P1-P2 final score difference."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    diff = scores[:, 0].astype(np.int32) - scores[:, 1].astype(np.int32)
    mean_diff = diff.mean()

    ax.hist(diff, bins=120, color=COLOR_BLUE, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=1.2, linestyle="--", label="Zero")
    ax.axvline(mean_diff, color=COLOR_RED, linewidth=1.5, linestyle="-", label=f"Mean = {mean_diff:+.1f}")

    ax.set_xlabel("Score Difference (P1 − P2)")
    ax.set_ylabel("Count")
    ax.set_title("Score Difference Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"mp_score_diff_hist.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_win_margin_distribution(
    scores: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Histogram of |P1-P2| for decisive games only."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    diff = scores[:, 0].astype(np.int32) - scores[:, 1].astype(np.int32)
    margins = np.abs(diff[diff != 0])

    mean_m = margins.mean()
    median_m = np.median(margins)

    ax.hist(margins, bins=100, color=COLOR_ORANGE, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(mean_m, color=COLOR_RED, linewidth=1.5, linestyle="-", label=f"Mean = {mean_m:.1f}")
    ax.axvline(median_m, color=COLOR_BLUE, linewidth=1.5, linestyle="--", label=f"Median = {median_m:.1f}")

    ax.set_xlabel("Win Margin (|P1 − P2|)")
    ax.set_ylabel("Count")
    ax.set_title("Win Margin Distribution (decisive games)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"mp_win_margin.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_winner_loser_trajectories(
    turn_totals: NDArray[np.int16],
    scores: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Average cumulative score trajectories for winner vs loser (retroactive classification)."""
    setup_theme()
    fig, ax = plt.subplots(figsize=(10, 6))

    s1 = scores[:, 0].astype(np.int32)
    s2 = scores[:, 1].astype(np.int32)
    tt = turn_totals.astype(np.float64)  # (N, 2, 15)

    # Decisive games only
    decisive = s1 != s2
    tt_d = tt[decisive]
    s1_d = s1[decisive]
    s2_d = s2[decisive]

    # For each game, pick winner/loser trajectory
    p1_wins = s1_d > s2_d
    winner_traj = np.where(p1_wins[:, None], tt_d[:, 0, :], tt_d[:, 1, :])  # (M, 15)
    loser_traj = np.where(p1_wins[:, None], tt_d[:, 1, :], tt_d[:, 0, :])   # (M, 15)

    turns = np.arange(1, 16)

    w_mean = winner_traj.mean(axis=0)
    w_p25 = np.percentile(winner_traj, 25, axis=0)
    w_p75 = np.percentile(winner_traj, 75, axis=0)

    l_mean = loser_traj.mean(axis=0)
    l_p25 = np.percentile(loser_traj, 25, axis=0)
    l_p75 = np.percentile(loser_traj, 75, axis=0)

    ax.fill_between(turns, w_p25, w_p75, alpha=0.2, color=COLOR_BLUE)
    ax.plot(turns, w_mean, color=COLOR_BLUE, linewidth=2, label="Winner (mean)")
    ax.fill_between(turns, l_p25, l_p75, alpha=0.2, color=COLOR_RED)
    ax.plot(turns, l_mean, color=COLOR_RED, linewidth=2, label="Loser (mean)")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative Score (excl. bonus)")
    ax.set_title("Winner vs Loser Score Trajectories")
    ax.legend()
    ax.set_xticks(turns)

    fig.tight_layout()
    fig.savefig(out_dir / f"mp_trajectories.{fmt}", dpi=dpi)
    plt.close(fig)


def plot_all_multiplayer(
    scores: NDArray[np.int16],
    turn_totals: NDArray[np.int16],
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> list[str]:
    """Generate all 5 multiplayer plots. Returns list of filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_score_difference_per_turn(turn_totals, scores, out_dir, dpi=dpi, fmt=fmt)
    plot_score_scatter(scores, out_dir, dpi=dpi, fmt=fmt)
    plot_score_difference_histogram(scores, out_dir, dpi=dpi, fmt=fmt)
    plot_win_margin_distribution(scores, out_dir, dpi=dpi, fmt=fmt)
    plot_winner_loser_trajectories(turn_totals, scores, out_dir, dpi=dpi, fmt=fmt)

    return [
        f"mp_score_diff_per_turn.{fmt}",
        f"mp_score_scatter.{fmt}",
        f"mp_score_diff_hist.{fmt}",
        f"mp_win_margin.{fmt}",
        f"mp_trajectories.{fmt}",
    ]
