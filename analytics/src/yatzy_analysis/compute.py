"""KDE, CDF, and summary statistics computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from .config import KDE_BANDWIDTH, KDE_POINTS, KDE_RANGE, KDE_SUBSAMPLE


def compute_cvar(scores: NDArray[np.int32], alpha: float) -> float:
    """Compute CVaR (Expected Shortfall) at level alpha.

    CVaR_alpha = mean of the worst alpha-fraction of scores.
    Scores must be sorted ascending.
    """
    n = max(1, int(alpha * len(scores)))
    return float(scores[:n].mean())


def compute_summary(theta: float, scores: NDArray[np.int32]) -> dict:
    """Compute percentiles, moments, and extremes for one theta."""
    n = len(scores)
    top5_pct_n = max(1, int(0.005 * n))
    return {
        "theta": theta,
        "n": n,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": int(scores[0]),
        "max": int(scores[-1]),
        "p1": int(np.percentile(scores, 1)),
        "p5": int(np.percentile(scores, 5)),
        "p10": int(np.percentile(scores, 10)),
        "p25": int(np.percentile(scores, 25)),
        "p50": int(np.percentile(scores, 50)),
        "p75": int(np.percentile(scores, 75)),
        "p90": int(np.percentile(scores, 90)),
        "p95": int(np.percentile(scores, 95)),
        "p99": int(np.percentile(scores, 99)),
        "p999": int(np.percentile(scores, 99.9)),
        "p9999": int(np.percentile(scores, 99.99)),
        "bot5_avg": float(scores[:5].mean()),
        "top5_avg": float(scores[-5:].mean()),
        "top5_pct_avg": float(scores[-top5_pct_n:].mean()),
        "cvar_1": compute_cvar(scores, 0.01),
        "cvar_5": compute_cvar(scores, 0.05),
        "cvar_10": compute_cvar(scores, 0.10),
    }


def compute_kde(
    theta: float,
    scores: NDArray[np.int32],
    *,
    n_points: int = KDE_POINTS,
    score_range: tuple[int, int] = KDE_RANGE,
    subsample: int = KDE_SUBSAMPLE,
    bandwidth: float = KDE_BANDWIDTH,
) -> pd.DataFrame:
    """Compute KDE density + integrated CDF + survival for one theta.

    Returns DataFrame with columns: theta, score, density, cdf, survival.
    """
    x_grid = np.linspace(score_range[0], score_range[1], n_points)
    dx = x_grid[1] - x_grid[0]

    rng = np.random.default_rng(42)
    sub = rng.choice(scores.astype(float), size=min(subsample, len(scores)), replace=False)
    kde = gaussian_kde(sub, bw_method=bandwidth)
    density = kde(x_grid)

    cdf_vals = np.cumsum(density) * dx
    cdf_vals = cdf_vals / cdf_vals[-1]

    return pd.DataFrame({
        "theta": np.full(n_points, theta, dtype=np.float64),
        "score": x_grid.astype(np.float32),
        "density": density.astype(np.float32),
        "cdf": cdf_vals.astype(np.float32),
        "survival": (1.0 - cdf_vals).astype(np.float32),
    })


def compute_exchange_rates(
    summary_df: pd.DataFrame,
    baseline: float = 0.0,
) -> pd.DataFrame:
    """Compute Marginal Exchange Rates for multiple quantile metrics.

    MER_q(theta) = [mean(0) - mean(theta)] / [q(theta) - q(0)]

    Returns DataFrame with theta and MER columns for each metric.
    """
    base = summary_df[summary_df["theta"] == baseline].iloc[0]
    metrics = ["p75", "p90", "p95", "p99", "max", "top5_avg", "top5_pct_avg"]
    rows = []
    for _, row in summary_df.iterrows():
        t = row["theta"]
        mean_cost = base["mean"] - row["mean"]
        r: dict[str, float] = {"theta": t, "mean_cost": mean_cost}
        for m in metrics:
            gain = row[m] - base[m]
            if abs(gain) < 1e-9:
                r[f"mer_{m}"] = float("inf") if mean_cost > 0 else 0.0
            else:
                r[f"mer_{m}"] = mean_cost / gain
        rows.append(r)
    return pd.DataFrame(rows)


def compute_sdva(
    kde_df: pd.DataFrame,
    theta: float,
    baseline: float = 0.0,
) -> dict:
    """Compute Stochastic Dominance Violation Area (CDF Gain-Loss Decomposition).

    D(x) = F_theta(x) - F_0(x)
    A_worse = integral of max(D(x), 0) dx  (region where theta has more mass below x)
    A_better = integral of max(-D(x), 0) dx (region where theta has less mass below x)
    A_worse - A_better = mean(0) - mean(theta) by identity.

    Returns dict with a_worse, a_better, ratio, x_cross.
    """
    base_cdf = kde_df[kde_df["theta"] == baseline]
    theta_cdf = kde_df[kde_df["theta"] == theta]

    if base_cdf.empty or theta_cdf.empty:
        return {"a_worse": 0.0, "a_better": 0.0, "ratio": 1.0, "x_cross": 0.0}

    scores = base_cdf["score"].values
    dx = float(scores[1] - scores[0])
    d = theta_cdf["cdf"].values - base_cdf["cdf"].values

    worse = np.maximum(d, 0)
    better = np.maximum(-d, 0)
    a_worse = float(np.sum(worse) * dx)
    a_better = float(np.sum(better) * dx)
    ratio = a_worse / a_better if a_better > 1e-9 else float("inf")

    # Find primary crossing point: where D goes from positive (worse) to negative (better)
    # in the main body of the distribution. Filter out edge noise by requiring |D| > threshold.
    sign_changes = np.where(np.diff(np.sign(d)))[0]
    x_cross = float(scores[0])
    if len(sign_changes) > 0:
        # Find crossings where D goes from positive to negative (the meaningful one)
        neg_crossings = [i for i in sign_changes if d[i] > 0 and d[min(i + 1, len(d) - 1)] <= 0]
        if neg_crossings:
            # Pick the one with largest |D| in its neighborhood (most significant crossing)
            best = max(neg_crossings, key=lambda i: abs(d[i]))
            x_cross = float(scores[best])
        else:
            x_cross = float(scores[sign_changes[0]])

    return {
        "a_worse": a_worse,
        "a_better": a_better,
        "ratio": ratio,
        "x_cross": x_cross,
    }


def compute_all_sdva(
    kde_df: pd.DataFrame,
    thetas: list[float],
    baseline: float = 0.0,
) -> pd.DataFrame:
    """Compute SDVA for all thetas. Returns DataFrame."""
    rows = []
    for t in thetas:
        if t == baseline:
            rows.append({"theta": t, "a_worse": 0.0, "a_better": 0.0, "ratio": 1.0, "x_cross": 0.0})
            continue
        r = compute_sdva(kde_df, t, baseline)
        r["theta"] = t
        rows.append(r)
    return pd.DataFrame(rows)


def compute_all(
    scores_dict: dict[float, NDArray[np.int32]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary + KDE for all thetas.

    Returns (summary_df, kde_df).
    """
    summaries = []
    kde_parts = []
    for t in sorted(scores_dict.keys()):
        scores = scores_dict[t]
        summaries.append(compute_summary(t, scores))
        kde_parts.append(compute_kde(t, scores))

    summary_df = pd.DataFrame(summaries)
    kde_df = pd.concat(kde_parts, ignore_index=True)
    return summary_df, kde_df
