"""KDE, CDF, and summary statistics computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from .config import KDE_BANDWIDTH, KDE_POINTS, KDE_RANGE, KDE_SUBSAMPLE


def compute_summary(theta: float, scores: NDArray[np.int32]) -> dict:
    """Compute percentiles, moments, and extremes for one theta."""
    return {
        "theta": theta,
        "n": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": int(scores[0]),
        "max": int(scores[-1]),
        "p5": int(np.percentile(scores, 5)),
        "p10": int(np.percentile(scores, 10)),
        "p25": int(np.percentile(scores, 25)),
        "p50": int(np.percentile(scores, 50)),
        "p75": int(np.percentile(scores, 75)),
        "p90": int(np.percentile(scores, 90)),
        "p95": int(np.percentile(scores, 95)),
        "p99": int(np.percentile(scores, 99)),
        "bot5_avg": float(scores[:5].mean()),
        "top5_avg": float(scores[-5:].mean()),
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
