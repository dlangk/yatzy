"""Read exact density evolution PMFs and produce summary + KDE DataFrames.

The density JSONs (outputs/density/density_*.json) contain exact score PMFs
from forward DP — zero variance, no KDE smoothing needed. This module converts
them into the same summary.parquet and kde.parquet schemas that the MC pipeline
produces, so all downstream plot modules work unchanged.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

from .config import KDE_POINTS, KDE_RANGE


def discover_density_thetas(base_path: str = ".") -> list[float]:
    """Scan outputs/density/ for density_*.json files.

    Args:
        base_path: Repository root directory (default ".").

    Returns:
        Sorted list of theta values found as density_<theta>.json files.
    """
    density_dir = Path(base_path) / "outputs" / "density"
    thetas: list[float] = []
    if not density_dir.is_dir():
        return thetas
    for p in density_dir.iterdir():
        m = re.match(r"density_(.+)\.json$", p.name)
        if m:
            try:
                thetas.append(float(m.group(1)))
            except ValueError:
                pass
    return sorted(thetas)


def read_density_json(path: Path) -> dict:
    """Read and parse a single density JSON file."""
    with open(path) as f:
        return json.load(f)


def _pmf_arrays(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (scores, probs) arrays from density JSON pmf field."""
    pmf = data["pmf"]
    scores = np.array([entry[0] for entry in pmf], dtype=np.float64)
    probs = np.array([entry[1] for entry in pmf], dtype=np.float64)
    return scores, probs


def _cdf_quantile(scores: np.ndarray, cdf: np.ndarray, q: float) -> float:
    """Invert CDF to find the score at quantile q (0-1).

    Returns the smallest score s such that CDF(s) >= q.
    For integer scores this gives the standard quantile definition.
    """
    idx = np.searchsorted(cdf, q, side="left")
    idx = min(idx, len(scores) - 1)
    return float(scores[idx])


def _conditional_expectation_below(
    scores: np.ndarray, probs: np.ndarray, cdf: np.ndarray, alpha: float,
) -> float:
    """Compute E[X | X <= VaR_alpha] = CVaR at level alpha.

    For exact PMF: sum over scores where CDF <= alpha, weighted by prob,
    then divide by alpha.
    """
    # Find the VaR threshold
    var_idx = np.searchsorted(cdf, alpha, side="left")
    var_idx = min(var_idx, len(scores) - 1)

    # All probability mass strictly below the VaR score
    if var_idx == 0:
        return float(scores[0])

    # Sum prob*score for indices 0..var_idx
    # But we need exactly alpha probability mass — handle the boundary
    mass_below = cdf[var_idx - 1] if var_idx > 0 else 0.0
    ev_below = float(np.sum(scores[:var_idx] * probs[:var_idx]))

    # Remaining mass needed from the VaR score itself
    remaining = alpha - mass_below
    if remaining > 0:
        ev_below += float(scores[var_idx]) * remaining

    return ev_below / alpha if alpha > 0 else float(scores[0])


def _conditional_expectation_above(
    scores: np.ndarray, probs: np.ndarray, cdf: np.ndarray, alpha: float,
) -> float:
    """Compute E[X | X >= quantile(1-alpha)].

    Upper tail expectation: mean of the top alpha fraction.
    """
    threshold_q = 1.0 - alpha
    var_idx = np.searchsorted(cdf, threshold_q, side="left")
    var_idx = min(var_idx, len(scores) - 1)

    # Mass at and above var_idx
    mass_above = 1.0 - (cdf[var_idx - 1] if var_idx > 0 else 0.0)
    ev_above = float(np.sum(scores[var_idx:] * probs[var_idx:]))

    if mass_above > 1e-15:
        return ev_above / mass_above
    return float(scores[-1])


def compute_summary_from_pmf(theta: float, data: dict) -> dict:
    """Compute summary dict with same 34 columns as compute_summary() from exact PMF."""
    scores, probs = _pmf_arrays(data)

    # Normalize (should already sum to 1, but be safe)
    total = probs.sum()
    if total > 0:
        probs = probs / total

    # Build CDF
    cdf = np.cumsum(probs)

    # Moments from JSON (exact)
    mean = data["mean"]
    std = data["std_dev"]
    variance = data["variance"]

    # Higher moments from PMF
    mu = mean
    m2 = variance
    m3 = float(np.sum(probs * (scores - mu) ** 3))
    m4 = float(np.sum(probs * (scores - mu) ** 4))
    skewness = m3 / (std**3) if std > 0 else 0.0
    kurt = (m4 / (std**4) - 3.0) if std > 0 else 0.0  # excess kurtosis

    # Percentiles via CDF inversion
    p1 = _cdf_quantile(scores, cdf, 0.01)
    p5 = _cdf_quantile(scores, cdf, 0.05)
    p10 = _cdf_quantile(scores, cdf, 0.10)
    p25 = _cdf_quantile(scores, cdf, 0.25)
    p50 = _cdf_quantile(scores, cdf, 0.50)
    p75 = _cdf_quantile(scores, cdf, 0.75)
    p90 = _cdf_quantile(scores, cdf, 0.90)
    p95 = _cdf_quantile(scores, cdf, 0.95)
    p99 = _cdf_quantile(scores, cdf, 0.99)
    p995 = _cdf_quantile(scores, cdf, 0.995)
    p999 = _cdf_quantile(scores, cdf, 0.999)
    p9999 = _cdf_quantile(scores, cdf, 0.9999)

    # Score range
    nonzero = probs > 0
    if nonzero.any():
        score_min = int(scores[nonzero][0])
        score_max = int(scores[nonzero][-1])
    else:
        score_min = int(scores[0])
        score_max = int(scores[-1])

    # CVaR (Expected Shortfall) — lower tail
    cvar_1 = _conditional_expectation_below(scores, probs, cdf, 0.01)
    cvar_5 = _conditional_expectation_below(scores, probs, cdf, 0.05)
    cvar_10 = _conditional_expectation_below(scores, probs, cdf, 0.10)

    # Upper tail (Expected Shortfall of gains)
    es_gain_1 = _conditional_expectation_above(scores, probs, cdf, 0.01)
    es_gain_5 = _conditional_expectation_above(scores, probs, cdf, 0.05)

    # top5_pct_avg: E[X | X >= p99.5]
    top5_pct_avg = _conditional_expectation_above(scores, probs, cdf, 0.005)

    # Trimmed mean (5%): E[X | p5 <= X <= p95]
    lo_idx = np.searchsorted(cdf, 0.05, side="left")
    hi_idx = np.searchsorted(cdf, 0.95, side="right")
    hi_idx = min(hi_idx, len(scores) - 1)
    trim_mask = np.zeros_like(probs)
    trim_mask[lo_idx : hi_idx + 1] = probs[lo_idx : hi_idx + 1]
    trim_total = trim_mask.sum()
    trimmed_mean = float(np.sum(scores * trim_mask) / trim_total) if trim_total > 0 else mean

    # Winsorized mean (5%): replace tails with boundary values
    win_probs = probs.copy()
    # Clip lower 5%
    lo_mass = float(cdf[lo_idx - 1]) if lo_idx > 0 else 0.0
    if lo_idx > 0:
        win_probs[:lo_idx] = 0.0
        win_probs[lo_idx] += lo_mass
    # Clip upper 5%
    hi_mass = 1.0 - float(cdf[hi_idx])
    if hi_idx < len(scores) - 1:
        win_probs[hi_idx + 1 :] = 0.0
        win_probs[hi_idx] += hi_mass
    winsorized_mean = float(np.sum(scores * win_probs))

    # Entropy
    nonzero_probs = probs[probs > 0]
    ent = -float(np.sum(nonzero_probs * np.log(nonzero_probs)))

    return {
        "theta": theta,
        "n": 0,  # signals exact data, not sampled
        "mean": mean,
        "std": std,
        "min": score_min,
        "max": score_max,
        "p1": int(p1),
        "p5": int(p5),
        "p10": int(p10),
        "p25": int(p25),
        "p50": int(p50),
        "p75": int(p75),
        "p90": int(p90),
        "p95": int(p95),
        "p99": int(p99),
        "p995": int(p995),
        "p999": int(p999),
        "p9999": int(p9999),
        "bot5_avg": float("nan"),  # MC-specific: literal 5 lowest games
        "top5_avg": float("nan"),  # MC-specific: literal 5 highest games
        "top5_pct_avg": top5_pct_avg,
        "cvar_1": cvar_1,
        "cvar_5": cvar_5,
        "cvar_10": cvar_10,
        "skewness": skewness,
        "kurtosis": kurt,
        "iqr": int(p75) - int(p25),
        "es_gain_1": es_gain_1,
        "es_gain_5": es_gain_5,
        "tail_ratio_95_5": p95 / p5 if p5 != 0 else float("inf"),
        "tail_ratio_99_1": p99 / p1 if p1 != 0 else float("inf"),
        "trimmed_mean_5": trimmed_mean,
        "winsorized_mean_5": winsorized_mean,
        "entropy": ent,
    }


def compute_kde_from_pmf(theta: float, data: dict) -> pl.DataFrame:
    """Convert exact PMF to kde-like DataFrame with columns: theta, score, density, cdf, survival.

    Interpolates onto the same grid as the MC KDE (linspace(50, 374, 1000)).
    The PMF gives P(score=k) for integer scores. We interpolate these as a
    continuous density curve on the standard KDE grid.
    """
    scores, probs = _pmf_arrays(data)

    # Target grid (matches KDE_RANGE and KDE_POINTS)
    x_grid = np.linspace(KDE_RANGE[0], KDE_RANGE[1], KDE_POINTS)

    # Interpolate PMF onto continuous grid
    # PMF values are at integer scores — interpolate linearly
    density = np.interp(x_grid, scores, probs, left=0.0, right=0.0)

    # Normalize so it integrates to ~1 over the grid
    dx = x_grid[1] - x_grid[0]
    area = np.sum(density) * dx
    if area > 0:
        density = density / area

    # CDF via cumulative trapezoid
    cdf_vals = np.cumsum(density) * dx
    # Ensure CDF ends at 1.0
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]

    return pl.DataFrame({
        "theta": np.full(KDE_POINTS, theta, dtype=np.float64),
        "score": x_grid.astype(np.float32),
        "density": density.astype(np.float32),
        "cdf": cdf_vals.astype(np.float32),
        "survival": (1.0 - cdf_vals).astype(np.float32),
    })


def compute_all_from_density(
    base_path: str = ".",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read all density JSONs and produce (summary_df, kde_df) with standard schemas."""
    density_dir = Path(base_path) / "outputs" / "density"
    thetas = discover_density_thetas(base_path)

    if not thetas:
        raise FileNotFoundError(f"No density_*.json files found in {density_dir}")

    summaries: list[dict] = []
    kde_parts: list[pl.DataFrame] = []

    for t in thetas:
        # Match Rust float formatting for filenames
        if t == 0:
            fname = "density_0.json"
        else:
            fname = f"density_{t:g}.json"
        path = density_dir / fname
        if not path.exists():
            # Try alternative formatting
            path = density_dir / f"density_{t}.json"
        if not path.exists():
            continue

        data = read_density_json(path)
        summaries.append(compute_summary_from_pmf(t, data))
        kde_parts.append(compute_kde_from_pmf(t, data))

    summary_df = pl.DataFrame(summaries)
    kde_df = pl.concat(kde_parts)
    return summary_df, kde_df
