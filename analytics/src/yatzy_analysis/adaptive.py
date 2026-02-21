"""Adaptive policy discovery, score extraction, and summary computation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .compute import compute_kde, compute_summary
from .config import HEADER_SIZE, RECORD_SIZE, TOTAL_SCORE_OFFSET, bin_files_dir
from .io import read_scores


def adaptive_base_dir(base_path: str = ".") -> Path:
    """Return the adaptive policy results directory."""
    return bin_files_dir(base_path) / "adaptive"


def discover_adaptive_policies(base_path: str = ".") -> list[str]:
    """Scan bin_files/adaptive/ for policy subdirs containing scores.bin or simulation_raw.bin."""
    base = adaptive_base_dir(base_path)
    policies: list[str] = []
    if not base.is_dir():
        return policies
    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            has_data = (entry / "scores.bin").exists() or (entry / "simulation_raw.bin").exists()
            if has_data:
                policies.append(entry.name)
    return policies


def read_adaptive_scores(
    policies: list[str], base_path: str = "."
) -> dict[str, NDArray[np.int32]]:
    """Read scores for each adaptive policy. Returns {policy_name: sorted_scores}.

    Prefers scores.bin over simulation_raw.bin.
    """
    result: dict[str, NDArray[np.int32]] = {}
    base = adaptive_base_dir(base_path)
    for name in policies:
        # Prefer compact format
        scores_path = base / name / "scores.bin"
        if scores_path.exists():
            scores = read_scores(scores_path)
            if scores is not None:
                result[name] = scores
                continue
        # Fall back to old format
        raw_path = base / name / "simulation_raw.bin"
        scores = read_scores(raw_path)
        if scores is not None:
            result[name] = scores
    return result


def compute_adaptive_summary(
    scores_dict: dict[str, NDArray[np.int32]],
) -> pl.DataFrame:
    """Compute summary stats for adaptive policies.

    Returns DataFrame with columns: policy, n, mean, std, min, max, p5..p99, etc.
    Same columns as theta summary but with 'policy' instead of 'theta'.
    """
    summaries = []
    for name in sorted(scores_dict.keys()):
        scores = scores_dict[name]
        # Reuse compute_summary with theta=0 (placeholder) and rename
        row = compute_summary(0.0, scores)
        row["policy"] = name
        del row["theta"]
        summaries.append(row)
    return pl.DataFrame(summaries)


def compute_adaptive_kde(
    scores_dict: dict[str, NDArray[np.int32]],
) -> pl.DataFrame:
    """Compute KDE for each adaptive policy."""
    parts = []
    for name in sorted(scores_dict.keys()):
        scores = scores_dict[name]
        kde_df = compute_kde(0.0, scores)
        kde_df = kde_df.drop("theta").with_columns(pl.lit(name).alias("policy"))
        parts.append(kde_df)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts)
