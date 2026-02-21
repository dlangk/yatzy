"""Save/load Parquet intermediates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def save_scores(scores_dict: dict[float, NDArray[np.int32]], path: Path) -> None:
    """Save {theta: scores_array} as a Parquet file with columns (theta, score)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for t in sorted(scores_dict.keys()):
        s = scores_dict[t]
        parts.append(pl.DataFrame({
            "theta": np.full(len(s), t, dtype=np.float64),
            "score": s,
        }))
    df = pl.concat(parts)
    df.write_parquet(path)


def load_scores(path: Path) -> dict[float, NDArray[np.int32]]:
    """Load scores parquet back into {theta: sorted_scores} dict."""
    df = pl.read_parquet(path)
    df = df.with_columns(pl.col("theta").round(4))
    result: dict[float, NDArray[np.int32]] = {}
    for t, group in df.group_by("theta", maintain_order=True):
        group = group.sort("theta")
        result[float(t[0])] = group["score"].to_numpy().astype(np.int32)
    # Sort by theta key
    return dict(sorted(result.items()))


def save_summary(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load_summary(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "theta" in df.columns:
        df = df.with_columns(pl.col("theta").round(4))
    return df


def save_kde(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def load_kde(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "theta" in df.columns:
        df = df.with_columns(pl.col("theta").round(4))
    return df
