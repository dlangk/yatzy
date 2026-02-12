"""Save/load Parquet intermediates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def save_scores(scores_dict: dict[float, NDArray[np.int32]], path: Path) -> None:
    """Save {theta: scores_array} as a Parquet file with columns (theta, score)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for t in sorted(scores_dict.keys()):
        s = scores_dict[t]
        parts.append(pd.DataFrame({
            "theta": np.full(len(s), t, dtype=np.float64),
            "score": s,
        }))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def load_scores(path: Path) -> dict[float, NDArray[np.int32]]:
    """Load scores parquet back into {theta: sorted_scores} dict."""
    df = pd.read_parquet(path, engine="pyarrow")
    df["theta"] = df["theta"].round(4)
    result: dict[float, NDArray[np.int32]] = {}
    for t, group in df.groupby("theta", sort=True):
        result[float(t)] = group["score"].to_numpy(dtype=np.int32)
    return result


def save_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "theta" in df.columns:
        df["theta"] = df["theta"].round(4)
    return df


def save_kde(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def load_kde(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "theta" in df.columns:
        df["theta"] = df["theta"].round(4)
    return df
