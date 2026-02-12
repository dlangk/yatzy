"""Read raw simulation binary files (numpy vectorized)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import HEADER_SIZE, RECORD_SIZE, TOTAL_SCORE_OFFSET, theta_base_dir


def fmt_theta(t: float, base_path: str = "results") -> str:
    """Format theta to match the actual directory name on disk."""
    if t == 0:
        return "0"
    s = f"{t:g}"
    if not (theta_base_dir(base_path) / f"theta_{s}").is_dir():
        alt = f"{t:.1f}"
        if (theta_base_dir(base_path) / f"theta_{alt}").is_dir():
            return alt
    return s


def read_scores(path: Path) -> NDArray[np.int32] | None:
    """Read all total_score values from a simulation_raw.bin file.

    Uses numpy vectorized strided reads — ~100x faster than struct.unpack loop.
    Returns sorted int32 array, or None if file doesn't exist.
    """
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(data[8:12], dtype=np.uint32)[0])
    offsets = HEADER_SIZE + np.arange(n) * RECORD_SIZE + TOTAL_SCORE_OFFSET
    # Read u16 little-endian: low byte | (high byte << 8)
    scores = data[offsets].astype(np.int32) | (data[offsets + 1].astype(np.int32) << 8)
    scores.sort()
    return scores


def read_all_scores(
    thetas: list[float], base_path: str = "results"
) -> dict[float, NDArray[np.int32]]:
    """Read scores for multiple thetas. Returns {theta: sorted_scores}."""
    result: dict[float, NDArray[np.int32]] = {}
    base = theta_base_dir(base_path)
    for t in thetas:
        tname = fmt_theta(t, base_path)
        path = base / f"theta_{tname}" / "simulation_raw.bin"
        scores = read_scores(path)
        if scores is not None:
            result[t] = scores
    return result
