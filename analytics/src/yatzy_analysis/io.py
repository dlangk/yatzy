"""Read simulation binary files (numpy vectorized)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import HEADER_SIZE, RECORD_SIZE, TOTAL_SCORE_OFFSET, theta_base_dir

# Magic numbers for format detection
RAW_SIM_MAGIC = 0x59545352  # "RTSZ" — old full GameRecord format
SCORES_MAGIC = 0x59545353  # "STSY" — new compact scores-only format
MULTIPLAYER_MAGIC = 0x594C504D  # "MPLY" — multiplayer recording format


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


def read_scores_compact(path: Path) -> NDArray[np.int32] | None:
    """Read scores from the compact scores.bin format (32-byte header + i16[]).

    Returns sorted int32 array, or None if file doesn't exist or is invalid.
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        header_data = f.read(32)
        if len(header_data) < 32:
            return None
        magic = struct.unpack_from("<I", header_data, 0)[0]
        if magic != SCORES_MAGIC:
            return None
        num_games = struct.unpack_from("<I", header_data, 8)[0]
        scores = np.frombuffer(f.read(num_games * 2), dtype=np.int16).astype(np.int32)
    scores.sort()
    return scores


def read_scores_raw(path: Path) -> NDArray[np.int32] | None:
    """Read total_score from old simulation_raw.bin format (32-byte header + GameRecord[]).

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


def read_scores(path: Path) -> NDArray[np.int32] | None:
    """Read scores from any supported binary format, auto-detecting by magic number.

    Supports:
    - scores.bin (compact: 32B header + i16[], magic 0x59545353)
    - simulation_raw.bin (full: 32B header + GameRecord[], magic 0x59545352)

    Returns sorted int32 array, or None if file doesn't exist or is invalid.
    """
    if not path.exists():
        return None

    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            return None
        magic = struct.unpack_from("<I", magic_bytes, 0)[0]

    if magic == SCORES_MAGIC:
        return read_scores_compact(path)
    elif magic == RAW_SIM_MAGIC:
        return read_scores_raw(path)
    else:
        return None


def read_multiplayer_recording(path: Path) -> dict | None:
    """Read multiplayer recording binary.

    Format: 32-byte header + MultiplayerGameRecord[N] (64 bytes each).
    Each record: scores i16[2] + turn_totals i16[2][15].

    Returns dict with keys: num_games, num_players, seed, scores (N,2), turn_totals (N,2,15).
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        header_data = f.read(32)
        if len(header_data) < 32:
            return None
        magic = struct.unpack_from("<I", header_data, 0)[0]
        if magic != MULTIPLAYER_MAGIC:
            return None
        version = struct.unpack_from("<I", header_data, 4)[0]
        if version != 1:
            return None
        num_games = struct.unpack_from("<I", header_data, 8)[0]
        num_players = struct.unpack_from("<B", header_data, 12)[0]
        seed = struct.unpack_from("<Q", header_data, 16)[0]

        # Read packed records: each is [scores i16[2], turn_totals i16[2][15]] = 64 bytes
        record_dt = np.dtype([
            ("scores", np.int16, (2,)),
            ("turn_totals", np.int16, (2, 15)),
        ])
        data = np.frombuffer(f.read(num_games * 64), dtype=record_dt)

    return {
        "num_games": int(num_games),
        "num_players": int(num_players),
        "seed": int(seed),
        "scores": data["scores"],          # (N, 2)
        "turn_totals": data["turn_totals"],  # (N, 2, 15)
    }


def read_all_scores(
    thetas: list[float], base_path: str = "results"
) -> dict[float, NDArray[np.int32]]:
    """Read scores for multiple thetas. Returns {theta: sorted_scores}.

    For each theta directory, prefers scores.bin over simulation_raw.bin.
    """
    result: dict[float, NDArray[np.int32]] = {}
    base = theta_base_dir(base_path)
    for t in thetas:
        tname = fmt_theta(t, base_path)
        tdir = base / f"theta_{tname}"

        # Prefer compact format
        scores_path = tdir / "scores.bin"
        if scores_path.exists():
            scores = read_scores(scores_path)
            if scores is not None:
                result[t] = scores
                continue

        # Fall back to old format
        raw_path = tdir / "simulation_raw.bin"
        scores = read_scores(raw_path)
        if scores is not None:
            result[t] = scores
    return result
