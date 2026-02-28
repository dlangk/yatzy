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

    Args:
        path: Path to a scores.bin file (magic 0x59545353).

    Returns:
        Sorted ascending int32 array of game scores, or None if the file
        doesn't exist or has an invalid magic number.
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

    Args:
        path: Path to a multiplayer recording file (magic 0x594C504D).

    Returns:
        Dict with keys num_games (int), num_players (int), seed (int),
        scores (N,2 int16 array), and turn_totals (N,2,15 int16 array).
        Returns None if the file doesn't exist or has an invalid format.
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


def read_full_recording(path: Path) -> dict | None:
    """Read per-game category data from simulation_raw.bin (full GameRecord format).

    Binary layout per game (289 bytes):
      turns[15]: each 19 bytes (dice[5], mask1, dice[5], mask2, dice[5], category u8, score u8)
      total_score: u16 LE at offset 285
      upper_total: u8 at offset 287
      got_bonus: u8 at offset 288

    Returns dict with:
      num_games: int
      total_scores: NDArray[np.uint16]   (N,)
      got_bonus: NDArray[np.bool_]       (N,)
      upper_totals: NDArray[np.uint8]    (N,)
      category_scores: NDArray[np.uint8] (N, 15) — score per category id
    """
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if len(data) < HEADER_SIZE + RECORD_SIZE:
        return None

    magic = int(np.frombuffer(data[0:4], dtype=np.uint32)[0])
    if magic != RAW_SIM_MAGIC:
        return None

    n = int(np.frombuffer(data[8:12], dtype=np.uint32)[0])
    if len(data) < HEADER_SIZE + n * RECORD_SIZE:
        return None

    # Vectorized extraction using strided offsets
    game_offsets = HEADER_SIZE + np.arange(n, dtype=np.int64) * RECORD_SIZE

    # total_score: u16 LE at offset 285 within record
    ts_off = game_offsets + TOTAL_SCORE_OFFSET
    total_scores = data[ts_off].astype(np.uint16) | (data[ts_off + 1].astype(np.uint16) << 8)

    # upper_total: u8 at offset 287
    upper_totals = data[game_offsets + 287]

    # got_bonus: u8 at offset 288
    got_bonus = data[game_offsets + 288].astype(np.bool_)

    # Per-category scores: scatter from temporal turns into category slots
    # Turn t: category at offset t*19+17, score at offset t*19+18
    category_scores = np.zeros((n, 15), dtype=np.uint8)
    for t in range(15):
        cat_off = game_offsets + t * 19 + 17
        score_off = game_offsets + t * 19 + 18
        cats = data[cat_off]      # (N,) category ids 0-14
        scores = data[score_off]  # (N,) scores
        # Scatter: category_scores[game_i, cats[game_i]] = scores[game_i]
        category_scores[np.arange(n), cats] = scores

    return {
        "num_games": n,
        "total_scores": total_scores,
        "got_bonus": got_bonus,
        "upper_totals": upper_totals,
        "category_scores": category_scores,
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
