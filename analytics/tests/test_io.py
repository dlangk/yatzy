"""Tests for analytics I/O module."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from yatzy_analysis.io import SCORES_MAGIC, read_scores_compact, read_scores_raw


def test_read_scores_compact_nonexistent(tmp_path: Path):
    result = read_scores_compact(tmp_path / "nonexistent.bin")
    assert result is None


def test_read_scores_raw_nonexistent(tmp_path: Path):
    result = read_scores_raw(tmp_path / "nonexistent.bin")
    assert result is None


def test_read_scores_compact_wrong_magic(tmp_path: Path):
    path = tmp_path / "bad_magic.bin"
    # Write 32-byte header with wrong magic
    header = struct.pack("<I", 0xDEADBEEF) + b"\x00" * 28
    path.write_bytes(header)
    result = read_scores_compact(path)
    assert result is None


def test_read_scores_compact_roundtrip(tmp_path: Path):
    path = tmp_path / "scores.bin"
    scores = np.array([150, 200, 250, 300], dtype=np.int16)
    n = len(scores)

    # Build 32-byte header: magic(4) + version(4) + num_games(4) + padding(20)
    header = struct.pack("<III", SCORES_MAGIC, 1, n) + b"\x00" * 20
    path.write_bytes(header + scores.tobytes())

    result = read_scores_compact(path)
    assert result is not None
    assert len(result) == n
    # Result is sorted
    np.testing.assert_array_equal(result, np.sort(scores.astype(np.int32)))


def test_magic_constants():
    from yatzy_analysis.io import RAW_SIM_MAGIC, SCORES_MAGIC

    assert SCORES_MAGIC == 0x59545353
    assert RAW_SIM_MAGIC == 0x59545352
