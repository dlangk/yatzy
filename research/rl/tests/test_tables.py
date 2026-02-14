"""Tests for tables.py — validates state value file loading."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from yatzy_rl.scoring import NUM_STATES
from yatzy_rl.tables import (
    HEADER_SIZE,
    STATE_FILE_MAGIC,
    STATE_FILE_VERSION_V4,
    load_state_values,
    state_file_path,
)


def create_test_bin(path: Path, version: int = STATE_FILE_VERSION_V4) -> np.ndarray:
    """Create a test .bin file with known values."""
    values = np.random.default_rng(42).standard_normal(NUM_STATES).astype(np.float32)
    header = struct.pack("<IIII", STATE_FILE_MAGIC, version, NUM_STATES, 0)
    with open(path, "wb") as f:
        f.write(header)
        f.write(values.tobytes())
    return values


class TestStateFilePath:
    def test_theta_zero(self):
        path = state_file_path("/some/base", 0.0)
        assert path == Path("/some/base/data/all_states.bin")

    def test_theta_nonzero(self):
        path = state_file_path("/some/base", 0.05)
        assert path == Path("/some/base/data/all_states_theta_0.050.bin")


class TestLoadStateValues:
    def test_round_trip(self, tmp_path):
        """Write and re-read should produce identical values."""
        bin_path = tmp_path / "test.bin"
        expected = create_test_bin(bin_path)
        loaded = load_state_values(bin_path)
        np.testing.assert_array_almost_equal(loaded, expected, decimal=6)

    def test_shape(self, tmp_path):
        bin_path = tmp_path / "test.bin"
        create_test_bin(bin_path)
        loaded = load_state_values(bin_path)
        assert loaded.shape == (NUM_STATES,)
        assert loaded.dtype == np.float32

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_state_values("/nonexistent/path.bin")

    def test_wrong_magic(self, tmp_path):
        bin_path = tmp_path / "bad_magic.bin"
        header = struct.pack("<IIII", 0xDEADBEEF, STATE_FILE_VERSION_V4, NUM_STATES, 0)
        values = np.zeros(NUM_STATES, dtype=np.float32)
        with open(bin_path, "wb") as f:
            f.write(header)
            f.write(values.tobytes())
        with pytest.raises(ValueError, match="Invalid magic"):
            load_state_values(bin_path)

    def test_wrong_size(self, tmp_path):
        bin_path = tmp_path / "bad_size.bin"
        header = struct.pack("<IIII", STATE_FILE_MAGIC, STATE_FILE_VERSION_V4, NUM_STATES, 0)
        with open(bin_path, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 100)  # Too small
        with pytest.raises(ValueError, match="File size mismatch"):
            load_state_values(bin_path)
