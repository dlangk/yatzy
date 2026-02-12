"""Load precomputed state-value .bin files via numpy memmap.

Port of backend/src/storage.rs — reads the 16-byte header + float32[2,097,152]
binary format.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .scoring import NUM_STATES

STATE_FILE_MAGIC = 0x59545A53
STATE_FILE_VERSION_V4 = 4
STATE_FILE_VERSION_V5 = 5
HEADER_SIZE = 16  # bytes


def load_state_values(path: str | Path) -> np.ndarray:
    """Load state values from a .bin file via numpy memmap.

    Returns a read-only float32 array of shape (NUM_STATES,).
    Validates the 16-byte header (magic, version, state count).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    expected_size = HEADER_SIZE + NUM_STATES * 4
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"File size mismatch: expected {expected_size}, got {actual_size}")

    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)

    magic, version, total_states, theta_bits = struct.unpack("<IIII", header)
    if magic != STATE_FILE_MAGIC:
        raise ValueError(f"Invalid magic: 0x{magic:08x}")
    if version not in (STATE_FILE_VERSION_V4, STATE_FILE_VERSION_V5):
        raise ValueError(f"Unsupported version: {version}")

    # Memory-map the data portion (skip header)
    return np.memmap(path, dtype=np.float32, mode="r", offset=HEADER_SIZE, shape=(NUM_STATES,))


def state_file_path(base_path: str | Path, theta: float) -> Path:
    """Return the state file path for a given theta value."""
    base = Path(base_path) / "data"
    if theta == 0.0:
        return base / "all_states.bin"
    return base / f"all_states_theta_{theta:.3f}.bin"


def load_theta_tables(
    base_path: str | Path, thetas: list[float]
) -> dict[float, np.ndarray]:
    """Load multiple theta tables. Returns {theta: state_values_array}."""
    tables = {}
    for theta in thetas:
        path = state_file_path(base_path, theta)
        tables[theta] = load_state_values(path)
    return tables
