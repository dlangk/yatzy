"""Shared fixtures for analytics tests."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def synthetic_scores() -> NDArray[np.int32]:
    """Sorted int32 array mimicking 10K game scores (range ~100-350)."""
    rng = np.random.default_rng(42)
    scores = rng.integers(100, 350, size=10_000, dtype=np.int32)
    scores.sort()
    return scores


@pytest.fixture
def constant_scores() -> NDArray[np.int32]:
    """All-same scores for degenerate-case testing."""
    return np.full(1000, 250, dtype=np.int32)
