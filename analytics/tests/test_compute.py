"""Tests for analytics compute module."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from yatzy_analysis.compute import compute_cvar, compute_summary


def test_cvar_bottom_third():
    scores = np.array([100, 200, 300], dtype=np.int32)
    result = compute_cvar(scores, 1 / 3)
    assert abs(result - 100.0) < 1e-6


def test_cvar_full_alpha_equals_mean():
    scores = np.array([100, 200, 300, 400, 500], dtype=np.int32)
    result = compute_cvar(scores, 1.0)
    assert abs(result - 300.0) < 1e-6


def test_compute_summary_keys(synthetic_scores: NDArray[np.int32]):
    summary = compute_summary(0.0, synthetic_scores)
    required_keys = {"theta", "n", "mean", "std", "min", "max", "p5", "p25", "p50", "p75", "p95"}
    assert required_keys.issubset(summary.keys())


def test_compute_summary_constant_mean(constant_scores: NDArray[np.int32]):
    summary = compute_summary(0.0, constant_scores)
    assert abs(summary["mean"] - 250.0) < 1e-6


def test_compute_summary_constant_std(constant_scores: NDArray[np.int32]):
    summary = compute_summary(0.0, constant_scores)
    assert abs(summary["std"]) < 1e-6


def test_compute_summary_min_max(synthetic_scores: NDArray[np.int32]):
    summary = compute_summary(0.0, synthetic_scores)
    assert summary["min"] == int(synthetic_scores[0])
    assert summary["max"] == int(synthetic_scores[-1])
