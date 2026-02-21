"""Tests for analytics config module."""
from __future__ import annotations

from pathlib import Path

from yatzy_analysis.config import (
    HEADER_SIZE,
    KDE_RANGE,
    MAX_SCORE,
    RECORD_SIZE,
    THETA_GRIDS,
    aggregates_dir,
    data_dir,
    plots_dir,
)


def test_data_dir():
    assert data_dir(".") == Path("./data")


def test_aggregates_dir():
    assert aggregates_dir(".") == Path("./outputs/aggregates/parquet")


def test_plots_dir():
    assert plots_dir(".") == Path("./outputs/plots")


def test_theta_grid_sorted():
    grid = THETA_GRIDS["all"]
    assert grid == sorted(grid)


def test_theta_grid_symmetric():
    grid = THETA_GRIDS["all"]
    negatives = sorted(-t for t in grid if t < 0)
    positives = sorted(t for t in grid if t > 0)
    assert negatives == positives


def test_kde_range_valid():
    assert len(KDE_RANGE) == 2
    assert KDE_RANGE[0] < KDE_RANGE[1]


def test_max_score():
    assert MAX_SCORE == 374


def test_binary_format_constants():
    assert isinstance(HEADER_SIZE, int) and HEADER_SIZE > 0
    assert isinstance(RECORD_SIZE, int) and RECORD_SIZE > 0
