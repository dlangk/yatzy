"""Single source of truth for paths, binary format, theta grids, and KDE params."""
from __future__ import annotations

import os
from pathlib import Path

# ── Binary format constants (simulation_raw.bin) ────────────────────────────
HEADER_SIZE = 32
RECORD_SIZE = 289
TOTAL_SCORE_OFFSET = 15 * 19  # offset of total_score (u16) within each GameRecord

# ── KDE parameters ──────────────────────────────────────────────────────────
KDE_POINTS = 1000
KDE_RANGE = (50, 374)
KDE_SUBSAMPLE = 100_000
KDE_BANDWIDTH = 0.04

MAX_SCORE = 374

# ── Named theta grids ──────────────────────────────────────────────────────
# Progressive spacing: dense near 0, sparse at tails. Symmetric around 0.
THETA_GRIDS: dict[str, list[float]] = {
    "all": [
        -3.00, -2.00, -1.50, -1.00, -0.75, -0.50, -0.30,
        -0.20, -0.15, -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.015, -0.01, -0.005,
        0,
        0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20,
        0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
    ],
    "dense": [
        -0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.015, -0.01, -0.005,
        0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10,
    ],
    "sparse": [
        -3.00, -2.00, -1.50, -1.00, -0.50, -0.20, -0.10, -0.05,
        0, 0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00,
    ],
}

# Plot subsets: name → list of thetas
PLOT_SUBSETS: dict[str, list[float]] = {
    "all": THETA_GRIDS["all"],
    "theta_dense": THETA_GRIDS["dense"],
    "theta_sparse": THETA_GRIDS["sparse"],
}


# ── Path resolution ─────────────────────────────────────────────────────────
# Layout (base_path defaults to repo root "."):
#   data/simulations/theta/theta_*/simulation_raw.bin
#   data/simulations/max_policy/scores.bin
#   outputs/aggregates/parquet/  (kde, summary, scores, mer, sdva)
#   outputs/aggregates/csv/
#   outputs/plots/               (flat, no subfolders)
#   outputs/scenarios/           (pivotal_scenarios.json, answers)


def data_dir(base_path: str = ".") -> Path:
    return Path(base_path) / "data"


def simulations_dir(base_path: str = ".") -> Path:
    return data_dir(base_path) / "simulations"


def aggregates_dir(base_path: str = ".") -> Path:
    return Path(base_path) / "outputs" / "aggregates" / "parquet"


def plots_dir(base_path: str = ".") -> Path:
    return Path(base_path) / "outputs" / "plots"


def scenarios_dir(base_path: str = ".") -> Path:
    return Path(base_path) / "outputs" / "scenarios"


def bin_files_dir(base_path: str = ".") -> Path:
    """Alias for simulations_dir (backwards compat)."""
    return simulations_dir(base_path)


def density_dir(base_path: str = ".") -> Path:
    return Path(base_path) / "outputs" / "density"


def multiplayer_dir(base_path: str = ".") -> Path:
    return simulations_dir(base_path) / "multiplayer"


def theta_base_dir(base_path: str = ".") -> Path:
    return simulations_dir(base_path) / "theta"


def theta_dir(theta: float, base_path: str = ".") -> Path:
    return theta_base_dir(base_path) / f"theta_{fmt_theta_dir(theta)}"


def fmt_theta_dir(t: float) -> str:
    """Format theta to match results directory naming (e.g. '0', '0.01', '0.1', '2.0')."""
    if t == 0:
        return "0"
    s = f"{t:g}"
    return s


def discover_thetas(base_path: str = ".") -> list[float]:
    """Scan data/simulations/theta/ for theta_* subdirs with simulation data.

    Args:
        base_path: Repository root directory (default ".").

    Returns:
        Sorted list of theta values whose directories contain scores.bin
        or simulation_raw.bin.
    """
    base = theta_base_dir(base_path)
    thetas = []
    if not base.is_dir():
        return thetas
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name.startswith("theta_"):
            has_data = (entry / "scores.bin").exists() or (entry / "simulation_raw.bin").exists()
            if has_data:
                try:
                    val = float(entry.name[len("theta_"):])
                    thetas.append(val)
                except ValueError:
                    pass
    return sorted(thetas)


# Backwards compat alias — old code used analysis_dir()
analysis_dir = aggregates_dir
