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
THETA_GRIDS: dict[str, list[float]] = {
    "all": [
        0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
        0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 3.0,
    ],
    "dense": [
        0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
    ],
    "sparse": [
        0, 0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.7, 0.9,
        1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 3.0,
    ],
}

# Plot subsets: name → list of thetas
PLOT_SUBSETS: dict[str, list[float]] = {
    "all": THETA_GRIDS["all"],
    "theta_dense": THETA_GRIDS["dense"],
    "theta_sparse": THETA_GRIDS["sparse"],
}


# ── Path resolution ─────────────────────────────────────────────────────────
def results_dir(base_path: str = "results") -> Path:
    return Path(base_path)


def analysis_dir(base_path: str = "results") -> Path:
    return results_dir(base_path) / "analysis"


def plots_dir(base_path: str = "results") -> Path:
    return results_dir(base_path) / "plots"


def theta_dir(theta: float, base_path: str = "results") -> Path:
    return results_dir(base_path) / f"theta_{fmt_theta_dir(theta)}"


def fmt_theta_dir(t: float) -> str:
    """Format theta to match results directory naming (e.g. '0', '0.01', '0.1', '2.0')."""
    if t == 0:
        return "0"
    s = f"{t:g}"
    # Check if directory exists with this name; if not, try .1f format
    # (deferred to runtime in io.py where base_path is known)
    return s


def discover_thetas(base_path: str = "results") -> list[float]:
    """Scan results directory for theta_* subdirs containing simulation_raw.bin."""
    base = results_dir(base_path)
    thetas = []
    if not base.is_dir():
        return thetas
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name.startswith("theta_"):
            raw = entry / "simulation_raw.bin"
            if raw.exists():
                try:
                    val = float(entry.name[len("theta_"):])
                    thetas.append(val)
                except ValueError:
                    pass
    return sorted(thetas)
