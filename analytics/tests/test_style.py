"""Tests for analytics plot styling."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from yatzy_analysis.plots.style import CMAP, save_fig


def test_cmap_type():
    assert isinstance(CMAP, mcolors.LinearSegmentedColormap)


def test_cmap_center_is_orange():
    r, g, b, _a = CMAP(0.5)
    # #F37021 ≈ (0.953, 0.439, 0.129)
    assert abs(r - 0.953) < 0.05
    assert abs(g - 0.439) < 0.05
    assert abs(b - 0.129) < 0.05


def test_save_fig_creates_file(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_fig(fig, tmp_path, "test_plot")
    assert path.exists()
    assert path.suffix == ".png"
