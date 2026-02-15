"""Shared plot styling: custom diverging colormap, theta_color, seaborn theme."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_TITLE = 14
FONT_SUPTITLE = 16
FONT_AXIS_LABEL = 12
FONT_LEGEND = 9
FONT_TICK = 10
FONT_ANNOTATION = 9

# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------
FIG_WIDE = (14, 7)
FIG_SQUARE = (10, 8)
FIG_TALL = (10, 14)
FIG_QUAD = (14, 12)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID_ALPHA = 0.3

# ---------------------------------------------------------------------------
# Shared colors
# ---------------------------------------------------------------------------
COLOR_BLUE = "#3b4cc0"
COLOR_RED = "#b40426"
COLOR_ORANGE = "#F37021"
COLOR_GREEN = "#2ca02c"

ADAPTIVE_COLORS = {
    "bonus-adaptive": "#e74c3c",
    "phase-based": "#2ecc71",
    "combined": "#9b59b6",
}

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]
CATEGORY_SHORT = [
    "1s", "2s", "3s", "4s", "5s", "6s",
    "Pair", "2Pair", "3Kind", "4Kind",
    "SmStr", "LgStr", "FHouse", "Chance", "Yatzy",
]

PERCENTILES_CORE = ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
PERCENTILES_EXTRA = ["p1", "p999", "p9999"]

# ---------------------------------------------------------------------------
# Custom diverging colormap: coolwarm endpoints with a distinct olive-yellow
# center (#C9CC3F) so that curves near θ=0 are visually distinguishable.
# ---------------------------------------------------------------------------
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "coolwarm_mid",
    ["#3b4cc0", "#8db0fe", "#F37021", "#f4987a", "#b40426"],
)
CMAP_R = CMAP.reversed()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig, out_dir: Path, name: str, *, dpi: int = 200, fmt: str = "png") -> Path:
    """Save + close with consistent bbox_inches='tight'."""
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def make_norm(thetas: list[float]) -> mcolors.Normalize:
    """Build a nonlinear color norm for the theta range.

    Uses SymLogNorm: linear in [-linthresh, +linthresh], logarithmic outside.
    This gives distinct colors to the dense region near 0 AND the sparse tails.
    """
    vmin, vmax = min(thetas), max(thetas)
    # If range is small and positive-only, use simple linear norm
    if vmin >= 0 and vmax <= 0.5:
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Symmetric log: linear below 0.05, log above
    vabs = max(abs(vmin), abs(vmax))
    return mcolors.SymLogNorm(linthresh=0.05, linscale=1.0, vmin=-vabs, vmax=vabs)


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)


def theta_color(t: float, norm: mcolors.Normalize):
    return CMAP(norm(t))


def theta_colorbar(
    ax, norm: mcolors.Normalize, *, label: str = "θ  (blue=risk-averse, red=risk-seeking)",
):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(label, fontsize=FONT_AXIS_LABEL)
    return cbar


def fmt_theta(t: float) -> str:
    """Display formatting for theta labels."""
    if t == int(t) and abs(t) >= 1:
        return f"{int(t)}"
    return f"{t:.2f}"
