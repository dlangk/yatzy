"""Shared plot styling: custom diverging colormap, theta_color, seaborn theme."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

# Custom diverging colormap: coolwarm endpoints with a distinct olive-yellow
# center (#C9CC3F) so that curves near θ=0 are visually distinguishable.
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "coolwarm_mid",
    ["#3b4cc0", "#8db0fe", "#F37021", "#f4987a", "#b40426"],
)
CMAP_R = CMAP.reversed()


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
    cbar.set_label(label, fontsize=11)
    return cbar


def fmt_theta(t: float) -> str:
    """Display formatting for theta labels."""
    if t == int(t) and abs(t) >= 1:
        return f"{int(t)}"
    return f"{t:.2f}"
