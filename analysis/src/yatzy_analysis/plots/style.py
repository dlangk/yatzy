"""Shared plot styling: coolwarm colormap, theta_color, seaborn theme."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

CMAP = sns.color_palette("coolwarm", as_cmap=True)


def make_norm(thetas: list[float]) -> mcolors.Normalize:
    """Build a color norm spanning the actual theta range."""
    return mcolors.Normalize(vmin=min(thetas), vmax=max(thetas))


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)


def theta_color(t: float, norm: mcolors.Normalize):
    return CMAP(norm(t))


def theta_colorbar(
    ax, norm: mcolors.Normalize, *, label: str = "θ  (blue=EV-optimal, red=risk-seeking)",
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
