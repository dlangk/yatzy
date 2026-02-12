"""Shared plot styling: coolwarm colormap, theta_color, seaborn theme."""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

# Global colormap: blue at θ=0 (EV-neutral), red at θ=3 (risk-seeking)
CMAP = sns.color_palette("coolwarm", as_cmap=True)
NORM = mcolors.Normalize(vmin=0.0, vmax=3.0)


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)


def theta_color(t: float):
    return CMAP(NORM(t))


def theta_colorbar(ax, *, label: str = "θ  (blue=EV-optimal, red=risk-seeking)"):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(label, fontsize=11)
    return cbar


def fmt_theta(t: float) -> str:
    """Display formatting for theta labels."""
    if t == int(t) and abs(t) >= 1:
        return f"{int(t)}"
    return f"{t:.2f}"
