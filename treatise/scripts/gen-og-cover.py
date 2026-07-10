"""Render the 1200x630 social share card (og:image) for the treatise.

A high-contrast static card: the theta=0.04 exact score distribution on a dark
background, with the title overlaid. Sized and composed per Open Graph / X card
best practice (1200x630, 1.91:1, headline in the center safe zone). Animated
GIFs freeze to their first frame in link previews, so shares need this static
asset; the on-page hero uses the video.

Run:
    uv run --with matplotlib,scipy,numpy treatise/scripts/gen-og-cover.py
Output: treatise/og-cover.png
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

BG = "#161311"       # matches the treatise dark chart background
TEXT = "#f3efe7"     # cream (matches treatise --text on dark)
MUTED = "#9a9488"
GRID = "#333333"
THETA = 0.04
SMOOTH = 1.5         # emulates the chart's D3 curveBasis
Y_MAX = 0.015        # fixed y-domain, same as the live risk-theta chart

THETA_STOPS = [
    (-0.30, "#3b4cc0"), (-0.15, "#8db0fe"), (0.00, "#F37021"),
    (0.15, "#f4987a"), (0.30, "#b40426"),
]


def theta_color(t):
    t = max(-0.30, min(0.30, t))
    for (t0, c0), (t1, c1) in zip(THETA_STOPS, THETA_STOPS[1:]):
        if t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
            return tuple(a + (b - a) * frac)
    return to_rgb(THETA_STOPS[-1][1])


def load_curve(theta):
    d = json.load(open(f"outputs/density/density_{theta:g}.json"))
    scores = np.array([p[0] for p in d["pmf"]], dtype=float)
    dens = gaussian_filter1d(np.array([p[1] for p in d["pmf"]], dtype=float), SMOOTH)
    return scores, dens, d["mean"], d["std_dev"]


def main():
    scores, dens, mean, std = load_curve(THETA)
    color = theta_color(THETA)
    peak = dens.max()

    # 2x pixel density (2400x1260). The design is deliberately BOLD and SIMPLE:
    # LinkedIn and X re-encode every share image to a downscaled JPEG, which
    # turns thin lines, fine detail and thin serif strokes to mush. Thick
    # shapes, soft fills and large heavy type survive that. So: no 85-curve
    # family, no thin dotted sigma/grid lines. One thick curve, a soft glow, a
    # translucent +-1 sigma band for context, a bold mean, a big title.
    fig = plt.figure(figsize=(12, 6.3), dpi=200)
    fig.patch.set_facecolor(BG)

    # Chart fills the lower portion, x-window centered on the mean.
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.66])
    ax.set_facecolor(BG)
    ax.set_xlim(mean - 150, mean + 150)
    ax.set_ylim(0, peak * 1.16)

    # Translucent +-1 sigma band = "the typical range" (a soft fill survives
    # compression; thin dotted sigma lines do not).
    ax.axvspan(mean - std, mean + std, color=color, alpha=0.11, zorder=0)

    # Filled area + soft glow + thick curve.
    ax.fill_between(scores, dens, color=color, alpha=0.28, zorder=1)
    ax.plot(scores, dens, color=color, lw=15, alpha=0.16, zorder=2,
            solid_capstyle="round")            # glow
    ax.plot(scores, dens, color=color, lw=6.5, zorder=3, solid_capstyle="round")

    # Bold mean marker.
    ax.axvline(mean, color=TEXT, lw=2.5, dashes=(6, 4), alpha=0.85, zorder=4)
    ax.text(mean + 6, peak * 1.10, f"mean {mean:.0f}", color=TEXT,
            fontsize=17, fontweight="bold", family="serif", va="top", zorder=5)
    ax.axis("off")

    # Headline (big + heavy; large thick type survives the re-encode).
    fig.text(0.5, 0.955, "Can You Be Skilled At Playing Yatzy?",
             ha="center", va="top", color=TEXT, fontsize=38,
             family="serif", fontweight="bold")
    fig.text(0.5, 0.815, "Computing the optimal strategy for Scandinavian Yatzy",
             ha="center", va="top", color=MUTED, fontsize=21, family="serif")

    out = Path("treatise/og-cover.png")
    fig.savefig(out, facecolor=BG)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
