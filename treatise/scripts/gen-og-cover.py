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

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

BG = "#14110d"       # deep warm near-black, makes the orange pop
TEXT = "#f3efe7"     # cream (matches treatise --text on dark)
MUTED = "#9a9488"
THETA = 0.04

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


def main():
    d = json.load(open(f"outputs/density/density_{THETA:g}.json"))
    scores = np.array([p[0] for p in d["pmf"]], dtype=float)
    dens = gaussian_filter1d(np.array([p[1] for p in d["pmf"]], dtype=float), 1.5)
    mean = d["mean"]
    color = theta_color(THETA)

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    # Chart occupies the lower ~62% of the card, full-bleed, as a backdrop.
    # Zoom the x-range to where the mass lives so the curve fills and centers.
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.62])
    ax.set_facecolor(BG)
    ax.set_xlim(80, 380)
    ax.set_ylim(0, dens.max() * 1.18)
    ax.fill_between(scores, dens, color=color, alpha=0.26, zorder=1)
    ax.plot(scores, dens, color=color, lw=4.5, zorder=3, solid_capstyle="round")
    ax.axvline(mean, color=color, lw=2.0, dashes=(5, 3), alpha=0.55, zorder=2)
    ax.text(mean + 5, dens.max() * 1.08, f"mean {mean:.0f}", color=color,
            fontsize=15, ha="left", va="top", family="serif")
    ax.axis("off")

    # Headline (kept centered, inside the safe zone; sized to fit the width).
    fig.text(0.5, 0.92, "Can You Be Skilled At Playing Yatzy?",
             ha="center", va="top", color=TEXT, fontsize=36,
             family="serif", fontweight="bold")
    fig.text(0.5, 0.76, "Computing the optimal strategy for Scandinavian Yatzy",
             ha="center", va="top", color=MUTED, fontsize=20, family="serif")

    # Brand mark.
    fig.text(0.984, 0.035, "langkilde.se/yatzy", ha="right", va="bottom",
             color=MUTED, fontsize=15, family="serif")

    out = Path("treatise/og-cover.png")
    fig.savefig(out, facecolor=BG)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
