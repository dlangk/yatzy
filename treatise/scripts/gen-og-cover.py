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
    # The full curve family (|theta| <= 0.4), same range the live chart shows.
    family = []
    for p in glob.glob("outputs/density/density_*.json"):
        m = re.search(r"density_(-?[\d.]+)\.json", p)
        t = float(m.group(1))
        if -0.401 <= t <= 0.401:
            family.append(t)
    family.sort()

    scores, dens, mean, std = load_curve(THETA)
    color = theta_color(THETA)

    # 2x pixel density (2400x1260) so serif text stays crisp on Retina and on
    # larger card renders; platforms downscale to the 1200x630 they display.
    fig = plt.figure(figsize=(12, 6.3), dpi=200)
    fig.patch.set_facecolor(BG)

    # Chart fills the lower ~70% of the card, like the live widget, with
    # equal left/right margins. The x-window is centered on the mean so the
    # distribution's mass sits in the middle of the card (a 0-400 axis would
    # push it right-of-center).
    ax = fig.add_axes([0.035, 0.06, 0.93, 0.68])
    ax.set_facecolor(BG)
    x_lo, x_hi = mean - 145.0, mean + 145.0
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, Y_MAX)

    # Faint dashed grid (as in the chart).
    for gy in np.linspace(0, Y_MAX, 6):
        ax.axhline(gy, color=GRID, lw=0.7, dashes=(2, 3), zorder=0)
    for gx in range(0, 401, 50):
        if x_lo < gx < x_hi:
            ax.axvline(gx, color=GRID, lw=0.7, dashes=(2, 3), zorder=0)

    # Ghost family: every computed curve, faint, coloured by its theta.
    for t in family:
        sc, de, _, _ = load_curve(t)
        ax.plot(sc, de, color=theta_color(t), lw=1.0, alpha=0.16, zorder=1)

    # Featured theta=0.04 curve: filled + bold.
    ax.fill_between(scores, dens, color=color, alpha=0.18, zorder=2)
    ax.plot(scores, dens, color=color, lw=4.0, zorder=5, solid_capstyle="round")

    # +-1..4 sigma lines (graded), matching the distribution chart.
    for k, alpha in [(1, 0.7), (2, 0.5), (3, 0.34), (4, 0.22)]:
        for sign in (-1, 1):
            sx = mean + sign * k * std
            if sx <= x_lo or sx >= x_hi:
                continue
            ax.axvline(sx, color=color, lw=1.4, dashes=(5, 3), alpha=alpha, zorder=4)
            ax.text(sx + (4 if sign > 0 else -4), Y_MAX * 0.03,
                    f"{'+' if sign > 0 else '−'}{k}σ", color=color,
                    alpha=min(alpha + 0.2, 0.95), fontsize=12,
                    ha="left" if sign > 0 else "right", va="bottom", zorder=6)

    # Mean line (heaviest).
    ax.axvline(mean, color=color, lw=2.4, dashes=(5, 3), zorder=6)
    ax.text(mean + 6, Y_MAX * 0.95, f"mean = {mean:.1f}", color=color,
            fontsize=15, ha="left", va="top", zorder=7)
    ax.axis("off")

    # Headline strip on top (share cards need a legible title).
    fig.text(0.5, 0.955, "Can You Be Skilled At Playing Yatzy?",
             ha="center", va="top", color=TEXT, fontsize=34,
             family="serif", fontweight="bold")
    fig.text(0.5, 0.825, "Computing the optimal strategy for Scandinavian Yatzy",
             ha="center", va="top", color=MUTED, fontsize=18, family="serif")

    out = Path("treatise/og-cover.png")
    fig.savefig(out, facecolor=BG)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB), family={len(family)} curves")


if __name__ == "__main__":
    main()
