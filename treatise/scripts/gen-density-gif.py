"""Render a GIF of the exact score-density distribution sweeping theta.

Reproduces the treatise's risk-theta distribution chart (dark theme) as an
animation: theta sweeps 0 -> 0.4 -> 0, one frame per computed theta, with the
mean and +-1/2/3 sigma lines. Frames are rendered with matplotlib and assembled
into a high-quality GIF with ffmpeg (palettegen/paletteuse).

Run:
    uv run --with matplotlib,scipy,numpy treatise/scripts/gen-density-gif.py \
        --out outputs/density_sweep.gif

Data source: outputs/density/density_*.json (exact PMFs with mean/std,
one per theta — the source of truth, independent of treatise data files).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# ── Theme (mirrors treatise dark theme + yatzy-viz.js thetaColor) ──────────
BG = "#1a1a1a"
GRID = "#333333"
MUTED = "#999999"
TEXT = "#e0ddd5"

THETA_STOPS = [
    (-0.30, "#3b4cc0"),
    (-0.15, "#8db0fe"),
    (0.00, "#F37021"),
    (0.15, "#f4987a"),
    (0.30, "#b40426"),
]

X_MAX = 400.0
Y_MAX = 0.015
SMOOTH_SIGMA = 1.5  # emulates D3 curveBasis smoothing


def theta_color(theta: float):
    t = max(-0.30, min(0.30, theta))
    for (t0, c0), (t1, c1) in zip(THETA_STOPS, THETA_STOPS[1:]):
        if t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
            return tuple(a + (b - a) * frac)
    return to_rgb(THETA_STOPS[-1][1])


def load_data():
    """Load exact PMFs straight from the density files (fail loudly if one
    is missing — never fall back to a nearby theta)."""
    base = Path("outputs/density")
    thetas = [round(0.01 * i, 2) for i in range(0, 41)]  # 0.00 .. 0.40
    curves, stats = {}, {}
    for t in thetas:
        fname = base / ("density_0.json" if t == 0 else f"density_{t:g}.json")
        if not fname.exists():
            raise SystemExit(f"ERROR: missing {fname}; compute it first "
                             f"(yatzy-density --thetas {t})")
        d = json.load(open(fname))
        scores = np.array([p[0] for p in d["pmf"]], dtype=float)
        dens = gaussian_filter1d(
            np.array([p[1] for p in d["pmf"]], dtype=float), SMOOTH_SIGMA)
        curves[t] = (scores, dens)
        stats[t] = (d["mean"], d["std_dev"])
    return curves, stats


def draw_frame(theta, curves, stats, path):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)

    # Horizontal grid
    for gy in np.linspace(0, Y_MAX, 6):
        ax.axhline(gy, color=GRID, lw=0.8, dashes=(2, 3), zorder=0)

    # Ghost family: every computed curve, faint, in its own theta color
    for th, (sc, de) in curves.items():
        ax.plot(sc, de, color=theta_color(th), lw=1.0, alpha=0.15, zorder=1)

    color = theta_color(theta)
    sc, de = curves[theta]

    # Selected curve: fill + bold line
    ax.fill_between(sc, de, color=color, alpha=0.12, zorder=2)
    ax.plot(sc, de, color=color, lw=2.6, zorder=5, solid_capstyle="round")

    mean, std = stats[theta]

    # +-1/2/3 sigma lines (graded so the mean stays dominant)
    for k, alpha in [(1, 0.70), (2, 0.52), (3, 0.36)]:
        for sign in (-1, 1):
            sx = mean + sign * k * std
            if sx <= 0 or sx >= X_MAX:
                continue
            ax.axvline(sx, color=color, lw=1.5, dashes=(5, 3), alpha=alpha, zorder=4)
            ax.text(sx + (4 if sign > 0 else -4), 0.0006,
                    f"{'+' if sign > 0 else '−'}{k}σ",
                    color=color, alpha=min(alpha + 0.2, 0.95), fontsize=9,
                    ha="left" if sign > 0 else "right", va="bottom", zorder=6)

    # Mean line (heaviest)
    ax.axvline(mean, color=color, lw=2.4, dashes=(5, 3), zorder=6)
    ax.text(mean + 5, Y_MAX * 0.955, f"mean = {mean:.1f}", color=color,
            fontsize=11, ha="left", va="top", zorder=7)

    # Threshold lines at 100 and 360
    for val, ha, dx in [(100, "left", 4), (360, "right", -4)]:
        ax.axvline(val, color=MUTED, lw=1.0, dashes=(2, 3), alpha=0.6, zorder=3)
        ax.text(val + dx, Y_MAX * 0.90, str(val), color=MUTED, fontsize=9,
                ha=ha, va="top", zorder=6)

    # Theta readout (top-left)
    sign = "+" if theta >= 0 else "−"
    ax.text(0.012, 0.955, f"θ = {sign}{abs(theta):.2f}", transform=ax.transAxes,
            color=TEXT, fontsize=20, ha="left", va="top",
            family="monospace", fontweight="bold", zorder=8)

    # Axes cosmetics
    ax.set_xlabel("Score", color=MUTED, fontsize=12)
    ax.set_ylabel("Density", color=MUTED, fontsize=12)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 401, 50))
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def build_sequence(available):
    """theta 0 -> 0.4 -> 0, one frame per computed theta, with endpoint holds."""
    up = [round(0.01 * i, 2) for i in range(0, 41)]        # 0.00 .. 0.40
    up = [t for t in up if t in available]
    seq = up + up[-2::-1]                                   # there and back
    frames = []
    for i, t in enumerate(seq):
        reps = 1
        if i == 0:
            reps = 4                      # hold at start (theta=0)
        elif t == max(up):
            reps = 5                      # hold at the peak (theta=0.4)
        elif i == len(seq) - 1:
            reps = 6                      # hold at the end (theta=0)
        frames.extend([t] * reps)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/density_sweep.gif")
    ap.add_argument("--fps", type=int, default=8, help="lower = slower")
    args = ap.parse_args()

    curves, stats = load_data()
    available = set(curves) & set(stats)
    frames = build_sequence(available)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, theta in enumerate(frames):
            draw_frame(theta, curves, stats, tmp / f"frame_{i:04d}.png")
        print(f"Rendered {len(frames)} frames")

        vf = ("split[s0][s1];[s0]palettegen=max_colors=256[p];"
              "[s1][p]paletteuse=dither=sierra2_4a")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%04d.png"),
            "-vf", vf, "-loop", "0", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.1f} MB, {args.fps} fps)")


if __name__ == "__main__":
    main()
