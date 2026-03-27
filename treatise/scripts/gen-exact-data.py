"""Generate treatise chart data from exact forward-DP density files.

Replaces Monte Carlo KDE curves and sweep summary with exact data:
- treatise/data/kde_curves.json → exact PMFs (no smoothing needed)
- treatise/data/sweep_summary.json → exact mean, std, percentiles, min, max
- treatise/data/tail_exact.json → exact tail probabilities

Run: python3 treatise/scripts/gen-exact-data.py
"""

import json
import numpy as np
from pathlib import Path

DENSITY_DIR = Path("outputs/density")
TREATISE_DATA = Path("treatise/data")

# Slider thetas (±0.4 range)
SLIDER_THETAS = [
    -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.07, -0.05, -0.04,
    -0.03, -0.02, -0.015, -0.01, -0.005,
    0,
    0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15,
    0.2, 0.25, 0.3, 0.35, 0.4,
]

# Tail chart thetas (wider range)
TAIL_THETAS = [0, 0.05, 0.1, 0.2, 0.5]
TAIL_EXTRA = [-0.1, -0.3]


def load_density(theta: float) -> dict | None:
    if theta == 0:
        fname = DENSITY_DIR / "density_0.json"
    else:
        fname = DENSITY_DIR / f"density_{theta}.json"
    if not fname.exists():
        # Try alternate formats
        for fmt in [f"{theta:g}", f"{theta:.1f}", f"{theta:.2f}", f"{theta:.3f}"]:
            alt = DENSITY_DIR / f"density_{fmt}.json"
            if alt.exists():
                fname = alt
                break
    if not fname.exists():
        return None
    return json.load(open(fname))


def pmf_to_kde_entry(theta: float, pmf_data: list) -> dict:
    """Convert exact PMF to KDE-compatible format for the chart."""
    scores = [p[0] for p in pmf_data]
    probs = [p[1] for p in pmf_data]
    return {"theta": theta, "score": scores, "density": probs}


def pmf_to_summary(theta: float, d: dict) -> dict:
    """Extract summary stats from exact density."""
    pmf = d["pmf"]
    scores = np.array([p[0] for p in pmf])
    probs = np.array([p[1] for p in pmf])

    # Min/max with meaningful probability threshold
    meaningful = probs > 1e-9
    score_min = int(scores[meaningful].min()) if meaningful.any() else int(scores[0])
    score_max = int(scores[meaningful].max()) if meaningful.any() else int(scores[-1])

    pcts = d["percentiles"]
    return {
        "theta": theta,
        "mean": d["mean"],
        "std": d["std_dev"],
        "p1": pcts["p1"],
        "p5": pcts["p5"],
        "p10": pcts["p10"],
        "p25": pcts["p25"],
        "p50": pcts["p50"],
        "p75": pcts["p75"],
        "p90": pcts["p90"],
        "p95": pcts["p95"],
        "p99": pcts["p99"],
        "skewness": 0.0,  # not in density files
        "kurtosis": 0.0,
        "cvar_5": 0.0,
        "min": score_min,
        "max": score_max,
    }


def pmf_to_tail(theta: float, pmf_data: list, mean: float) -> dict:
    """Compute tail probabilities from exact PMF."""
    scores = [p[0] for p in pmf_data]
    probs = [p[1] for p in pmf_data]
    tail = []
    for i in range(len(scores)):
        if scores[i] < 250 or probs[i] <= 0:
            continue
        surv = sum(probs[j] for j in range(i, len(scores)))
        tail.append({"score": scores[i], "prob": probs[i], "survival": surv})
    return {"theta": theta, "mean": mean, "tail": tail}


def main():
    # --- KDE curves (exact PMFs) ---
    kde_curves = []
    missing_kde = []
    for theta in SLIDER_THETAS:
        d = load_density(theta)
        if d is None:
            missing_kde.append(theta)
            continue
        kde_curves.append(pmf_to_kde_entry(theta, d["pmf"]))
    kde_curves.sort(key=lambda x: x["theta"])

    if missing_kde:
        print(f"WARNING: Missing density for KDE: {missing_kde}")
    else:
        print(f"KDE curves: {len(kde_curves)} thetas (all exact)")

    out = TREATISE_DATA / "kde_curves.json"
    json.dump(kde_curves, open(out, "w"))
    print(f"  → {out}")

    # --- Sweep summary (exact stats) ---
    # Keep existing entries for thetas outside slider range, update slider range
    existing = json.load(open(TREATISE_DATA / "sweep_summary.json"))
    existing_map = {round(d["theta"], 4): d for d in existing}

    for theta in SLIDER_THETAS:
        d = load_density(theta)
        if d is None:
            continue
        existing_map[round(theta, 4)] = pmf_to_summary(theta, d)

    summary = sorted(existing_map.values(), key=lambda x: x["theta"])
    out = TREATISE_DATA / "sweep_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"Sweep summary: {len(summary)} thetas → {out}")

    # --- Tail probabilities ---
    tail_data = []
    for theta in TAIL_THETAS + TAIL_EXTRA:
        d = load_density(theta)
        if d is None:
            print(f"  WARNING: Missing density for tail theta={theta}")
            continue
        tail_data.append(pmf_to_tail(theta, d["pmf"], d["mean"]))

    out = TREATISE_DATA / "tail_exact.json"
    json.dump(tail_data, open(out, "w"))
    print(f"Tail data: {len(tail_data)} thetas → {out}")

    # --- Verify ---
    t0 = next(d for d in summary if abs(d["theta"]) < 0.001)
    print(f"\nVerification (θ=0):")
    print(f"  Mean: {t0['mean']:.2f}")
    print(f"  Std:  {t0['std']:.2f}")
    print(f"  Min:  {t0['min']} (P>1e-9 threshold)")
    print(f"  Max:  {t0['max']} (P>1e-9 threshold)")


if __name__ == "__main__":
    main()
