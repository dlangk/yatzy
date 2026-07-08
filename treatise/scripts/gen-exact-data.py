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
SLIDER_THETAS = sorted(set(
    [round(-0.4 + i * 0.01, 3) for i in range(81)]  # -0.40 to +0.40 in 0.01 steps
))

# Tail chart thetas (wider range)
TAIL_THETAS = [0, 0.05, 0.1, 0.2, 0.5]
TAIL_EXTRA = [-0.1, -0.3]


def load_density(theta: float) -> dict | None:
    if theta == 0:
        fname = DENSITY_DIR / "density_0.json"
    else:
        fname = DENSITY_DIR / f"density_{theta}.json"
    if not fname.exists():
        # Try alternate filename formats, but ONLY ones that represent the exact
        # same theta value. A lossy format like f"{0.06:.1f}" == "0.1" would load
        # a DIFFERENT theta's density (0.06 silently became 0.1), putting that
        # point off the frontier curve. Guard every candidate with a value check.
        for fmt in [f"{theta:g}", f"{theta:.2f}", f"{theta:.3f}", f"{theta:.4f}"]:
            alt = DENSITY_DIR / f"density_{fmt}.json"
            if alt.exists() and abs(float(fmt) - theta) < 1e-9:
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


def require(theta: float) -> dict:
    """Load exact density for theta, or fail loudly. We never fall back to a
    nearby theta or silently skip: emitting a chart point for a theta we did not
    actually compute is how a wrong value (e.g. theta=0.06 showing 0.1's data)
    ends up on the frontier. If a file is missing, compute it, don't approximate."""
    d = load_density(theta)
    if d is None:
        raise SystemExit(
            f"ERROR: no exact density for theta={theta} "
            f"(expected {DENSITY_DIR}/density_{theta}.json). "
            f"Compute it first:  yatzy-density --thetas {theta}\n"
            f"Refusing to emit incomplete/approximate chart data."
        )
    return d


def main():
    # Every theta below is REQUIRED; require() raises if its density is missing,
    # so we can never ship a chart point that was faked from another theta.

    # --- KDE curves (exact PMFs) ---
    kde_curves = sorted(
        (pmf_to_kde_entry(t, require(t)["pmf"]) for t in SLIDER_THETAS),
        key=lambda x: x["theta"],
    )
    json.dump(kde_curves, open(TREATISE_DATA / "kde_curves.json", "w"))
    print(f"KDE curves: {len(kde_curves)} thetas (all exact)")

    # --- Sweep summary (exact stats) --- built purely from computed density,
    # never seeded from an existing file (that preserved stale/wrong entries).
    summary = sorted(
        (pmf_to_summary(t, require(t)) for t in SLIDER_THETAS),
        key=lambda x: x["theta"],
    )
    json.dump(summary, open(TREATISE_DATA / "sweep_summary.json", "w"), indent=2)
    print(f"Sweep summary: {len(summary)} thetas")

    # --- Tail probabilities ---
    tail_data = []
    for theta in TAIL_THETAS + TAIL_EXTRA:
        d = require(theta)
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
