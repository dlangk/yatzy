"""Tail distribution analysis for max-policy Yatzy simulations.

Provides empirical tail statistics, log-linear extrapolation, and analytical
probability estimates for the theoretical maximum score (374).
"""
from __future__ import annotations

import math
import random
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# IO: load scores from legacy scores.bin format (u32 count + i32[count])
# ---------------------------------------------------------------------------


def load_scores_bin(path: Path) -> NDArray[np.int32]:
    """Load scores from legacy scores.bin format (u32 count + i32[count]).

    Returns sorted int32 array.
    """
    with open(path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        raw = struct.unpack(f"<{count}i", f.read(count * 4))
    arr = np.array(raw, dtype=np.int32)
    arr.sort()
    return arr


# ---------------------------------------------------------------------------
# Empirical tail distribution
# ---------------------------------------------------------------------------


def tail_distribution(scores: NDArray[np.int32]) -> dict:
    """Compute empirical tail statistics at standard thresholds.

    Returns a dict with keys:
      - n: total number of scores
      - mean: mean score
      - max: maximum score
      - thresholds: list of dicts with keys (threshold, count, fraction, inv_fraction)
    """
    n = len(scores)
    mean = float(scores.mean())
    max_score = int(scores[-1]) if len(scores) > 0 else 0

    thresholds = []
    for t in range(200, 325, 10):
        above = int(np.sum(scores >= t))
        frac = above / n if n > 0 else 0.0
        inv = 1.0 / frac if frac > 0 else float("inf")
        thresholds.append({
            "threshold": t,
            "count": above,
            "fraction": frac,
            "inv_fraction": inv,
        })

    return {
        "n": n,
        "mean": mean,
        "max": max_score,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Log-linear tail fit
# ---------------------------------------------------------------------------


def fit_log_linear(scores: NDArray[np.int32]) -> dict:
    """Fit log10(P(X >= t)) = a + b*t via OLS on thresholds with >= 10 samples.

    Returns a dict with keys:
      - a, b: intercept and slope of the log-linear fit
      - decay_per_point: 10^b (multiplier per score point)
      - log10_p374: extrapolated log10(P(score >= 374))
      - p374: extrapolated P(score >= 374)
      - games_needed: 1 / p374
      - n_points: number of threshold points used in the fit
    """
    n = len(scores)
    ts: list[float] = []
    log_fracs: list[float] = []
    for threshold in range(220, 315, 5):
        above = int(np.sum(scores >= threshold))
        if above >= 10:
            frac = above / n
            ts.append(float(threshold))
            log_fracs.append(math.log10(frac))

    n_pts = len(ts)
    if n_pts < 2:
        return {
            "a": 0.0, "b": 0.0, "decay_per_point": 1.0,
            "log10_p374": 0.0, "p374": 0.0, "games_needed": float("inf"),
            "n_points": n_pts,
        }

    sx = sum(ts)
    sy = sum(log_fracs)
    sxx = sum(t * t for t in ts)
    sxy = sum(t * y for t, y in zip(ts, log_fracs))
    b = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx * sx)
    a = (sy - b * sx) / n_pts

    log10_p374 = a + b * 374
    p374 = 10**log10_p374

    return {
        "a": a,
        "b": b,
        "decay_per_point": 10**b,
        "log10_p374": log10_p374,
        "p374": p374,
        "games_needed": 1.0 / p374 if p374 > 0 else float("inf"),
        "n_points": n_pts,
    }


# ---------------------------------------------------------------------------
# Analytical probability calculations
# ---------------------------------------------------------------------------


def p_five_of_specific_face() -> float:
    """P(all 5 dice show a specific face) with 3 rolls, keeping matches.

    Derivation:
      After roll 1 with k matches, remaining n=5-k dice need all to match
      in 2 more rolls. P_remain(n) = (11/36)^n.

      P = Sum_{k=0}^{5} C(5,k) (1/6)^k (5/6)^{5-k} (11/36)^{5-k}
        = (1/6 + 5/6 * 11/36)^5
        = (91/216)^5
    """
    return (91 / 216) ** 5


def p_at_least_n_of_face(need: int, n_dice: int = 5) -> float:
    """P(at least `need` dice show a specific face) in 3 rolls, keeping matches."""
    p = 1 / 6
    q = 5 / 6
    total = 0.0
    for k1 in range(n_dice + 1):
        p_k1 = math.comb(n_dice, k1) * p**k1 * q ** (n_dice - k1)
        have1 = min(k1, need)
        remaining1 = n_dice - have1
        if have1 >= need:
            total += p_k1
            continue
        for k2 in range(remaining1 + 1):
            p_k2 = math.comb(remaining1, k2) * p**k2 * q ** (remaining1 - k2)
            have2 = have1 + k2
            if have2 >= need:
                total += p_k1 * p_k2
                continue
            remaining2 = n_dice - have2
            still_need2 = need - have2
            p_enough = sum(
                math.comb(remaining2, j) * p**j * q ** (remaining2 - j)
                for j in range(still_need2, remaining2 + 1)
            )
            total += p_k1 * p_k2 * p_enough
    return total


def _p_specific_combo_mc(
    dice_needed: tuple[int, ...], n_trials: int = 5_000_000
) -> float:
    """Monte Carlo estimate: P(getting exact sorted dice combo) in 3 rolls."""
    rng = random.Random(42)
    hits = 0
    target = tuple(sorted(dice_needed))
    for _ in range(n_trials):
        dice = [rng.randint(1, 6) for _ in range(5)]
        for _ in range(2):  # 2 reroll opportunities
            needed = list(target)
            keep: list[int] = []
            for d in sorted(dice):
                if d in needed:
                    keep.append(d)
                    needed.remove(d)
            n_reroll = 5 - len(keep)
            if n_reroll == 0:
                break
            dice = keep + [rng.randint(1, 6) for _ in range(n_reroll)]
        if tuple(sorted(dice)) == target:
            hits += 1
    return hits / n_trials


def _p_two_pairs_max_mc(n_trials: int = 2_000_000) -> float:
    """Monte Carlo: P(at least 2 sixes and 2 fives) in 3 rolls."""
    rng = random.Random(42)
    hits = 0
    for _ in range(n_trials):
        dice = [rng.randint(1, 6) for _ in range(5)]
        for _ in range(2):
            if dice.count(6) >= 2 and dice.count(5) >= 2:
                break
            keep: list[int] = []
            s6, s5 = 0, 0
            for d in dice:
                if d == 6 and s6 < 2:
                    keep.append(d)
                    s6 += 1
                elif d == 5 and s5 < 2:
                    keep.append(d)
                    s5 += 1
            n_reroll = 5 - len(keep)
            dice = keep + [rng.randint(1, 6) for _ in range(n_reroll)]
        if dice.count(6) >= 2 and dice.count(5) >= 2:
            hits += 1
    return hits / n_trials


def analytical_p374() -> dict:
    """Compute per-category probabilities and combined P(374) estimate.

    Returns a dict with keys:
      - categories: list of dicts with (name, probability, inv_probability)
      - overall: product of all category probabilities (assuming independence)
      - games_needed: 1 / overall
      - games_per_sec: assumed simulation throughput
      - years: estimated wall-clock time at games_per_sec
    """
    p5 = p_five_of_specific_face()

    categories = [
        ("Ones (5x1)", p5),
        ("Twos (5x2)", p5),
        ("Threes (5x3)", p5),
        ("Fours (5x4)", p5),
        ("Fives (5x5)", p5),
        ("Sixes (5x6)", p5),
        ("One pair (>=2 sixes)", p_at_least_n_of_face(2)),
        ("Two pairs (>=2x5 + >=2x6)", _p_two_pairs_max_mc()),
        ("Three-of-kind (>=3 sixes)", p_at_least_n_of_face(3)),
        ("Four-of-kind (>=4 sixes)", p_at_least_n_of_face(4)),
        ("Small straight (1,2,3,4,5)", _p_specific_combo_mc((1, 2, 3, 4, 5))),
        ("Large straight (2,3,4,5,6)", _p_specific_combo_mc((2, 3, 4, 5, 6))),
        ("Full house (5,5,6,6,6)", _p_specific_combo_mc((5, 5, 6, 6, 6))),
        ("Chance (5x6)", p5),
        ("Yatzy (any 5-of-kind)", 6 * p5),
    ]

    cat_dicts = []
    probs = []
    for name, p in categories:
        inv = 1.0 / p if p > 0 else float("inf")
        cat_dicts.append({"name": name, "probability": p, "inv_probability": inv})
        probs.append(p)

    overall = math.prod(probs)
    games_needed = 1.0 / overall if overall > 0 else float("inf")
    games_per_sec = 50_000
    seconds = games_needed / games_per_sec
    years = seconds / (365.25 * 24 * 3600)

    return {
        "p_five_of_specific_face": p5,
        "categories": cat_dicts,
        "overall": overall,
        "games_needed": games_needed,
        "games_per_sec": games_per_sec,
        "years": years,
    }
