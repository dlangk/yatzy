"""Generate score spray data for the treatise prologue visualization.

Reads simulation_raw.bin (1M games), samples 10K, outputs:
  - treatise/data/score_spray.json  (~400 KB, per-game data)
  - treatise/data/score_spray_meta.json  (~2 KB, KDE + population stats)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Add analytics to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analytics" / "src"))

from yatzy_analysis.io import read_full_recording

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

POPULATION_COLORS = [
    {"label": "No bonus, no Yatzy", "color": "#8c8c8c"},
    {"label": "No bonus, Yatzy",    "color": "#f4987a"},
    {"label": "Bonus, no Yatzy",    "color": "#3b4cc0"},
    {"label": "Bonus + Yatzy",      "color": "#F37021"},
]

N_SAMPLE = 10_000
SEED = 42


def classify_population(bonus: bool, yatzy_score: int) -> int:
    """0=no bonus/no Yatzy, 1=no bonus/Yatzy, 2=bonus/no Yatzy, 3=bonus+Yatzy."""
    has_yatzy = yatzy_score > 0
    return int(bonus) * 2 + int(has_yatzy)


def main() -> None:
    raw_path = REPO_ROOT / "data" / "simulations" / "simulation_raw.bin"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {raw_path} ...")
    rec = read_full_recording(raw_path)
    if rec is None:
        print("Error: failed to read recording", file=sys.stderr)
        sys.exit(1)

    n_total = rec["num_games"]
    print(f"  {n_total:,} games loaded")

    # Sample
    rng = np.random.default_rng(SEED)
    indices = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)
    indices.sort()

    totals = rec["total_scores"][indices].astype(int)
    bonuses = rec["got_bonus"][indices]
    uppers = rec["upper_totals"][indices].astype(int)
    cats = rec["category_scores"][indices]  # (N, 15)

    # Build per-game records
    games = []
    pop_counts = [0, 0, 0, 0]
    pop_totals = [0.0, 0.0, 0.0, 0.0]

    for i in range(len(indices)):
        total = int(totals[i])
        bonus = bool(bonuses[i])
        upper = int(uppers[i])
        cat_scores = cats[i].tolist()
        pop = classify_population(bonus, cat_scores[14])  # Yatzy is index 14
        pop_counts[pop] += 1
        pop_totals[pop] += total

        games.append({
            "total": total,
            "upper": upper,
            "bonus": bonus,
            "cats": cat_scores,
            "pop": pop,
        })

    # Compute meta stats
    all_totals = np.array([g["total"] for g in games], dtype=float)
    mean_val = float(np.mean(all_totals))
    std_val = float(np.std(all_totals))
    median_val = int(np.median(all_totals))
    min_val = int(np.min(all_totals))
    max_val = int(np.max(all_totals))

    # KDE
    kde = stats.gaussian_kde(all_totals, bw_method=0.04)
    x_kde = np.linspace(max(min_val - 20, 0), min(max_val + 20, 400), 300)
    y_kde = kde(x_kde)
    kde_points = [[round(float(x), 1), round(float(y), 6)] for x, y in zip(x_kde, y_kde)]

    # Population stats
    n = len(games)
    populations = []
    for p_idx in range(4):
        frac = pop_counts[p_idx] / n if n > 0 else 0
        p_mean = pop_totals[p_idx] / pop_counts[p_idx] if pop_counts[p_idx] > 0 else 0
        populations.append({
            **POPULATION_COLORS[p_idx],
            "fraction": round(frac, 4),
            "mean": round(p_mean, 1),
        })

    # Write outputs
    out_dir = REPO_ROOT / "treatise" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    spray_path = out_dir / "score_spray.json"
    with open(spray_path, "w") as f:
        json.dump({"games": games}, f, separators=(",", ":"))
    print(f"  Wrote {spray_path} ({spray_path.stat().st_size / 1024:.0f} KB)")

    meta_path = out_dir / "score_spray_meta.json"
    meta = {
        "mean": round(mean_val, 1),
        "std": round(std_val, 1),
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "n": n,
        "kde": kde_points,
        "populations": populations,
        "category_names": CATEGORY_NAMES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
