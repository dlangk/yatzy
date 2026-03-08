#!/usr/bin/env python3
"""Generate treatise JSON data files from real simulation data.

Reads:
  - data/simulations/theta/theta_0.000/simulation_raw.bin (1M games, full recording)
  - outputs/aggregates/csv/category_stats.csv (per-category stats from category_sweep)

Produces:
  - treatise/data/category_landscape.json
  - treatise/data/category_pmfs.json
  - treatise/data/mixture.json
  - treatise/data/bonus_covariance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add analytics package to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "analytics" / "src"))

from yatzy_analysis.io import read_full_recording

# Category metadata
CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

CATEGORY_SECTIONS = ["upper"] * 6 + ["lower"] * 9

# Maximum possible scores per category (Scandinavian Yatzy)
CATEGORY_CEILINGS = [5, 10, 15, 20, 25, 30, 12, 22, 18, 24, 15, 20, 28, 30, 50]

# PMF type classification for chart styling
CATEGORY_PMF_TYPES = [
    "continuous", "continuous", "continuous", "continuous", "continuous", "continuous",
    "discrete", "discrete", "discrete", "discrete",
    "binary", "binary", "discrete", "continuous", "binary",
]


def read_category_stats_csv(csv_path: Path) -> dict:
    """Read category_stats.csv and extract theta=0 rows."""
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta = float(row["theta"])
            if abs(theta) < 1e-9:  # theta = 0
                rows.append(row)
    return rows


def generate_category_landscape(csv_path: Path, rec: dict, out_path: Path):
    """Generate category_landscape.json from category_stats.csv (theta=0) and simulation data."""
    csv_rows = read_category_stats_csv(csv_path)

    # Compute variance contributions from the simulation data
    cat_scores = rec["category_scores"]  # (N, 15) uint8
    total_scores = rec["total_scores"].astype(np.float64)  # (N,)
    n = rec["num_games"]

    # Variance contribution: Cov(cat_score, total_score) for each category
    # This measures how much each category contributes to total score variance
    total_mean = total_scores.mean()
    total_var = total_scores.var()

    result = []
    for row in csv_rows:
        cat_id = int(row["category_id"])
        cat_vals = cat_scores[:, cat_id].astype(np.float64)
        cat_mean = cat_vals.mean()

        # Covariance contribution: Cov(X_i, Total) = sum of all covariances involving X_i
        # This is the proper decomposition: Var(Total) = sum_i Cov(X_i, Total)
        cov_with_total = np.mean((cat_vals - cat_mean) * (total_scores - total_mean))

        result.append({
            "id": cat_id,
            "name": CATEGORY_NAMES[cat_id],
            "section": CATEGORY_SECTIONS[cat_id],
            "ceiling": CATEGORY_CEILINGS[cat_id],
            "mean_score": round(float(row["mean_score"]), 2),
            "zero_rate": round(float(row["zero_rate"]), 4),
            "mean_fill_turn": round(float(row["mean_fill_turn"]), 2),
            "score_pct_ceiling": round(float(row["score_pct_ceiling"]) / 100.0, 4),
            "variance_contribution": round(float(cov_with_total), 1),
            "hit_rate": round(float(row["hit_rate"]), 4),
        })

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  category_landscape.json: {len(result)} categories")


def generate_category_pmfs(rec: dict, out_path: Path):
    """Generate category_pmfs.json from simulation category_scores."""
    cat_scores = rec["category_scores"]  # (N, 15) uint8
    n = rec["num_games"]

    categories = []
    for cat_id in range(15):
        vals = cat_scores[:, cat_id]
        ceiling = CATEGORY_CEILINGS[cat_id]

        # Build PMF: count occurrences of each score value
        max_val = int(vals.max())
        counts = np.bincount(vals, minlength=max_val + 1)
        pmf = []
        for score_val in range(len(counts)):
            prob = counts[score_val] / n
            if prob > 1e-6:  # skip negligible probabilities
                pmf.append({
                    "score": int(score_val),
                    "prob": round(float(prob), 6),
                })

        categories.append({
            "id": cat_id,
            "name": CATEGORY_NAMES[cat_id],
            "ceiling": ceiling,
            "type": CATEGORY_PMF_TYPES[cat_id],
            "pmf": pmf,
        })

    with open(out_path, "w") as f:
        json.dump({"categories": categories}, f, indent=2)
    print(f"  category_pmfs.json: {len(categories)} categories, {sum(len(c['pmf']) for c in categories)} PMF entries")


def generate_mixture(rec: dict, out_path: Path):
    """Generate mixture.json: four sub-populations (bonus x yatzy)."""
    total_scores = rec["total_scores"].astype(np.float64)
    got_bonus = rec["got_bonus"]
    cat_scores = rec["category_scores"]
    n = rec["num_games"]

    # Yatzy is category 14; scored > 0 means hit
    got_yatzy = cat_scores[:, 14] > 0

    populations = []
    labels = [
        ("No bonus, no Yatzy", False, False),
        ("No bonus, scored Yatzy", False, True),
        ("Hit bonus, no Yatzy", True, False),
        ("Hit bonus, scored Yatzy", True, True),
    ]

    for label, bonus_flag, yatzy_flag in labels:
        mask = (got_bonus == bonus_flag) & (got_yatzy == yatzy_flag)
        count = int(mask.sum())
        if count == 0:
            populations.append({
                "label": label,
                "fraction": 0.0,
                "mean": 0.0,
                "std": 1.0,
            })
            continue

        subset = total_scores[mask]
        populations.append({
            "label": label,
            "fraction": round(count / n, 4),
            "mean": round(float(subset.mean()), 1),
            "std": round(float(subset.std()), 1),
        })

    with open(out_path, "w") as f:
        json.dump({"populations": populations}, f, indent=2)

    print(f"  mixture.json: {len(populations)} populations")
    for p in populations:
        print(f"    {p['label']}: {p['fraction']*100:.1f}%, mean={p['mean']}, std={p['std']}")


def generate_bonus_covariance(rec: dict, out_path: Path):
    """Generate bonus_covariance.json: decomposition of bonus advantage.

    The bonus is worth more than 50 points because games that hit the bonus
    tend to have rolled better overall. We decompose the gap:
      total_gap = bonus_itself (50) + upper_lift + lower_lift
    where:
      upper_lift = E[upper_total | bonus] - E[upper_total | no bonus] - 50
        (games hitting bonus have higher upper scores beyond just the 50 pts)
      lower_lift = E[lower_total | bonus] - E[lower_total | no bonus]
        (positive cross-section correlation)
    """
    total_scores = rec["total_scores"].astype(np.float64)
    got_bonus = rec["got_bonus"]
    cat_scores = rec["category_scores"].astype(np.float64)

    bonus_mask = got_bonus
    no_bonus_mask = ~got_bonus

    n_bonus = int(bonus_mask.sum())
    n_no_bonus = int(no_bonus_mask.sum())

    if n_no_bonus == 0 or n_bonus == 0:
        print("  WARNING: Cannot compute bonus decomposition (all or no games hit bonus)")
        return

    # Upper section total (categories 0-5, raw scores without bonus)
    upper_raw_bonus = cat_scores[bonus_mask, :6].sum(axis=1).mean()
    upper_raw_no_bonus = cat_scores[no_bonus_mask, :6].sum(axis=1).mean()
    upper_lift = round(upper_raw_bonus - upper_raw_no_bonus, 1)

    # Lower section total (categories 6-14)
    lower_bonus = cat_scores[bonus_mask, 6:].sum(axis=1).mean()
    lower_no_bonus = cat_scores[no_bonus_mask, 6:].sum(axis=1).mean()
    lower_lift = round(lower_bonus - lower_no_bonus, 1)

    total_gap = round(50 + upper_lift + lower_lift, 1)

    result = {
        "total_gap": total_gap,
        "components": [
            {"label": "Bonus itself", "value": 50},
            {"label": "Better upper scores", "value": upper_lift},
            {"label": "Better lower scores", "value": lower_lift},
        ],
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  bonus_covariance.json: 50 + {upper_lift} + {lower_lift} = {total_gap}")


def main():
    raw_path = repo_root / "data" / "simulations" / "theta" / "theta_0.000" / "simulation_raw.bin"
    csv_path = repo_root / "outputs" / "aggregates" / "csv" / "category_stats.csv"
    out_dir = repo_root / "treatise" / "data"

    print(f"Reading simulation data from {raw_path}...")
    rec = read_full_recording(raw_path)
    if rec is None:
        print("ERROR: Could not read simulation_raw.bin")
        sys.exit(1)
    print(f"  {rec['num_games']:,} games loaded")

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print(f"\nGenerating treatise data files in {out_dir}/...")

    generate_category_landscape(csv_path, rec, out_dir / "category_landscape.json")
    generate_category_pmfs(rec, out_dir / "category_pmfs.json")
    generate_mixture(rec, out_dir / "mixture.json")
    generate_bonus_covariance(rec, out_dir / "bonus_covariance.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
