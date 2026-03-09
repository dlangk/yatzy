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
  - treatise/data/category_correlations.json
  - treatise/data/fill_turn_heatmap.json
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
    fill_turns = rec["fill_turns"]  # (N, 15) uint8
    total_scores = rec["total_scores"].astype(np.float64)  # (N,)
    got_bonus = rec["got_bonus"]
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
        cat_std = cat_vals.std()

        # Covariance contribution: Cov(X_i, Total) = sum of all covariances involving X_i
        # This is the proper decomposition: Var(Total) = sum_i Cov(X_i, Total)
        cov_with_total = np.mean((cat_vals - cat_mean) * (total_scores - total_mean))

        # Score skewness: mean(((x - mean) / std)^3), guarded for std=0
        if cat_std > 0:
            score_skewness = float(np.mean(((cat_vals - cat_mean) / cat_std) ** 3))
        else:
            score_skewness = 0.0

        # Fill turn statistics
        ft = fill_turns[:, cat_id].astype(np.float64)
        fill_turn_std = float(np.std(ft))

        # Fill turn entropy: -sum(p * log2(p)) over turn distribution
        turn_counts = np.bincount(fill_turns[:, cat_id], minlength=16)[1:]  # turns 1-15
        turn_probs = turn_counts / n
        nonzero = turn_probs > 0
        fill_turn_entropy = float(-np.sum(turn_probs[nonzero] * np.log2(turn_probs[nonzero])))

        # Bonus dependency: mean(score | bonus) - mean(score | no bonus)
        bonus_mask = got_bonus
        no_bonus_mask = ~got_bonus
        if bonus_mask.sum() > 0 and no_bonus_mask.sum() > 0:
            bonus_dependency = float(cat_vals[bonus_mask].mean() - cat_vals[no_bonus_mask].mean())
        else:
            bonus_dependency = 0.0

        # Opportunity cost: mean(score | score > 0) - mean(score)
        nonzero_mask = cat_vals > 0
        if nonzero_mask.sum() > 0:
            opportunity_cost = float(cat_vals[nonzero_mask].mean() - cat_mean)
        else:
            opportunity_cost = 0.0

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
            "score_std": round(cat_std, 2),
            "score_skewness": round(score_skewness, 2),
            "fill_turn_std": round(fill_turn_std, 2),
            "fill_turn_entropy": round(fill_turn_entropy, 2),
            "mean_score_bonus": round(float(cat_vals[bonus_mask].mean()), 2) if bonus_mask.sum() > 0 else round(float(cat_mean), 2),
            "mean_score_no_bonus": round(float(cat_vals[no_bonus_mask].mean()), 2) if no_bonus_mask.sum() > 0 else round(float(cat_mean), 2),
            "bonus_dependency": round(bonus_dependency, 2),
            "opportunity_cost": round(opportunity_cost, 2),
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
    """Generate mixture.json: 16 sub-populations (bonus x yatzy x SS x LS)."""
    total_scores = rec["total_scores"].astype(np.float64)
    got_bonus = rec["got_bonus"]
    cat_scores = rec["category_scores"]
    n = rec["num_games"]

    got_yatzy = cat_scores[:, 14] > 0
    got_ss = cat_scores[:, 10] > 0  # Small Straight (cat 10)
    got_ls = cat_scores[:, 11] > 0  # Large Straight (cat 11)

    flags = {
        "bonus": got_bonus,
        "yatzy": got_yatzy,
        "ss": got_ss,
        "ls": got_ls,
    }

    populations = []
    for bonus_hit in [False, True]:
        for yatzy_hit in [False, True]:
            for ss_hit in [False, True]:
                for ls_hit in [False, True]:
                    mask = (
                        (got_bonus == bonus_hit)
                        & (got_yatzy == yatzy_hit)
                        & (got_ss == ss_hit)
                        & (got_ls == ls_hit)
                    )
                    count = int(mask.sum())
                    if count == 0:
                        populations.append({
                            "bonus": bonus_hit, "yatzy": yatzy_hit,
                            "ss": ss_hit, "ls": ls_hit,
                            "fraction": 0.0, "mean": 0.0, "std": 1.0,
                        })
                        continue

                    subset = total_scores[mask]
                    populations.append({
                        "bonus": bonus_hit, "yatzy": yatzy_hit,
                        "ss": ss_hit, "ls": ls_hit,
                        "fraction": round(count / n, 4),
                        "mean": round(float(subset.mean()), 1),
                        "std": round(float(subset.std()), 1),
                    })

    with open(out_path, "w") as f:
        json.dump({"populations": populations}, f, indent=2)

    print(f"  mixture.json: {len(populations)} populations")
    for p in populations:
        parts = []
        parts.append("Bonus" if p["bonus"] else "No bonus")
        parts.append("Yatzy" if p["yatzy"] else "No Yatzy")
        parts.append("SS" if p["ss"] else "No SS")
        parts.append("LS" if p["ls"] else "No LS")
        label = ", ".join(parts)
        print(f"    {label}: {p['fraction']*100:.1f}%, mean={p['mean']}, std={p['std']}")


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


def _landscape_sort_order(out_path: Path) -> list[int]:
    """Return category IDs in scorecard order (0-14)."""
    return list(range(15))


def generate_correlation_matrix(rec: dict, out_path: Path):
    """Generate category_correlations.json: 16x16 Pearson correlation matrix (15 cats + bonus)."""
    cat_scores = rec["category_scores"].astype(np.float64)  # (N, 15)
    got_bonus = rec["got_bonus"].astype(np.float64)  # (N,)

    # Stack bonus as column 15, then compute 16x16 correlation
    all_cols = np.column_stack([cat_scores, got_bonus])
    corr = np.corrcoef(all_cols.T)

    # Scorecard order: 0-5 (upper), bonus (col 15), 6-14 (lower)
    sorted_ids = list(range(6)) + [15] + list(range(6, 15))
    all_names = CATEGORY_NAMES + ["Bonus"]
    sorted_names = [all_names[i] for i in sorted_ids]

    # Reorder matrix rows and columns
    sorted_corr = corr[np.ix_(sorted_ids, sorted_ids)]

    matrix = [[round(float(v), 4) for v in row] for row in sorted_corr]

    result = {
        "categories": sorted_names,
        "matrix": matrix,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  category_correlations.json: {len(sorted_names)}x{len(sorted_names)} matrix")


def _build_fill_turn_matrix(
    fill_turns: np.ndarray, sorted_ids: list[int], n: int
) -> tuple[list[str], list[list[float]]]:
    """Build fill-turn probability matrix with bonus row after Sixes."""
    matrix = []
    names = []
    for cat_id in sorted_ids:
        turn_counts = np.bincount(fill_turns[:, cat_id], minlength=16)[1:]
        matrix.append([round(float(p), 6) for p in turn_counts / n])
        names.append(CATEGORY_NAMES[cat_id])

        if cat_id == 5:
            bonus_turn = fill_turns[:, :6].max(axis=1)
            bonus_counts = np.bincount(bonus_turn, minlength=16)[1:]
            matrix.append([round(float(p), 6) for p in bonus_counts / n])
            names.append("Bonus (decided)")
    return names, matrix


def _build_fill_turn_matrix_filtered(
    fill_turns: np.ndarray,
    category_scores: np.ndarray,
    got_bonus: np.ndarray,
    sorted_ids: list[int],
    score_filter: str,
    bonus_filter: str,
) -> list[list[float]]:
    """Build fill-turn probability matrix with combined bonus + score filters.

    score_filter: 'all', 'scored' (>0), or 'zeroed' (==0). Applied per-category.
    bonus_filter: 'all', 'bonus', or 'no_bonus'. Applied globally.
    """
    # Global bonus mask
    if bonus_filter == "bonus":
        global_mask = got_bonus
    elif bonus_filter == "no_bonus":
        global_mask = ~got_bonus
    else:
        global_mask = np.ones(len(got_bonus), dtype=bool)

    matrix = []
    for cat_id in sorted_ids:
        # Per-category score mask
        if score_filter == "scored":
            score_mask = category_scores[:, cat_id] > 0
        elif score_filter == "zeroed":
            score_mask = category_scores[:, cat_id] == 0
        else:
            score_mask = np.ones(len(got_bonus), dtype=bool)

        mask = global_mask & score_mask
        count = int(mask.sum())
        if count == 0:
            matrix.append([0.0] * 15)
        else:
            turn_counts = np.bincount(fill_turns[mask, cat_id], minlength=16)[1:]
            matrix.append([round(float(p), 6) for p in turn_counts / count])

        if cat_id == 5:
            # Bonus row: for score_filter, use bonus hit/miss as the "score"
            if score_filter == "scored":
                bonus_score_mask = got_bonus
            elif score_filter == "zeroed":
                bonus_score_mask = ~got_bonus
            else:
                bonus_score_mask = np.ones(len(got_bonus), dtype=bool)
            bmask = global_mask & bonus_score_mask
            bcount = int(bmask.sum())
            bonus_turn = fill_turns[:, :6].max(axis=1)
            if bcount == 0:
                matrix.append([0.0] * 15)
            else:
                bonus_counts = np.bincount(bonus_turn[bmask], minlength=16)[1:]
                matrix.append([round(float(p), 6) for p in bonus_counts / bcount])
    return matrix


def generate_fill_turn_heatmap(rec: dict, out_path: Path):
    """Generate fill_turn_heatmap.json: 3x3 grid of bonus x score filters."""
    fill_turns = rec["fill_turns"]  # (N, 15) uint8
    category_scores = rec["category_scores"]  # (N, 15) uint8
    got_bonus = rec["got_bonus"]

    sorted_ids = _landscape_sort_order(out_path)
    names, matrix_all = _build_fill_turn_matrix(fill_turns, sorted_ids, rec["num_games"])

    # Generate all 9 combinations (3 bonus x 3 score)
    bonus_keys = ["all", "bonus", "no_bonus"]
    score_keys = ["all", "scored", "zeroed"]
    matrices = {}
    for bk in bonus_keys:
        for sk in score_keys:
            key = f"{bk}__{sk}"
            matrices[key] = _build_fill_turn_matrix_filtered(
                fill_turns, category_scores, got_bonus, sorted_ids, sk, bk
            )

    result = {
        "categories": names,
        "turns": list(range(1, 16)),
        "matrices": matrices,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    combos = len(matrices)
    print(f"  fill_turn_heatmap.json: {len(names)} rows x 15 turns ({combos} filter combos)")


def main():
    # Try multiple known locations for simulation_raw.bin
    candidates = [
        repo_root / "data" / "simulations" / "theta" / "theta_0.000" / "simulation_raw.bin",
        repo_root / "data" / "simulations" / "theta" / "theta_0" / "simulation_raw.bin",
        repo_root / "data" / "simulations" / "simulation_raw.bin",
    ]
    raw_path = next((p for p in candidates if p.exists()), candidates[0])
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
    generate_correlation_matrix(rec, out_dir / "category_correlations.json")
    generate_fill_turn_heatmap(rec, out_dir / "fill_turn_heatmap.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
