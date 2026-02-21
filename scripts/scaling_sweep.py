#!/usr/bin/env python3
"""Scaling law sweep: evaluate category-only rules from N=0 to N=100.

Uses optimal DP rerolls + truncated category rule list.
Greedy sequential covering guarantees first-N rules == max_rules=N.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EVAL_BINARY = "solver/target/release/yatzy-eval-policy"
LADDER_JSON = "outputs/rosetta/skill_ladder.json"
GAMES = 200_000
SEED = 42

SWEEP_POINTS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]


def main() -> None:
    base = Path(".")

    # Load full ladder
    with open(base / LADDER_JSON) as f:
        full_ladder = json.load(f)

    all_cat_rules = full_ladder["category_rules"]
    print(f"Loaded {len(all_cat_rules)} category rules from {LADDER_JSON}")
    print(f"Sweep: {len(SWEEP_POINTS)} points, {GAMES:,d} games each")
    print(f"{'='*70}")

    results: list[dict] = []
    t_total = time.time()

    for n_rules in SWEEP_POINTS:
        # Create truncated ladder
        truncated = {
            "category_rules": all_cat_rules[:n_rules],
            "reroll1_rules": [],
            "reroll2_rules": [],
            "meta": {
                "total_rules": n_rules,
                "default_category": full_ladder["meta"]["default_category"],
                "default_reroll1": "Reroll_All",
                "default_reroll2": "Reroll_All",
                "oracle_ev": full_ladder["meta"]["oracle_ev"],
            },
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        ) as tf:
            json.dump(truncated, tf)
            tmp_path = tf.name

        # Run evaluation
        t0 = time.time()
        result = subprocess.run(
            [
                str(base / EVAL_BINARY),
                "--category-only",
                "--games", str(GAMES),
                "--seed", str(SEED),
                "--output", f"/tmp/scaling_n{n_rules}.json",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            env={"YATZY_BASE_PATH": ".", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  N={n_rules:3d}: FAILED ({result.stderr.strip()})")
            continue

        # Parse output JSON
        with open(f"/tmp/scaling_n{n_rules}.json") as f:
            eval_result = json.load(f)

        ev = eval_result["policy_ev"]
        gap = eval_result["ev_gap"]
        std = eval_result["std_dev"]
        p50 = eval_result["p50"]
        se = std / (GAMES ** 0.5)

        results.append({
            "n_rules": n_rules,
            "ev": round(ev, 2),
            "gap": round(gap, 2),
            "std": round(std, 2),
            "p50": p50,
            "se": round(se, 3),
        })

        print(
            f"  N={n_rules:3d}: EV={ev:6.2f}  gap={gap:5.2f}  "
            f"p50={p50:3d}  std={std:5.2f}  ({elapsed:.1f}s)"
        )

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"Done in {total_time:.0f}s")

    # Save results
    out_path = base / "outputs/rosetta/scaling_law.json"
    with open(out_path, "w") as f:
        json.dump({"sweep": results, "games_per_point": GAMES, "seed": SEED}, f, indent=2)
    print(f"Wrote {out_path}")

    # Print summary table
    print(f"\n{'N':>5s} {'EV':>8s} {'Gap':>7s} {'p50':>5s} {'SE':>6s}")
    print("-" * 35)
    for r in results:
        marker = ""
        if r["ev"] >= 221:
            marker = " <-- beats NN baseline (221)"
        print(f"{r['n_rules']:5d} {r['ev']:8.2f} {r['gap']:7.2f} {r['p50']:5d} {r['se']:6.3f}{marker}")


if __name__ == "__main__":
    main()
