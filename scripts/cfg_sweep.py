#!/usr/bin/env python3
"""Run CFG skill ladder induction at multiple lambda values and evaluate each.

Usage:
    python scripts/cfg_sweep.py [--base-path .] [--games 200000]

Produces outputs/rosetta/lambda_<X>/skill_ladder.json for each lambda,
then calls yatzy-eval-policy on each.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics" / "src"))

LAMBDAS = [0.00, 0.05, 0.10]


def main():
    parser = argparse.ArgumentParser(description="CFG lambda sweep")
    parser.add_argument("--base-path", default=".", help="Repo root")
    parser.add_argument("--games", default=200_000, type=int, help="Games for eval")
    args = parser.parse_args()

    base = Path(args.base_path)
    rosetta_dir = base / "outputs" / "rosetta"

    if not rosetta_dir.exists():
        print(f"{rosetta_dir} not found. Run yatzy-regret-export first.")
        sys.exit(1)

    from yatzy_analysis.skill_ladder import generate_skill_ladder

    eval_binary = base / "solver" / "target" / "release" / "yatzy-eval-policy"
    if not eval_binary.exists():
        print(f"Eval binary not found: {eval_binary}")
        print("Run: cd solver && cargo build --release")
        sys.exit(1)

    results = []
    for lam in LAMBDAS:
        print(f"\n{'=' * 60}")
        print(f"  λ = {lam}")
        print(f"{'=' * 60}")

        output_dir = rosetta_dir / f"lambda_{lam:.2f}"
        generate_skill_ladder(rosetta_dir, output_dir, lam=lam)

        json_path = output_dir / "skill_ladder.json"
        eval_output = output_dir / "eval_results.json"

        cmd = [
            str(eval_binary),
            "--games",
            str(args.games),
            "--output",
            str(eval_output),
            str(json_path),
        ]
        print(f"\nEvaluating: {' '.join(cmd)}")
        subprocess.run(cmd, env={"YATZY_BASE_PATH": str(base), "PATH": "/usr/bin:/bin"})
        results.append((lam, eval_output))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    import json

    for lam, path in results:
        if path.exists():
            data = json.loads(path.read_text())
            print(
                f"  λ={lam:.2f}: EV={data['policy_ev']:.2f}, "
                f"gap={data['ev_gap']:.2f}, rules={data['total_rules']}"
            )
        else:
            print(f"  λ={lam:.2f}: [no results]")


if __name__ == "__main__":
    main()
