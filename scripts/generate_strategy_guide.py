#!/usr/bin/env python3
"""Generate human strategy guide from skill ladder using Claude API.

Prepares enriched JSON from skill_ladder.json, calls Anthropic API with
the Rosetta Stone translation prompt, writes output to theory/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]


def condition_to_human(cond: dict) -> str:
    """Convert a condition dict to human-readable string."""
    feat = cond["feature"]
    op = cond["op"]
    thresh = cond["threshold"]

    # Format threshold
    if thresh == int(thresh):
        t = str(int(thresh))
    else:
        t = f"{thresh:.4f}".rstrip("0").rstrip(".")

    # Boolean features
    bool_feats = {
        "bonus_secured", "has_pair", "has_two_pair", "has_three_of_kind",
        "has_four_of_kind", "has_full_house", "has_small_straight",
        "has_large_straight", "has_yatzy",
    }
    if feat.startswith("cat_avail_"):
        if op == "==" and float(thresh) > 0.5:
            return f"cat_avail_{feat[10:]}"
        elif op == "==" and float(thresh) < 0.5:
            return f"!cat_avail_{feat[10:]}"
    if feat in bool_feats:
        if op == "==" and float(thresh) > 0.5:
            return feat
        elif op == "==" and float(thresh) < 0.5:
            return f"!{feat}"

    return f"{feat} {op} {t}"


def enrich_rules(
    ladder: dict,
    max_rules: int = 50,
    regret_path: Path | None = None,
) -> list[dict]:
    """Convert skill ladder rules to enriched format for the prompt.

    When regret_path is provided, adds runner_up and regret_prevented fields
    by loading the regret binary and computing per-category mean regret for
    each rule's matching records.
    """
    # Conditionally load regret data
    regret_ds = None
    feature_index: dict[str, int] = {}
    bool_features: set[str] = set()
    if regret_path is not None:
        from yatzy_analysis.skill_ladder import (
            BOOL_FEATURES,
            FEATURE_NAMES,
            load_regret_file,
        )

        regret_ds = load_regret_file(regret_path, "category")
        feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        bool_features = BOOL_FEATURES
        print(f"Loaded regret file: {len(regret_ds.features):,d} records, "
              f"{regret_ds.num_actions} actions")

    enriched = []
    for i, rule in enumerate(ladder["category_rules"][:max_rules]):
        conditions = [condition_to_human(c) for c in rule["conditions"]]
        condition_str = " AND ".join(conditions)
        action_idx = rule["action"]
        action_name = CATEGORY_NAMES[action_idx] if action_idx < len(CATEGORY_NAMES) else f"Category {action_idx}"

        entry: dict = {
            "rank": i + 1,
            "action": action_name,
            "action_index": action_idx,
            "condition": condition_str,
            "coverage": rule["coverage"],
            "mean_regret": round(rule["mean_regret"], 6),
        }

        # Runner-up enrichment from regret data
        if regret_ds is not None:
            import numpy as np

            mask = np.ones(len(regret_ds.features), dtype=bool)
            for cond in rule["conditions"]:
                feat = cond["feature"]
                op = cond["op"]
                thresh = cond["threshold"]
                if feat not in feature_index:
                    continue
                col = regret_ds.features[:, feature_index[feat]]
                if feat in bool_features or feat.startswith("cat_avail_"):
                    if op == "==" and thresh > 0.5:
                        mask &= col > 0.5
                    elif op == "==" and thresh < 0.5:
                        mask &= col < 0.5
                elif op == "==":
                    mask &= np.abs(col - thresh) < 1e-6
                elif op == ">=":
                    mask &= col >= thresh - 1e-9
                elif op == "<=":
                    mask &= col <= thresh + 1e-9

            n_match = int(mask.sum())
            if n_match > 0:
                subset_regret = regret_ds.regret[mask]  # (n_match, num_actions)
                # Mean regret per category across matching records
                mean_per_cat = np.where(
                    subset_regret > -1e30,
                    subset_regret,
                    np.nan,
                ).astype(np.float64)
                with np.errstate(all="ignore"):
                    cat_means = np.nanmean(mean_per_cat, axis=0)

                # Best action is the one with lowest mean regret (should match action_idx)
                # Runner-up is second-lowest
                sorted_indices = np.argsort(cat_means)
                runner_up_idx = None
                for idx in sorted_indices:
                    if idx != action_idx and not np.isnan(cat_means[idx]):
                        runner_up_idx = idx
                        break

                if runner_up_idx is not None:
                    runner_up_name = (
                        CATEGORY_NAMES[runner_up_idx]
                        if runner_up_idx < len(CATEGORY_NAMES)
                        else f"Category {runner_up_idx}"
                    )
                    gap = float(cat_means[runner_up_idx] - cat_means[action_idx])
                    entry["runner_up"] = runner_up_name
                    entry["regret_prevented"] = round(gap, 1)

        enriched.append(entry)

    return enriched


PROMPT = """
<system_role>
You are a Data Translator and Technical Writer. Your job is to convert structured machine-generated decision rules into clear, accurate, human-readable strategy heuristics for Scandinavian Yatzy.
</system_role>

<physics_engine>
Scandinavian Yatzy scoring rules — these are absolute constraints. Every claim you make must be consistent with them.

Upper Section (categories 0-5):
- Ones: sum of all dice showing 1 (max 5)
- Twos: sum of all dice showing 2 (max 10)
- Threes: sum of all dice showing 3 (max 15)
- Fours: sum of all dice showing 4 (max 20)
- Fives: sum of all dice showing 5 (max 25)
- Sixes: sum of all dice showing 6 (max 30)
- Upper bonus: 50 points if upper total >= 63

Lower Section (categories 6-14):
- One Pair: sum of the two highest matching dice (max 12)
- Two Pairs: sum of two different pairs (max 22)
- Three of a Kind: sum of three matching dice (max 18)
- Four of a Kind: sum of four matching dice (max 24)
- Small Straight: 1-2-3-4-5 = 15 points (fixed)
- Large Straight: 2-3-4-5-6 = 20 points (fixed)
- Full House: three of a kind + a pair = sum of all five dice (max 28)
- Chance: sum of all five dice (max 30)
- Yatzy: five of a kind = 50 points (fixed)

Each category is used exactly once per game (15 turns). A category scored as 0 is "burned."
</physics_engine>

<glossary>
Machine DSL:
- face_count_X == Y: "Exactly Y dice show face X"
- !cat_avail_X / cat_avail_X: "Category X is burned / still open"
- categories_left: empty boxes remaining (15 = turn 1, 1 = final turn)
- upper_cats_left: empty upper section boxes remaining
- zeros_available: low-cost dump categories still open
- has_X: boolean — dice form pattern X (e.g., has_small_straight)
- bonus_pace: ratio of upper progress to target (>1.0 = ahead)
- upper_score: accumulated upper section points (target 63)
- best_available_score: highest immediate score across open categories
- turn: current turn number (0 = first, 14 = last)
- runner_up: second-best category for this situation
- regret_prevented: EV gap between best action and runner-up (how much choosing wrong costs)
</glossary>

<strict_constraints>
1. NO MATH HALLUCINATIONS: Never state a score, maximum, or probability that contradicts the physics engine above. If a rule mentions a specific score, verify it against the category formulas.
2. NO IMPOSSIBLE SCORES: One Pair of 6s = 12, not 36. Full House max = 28 (6+6+6+5+5), not 30. Small Straight = exactly 15. Verify every number.
3. THE "ELIF" SHADOW: Rules are evaluated strictly in priority order (IF-ELIF-ELSE). Rule N only fires if rules 1 through N-1 all failed. Reflect this in your prose: "Assuming no higher-priority play applies..." or "When your primary targets are blocked..."
4. AGGREGATE BY CONCEPT: Do not list "Rule 1, Rule 2, Rule 3." Group rules by target category or strategic intent into cohesive heuristics.
5. SILICON RATIONALE: Explain *why* the solver chose each action using concepts like option value, variance sinks, upper bonus protection, garbage routing, and opportunity cost. Use the regret_prevented field to quantify the cost of the wrong choice.
6. ACTIVE VERBS: Use direct tabletop language — "Lock in," "Dump," "Sacrifice," "Pivot," "Slam," "Route."
</strict_constraints>

<task>
Translate the boolean conditions in the input rules into plain English heuristics, grouped by target category. For each heuristic group:

1. A descriptive name
2. When it applies (the trigger conditions in natural language)
3. What to do (the action, with examples)
4. Why (the silicon rationale, referencing regret_prevented where available)
5. Source rule numbers and coverage/regret stats

End with a "Default Play" section for when no rule matches.
</task>

<input_rules>
{rules_json}
</input_rules>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate human strategy guide from skill ladder rules"
    )
    parser.add_argument(
        "--max-rules", type=int, default=50,
        help="Maximum number of rules to include (default: 50)",
    )
    parser.add_argument(
        "--regret-file", type=str, default=None,
        help="Path to regret_category.bin for runner-up enrichment",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load and enrich rules
    ladder_path = Path("outputs/rosetta/skill_ladder.json")
    with open(ladder_path) as f:
        ladder = json.load(f)

    regret_path = Path(args.regret_file) if args.regret_file else None
    enriched = enrich_rules(ladder, max_rules=args.max_rules, regret_path=regret_path)
    rules_json = json.dumps(enriched, indent=2)

    print(f"Prepared {len(enriched)} enriched rules")
    print(f"Calling Claude claude-opus-4-20250514...")

    # Build the prompt
    full_prompt = PROMPT.format(rules_json=rules_json)

    # Call the API with streaming (required for Opus long requests)
    client = anthropic.Anthropic(api_key=api_key)
    response_text = ""
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16000,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
    ) as stream:
        for text in stream.text_stream:
            response_text += text
            print(text, end="", flush=True)

    print(f"\n\nReceived {len(response_text)} characters")

    # Write output
    output_path = Path("theory/yatzy_grandmaster_playbook.md")
    output_path.write_text(response_text)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
