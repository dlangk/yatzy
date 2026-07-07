#!/usr/bin/env python3
"""Generate the two Section-7 Rosetta chart data files from skill_ladder.json.

Reads:  outputs/rosetta/skill_ladder.json   (from `just rosetta`)
Writes: treatise/data/rosetta_rules.json    (top category rules, for rosetta-rules.js)
        treatise/data/filter_grammar.json   (reroll rule counts, for filter-grammar.js)

Run:    analytics/.venv/bin/python treatise/scripts/gen-rosetta-charts.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analytics" / "src"))
from yatzy_analysis.skill_ladder import CATEGORY_NAMES, Condition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "rosetta" / "skill_ladder.json"
OUT = ROOT / "treatise" / "data"

# Chart color buckets (treatise/js/charts/rosetta-rules.js categoryColors)
UPPER = set(CATEGORY_NAMES[:6])           # Ones..Sixes
COMBO = set(CATEGORY_NAMES[6:13])         # One Pair..Full House


def bucket(cat_name: str) -> str:
    if cat_name in UPPER:
        return "upper"
    if cat_name == "Yatzy":
        return "jackpot"
    if cat_name == "Chance":
        return "dump"
    if cat_name in COMBO:
        return "combo"
    return "speculative"


def render_condition(conds: list[dict]) -> str:
    parts = [Condition(c["feature"], c["op"], c["threshold"]).to_human() for c in conds]
    return " AND ".join(parts) if parts else "always"


def action_name(action) -> str:
    # Category rules store an int index; reroll rules store a CFG action name.
    if isinstance(action, int):
        return CATEGORY_NAMES[action] if 0 <= action < len(CATEGORY_NAMES) else str(action)
    return str(action)


def main() -> None:
    data = json.loads(SRC.read_text())

    # --- rosetta_rules.json: top category rules by regret prevented ---
    cat_rules = data.get("category_rules", [])
    ranked = sorted(cat_rules, key=lambda r: r.get("mean_regret", 0.0), reverse=True)
    rules = []
    for rank, r in enumerate(ranked[:15]):
        name = action_name(r["action"])
        rules.append({
            "rank": rank + 1,
            "action": name,
            "category": bucket(name),
            "regret": round(float(r.get("mean_regret", 0.0)), 4),
            "condition": render_condition(r.get("conditions", [])),
            "coverage": r.get("coverage", 0),
        })
    (OUT / "rosetta_rules.json").write_text(json.dumps({"rules": rules}, indent=2))
    print(f"rosetta_rules.json: {len(rules)} rules")

    # --- filter_grammar.json: reroll rule counts per CFG action ---
    c1 = Counter(action_name(r["action"]) for r in data.get("reroll1_rules", []))
    c2 = Counter(action_name(r["action"]) for r in data.get("reroll2_rules", []))
    acts = sorted(set(c1) | set(c2), key=lambda a: -(c1[a] + c2[a]))
    actions = [
        {"action": a, "reroll1_rules": c1.get(a, 0), "reroll2_rules": c2.get(a, 0)}
        for a in acts
    ]
    (OUT / "filter_grammar.json").write_text(json.dumps({"actions": actions}, indent=2))
    print(f"filter_grammar.json: {len(actions)} actions")


if __name__ == "__main__":
    main()
