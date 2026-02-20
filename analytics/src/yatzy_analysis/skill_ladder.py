"""Greedy sequential covering algorithm for skill ladder induction.

Reads regret export binary files (from yatzy-regret-export) and induces
a human-readable decision list optimized for weighted regret minimization.

Output: skill_ladder.json + YATZY_SKILL_LADDER.md
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Binary format constants ──────────────────────────────────────────────

MAGIC = 0x52455052  # "REPR"
VERSION = 1
HEADER_SIZE = 32

NUM_SEMANTIC_FEATURES = 56

FEATURE_NAMES = [
    "turn", "categories_left", "rerolls_remaining", "upper_score",
    "bonus_secured", "bonus_pace", "upper_cats_left",
    "face_count_1", "face_count_2", "face_count_3",
    "face_count_4", "face_count_5", "face_count_6",
    "max_count", "num_distinct", "dice_sum",
    "has_pair", "has_two_pair", "has_three_of_kind", "has_four_of_kind",
    "has_full_house", "has_small_straight", "has_large_straight", "has_yatzy",
    "cat_avail_ones", "cat_avail_twos", "cat_avail_threes",
    "cat_avail_fours", "cat_avail_fives", "cat_avail_sixes",
    "cat_avail_one_pair", "cat_avail_two_pairs", "cat_avail_three_of_kind",
    "cat_avail_four_of_kind", "cat_avail_small_straight",
    "cat_avail_large_straight", "cat_avail_full_house",
    "cat_avail_chance", "cat_avail_yatzy",
    "cat_score_ones", "cat_score_twos", "cat_score_threes",
    "cat_score_fours", "cat_score_fives", "cat_score_sixes",
    "cat_score_one_pair", "cat_score_two_pairs", "cat_score_three_of_kind",
    "cat_score_four_of_kind", "cat_score_small_straight",
    "cat_score_large_straight", "cat_score_full_house",
    "cat_score_chance", "cat_score_yatzy",
    "zeros_available", "best_available_score",
]

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

BOOL_FEATURES = {
    "bonus_secured", "has_pair", "has_two_pair", "has_three_of_kind",
    "has_four_of_kind", "has_full_house", "has_small_straight",
    "has_large_straight", "has_yatzy",
} | {f"cat_avail_{n}" for n in [
    "ones", "twos", "threes", "fours", "fives", "sixes",
    "one_pair", "two_pairs", "three_of_kind", "four_of_kind",
    "small_straight", "large_straight", "full_house", "chance", "yatzy",
]}


# ── Data loading ─────────────────────────────────────────────────────────

@dataclass
class RegretDataset:
    """Loaded regret export data for one decision type."""
    features: np.ndarray     # (N, NUM_SEMANTIC_FEATURES) float32
    q_values: np.ndarray     # (N, num_actions) float32
    best_action: np.ndarray  # (N,) int32
    regret: np.ndarray       # (N, num_actions) float32
    num_actions: int
    decision_type: str


def load_regret_file(path: Path, decision_type: str) -> RegretDataset:
    """Load a regret binary file."""
    data = path.read_bytes()
    magic, version, count, n_features, n_actions = struct.unpack_from("<IIqII", data, 0)
    assert magic == MAGIC, f"Bad magic: {magic:#x}"
    assert version == VERSION, f"Bad version: {version}"
    assert n_features == NUM_SEMANTIC_FEATURES, f"Feature count mismatch: {n_features}"

    record_floats = n_features + n_actions + 1 + n_actions
    expected = HEADER_SIZE + count * record_floats * 4
    assert len(data) >= expected, f"File too small: {len(data)} < {expected}"

    raw = np.frombuffer(data, dtype=np.float32, offset=HEADER_SIZE)
    raw = raw[:count * record_floats].reshape(count, record_floats)

    features = raw[:, :n_features].copy()
    q_values = raw[:, n_features:n_features + n_actions].copy()
    best_action = raw[:, n_features + n_actions].astype(np.int32)
    regret = raw[:, n_features + n_actions + 1:].copy()

    return RegretDataset(
        features=features,
        q_values=q_values,
        best_action=best_action,
        regret=regret,
        num_actions=n_actions,
        decision_type=decision_type,
    )


# ── Rule induction (vectorized) ──────────────────────────────────────────

@dataclass
class Condition:
    feature: str
    op: str
    threshold: float

    def to_human(self) -> str:
        if self.feature in BOOL_FEATURES:
            if self.op == "==" and self.threshold > 0.5:
                return self.feature
            elif self.op == "==" and self.threshold < 0.5:
                return f"NOT {self.feature}"
        # Raw values — display as integers where appropriate
        t = self.threshold
        if t == int(t) and abs(t) < 1000:
            t_str = str(int(t))
        else:
            t_str = f"{t:.4f}".rstrip("0").rstrip(".")
        return f"{self.feature} {self.op} {t_str}"

    def to_dict(self) -> dict:
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold}


@dataclass
class Rule:
    conditions: list[Condition]
    action: int
    decision_type: str
    coverage: int = 0
    mean_regret: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "action": self.action,
            "decision_type": self.decision_type,
            "coverage": self.coverage,
            "mean_regret": round(self.mean_regret, 6),
        }


def _build_condition_masks(
    features: np.ndarray,
) -> tuple[np.ndarray, list[Condition]]:
    """Pre-compute all candidate condition masks as a (num_conds, N) bool matrix.

    Returns (masks, conditions) where masks[i] is the bool mask for conditions[i].
    """
    conditions: list[Condition] = []
    mask_list: list[np.ndarray] = []

    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]

        if name in BOOL_FEATURES:
            # Normalized booleans: 1/max → 1.0 or 0.0 effectively
            conditions.append(Condition(name, "==", 1.0))
            mask_list.append(col > 0.5)
            conditions.append(Condition(name, "==", 0.0))
            mask_list.append(col < 0.5)
        elif name.startswith("cat_score_"):
            # Skip cat_score features — too many unique values, not great for rules
            continue
        else:
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                for v in unique_vals:
                    fv = float(round(v, 6))
                    conditions.append(Condition(name, "==", fv))
                    mask_list.append(np.abs(col - v) < 1e-6)
                    conditions.append(Condition(name, ">=", fv))
                    mask_list.append(col >= v - 1e-9)
                    conditions.append(Condition(name, "<=", fv))
                    mask_list.append(col <= v + 1e-9)
            else:
                for pct in [10, 25, 50, 75, 90]:
                    t = float(np.percentile(col, pct))
                    t_r = round(t, 4)
                    conditions.append(Condition(name, ">=", t_r))
                    mask_list.append(col >= t - 1e-9)
                    conditions.append(Condition(name, "<=", t_r))
                    mask_list.append(col <= t + 1e-9)

    # Stack into (num_conds, N) bool array — memory: ~conditions × N bits
    # For 500 conditions × 3M records = ~190 MB (uint8), manageable
    masks = np.array(mask_list, dtype=np.uint8)
    return masks, conditions


def _best_action_for_masked(
    best_actions: np.ndarray,
    regret: np.ndarray,
    mask: np.ndarray,
    num_actions: int,
) -> tuple[int, float]:
    """Find best action for subset using majority-vote on oracle actions.

    Much faster than scanning all actions' regret columns.
    Returns (action, mean_regret_of_that_action).
    """
    subset_actions = best_actions[mask]
    n = len(subset_actions)
    if n == 0:
        return 0, float("inf")

    # Mode of oracle actions = best action for this subset
    counts = np.bincount(subset_actions, minlength=num_actions)
    action = int(counts.argmax())

    # Mean regret of picking this action for all records in subset
    regret_col = regret[mask, action]
    valid = regret_col > -1e30
    if valid.sum() == 0:
        return action, 0.0
    mean_reg = float(regret_col[valid].mean())
    return action, mean_reg


def induce_rules(
    ds: RegretDataset,
    max_rules: int = 100,
    min_coverage: int = 100,
    subsample: int = 300_000,
) -> tuple[list[Rule], int]:
    """Run greedy sequential covering to induce a decision list.

    Subsamples to `subsample` records for tractable rule search.
    """
    n_full = len(ds.features)
    # Subsample for speed if dataset is large
    if n_full > subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_full, subsample, replace=False)
        idx.sort()
        features = ds.features[idx]
        regret = ds.regret[idx]
        best_action = ds.best_action[idx]
    else:
        features = ds.features
        regret = ds.regret
        best_action = ds.best_action

    n = len(features)
    alive = np.ones(n, dtype=np.uint8)
    rules: list[Rule] = []

    print(f"  Inducing rules for {ds.decision_type} "
          f"({n_full:,d} records, subsampled to {n:,d}, {ds.num_actions} actions)...")

    t0 = time.time()
    cond_masks, conditions = _build_condition_masks(features)
    num_conds = len(conditions)
    print(f"    {num_conds} candidate conditions, precomputed in {time.time() - t0:.1f}s")

    for rule_idx in range(max_rules):
        alive_count = int(alive.sum())
        if alive_count < min_coverage:
            break

        t1 = time.time()

        # Phase 1: score all single conditions vectorized
        match_counts = cond_masks.dot(alive).astype(np.int64)

        viable = np.where(match_counts >= min_coverage)[0]
        if len(viable) == 0:
            break

        best_rule: Rule | None = None
        best_score = float("inf")

        order = viable[np.argsort(-match_counts[viable])]

        for ci in order[:200]:
            mask = (cond_masks[ci] & alive).astype(bool)
            n_match = int(mask.sum())
            if n_match < min_coverage:
                continue

            action, mean_reg = _best_action_for_masked(
                best_action, regret, mask, ds.num_actions,
            )
            score = mean_reg

            if score < best_score:
                best_score = score
                best_rule = Rule(
                    conditions=[conditions[ci]],
                    action=action,
                    decision_type=ds.decision_type,
                    coverage=n_match,
                    mean_regret=mean_reg,
                )

        # Phase 2: conjunctions (top 20 bases × top 30 second conditions)
        base_candidates: list[tuple[int, np.ndarray]] = []
        for ci in order[:50]:
            mask = (cond_masks[ci] & alive).astype(bool)
            if int(mask.sum()) >= min_coverage:
                base_candidates.append((ci, mask))
            if len(base_candidates) >= 20:
                break

        for base_ci, base_mask in base_candidates:
            base_alive = base_mask.astype(np.uint8)
            inner_counts = cond_masks.dot(base_alive).astype(np.int64)
            inner_viable = np.where(inner_counts >= min_coverage)[0]
            inner_order = inner_viable[np.argsort(-inner_counts[inner_viable])]

            for ci2 in inner_order[:30]:
                if ci2 == base_ci:
                    continue
                mask2 = cond_masks[ci2].astype(bool) & base_mask
                n_match = int(mask2.sum())
                if n_match < min_coverage:
                    continue

                action, mean_reg = _best_action_for_masked(
                    best_action, regret, mask2, ds.num_actions,
                )
                score = mean_reg

                if score < best_score:
                    best_score = score
                    best_rule = Rule(
                        conditions=[conditions[base_ci], conditions[ci2]],
                        action=action,
                        decision_type=ds.decision_type,
                        coverage=n_match,
                        mean_regret=mean_reg,
                    )

        if best_rule is None or best_rule.coverage < min_coverage:
            break

        # Apply rule
        mask = np.ones(n, dtype=bool)
        for c in best_rule.conditions:
            ci = conditions.index(c)
            mask &= cond_masks[ci].astype(bool)
        mask &= alive.astype(bool)
        covered = int(mask.sum())
        if covered < min_coverage:
            break

        best_rule.coverage = covered
        rules.append(best_rule)
        alive[mask] = 0

        remaining = int(alive.sum())
        elapsed = time.time() - t1
        print(
            f"    Rule {rule_idx + 1:3d}: "
            f"action={best_rule.action:2d}, "
            f"coverage={covered:7,d}, "
            f"mean_regret={best_rule.mean_regret:.4f}, "
            f"remaining={remaining:7,d} "
            f"({elapsed:.1f}s)"
        )

    # Default action
    uncovered_mask = alive.astype(bool)
    if uncovered_mask.sum() > 0:
        default_action, _ = _best_action_for_masked(
            best_action, regret, uncovered_mask, ds.num_actions,
        )
    else:
        default_action = 0

    print(f"    -> {len(rules)} rules, default={default_action}, uncovered={int(alive.sum()):,d}")
    return rules, default_action


# ── Output generation ────────────────────────────────────────────────────

def _action_label(action: int, decision_type: str) -> str:
    if decision_type == "category":
        if 0 <= action < len(CATEGORY_NAMES):
            return CATEGORY_NAMES[action]
        return f"Category {action}"
    else:
        if action == 0:
            return "Keep all"
        kept = []
        for i in range(5):
            if action & (1 << i) == 0:
                kept.append(str(i + 1))
        if not kept:
            return "Reroll all"
        return f"Reroll mask {action} (keep positions {','.join(kept)})"


def generate_skill_ladder(
    rosetta_dir: Path,
    output_dir: Path,
    max_rules: int = 100,
    min_coverage: int = 100,
) -> dict:
    """Load regret data, induce rules, write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, RegretDataset] = {}
    for dtype, filename in [
        ("category", "regret_category.bin"),
        ("reroll1", "regret_reroll1.bin"),
        ("reroll2", "regret_reroll2.bin"),
    ]:
        path = rosetta_dir / filename
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        t0 = time.time()
        datasets[dtype] = load_regret_file(path, dtype)
        print(f"  Loaded {dtype}: {len(datasets[dtype].features):,d} records ({time.time() - t0:.1f}s)")

    if not datasets:
        raise FileNotFoundError("No regret data files found")

    all_rules: dict[str, list[Rule]] = {}
    defaults: dict[str, int] = {}

    for dtype, ds in datasets.items():
        rules, default = induce_rules(ds, max_rules=max_rules, min_coverage=min_coverage)
        all_rules[dtype] = rules
        defaults[dtype] = default

    total_rules = sum(len(r) for r in all_rules.values())

    result = {
        "category_rules": [r.to_dict() for r in all_rules.get("category", [])],
        "reroll1_rules": [r.to_dict() for r in all_rules.get("reroll1", [])],
        "reroll2_rules": [r.to_dict() for r in all_rules.get("reroll2", [])],
        "meta": {
            "total_rules": total_rules,
            "default_category": defaults.get("category", 0),
            "default_reroll1": defaults.get("reroll1", 0),
            "default_reroll2": defaults.get("reroll2", 0),
            "oracle_ev": 245.87,
        },
    }

    json_path = output_dir / "skill_ladder.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {json_path}")

    md_path = output_dir / "YATZY_SKILL_LADDER.md"
    md_lines = [
        "# Yatzy Skill Ladder",
        "",
        "Human-readable decision rules distilled from the exact DP oracle.",
        f"Total rules: {total_rules}",
        "",
    ]

    for dtype, label in [
        ("category", "Category Selection"),
        ("reroll1", "First Reroll (2 rerolls left)"),
        ("reroll2", "Second Reroll (1 reroll left)"),
    ]:
        rules = all_rules.get(dtype, [])
        if not rules:
            continue
        md_lines.append(f"## {label}")
        md_lines.append("")
        for i, rule in enumerate(rules):
            conds = " AND ".join(c.to_human() for c in rule.conditions)
            action = _action_label(rule.action, dtype)
            md_lines.append(
                f"{i + 1}. **IF** {conds} **THEN** {action} "
                f"(covers {rule.coverage:,d} states, regret {rule.mean_regret:.4f})"
            )
        default = defaults.get(dtype, 0)
        md_lines.append(
            f"{len(rules) + 1}. **ELSE** {_action_label(default, dtype)}"
        )
        md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")

    return result
