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

SEMANTIC_REROLL_ACTIONS = [
    "Reroll_All", "Keep_Face_1", "Keep_Face_2", "Keep_Face_3",
    "Keep_Face_4", "Keep_Face_5", "Keep_Face_6", "Keep_Pair",
    "Keep_Two_Pairs", "Keep_Triple", "Keep_Quad",
    "Keep_Triple_Plus_Highest", "Keep_Pair_Plus_Kicker",
    "Keep_Straight_Draw", "Keep_All",
]
NUM_SEMANTIC_ACTIONS = 15


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


# ── Semantic reroll projection ────────────────────────────────────────────

# Face count feature indices in the feature vector (face_count_1 .. face_count_6)
_FACE_COUNT_START = FEATURE_NAMES.index("face_count_1")  # 7


def reconstruct_sorted_dice(face_counts: np.ndarray) -> np.ndarray:
    """Rebuild sorted 5-dice array from 6 face counts.

    E.g. counts [0,2,0,1,2,0] → dice [2,2,4,5,5].
    """
    dice = []
    for face_val in range(1, 7):
        cnt = int(face_counts[face_val - 1])
        dice.extend([face_val] * cnt)
    return np.array(dice, dtype=np.int32)


def reconstruct_all_dice(features: np.ndarray) -> np.ndarray:
    """Vectorized: reconstruct (N, 5) sorted dice from feature matrix."""
    counts = features[:, _FACE_COUNT_START:_FACE_COUNT_START + 6].astype(np.int32)
    n = len(counts)
    result = np.empty((n, 5), dtype=np.int32)
    for i in range(n):
        pos = 0
        for face_val in range(1, 7):
            cnt = counts[i, face_val - 1]
            if cnt > 0:
                result[i, pos:pos + cnt] = face_val
                pos += cnt
    return result


def semantic_to_bitmask(sorted_dice: np.ndarray, action_id: int) -> int:
    """Map one semantic action to one bitmask for one dice set.

    Convention: bit i set = REROLL position i. Dice are sorted ascending.
    Returns -1 if action is invalid for this dice.
    """
    dice = sorted_dice  # length 5, sorted ascending

    if action_id == 0:  # Reroll_All
        return 31

    if action_id == 14:  # Keep_All
        return 0

    if 1 <= action_id <= 6:  # Keep_Face_X
        face = action_id  # 1-6
        positions = [i for i in range(5) if dice[i] == face]
        if not positions:
            return -1
        mask = 31
        for p in positions:
            mask &= ~(1 << p)
        return mask

    # Count faces for pattern-based actions
    from collections import Counter
    counts = Counter(int(d) for d in dice)

    if action_id == 7:  # Keep_Pair — highest face with count≥2
        pair_faces = sorted([f for f, c in counts.items() if c >= 2], reverse=True)
        if not pair_faces:
            return -1
        target = pair_faces[0]
        # Keep 2 rightmost positions with that face
        positions = [i for i in range(5) if dice[i] == target]
        keep = positions[-2:]  # rightmost 2
        mask = 31
        for p in keep:
            mask &= ~(1 << p)
        return mask

    if action_id == 8:  # Keep_Two_Pairs — two highest pair faces
        pair_faces = sorted([f for f, c in counts.items() if c >= 2], reverse=True)
        if len(pair_faces) < 2:
            return -1
        keep_faces = pair_faces[:2]
        mask = 31
        for face in keep_faces:
            positions = [i for i in range(5) if dice[i] == face]
            for p in positions[-2:]:  # keep 2 of each
                mask &= ~(1 << p)
        return mask

    if action_id == 9:  # Keep_Triple — highest face with count≥3
        triple_faces = sorted([f for f, c in counts.items() if c >= 3], reverse=True)
        if not triple_faces:
            return -1
        target = triple_faces[0]
        positions = [i for i in range(5) if dice[i] == target]
        keep = positions[-3:]  # rightmost 3
        mask = 31
        for p in keep:
            mask &= ~(1 << p)
        return mask

    if action_id == 10:  # Keep_Quad — highest face with count≥4
        quad_faces = sorted([f for f, c in counts.items() if c >= 4], reverse=True)
        if not quad_faces:
            return -1
        target = quad_faces[0]
        positions = [i for i in range(5) if dice[i] == target]
        keep = positions[-4:]  # rightmost 4
        mask = 31
        for p in keep:
            mask &= ~(1 << p)
        return mask

    if action_id == 11:  # Keep_Triple_Plus_Highest — triple + best remaining
        triple_faces = sorted([f for f, c in counts.items() if c >= 3], reverse=True)
        if not triple_faces:
            return -1
        target = triple_faces[0]
        triple_pos = [i for i in range(5) if dice[i] == target]
        keep_triple = triple_pos[-3:]
        remaining = [i for i in range(5) if i not in keep_triple]
        if not remaining:
            return -1
        # Pick highest remaining die (rightmost = highest since sorted)
        best_remaining = remaining[-1]
        mask = 31
        for p in keep_triple + [best_remaining]:
            mask &= ~(1 << p)
        return mask

    if action_id == 12:  # Keep_Pair_Plus_Kicker — pair + highest remaining
        pair_faces = sorted([f for f, c in counts.items() if c >= 2], reverse=True)
        if not pair_faces:
            return -1
        target = pair_faces[0]
        pair_pos = [i for i in range(5) if dice[i] == target]
        keep_pair = pair_pos[-2:]
        remaining = [i for i in range(5) if i not in keep_pair]
        if not remaining:
            return -1
        best_remaining = remaining[-1]
        mask = 31
        for p in keep_pair + [best_remaining]:
            mask &= ~(1 << p)
        return mask

    if action_id == 13:  # Keep_Straight_Draw — longest consecutive run
        unique_faces = sorted(set(int(d) for d in dice))
        # Find longest consecutive subsequence
        best_run: list[int] = []
        current_run = [unique_faces[0]]
        for j in range(1, len(unique_faces)):
            if unique_faces[j] == unique_faces[j - 1] + 1:
                current_run.append(unique_faces[j])
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run[:]
                current_run = [unique_faces[j]]
        if len(current_run) > len(best_run):
            best_run = current_run[:]
        if len(best_run) < 2:
            return -1
        # Keep 1 die per face in the run
        mask = 31
        run_set = set(best_run)
        used: set[int] = set()
        for i in range(5):
            if int(dice[i]) in run_set and int(dice[i]) not in used:
                mask &= ~(1 << i)
                used.add(int(dice[i]))
        return mask

    return -1


def build_semantic_regret_matrix(
    features: np.ndarray, bitmask_regret: np.ndarray,
) -> np.ndarray:
    """Project (N, 32) bitmask regret to (N, 15) semantic regret.

    For each record and semantic action, looks up the corresponding bitmask
    and copies the regret. Invalid actions get 999.0.
    """
    n = len(features)
    all_dice = reconstruct_all_dice(features)
    semantic_regret = np.full((n, NUM_SEMANTIC_ACTIONS), 999.0, dtype=np.float32)

    for i in range(n):
        dice_i = all_dice[i]
        for a in range(NUM_SEMANTIC_ACTIONS):
            mask = semantic_to_bitmask(dice_i, a)
            if mask == -1:
                continue
            if mask >= bitmask_regret.shape[1]:
                continue
            val = bitmask_regret[i, mask]
            if val <= -1e30:
                continue
            semantic_regret[i, a] = val

    return semantic_regret


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
    action: int | str
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
    semantic: bool = False,
) -> tuple[int, float]:
    """Find best action for subset.

    When semantic=False: majority-vote on oracle actions (fast, for category rules).
    When semantic=True: regret-minimizing selection across all actions.
    Returns (action, mean_regret_of_that_action).
    """
    n = int(mask.sum())
    if n == 0:
        return 0, float("inf")

    if semantic:
        # Regret-minimizing: for each action, compute mean of valid regrets
        best_action = 0
        best_mean = float("inf")
        subset_regret = regret[mask]  # (n_match, num_actions)
        for a in range(num_actions):
            col = subset_regret[:, a]
            valid = col < 998.0  # 999.0 = invalid
            valid_count = int(valid.sum())
            if valid_count < n * 0.5:
                continue  # require ≥50% valid
            mean_reg = float(col[valid].mean())
            if mean_reg < best_mean:
                best_mean = mean_reg
                best_action = a
        if best_mean == float("inf"):
            return 0, float("inf")
        return best_action, best_mean
    else:
        subset_actions = best_actions[mask]
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
    semantic: bool = False,
) -> tuple[list[Rule], int | str]:
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
                semantic=semantic,
            )
            score = mean_reg
            act_out: int | str = SEMANTIC_REROLL_ACTIONS[action] if semantic else action

            if score < best_score:
                best_score = score
                best_rule = Rule(
                    conditions=[conditions[ci]],
                    action=act_out,
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
                    semantic=semantic,
                )
                score = mean_reg
                act_out2: int | str = SEMANTIC_REROLL_ACTIONS[action] if semantic else action

                if score < best_score:
                    best_score = score
                    best_rule = Rule(
                        conditions=[conditions[base_ci], conditions[ci2]],
                        action=act_out2,
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
        act_str = (
            best_rule.action if isinstance(best_rule.action, str)
            else f"{best_rule.action:2d}"
        )
        print(
            f"    Rule {rule_idx + 1:3d}: "
            f"action={act_str}, "
            f"coverage={covered:7,d}, "
            f"mean_regret={best_rule.mean_regret:.4f}, "
            f"remaining={remaining:7,d} "
            f"({elapsed:.1f}s)"
        )

    # Default action
    uncovered_mask = alive.astype(bool)
    if uncovered_mask.sum() > 0:
        default_idx, _ = _best_action_for_masked(
            best_action, regret, uncovered_mask, ds.num_actions,
            semantic=semantic,
        )
    else:
        default_idx = 0

    default_action: int | str = (
        SEMANTIC_REROLL_ACTIONS[default_idx] if semantic else default_idx
    )

    print(f"    -> {len(rules)} rules, default={default_action}, uncovered={int(alive.sum()):,d}")
    return rules, default_action


# ── Output generation ────────────────────────────────────────────────────

def _action_label(action: int | str, decision_type: str) -> str:
    if isinstance(action, str):
        return action.replace("_", " ")
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
    defaults: dict[str, int | str] = {}

    for dtype, ds in datasets.items():
        if dtype in ("reroll1", "reroll2"):
            # Project bitmask regret to semantic regret
            print(f"  Projecting {dtype} to semantic actions...")
            t_proj = time.time()
            semantic_regret = build_semantic_regret_matrix(ds.features, ds.regret)
            print(f"    Done in {time.time() - t_proj:.1f}s")
            sem_ds = RegretDataset(
                features=ds.features,
                q_values=ds.q_values[:, :NUM_SEMANTIC_ACTIONS] if ds.q_values.shape[1] >= NUM_SEMANTIC_ACTIONS else ds.q_values,
                best_action=np.zeros(len(ds.features), dtype=np.int32),
                regret=semantic_regret,
                num_actions=NUM_SEMANTIC_ACTIONS,
                decision_type=ds.decision_type,
            )
            rules, default = induce_rules(
                sem_ds, max_rules=max_rules, min_coverage=min_coverage,
                semantic=True,
            )
        else:
            rules, default = induce_rules(
                ds, max_rules=max_rules, min_coverage=min_coverage,
            )
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
            "default_reroll1": defaults.get("reroll1", "Reroll_All"),
            "default_reroll2": defaults.get("reroll2", "Reroll_All"),
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
