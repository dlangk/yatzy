"""Game-level evaluation of surrogate models.

Simulates full Yatzy games using trained sklearn classifiers (DTs/MLPs)
for all 3 decision types, measuring actual mean scores rather than
per-decision EV loss.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


# ── Yatzy constants ──────────────────────────────────────────────────────

CATEGORY_COUNT = 15

# Category indices (must match solver/src/constants.rs)
CAT_ONES = 0
CAT_TWOS = 1
CAT_THREES = 2
CAT_FOURS = 3
CAT_FIVES = 4
CAT_SIXES = 5
CAT_ONE_PAIR = 6
CAT_TWO_PAIRS = 7
CAT_THREE_OF_A_KIND = 8
CAT_FOUR_OF_A_KIND = 9
CAT_SMALL_STRAIGHT = 10
CAT_LARGE_STRAIGHT = 11
CAT_FULL_HOUSE = 12
CAT_CHANCE = 13
CAT_YATZY = 14


# ── Scoring rules ────────────────────────────────────────────────────────


def count_faces(dice: np.ndarray) -> np.ndarray:
    """Count face occurrences. Returns array of length 7 (index 0 unused)."""
    fc = np.zeros(7, dtype=np.int32)
    for d in dice:
        fc[d] += 1
    return fc


def calculate_score(dice: np.ndarray, category: int) -> int:
    """Compute Scandinavian Yatzy score for dice in given category."""
    fc = count_faces(dice)
    dice_sum = int(dice.sum())

    if category <= CAT_SIXES:
        face = category + 1
        return int(fc[face] * face)

    if category == CAT_ONE_PAIR:
        for f in range(6, 0, -1):
            if fc[f] >= 2:
                return 2 * f
        return 0

    if category == CAT_TWO_PAIRS:
        pairs = []
        for f in range(6, 0, -1):
            if fc[f] >= 2:
                pairs.append(f)
                if len(pairs) == 2:
                    return 2 * pairs[0] + 2 * pairs[1]
        return 0

    if category == CAT_THREE_OF_A_KIND:
        for f in range(6, 0, -1):
            if fc[f] >= 3:
                return 3 * f
        return 0

    if category == CAT_FOUR_OF_A_KIND:
        for f in range(6, 0, -1):
            if fc[f] >= 4:
                return 4 * f
        return 0

    if category == CAT_SMALL_STRAIGHT:
        if fc[1] == 1 and fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and fc[5] == 1:
            return 15
        return 0

    if category == CAT_LARGE_STRAIGHT:
        if fc[2] == 1 and fc[3] == 1 and fc[4] == 1 and fc[5] == 1 and fc[6] == 1:
            return 20
        return 0

    if category == CAT_FULL_HOUSE:
        three_face = pair_face = 0
        for f in range(1, 7):
            if fc[f] == 3:
                three_face = f
            elif fc[f] == 2:
                pair_face = f
        if three_face and pair_face:
            return dice_sum
        return 0

    if category == CAT_CHANCE:
        return dice_sum

    if category == CAT_YATZY:
        for f in range(1, 7):
            if fc[f] == 5:
                return 50
        return 0

    return 0


def update_upper_score(upper_score: int, category: int, score: int) -> int:
    """Update upper score (capped at 63)."""
    if category < 6:
        return min(upper_score + score, 63)
    return upper_score


def is_scored(scored: int, cat: int) -> bool:
    return bool(scored & (1 << cat))


# ── Feature extraction (must match export_training_data.rs exactly) ──────


def build_features(
    turn: int, upper_score: int, scored: int, dice: np.ndarray, rerolls_remaining: int | None = None
) -> np.ndarray:
    """Build feature vector matching Rust's build_shared_features.

    29 features for category decisions, 30 for reroll (appends rerolls_remaining).
    """
    fc = count_faces(dice)
    dice_sum = int(dice.sum())
    max_face = int(fc[1:7].max())
    num_distinct = int((fc[1:7] > 0).sum())

    upper_cats_left = sum(1 for c in range(6) if not is_scored(scored, c))
    bonus_secured = upper_score >= 63
    bonus_deficit = 0 if bonus_secured else 63 - upper_score

    features = [
        turn / 14.0,                          # 0: turn normalized
        upper_score / 63.0,                   # 1: upper_score normalized
        upper_cats_left / 6.0,                # 2: upper categories left
        1.0 if bonus_secured else 0.0,        # 3: bonus secured
        bonus_deficit / 63.0,                 # 4: bonus deficit
    ]
    # 5-10: face counts
    for f in range(1, 7):
        features.append(fc[f] / 5.0)
    # 11: dice sum
    features.append(dice_sum / 30.0)
    # 12: max face count
    features.append(max_face / 5.0)
    # 13: num distinct faces
    features.append(num_distinct / 6.0)
    # 14-28: category availability
    for c in range(CATEGORY_COUNT):
        features.append(0.0 if is_scored(scored, c) else 1.0)

    if rerolls_remaining is not None:
        features.append(rerolls_remaining / 2.0)  # 29: normalized

    return np.array(features, dtype=np.float32)


# ── Reroll mask application ──────────────────────────────────────────────


def apply_reroll(dice: np.ndarray, mask: int, rng: np.random.RandomState) -> np.ndarray:
    """Apply 5-bit reroll mask. Bit i set = reroll die at position i."""
    new_dice = dice.copy()
    for i in range(5):
        if mask & (1 << i):
            new_dice[i] = rng.randint(1, 7)
    new_dice.sort()
    return new_dice


# ── Fallback category selection ──────────────────────────────────────────


def pick_best_available_category(dice: np.ndarray, scored: int) -> int:
    """Greedy fallback: pick the available category with the highest score."""
    best_cat = -1
    best_score = -1
    for c in range(CATEGORY_COUNT):
        if not is_scored(scored, c):
            s = calculate_score(dice, c)
            if s > best_score:
                best_score = s
                best_cat = c
    if best_cat >= 0:
        return best_cat
    # All have score 0 — pick first available
    for c in range(CATEGORY_COUNT):
        if not is_scored(scored, c):
            return c
    raise RuntimeError("No available category")


# ── Game simulation ──────────────────────────────────────────────────────


@dataclass
class ModelTriple:
    """Three models (one per decision type) forming a complete player."""
    category: object  # sklearn classifier
    reroll1: object
    reroll2: object
    name: str = ""
    total_params: int = 0


def simulate_game(
    models: ModelTriple, rng: np.random.RandomState
) -> tuple[int, bool, dict[int, int]]:
    """Play one full Yatzy game using surrogate models.

    Returns (total_score, got_bonus, category_scores).
    """
    upper_score = 0
    scored = 0
    total = 0
    category_scores: dict[int, int] = {}

    for turn in range(CATEGORY_COUNT):
        # Roll 5 dice
        dice = np.sort(rng.randint(1, 7, size=5))

        # Reroll 1
        features = build_features(turn, upper_score, scored, dice, rerolls_remaining=2)
        mask = int(models.reroll1.predict(features.reshape(1, -1))[0])
        if mask != 0:
            dice = apply_reroll(dice, mask, rng)

        # Reroll 2
        features = build_features(turn, upper_score, scored, dice, rerolls_remaining=1)
        mask = int(models.reroll2.predict(features.reshape(1, -1))[0])
        if mask != 0:
            dice = apply_reroll(dice, mask, rng)

        # Category selection
        features = build_features(turn, upper_score, scored, dice)
        cat = int(models.category.predict(features.reshape(1, -1))[0])

        # Handle invalid predictions (category already scored)
        if is_scored(scored, cat):
            cat = pick_best_available_category(dice, scored)

        score = calculate_score(dice, cat)
        category_scores[cat] = score
        upper_score = update_upper_score(upper_score, cat, score)
        scored |= 1 << cat
        total += score

    got_bonus = upper_score >= 63
    if got_bonus:
        total += 50

    return total, got_bonus, category_scores


@dataclass
class BatchCategoryStats:
    """Per-category aggregate stats from a batch simulation."""
    cat_mean_scores: dict[int, float]
    cat_hit_rates: dict[int, float]
    mean_upper_total: float
    mean_chance: float
    yatzy_rate: float
    small_straight_rate: float
    large_straight_rate: float
    full_house_rate: float


def simulate_batch(
    models: ModelTriple, n_games: int, seed: int = 42
) -> tuple[np.ndarray, float, BatchCategoryStats]:
    """Simulate n_games and return (scores_array, bonus_rate, category_stats)."""
    rng = np.random.RandomState(seed)
    scores = np.empty(n_games, dtype=np.int32)
    bonus_count = 0
    cat_score_sums: dict[int, int] = {c: 0 for c in range(CATEGORY_COUNT)}
    cat_hit_counts: dict[int, int] = {c: 0 for c in range(CATEGORY_COUNT)}

    for i in range(n_games):
        scores[i], got_bonus, cat_scores = simulate_game(models, rng)
        if got_bonus:
            bonus_count += 1
        for cat, scr in cat_scores.items():
            cat_score_sums[cat] += scr
            if scr > 0:
                cat_hit_counts[cat] += 1

    cat_stats = BatchCategoryStats(
        cat_mean_scores={c: cat_score_sums[c] / n_games for c in range(CATEGORY_COUNT)},
        cat_hit_rates={c: cat_hit_counts[c] / n_games for c in range(CATEGORY_COUNT)},
        mean_upper_total=sum(cat_score_sums[c] for c in range(6)) / n_games,
        mean_chance=cat_score_sums[CAT_CHANCE] / n_games,
        yatzy_rate=cat_hit_counts[CAT_YATZY] / n_games,
        small_straight_rate=cat_hit_counts[CAT_SMALL_STRAIGHT] / n_games,
        large_straight_rate=cat_hit_counts[CAT_LARGE_STRAIGHT] / n_games,
        full_house_rate=cat_hit_counts[CAT_FULL_HOUSE] / n_games,
    )
    return scores, bonus_count / n_games, cat_stats


# ── Heuristic baseline (Python port of heuristic.rs) ─────────────────────


class HeuristicPlayer:
    """Stateless heuristic player matching heuristic.rs logic."""

    def reroll_mask(self, dice: np.ndarray, scored: int, upper_score: int) -> int:
        fc = count_faces(dice)

        # 1. Yatzy chase
        if not is_scored(scored, CAT_YATZY):
            for f in range(6, 0, -1):
                if fc[f] >= 3:
                    return self._mask_keeping_face(dice, f)

        # 2. Large straight [2,3,4,5,6]
        if not is_scored(scored, CAT_LARGE_STRAIGHT):
            m = self._straight_chase(dice, fc, [2, 3, 4, 5, 6])
            if m >= 0:
                return m

        # 3. Small straight [1,2,3,4,5]
        if not is_scored(scored, CAT_SMALL_STRAIGHT):
            m = self._straight_chase(dice, fc, [1, 2, 3, 4, 5])
            if m >= 0:
                return m

        # 4. Full house
        if not is_scored(scored, CAT_FULL_HOUSE):
            three_face = None
            pair_face = None
            for f in range(6, 0, -1):
                if fc[f] == 3:
                    three_face = f
                elif fc[f] == 2:
                    pair_face = f
            if three_face is not None:
                if pair_face is not None:
                    return 0  # full house, keep all
                return self._mask_keeping_face(dice, three_face)

        # 5. Four of a kind
        if not is_scored(scored, CAT_FOUR_OF_A_KIND):
            for f in range(6, 0, -1):
                if fc[f] >= 4:
                    return self._mask_keeping_n(dice, f, 4)

        # 6. Three of a kind
        if not is_scored(scored, CAT_THREE_OF_A_KIND):
            for f in range(6, 0, -1):
                if fc[f] >= 3:
                    return self._mask_keeping_face(dice, f)

        # 7. Two pairs
        if not is_scored(scored, CAT_TWO_PAIRS):
            pairs = [f for f in range(6, 0, -1) if fc[f] >= 2]
            if len(pairs) >= 2:
                return self._mask_keeping_two_faces(dice, pairs[0], pairs[1])

        # 8. One pair >= 8
        if not is_scored(scored, CAT_ONE_PAIR):
            for f in range(6, 0, -1):
                if fc[f] >= 2:
                    return self._mask_keeping_n(dice, f, 2)

        # 9. Fallback: keep highest die
        mask = 0
        for i in range(4):
            mask |= 1 << i
        return mask

    def pick_category(self, dice: np.ndarray, scored: int, upper_score: int) -> int:
        fc = count_faces(dice)

        def open_cat(c: int) -> bool:
            return not is_scored(scored, c)

        # 1-7: Pattern matching cascade
        if open_cat(CAT_YATZY) and calculate_score(dice, CAT_YATZY) == 50:
            return CAT_YATZY
        if open_cat(CAT_LARGE_STRAIGHT) and calculate_score(dice, CAT_LARGE_STRAIGHT) == 20:
            return CAT_LARGE_STRAIGHT
        if open_cat(CAT_SMALL_STRAIGHT) and calculate_score(dice, CAT_SMALL_STRAIGHT) == 15:
            return CAT_SMALL_STRAIGHT
        if open_cat(CAT_FULL_HOUSE) and calculate_score(dice, CAT_FULL_HOUSE) > 0:
            return CAT_FULL_HOUSE
        if open_cat(CAT_FOUR_OF_A_KIND) and calculate_score(dice, CAT_FOUR_OF_A_KIND) > 0:
            return CAT_FOUR_OF_A_KIND
        if open_cat(CAT_THREE_OF_A_KIND) and calculate_score(dice, CAT_THREE_OF_A_KIND) > 0:
            return CAT_THREE_OF_A_KIND
        if open_cat(CAT_TWO_PAIRS) and calculate_score(dice, CAT_TWO_PAIRS) > 0:
            return CAT_TWO_PAIRS
        if open_cat(CAT_ONE_PAIR) and calculate_score(dice, CAT_ONE_PAIR) >= 8:
            return CAT_ONE_PAIR

        # 9. Upper bonus chase
        if upper_score < 63:
            best_c = None
            best_s = 0
            for c in range(6):
                if open_cat(c):
                    s = calculate_score(dice, c)
                    face = c + 1
                    if s >= face * 3 and s > best_s:
                        best_c = c
                        best_s = s
            if best_c is not None:
                return best_c

        # 10. Chance >= 20
        if open_cat(CAT_CHANCE) and calculate_score(dice, CAT_CHANCE) >= 20:
            return CAT_CHANCE

        # 11. Any upper with score > 0
        best_c = None
        best_ratio = -1.0
        for c in range(6):
            if open_cat(c):
                s = calculate_score(dice, c)
                if s > 0:
                    ratio = s / ((c + 1) * 3)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_c = c
        if best_c is not None:
            return best_c

        # 12. Chance fallback
        if open_cat(CAT_CHANCE):
            return CAT_CHANCE

        # 13. One pair (any)
        if open_cat(CAT_ONE_PAIR) and calculate_score(dice, CAT_ONE_PAIR) > 0:
            return CAT_ONE_PAIR

        # 14. Dump order
        dump_order = [
            CAT_ONES, CAT_TWOS, CAT_THREES, CAT_YATZY,
            CAT_SMALL_STRAIGHT, CAT_LARGE_STRAIGHT, CAT_FULL_HOUSE,
            CAT_TWO_PAIRS, CAT_THREE_OF_A_KIND, CAT_FOUR_OF_A_KIND,
            CAT_ONE_PAIR, CAT_FOURS, CAT_FIVES, CAT_SIXES, CAT_CHANCE,
        ]
        for c in dump_order:
            if open_cat(c):
                return c

        raise RuntimeError("No open category")

    @staticmethod
    def _mask_keeping_face(dice: np.ndarray, face: int) -> int:
        mask = 0
        for i in range(5):
            if dice[i] != face:
                mask |= 1 << i
        return mask

    @staticmethod
    def _mask_keeping_n(dice: np.ndarray, face: int, n: int) -> int:
        mask = 0
        kept = 0
        for i in range(5):
            if dice[i] == face and kept < n:
                kept += 1
            else:
                mask |= 1 << i
        return mask

    @staticmethod
    def _mask_keeping_two_faces(dice: np.ndarray, f1: int, f2: int) -> int:
        mask = 0
        k1 = k2 = 0
        for i in range(5):
            if dice[i] == f1 and k1 < 2:
                k1 += 1
            elif dice[i] == f2 and k2 < 2:
                k2 += 1
            else:
                mask |= 1 << i
        return mask

    @staticmethod
    def _straight_chase(dice: np.ndarray, fc: np.ndarray, target: list[int]) -> int:
        present = sum(1 for f in target if fc[f] >= 1)
        if present < 4:
            return -1
        if present == 5:
            return 0
        missing = [f for f in target if fc[f] == 0][0]
        active = [f for f in target if f != missing]
        mask = 0
        remaining = list(active)
        for i in range(5):
            if dice[i] in remaining:
                remaining.remove(dice[i])
            else:
                mask |= 1 << i
        return mask


def simulate_heuristic_game(
    player: HeuristicPlayer, rng: np.random.RandomState
) -> tuple[int, bool, dict[int, int]]:
    """Play one full game using heuristic strategy.

    Returns (total_score, got_bonus, category_scores).
    """
    upper_score = 0
    scored = 0
    total = 0
    category_scores: dict[int, int] = {}

    for _turn in range(CATEGORY_COUNT):
        dice = np.sort(rng.randint(1, 7, size=5))

        # Two rerolls
        for _ in range(2):
            mask = player.reroll_mask(dice, scored, upper_score)
            if mask != 0:
                dice = apply_reroll(dice, mask, rng)

        cat = player.pick_category(dice, scored, upper_score)
        score = calculate_score(dice, cat)
        category_scores[cat] = score
        upper_score = update_upper_score(upper_score, cat, score)
        scored |= 1 << cat
        total += score

    got_bonus = upper_score >= 63
    if got_bonus:
        total += 50
    return total, got_bonus, category_scores


def simulate_heuristic_batch(
    n_games: int, seed: int = 42
) -> tuple[np.ndarray, float, BatchCategoryStats]:
    """Simulate n_games with heuristic strategy. Returns (scores, bonus_rate, category_stats)."""
    player = HeuristicPlayer()
    rng = np.random.RandomState(seed)
    scores = np.empty(n_games, dtype=np.int32)
    bonus_count = 0
    cat_score_sums: dict[int, int] = {c: 0 for c in range(CATEGORY_COUNT)}
    cat_hit_counts: dict[int, int] = {c: 0 for c in range(CATEGORY_COUNT)}

    for i in range(n_games):
        scores[i], got_bonus, cat_scores = simulate_heuristic_game(player, rng)
        if got_bonus:
            bonus_count += 1
        for cat, scr in cat_scores.items():
            cat_score_sums[cat] += scr
            if scr > 0:
                cat_hit_counts[cat] += 1

    cat_stats = BatchCategoryStats(
        cat_mean_scores={c: cat_score_sums[c] / n_games for c in range(CATEGORY_COUNT)},
        cat_hit_rates={c: cat_hit_counts[c] / n_games for c in range(CATEGORY_COUNT)},
        mean_upper_total=sum(cat_score_sums[c] for c in range(6)) / n_games,
        mean_chance=cat_score_sums[CAT_CHANCE] / n_games,
        yatzy_rate=cat_hit_counts[CAT_YATZY] / n_games,
        small_straight_rate=cat_hit_counts[CAT_SMALL_STRAIGHT] / n_games,
        large_straight_rate=cat_hit_counts[CAT_LARGE_STRAIGHT] / n_games,
        full_house_rate=cat_hit_counts[CAT_FULL_HOUSE] / n_games,
    )
    return scores, bonus_count / n_games, cat_stats


# ── Model loading and evaluation ─────────────────────────────────────────


@dataclass
class EvalResult:
    """Results from evaluating one model combo."""
    name: str
    total_params: int
    n_games: int
    mean: float
    std: float
    min: int
    p5: int
    p10: int
    p25: int
    p50: int
    p75: int
    p90: int
    p95: int
    p99: int
    max: int
    bonus_rate: float
    yatzy_rate: float = 0.0
    small_straight_rate: float = 0.0
    large_straight_rate: float = 0.0
    full_house_rate: float = 0.0
    mean_upper_total: float = 0.0
    mean_chance: float = 0.0


def compute_eval_result(name: str, total_params: int, scores: np.ndarray) -> EvalResult:
    """Compute statistics from score array."""
    n = len(scores)
    return EvalResult(
        name=name,
        total_params=total_params,
        n_games=n,
        mean=float(scores.mean()),
        std=float(scores.std()),
        min=int(scores.min()),
        p5=int(np.percentile(scores, 5)),
        p10=int(np.percentile(scores, 10)),
        p25=int(np.percentile(scores, 25)),
        p50=int(np.percentile(scores, 50)),
        p75=int(np.percentile(scores, 75)),
        p90=int(np.percentile(scores, 90)),
        p95=int(np.percentile(scores, 95)),
        p99=int(np.percentile(scores, 99)),
        max=int(scores.max()),
        bonus_rate=float((scores >= 0).mean()),  # placeholder, recomputed below
    )


def discover_model_combos(models_dir: Path) -> list[tuple[str, str]]:
    """Discover available DT model combos from saved .pkl files.

    Returns list of (combo_name, model_name_stem) pairs where all 3 decision
    types have a matching model file.
    """
    combos = []
    # Find all category models
    cat_models = sorted(models_dir.glob("category_dt_*.pkl"))
    for cat_path in cat_models:
        stem = cat_path.stem.replace("category_", "")  # e.g. "dt_d5"
        r1_path = models_dir / f"reroll1_{stem}.pkl"
        r2_path = models_dir / f"reroll2_{stem}.pkl"
        if r1_path.exists() and r2_path.exists():
            combos.append((stem, stem))

    # Also find MLP combos
    cat_mlps = sorted(models_dir.glob("category_mlp_*.pkl"))
    for cat_path in cat_mlps:
        stem = cat_path.stem.replace("category_", "")
        r1_path = models_dir / f"reroll1_{stem}.pkl"
        r2_path = models_dir / f"reroll2_{stem}.pkl"
        if r1_path.exists() and r2_path.exists():
            combos.append((stem, stem))

    return combos


def load_model_triple(
    models_dir: Path, model_stem: str, results_dir: Path | None = None
) -> ModelTriple:
    """Load 3 models forming a complete player."""
    cat = joblib.load(models_dir / f"category_{model_stem}.pkl")
    r1 = joblib.load(models_dir / f"reroll1_{model_stem}.pkl")
    r2 = joblib.load(models_dir / f"reroll2_{model_stem}.pkl")

    # Count params
    total_params = 0
    if results_dir is not None:
        for dtype in ["category", "reroll1", "reroll2"]:
            csv_path = results_dir / f"results_{dtype}.csv"
            if csv_path.exists():
                import polars as pl
                df = pl.read_csv(csv_path)
                row = df.filter(pl.col("name") == model_stem)
                if not row.is_empty():
                    total_params += int(row.row(0, named=True)["n_params"])

    return ModelTriple(
        category=cat, reroll1=r1, reroll2=r2,
        name=model_stem, total_params=total_params,
    )


def run_evaluation(
    models_dir: Path,
    results_dir: Path,
    output_dir: Path,
    n_games: int = 100_000,
    seed: int = 42,
) -> list[EvalResult]:
    """Run full game-level evaluation of all discovered model combos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[EvalResult] = []

    # 1. Heuristic baseline
    print("Evaluating heuristic baseline...")
    t0 = time.time()
    heur_scores, bonus_rate, heur_cat_stats = simulate_heuristic_batch(n_games, seed=seed)
    r = compute_eval_result("heuristic", 0, heur_scores)
    r.bonus_rate = bonus_rate
    _apply_category_stats(r, heur_cat_stats)
    all_results.append(r)
    print(
        f"  heuristic: mean={r.mean:.1f}, std={r.std:.1f}, "
        f"p5={r.p5}, p50={r.p50}, p95={r.p95}, bonus={r.bonus_rate:.1%} "
        f"({time.time()-t0:.1f}s)"
    )

    # 2. Surrogate model combos
    combos = discover_model_combos(models_dir)
    if not combos:
        print("No model combos found. Run surrogate-train first.")
        return all_results

    print(f"\nFound {len(combos)} model combos")
    for combo_name, model_stem in combos:
        print(f"\nEvaluating {combo_name}...")
        t0 = time.time()

        triple = load_model_triple(models_dir, model_stem, results_dir)
        scores, bonus_rate, cat_stats = simulate_batch(triple, n_games, seed=seed)
        r = compute_eval_result(combo_name, triple.total_params, scores)
        r.bonus_rate = bonus_rate
        _apply_category_stats(r, cat_stats)
        all_results.append(r)

        print(
            f"  {combo_name}: mean={r.mean:.1f}, std={r.std:.1f}, "
            f"p5={r.p5}, p50={r.p50}, p95={r.p95}, bonus={r.bonus_rate:.1%}, "
            f"params={r.total_params:,d} ({time.time()-t0:.1f}s)"
        )

    # Save results
    _save_eval_csv(all_results, output_dir / "game_eval_results.csv")
    _save_eval_json(all_results, output_dir / "game_eval_results.json")
    print(f"\nSaved results to {output_dir}/")

    return all_results


def _apply_category_stats(r: EvalResult, stats: BatchCategoryStats) -> None:
    """Copy category stats into an EvalResult."""
    r.yatzy_rate = stats.yatzy_rate
    r.small_straight_rate = stats.small_straight_rate
    r.large_straight_rate = stats.large_straight_rate
    r.full_house_rate = stats.full_house_rate
    r.mean_upper_total = stats.mean_upper_total
    r.mean_chance = stats.mean_chance


def _save_eval_csv(results: list[EvalResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "total_params", "n_games", "mean", "std",
            "min", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max",
            "bonus_rate", "yatzy_rate", "small_straight_rate", "large_straight_rate",
            "full_house_rate", "mean_upper_total", "mean_chance",
        ])
        for r in results:
            w.writerow([
                r.name, r.total_params, r.n_games,
                f"{r.mean:.2f}", f"{r.std:.2f}",
                r.min, r.p5, r.p10, r.p25, r.p50, r.p75, r.p90, r.p95, r.p99, r.max,
                f"{r.bonus_rate:.4f}", f"{r.yatzy_rate:.4f}",
                f"{r.small_straight_rate:.4f}", f"{r.large_straight_rate:.4f}",
                f"{r.full_house_rate:.4f}", f"{r.mean_upper_total:.2f}",
                f"{r.mean_chance:.2f}",
            ])


def _save_eval_json(results: list[EvalResult], path: Path) -> None:
    data = [
        {
            "name": r.name,
            "total_params": r.total_params,
            "n_games": r.n_games,
            "mean": round(r.mean, 2),
            "std": round(r.std, 2),
            "min": r.min,
            "p5": r.p5,
            "p10": r.p10,
            "p25": r.p25,
            "p50": r.p50,
            "p75": r.p75,
            "p90": r.p90,
            "p95": r.p95,
            "p99": r.p99,
            "max": r.max,
            "bonus_rate": round(r.bonus_rate, 4),
            "yatzy_rate": round(r.yatzy_rate, 4),
            "small_straight_rate": round(r.small_straight_rate, 4),
            "large_straight_rate": round(r.large_straight_rate, 4),
            "full_house_rate": round(r.full_house_rate, 4),
            "mean_upper_total": round(r.mean_upper_total, 2),
            "mean_chance": round(r.mean_chance, 2),
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2))
