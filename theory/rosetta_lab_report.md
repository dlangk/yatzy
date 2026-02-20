# Rosetta Stone Lab Report: Policy Distillation into Human-Readable Rules

## Goal

Distill the 8MB exact DP oracle (2M states × 3 decision types) into a human-readable IF-ELIF-ELSE rule set (Sequential Decision List) optimized for EV-regret minimization.

## Architecture

Four-phase pipeline:

1. **Semantic Feature DSL** (Rust) — 56 human-interpretable features from game state + dice
2. **Regret Data Export** (Rust) — simulate 200K games under optimal policy, export features + full Q-vectors + regret vectors
3. **Greedy Rule Induction** (Python) — sequential covering algorithm over subsampled records
4. **Policy Evaluation** (Rust) — Monte Carlo simulation with rule-based policy

## Phase 1+2: Feature Design and Regret Export

### Feature Set (56 features)

| Group | Features | Count |
|-------|----------|-------|
| Game progress | turn, categories_left, rerolls_remaining | 3 |
| Upper bonus | upper_score, bonus_secured, bonus_pace, upper_cats_left | 4 |
| Dice topology | face_counts[6], max_count, num_distinct, dice_sum | 9 |
| Pattern booleans | has_pair, has_two_pair, has_three/four_of_kind, has_full_house, has_small/large_straight, has_yatzy | 8 |
| Category availability | cat_available[15] | 15 |
| Category scores | cat_scores[15] | 15 |
| Safety valves | zeros_available, best_available_score | 2 |

All features exported as raw (unnormalized) values matching the evaluator's `get_feature_value()` function. This is critical — an earlier version normalized features (turn/14, face_count/5, etc.) during export but the evaluator read raw values, causing a complete mismatch that reduced policy EV from 228 to 86.

### Export Format

Binary files with 32-byte header (magic `0x52455052`, version 1):

| File | Records | Floats/record | Size |
|------|---------|---------------|------|
| regret_category.bin | 3M | 87 (56 + 15 + 1 + 15) | 1.0 GB |
| regret_reroll1.bin | 3M | 121 (56 + 32 + 1 + 32) | 1.5 GB |
| regret_reroll2.bin | 3M | 121 (56 + 32 + 1 + 32) | 1.5 GB |

Each record: `[features | q_values | best_action | regret_vector]` where `regret[a] = q_best - q[a]`.

**Performance:** 200K games exported in 8.9s (22K games/s) on M1 Max, 8 threads.

## Phase 3: Greedy Sequential Covering

### Algorithm

```
rules = []
uncovered = all_records (subsampled to 300K)
while uncovered is not empty and len(rules) < max_rules:
    best_rule = search(single conditions, then top-20 × top-30 conjunctions)
    apply best_rule, remove covered records
default_action = majority vote on remaining uncovered records
```

### Condition Search

- 370 candidate conditions generated from feature data:
  - Boolean features: `== 1` and `== 0`
  - Discrete features (≤20 unique values): `==`, `>=`, `<=` for each value
  - Continuous features: percentile thresholds at p10/p25/p50/p75/p90
  - `cat_score_*` features excluded (too many unique values)
- Condition masks pre-computed as `(370, 300K)` uint8 matrix
- Best action per masked subset via `np.bincount` majority vote on oracle actions

### Performance Optimizations

The naive approach (3M records × 500 conditions × 100 rules) was intractable. Three optimizations made it feasible:

1. **Subsampling**: 300K records from 3M (seed=42), preserving the action distribution
2. **Pre-computed condition masks**: Dense `(num_conds, N)` uint8 matrix, enabling vectorized `dot(mask, alive)` for coverage counting
3. **Majority-vote action selection**: `np.bincount` on oracle actions instead of scanning all action columns' regret

Total induction time: 673s (~11 min) for 300 rules across 3 decision types.

### Rule Counts

| Decision Type | Rules | Default Action | Uncovered |
|---------------|-------|----------------|-----------|
| Category | 100 | Yatzy (14) | 2,372 (0.8%) |
| Reroll 1 | 100 | mask 19 | 22,386 (7.5%) |
| Reroll 2 | 100 | mask 19 | 8,906 (3.0%) |

## Phase 4: Evaluation Results

### Summary

| Mode | Policy EV | Oracle EV | Gap | Notes |
|------|-----------|-----------|-----|-------|
| Category-only (optimal rerolls) | **227.5** | 248.4 | **20.9** | Isolates category rule quality |
| Full rules (all 300) | **168.7** | 248.4 | **79.8** | Reroll rules are the bottleneck |

### Category-Only Distribution (1M games)

| Statistic | Value |
|-----------|-------|
| Mean | 227.5 |
| Std | 45.9 |
| Min | 70 |
| p5 | 150 |
| p25 | 194 |
| p50 | 229 |
| p75 | 260 |
| p95 | 302 |
| Max | 357 |

### Comparison to Other Approaches

| Policy | Params | Mean | Gap to Oracle |
|--------|--------|------|---------------|
| Oracle (DP lookup) | 2M entries | 248.4 | 0 |
| DT depth-20 | 413K | 245 | 3.4 |
| DT depth-15 | 81K | 239 | 9.4 |
| **Category rules (100 rules)** | **~100 rules** | **227.5** | **20.9** |
| MLP [64,32] | 5K | 221 | 27.4 |
| DT depth-10 | 6K | 216 | 32.4 |
| Heuristic | 0 | 166 | 82.4 |

The 100 category rules (with optimal rerolls) place between dt_d10 and mlp_64 in performance, but are **human-readable** — a qualitative advantage no other compressed policy achieves.

## Sample Category Rules (Top 15)

```
 1. IF has_large_straight           THEN Large Straight    (regret 0.0000)
 2. IF face_count_1 == 5            THEN Yatzy             (regret 0.0000)
 3. IF categories_left <= 1         THEN Four of a Kind    (regret 0.0000)
 4. IF face_count_1 == 4            THEN Ones              (regret 0.0000)
 5. IF !cat_avail_fives AND
       face_count_5 >= 5            THEN Yatzy             (regret 0.0000)
 6. IF categories_left >= 12 AND
       face_count_3 == 4            THEN Threes            (regret 0.0000)
 7. IF upper_cats_left == 6 AND
       face_count_3 == 5            THEN Yatzy             (regret 0.0000)
 8. IF categories_left <= 13 AND
       face_count_2 == 4            THEN Twos              (regret 0.0000)
 9. IF face_count_3 == 3 AND
       face_count_2 >= 1            THEN Threes            (regret 0.0042)
10. IF zeros_available >= 2 AND
       face_count_2 == 5            THEN Yatzy             (regret 0.0000)
11. IF zeros_available >= 2 AND
       face_count_3 == 4            THEN Threes            (regret 0.0006)
12. IF face_count_2 >= 2 AND
       face_count_1 >= 3            THEN Ones              (regret 0.0000)
13. IF !cat_avail_full_house AND
       face_count_3 == 3            THEN Threes            (regret 0.0060)
14. IF upper_cats_left <= 3 AND
       has_small_straight            THEN Small Straight    (regret 0.0000)
15. IF face_count_2 == 1 AND
       face_count_1 >= 3            THEN Ones              (regret 0.0000)
```

The first 15 rules have near-zero regret and capture clear strategic patterns: take completed patterns immediately, score upper categories with 4+ matching dice, and handle Yatzy opportunities.

## Why Reroll Rules Fail

The reroll action space uses 5-bit position bitmasks (0-31): mask bit `i` = reroll die at position `i`. Since dice are sorted, mask 7 ("keep positions 0,1,2") always keeps the three lowest dice — but [1,1,1] and [2,3,4] require completely different strategic reasoning.

The induced rules learn statistical associations (e.g., "IF has_three_of_kind THEN mask 24") but these don't generalize because the same semantic intent ("keep the triple") maps to different bitmasks depending on which positions the triple occupies. The evaluator applies the literal bitmask regardless of the actual dice, causing catastrophic mismatches.

### Path Forward

Reroll rules need a semantic action space:
- "Keep all dice matching face X" (6 actions)
- "Keep the pair/triple/quad" (pattern-based)
- "Keep the straight draw" (sequence-based)
- "Keep all" / "Reroll all"

This requires reworking the export to map bitmasks to semantic descriptions and the evaluator to map semantic descriptions back to position-specific bitmasks at runtime.

## Bug Log

| Bug | Symptom | Fix |
|-----|---------|-----|
| Normalized features in export | Rules used thresholds like `face_count_3 >= 0.6` but evaluator returned raw values (3) | Removed all normalization from `features_to_f32()` |
| Phase 3 performance | 3M records × 500 conditions hung for 30+ min | Subsampling (300K) + pre-computed mask matrix + majority-vote |
| Type inference in Rust | Single-element array `[("category", &records)]` couldn't infer types | Replaced loop with direct block |

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `solver/src/rosetta/mod.rs` | 7 | Module declaration |
| `solver/src/rosetta/dsl.rs` | 477 | SemanticFeatures + compute + serialize |
| `solver/src/rosetta/policy.rs` | 400 | Rule evaluation + game simulation |
| `solver/src/bin/regret_export.rs` | 557 | Simulate + export regret data |
| `solver/src/bin/eval_policy.rs` | 210 | MC evaluation of rule policy |
| `analytics/src/yatzy_analysis/skill_ladder.py` | 490 | Greedy sequential covering |

## Timing (M1 Max, 8 threads)

| Phase | Time |
|-------|------|
| Regret export (200K games) | 8.9s |
| Rule induction (3 × 100 rules) | 673s |
| Evaluation (1M games, category-only) | 21.6s |
| Evaluation (1M games, full rules) | 2.5s |
| **Total pipeline** | **~12 min** |
