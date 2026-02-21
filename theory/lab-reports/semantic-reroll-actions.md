# Semantic Reroll Actions: Experiment Report

## Motivation

The Rosetta Stone pipeline distills the DP oracle into human-readable IF-THEN rules.
Category rules (15 position-independent actions) performed well: **227.5 EV** (gap of 20.9 vs oracle 248.4).
However, reroll rules used 32 position-dependent bitmask actions (bit i = reroll position i).
The same intent ("keep the triple") maps to different masks for different dice, so induced rules didn't generalize.
Full bitmask rules: **168.7 EV** — reroll rules *hurt* rather than helped.

## Approach

Replace 32 bitmask actions with 15 semantic (position-independent) actions:

| ID | Action | Logic |
|----|--------|-------|
| 0 | Reroll All | mask=31 |
| 1-6 | Keep Face 1..6 | Keep all dice showing face X |
| 7 | Keep Pair | Keep 2 of highest face with count>=2 |
| 8 | Keep Two Pairs | Keep 4 dice forming two highest pairs |
| 9 | Keep Triple | Keep 3 of face with count>=3 |
| 10 | Keep Quad | Keep 4 of face with count>=4 |
| 11 | Keep Triple+Highest | Triple + best remaining die |
| 12 | Keep Pair+Kicker | Pair + highest remaining singleton |
| 13 | Keep Straight Draw | Longest consecutive unique-face run |
| 14 | Keep All | mask=0 |

**Key design**: no re-export needed. Project existing `(N, 32)` bitmask regret to `(N, 15)` semantic regret in Python; compile semantic intents back to position-specific bitmasks at Rust evaluation time.

Rule selection uses **regret-minimizing** selection (mean regret across subset) instead of majority-vote, since oracle best-action indices no longer exist in the projected space.

## Implementation

### Python (`skill_ladder.py`)
- `reconstruct_sorted_dice()` / `reconstruct_all_dice()`: rebuild dice from face count features
- `semantic_to_bitmask()`: map semantic action + dice -> bitmask (or -1 if invalid)
- `build_semantic_regret_matrix()`: project (N, 32) -> (N, 15) regret
- Updated `_best_action_for_masked()` with `semantic=True` mode
- Rule actions emitted as strings (e.g. `"Keep_Triple"`) in JSON

### Rust (`rosetta/policy.rs`)
- `SemanticRerollAction` enum (15 variants) with `parse()` method
- `RuleAction` enum: `CategoryIndex` | `SemanticReroll` | `BitmaskReroll`
- `compile_to_mask(dice, action)`: compile semantic intent to position-specific bitmask
- Updated JSON parsing (backward compatible — old integer actions still work)
- 11 unit tests covering all semantic actions

## Results

| Configuration | Policy EV | Oracle EV | EV Gap | Std Dev | p50 |
|---------------|-----------|-----------|--------|---------|-----|
| Oracle (exact DP) | 248.4 | 248.4 | 0.0 | — | — |
| Category-only (optimal rerolls) | 227.5 | 248.4 | 20.9 | 45.9 | 229 |
| Full rules — semantic rerolls | **184.4** | 248.4 | **64.0** | 45.1 | 183 |
| Full rules — bitmask rerolls (prior) | 168.7 | 248.4 | 79.7 | — | — |

### Distribution (semantic full rules, 1M games)

| Stat | Value |
|------|-------|
| Min | 52 |
| p5 | 118 |
| p25 | 148 |
| p50 | 183 |
| p75 | 215 |
| p95 | 266 |
| Max | 337 |

## Action Distribution in Induced Rules

### Reroll 1 (2 rerolls remaining) — 100 rules
| Action | Rules |
|--------|-------|
| Keep Face 6 | 38 |
| Keep Face 5 | 15 |
| Keep Face 4 | 10 |
| Keep Face 3 | 9 |
| Keep Pair | 7 |
| Keep Face 2 | 6 |
| Keep Face 1 | 5 |
| Keep Triple | 4 |
| Keep Quad | 2 |
| Keep Straight Draw | 2 |
| Keep All | 1 |
| Reroll All | 1 |

Default: **Keep Pair+Kicker**

### Reroll 2 (1 reroll remaining) — 100 rules
| Action | Rules |
|--------|-------|
| Keep Triple | 18 |
| Keep Face 6 | 12 |
| Keep Face 5 | 11 |
| Keep Pair | 11 |
| Keep Face 2 | 9 |
| Keep Face 3 | 9 |
| Keep Straight Draw | 6 |
| Keep Face 4 | 5 |
| Keep Face 1 | 5 |
| Keep Quad | 5 |
| Keep All | 5 |
| Keep Pair+Kicker | 3 |
| Keep Two Pairs | 1 |

Default: **Keep Face 6**

## Analysis

### Improvement over bitmask (+15.7 EV)

Semantic actions eliminated the position-dependence problem. The 15.7-point improvement (168.7 -> 184.4) confirms that bitmask fragmentation was real.

### Still far from category-only (-43.1 EV)

The full rules (184.4) are still 43 points below category-only (227.5). This means the semantic reroll rules are actively *hurting* compared to optimal rerolls. Two root causes:

1. **Vocabulary coverage gap**: the 15 semantic actions cannot represent every oracle decision. When the oracle's optimal mask doesn't correspond to any semantic action, all 15 columns get 999.0 (invalid). The regret-minimizing selection picks the least-bad option, but it may still be far from optimal.

2. **Rule granularity**: reroll decisions are inherently more context-dependent than category selection. The 56 semantic features may not capture enough state for a 100-rule decision list to approximate optimal rerolls. Category selection has 15 actions with clear feature correlates (has_yatzy -> pick Yatzy). Reroll selection requires reasoning about potential future states.

3. **Keep Face X dominance**: 83/100 reroll1 rules and 42/100 reroll2 rules use Keep Face X actions. This is essentially "collect face X" which is a reasonable heuristic but misses nuanced multi-category strategies (e.g. keeping a partial straight while also having a pair).

### Reroll 2 > Reroll 1

Reroll 2 rules are more diverse (18 Keep Triple, 11 Keep Pair, 6 Keep Straight Draw) and have lower mean regrets. This makes sense: with 1 reroll left, the decision is more about the current dice pattern. With 2 rerolls left, the decision is more speculative.

## Next Steps

Possible improvements to explore:

1. **Expand vocabulary** — add composite actions like Keep_Full_House, Keep_Pair_Plus_Straight_Draw, or more parametric actions (Keep_Best_N for N=1..4).

2. **More features** — add features like "expected score improvement from reroll", "probability of completing straight", "upper bonus distance" to give rules more signal.

3. **Deeper search** — allow 3-condition conjunctions (currently max 2) or increase rule budget beyond 100.

4. **Hybrid approach** — use semantic reroll rules only when confidence is high (low mean regret), fall back to a simple heuristic (e.g. always reroll all) for low-confidence states.

5. **Skip reroll rules entirely** — the category-only result (227.5) with optimal rerolls may be the practical ceiling for this rule-based framework. Reroll decisions may simply be too complex for a shallow decision list.

## Verification

- `cargo test`: 116 passed, 0 failed (11 new `compile_to_mask` tests)
- `cargo clippy`: no new warnings
- Category-only baseline: unchanged at 227.5 (confirms no regression)
- All reroll rules in JSON use semantic string names
- Backward compatibility: old integer actions still parsed via `BitmaskReroll`
