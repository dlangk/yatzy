# Scaling Laws of Strategic Compression

## The Experiment

How many English-language rules does it take to play Yatzy well?

We swept from N=0 to N=100 category selection rules, each evaluated with **optimal DP rerolls** (category-only mode). This isolates the pure strategic compression question: how efficiently can greedy sequential covering extract human-readable decision rules from the exact DP oracle?

Each point: 200,000 games, seed=42, 8 threads. Standard error ~0.1 EV.

## The Scaling Curve

```
EV
230 ┤                                                          ●──●
    │                                                     ●
225 ┤                                                ●
    │                                           ●
221 ┤· · · · · · · · · · · · · · · · · · · · ● · · · · · · · · · NN baseline
220 ┤                                    ●
    │
215 ┤                               ●
    │                          ●
210 ┤
    │
205 ┤                     ●
    │                ●
200 ┤           ●
    │  ● ● ● ● ● ●
197 ┤●
    └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→ N
       0  1  3  5  7 10 15 20 25 30    40    50    60    70 80 90 100
```

| N rules | EV | Oracle gap | p50 | Phase |
|--------:|-------:|----------:|----:|-------|
| 0 | 196.65 | 51.79 | 190 | Baseline (default action only) |
| 1 | 196.94 | 51.50 | 191 | Plateau |
| 5 | 197.67 | 50.77 | 191 | Plateau |
| 10 | 198.67 | 49.77 | 193 | Slow climb |
| 15 | 199.35 | 49.09 | 194 | Slow climb |
| 20 | 199.35 | 49.09 | 194 | Slow climb |
| **25** | **203.51** | **44.93** | **200** | **Phase transition begins** |
| 30 | 204.54 | 43.90 | 202 | Acceleration |
| 40 | 214.62 | 33.82 | 218 | Steep climb |
| 50 | 216.52 | 31.92 | 221 | Steep climb |
| 60 | 219.94 | 28.50 | 224 | Steep climb |
| **70** | **221.29** | **27.15** | **224** | **Crosses NN baseline (221)** |
| 80 | 224.61 | 23.83 | 227 | Diminishing returns |
| 90 | 227.24 | 21.20 | 229 | Saturation |
| 100 | 227.52 | 20.92 | 229 | Saturation |

## Three Phases of Strategic Emergence

### Phase 1: The Foundation (N=0..20) — +2.7 EV

The first 20 rules barely move the needle. The default action (Yatzy, category 14) handles the degenerate cases — five-of-a-kind, large straight when present. These rules cover rare, high-certainty situations. EV creeps from 196.7 to 199.4.

**What's happening**: the algorithm is picking off zero-regret corner cases. "If you have a large straight, score it." "If you rolled Yatzy, score Yatzy." Important for correctness, negligible for performance.

### Phase 2: The Acceleration (N=25..70) — +21.9 EV

Between rules 25 and 70, EV surges from 203.5 to 221.3. This is where the algorithm discovers the *structural* strategies:

- **Upper section management** (rules 22-24, 30, 34, 40): when to chase Fives vs Fours, when to take safe upper-section points
- **Full House / Chance arbitrage** (rules 35, 42, 46): the non-obvious trade-offs between Full House and Chance
- **One Pair dominance** (rules 49, 53, 59, 61): One Pair as the high-frequency workhorse category
- **Sixes accumulation** (rule 37): the single highest-impact rule — take Sixes when available and dice support it

This phase contains the Pareto knee. The marginal value per rule peaks around N=30-50 where each additional rule adds ~0.4-0.5 EV.

### Phase 3: Saturation (N=70..100) — +6.2 EV

After 70 rules, returns diminish sharply. Rules 70-100 are mopping up context-dependent edge cases — specific combinations of remaining categories, upper bonus proximity, and dice patterns. Each rule adds ~0.2 EV.

## The Key Numbers

| Milestone | N rules | EV |
|-----------|--------:|------:|
| No rules (default only) | 0 | 196.7 |
| Crosses 200 EV | 25 | 203.5 |
| Crosses 210 EV | ~35 | ~210 |
| **Crosses NN baseline (221)** | **~67** | **221** |
| 90% of max rule EV | 80 | 224.6 |
| Full 100 rules | 100 | 227.5 |
| Oracle (exact DP) | — | 248.4 |

## Strategic Compressibility

The oracle has 2,097,152 state values. We compress this to 100 English sentences and recover **59.7%** of the oracle gap:

```
Oracle gap (default → oracle):  51.79 EV
Rules close:                    30.87 EV  (59.7%)
Residual gap (100 rules → oracle): 20.92 EV  (40.3%)
```

The residual 20.9 EV gap is the **incompressible** component under this representation — the strategic knowledge that requires state-level lookup tables rather than human-parseable conditions.

## Beating the Neural Network

The 221 EV neural network baseline falls at **~67 rules**. This means:

> **67 English sentences, each a simple IF-THEN rule with 1-2 conditions, outperform a trained neural network at Yatzy category selection.**

This is a remarkable compression ratio. The NN has thousands of parameters encoding implicit state representations. The rule list has 67 explicit conditions that a human can read, understand, and memorize.

## Marginal Value Curve

| Rule range | EV gain | Per-rule gain | Character |
|------------|---------|---------------|-----------|
| 1-10 | 2.0 | 0.20 | Corner cases |
| 11-25 | 4.9 | 0.33 | Structural setup |
| 26-40 | 11.1 | **0.74** | Peak marginal value |
| 41-60 | 5.3 | 0.27 | Core strategy |
| 61-80 | 4.7 | 0.23 | Refinement |
| 81-100 | 2.9 | 0.15 | Edge cases |

The peak marginal value band (rules 26-40) is where the algorithm discovers the *shape* of good Yatzy play. This is where "intelligence" concentrates.

## What The First 10 Rules Say

For reference, the most valuable early rules that the algorithm discovers:

1. IF has_large_straight THEN Large Straight (10K states, 0.000 regret)
2. IF face_count_1 == 5 THEN Yatzy (935 states, 0.000 regret)
3. IF has_four_of_kind AND cat_avail_four_of_kind THEN Four of a Kind (19K states, 0.000 regret)
4. IF has_small_straight AND cat_avail_small_straight THEN Ones (3.8K states, 0.000 regret)
5. IF face_count_2 == 5 THEN Yatzy (757 states, 0.000 regret)
6. IF face_count_3 >= 4 AND cat_avail_threes THEN Threes (2.6K states, 0.000 regret)
7. IF face_count_3 == 5 THEN Yatzy (253 states, 0.000 regret)
8. IF max_count == 5 AND cat_avail_twos THEN Twos (4.6K states, 0.000 regret)
9. IF max_count >= 4 AND cat_avail_threes THEN Threes (4.4K states, 0.004 regret)
10. IF face_count_4 == 5 THEN Yatzy (1.0K states, 0.000 regret)

The pattern: sweep up the zero-regret certainties first (Yatzy, Large Straight, Four of a Kind), then begin the upper-section management.

## Methodology

- **Rule induction**: greedy sequential covering with regret minimization
- **Evaluation**: category-only mode — optimal DP rerolls + rule-based category selection
- **Games**: 200,000 per sweep point (SE ~0.1 EV)
- **Seed**: 42 (deterministic)
- **Rule budget**: 100 rules max, min coverage 100 states per rule
- **Feature space**: 56 semantic features (dice topology, bonus tracking, category availability, scores)
- **Condition depth**: max 2 conjuncts per rule
