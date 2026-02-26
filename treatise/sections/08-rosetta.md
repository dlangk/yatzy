:::section{#part-vii}

:::part-title
Part VII
:::

## The Rosetta Stone

The solver's 16 MB strategy table encodes the optimal action for every reachable
state. No human can read it. Can we distill that knowledge into something a
person could memorize? A
::concept[sequential decision list]{sequential-decision-list}
of 100 category-selection rules --each a simple IF-THEN with one or two
conditions --recovers a mean score of 227.5 when paired with optimal
rerolls. Against the oracle's 248.4, that closes 59.7% of the gap, leaving a
40.3% incompressible residual of 20.9 EV points.

The rules emerge from greedy sequential covering over 370 candidate conditions
drawn from 56 semantic features (dice topology, bonus tracking, category
availability). Each condition is a simple threshold or equality test; rules
combine at most two conditions via conjunction. The algorithm searches
single conditions, then top-20 by top-30 conjunctions, selecting whichever
minimizes mean regret over uncovered states. Early rules are obvious ---
"if you have a large straight, score it." By rule 30, the algorithm discovers
structural strategies that experienced players miss: when to sacrifice
upper-section points for a better bonus trajectory, how to arbitrage Full
House against Chance, and which upper category to prioritize when multiple
score equally.

The scaling curve has three phases. The first 20 rules add only 2.7 EV,
picking off zero-regret corner cases like Yatzy and completed straights.
Then the curve accelerates: rules 26–40 deliver peak marginal value
of 0.74 EV/rule, discovering the structural core of Yatzy ---
upper-section management, Sixes accumulation, One Pair as the high-frequency
workhorse category. After 70 rules, returns diminish to 0.15 EV/rule as the
algorithm mops up context-dependent edge cases. The key landmarks: 0 rules
gives 196.7, 50 rules gives 216.5, 100 rules gives 227.5. Notably, roughly
67 rules suffice to surpass a trained neural network baseline at 221 EV.

:::html
<div class="chart-container" id="chart-rosetta-rules"><div id="chart-rosetta-rules-svg"></div></div>
:::

Reroll rules are harder. The natural action space is a 5-bit position bitmask,
but "keep positions 0, 1, 2" means entirely different things for [1,1,1,4,6]
versus [2,3,4,5,6]. Bitmask rules achieve only 168.7 EV. The fix is a
::concept[composable filter grammar]{composable-filter-grammar}
of 15 semantic actions --"keep the triple," "keep face 6," "keep the
straight draw" --that describe intent rather than position. Each verb
compiles to the correct bitmask at runtime. This recovers 184.4 EV, a +15.7
improvement that confirms position-dependence was the bottleneck.

:::html
<div class="chart-container" id="chart-filter-grammar"><div id="chart-filter-grammar-svg"></div></div>
:::

Yet semantic rules (184.4) still trail category-only mode (227.5) by 43 points.
Reroll decisions are inherently more speculative --they require reasoning
about potential future states, not just the current hand. The 15-verb vocabulary
cannot represent every oracle decision, and the 56 features may not carry
enough signal for a 100-rule list to approximate forward-looking reroll
strategy. The practical ceiling for this rule-based framework may be
category selection alone, with reroll decisions remaining in the domain
of lookup tables or learned surrogates.

The incompressible residual of 20.9 EV concentrates near the upper bonus
threshold (upper scores 40–62), where the value function is discontinuous.
Over half of all surrogate model errors land in this zone. No short rule list
captures the fine-grained state-dependence where a single category assignment
swings the 50-point bonus. This is the knowledge that resists compression
into language.

:::depth-2

The scaling law follows a three-phase structure. Let <var>N</var> be the number
of category rules and EV(<var>N</var>) the mean score under category-only
evaluation with optimal rerolls:

:::equation
EV(0) = 196.7 &nbsp;&rarr;&nbsp; EV(50) = 216.5 &nbsp;&rarr;&nbsp; EV(100) = 227.5
:::

Phase 1 (<var>N</var> = 0–20): foundation, +2.7 EV. Phase 2
(<var>N</var> = 25–70): acceleration, +21.9 EV with peak marginal value
of 0.74 EV/rule in the 26–40 band. Phase 3 (<var>N</var> = 70–100):
saturation, +6.2 EV. The phase transition at <var>N</var> &approx; 25
coincides with the algorithm's discovery of upper-section management rules
--- the structural backbone of good Yatzy play.

The greedy covering algorithm operates on a condition space of 370 candidates
generated from the 56-feature DSL: boolean features produce `== 0`
and `== 1` conditions, discrete features produce `==`,
`≥`, `≤` comparisons, and continuous features use
percentile thresholds at p10/p25/p50/p75/p90. Condition masks are pre-computed
as a dense (370 × 300K) uint8 matrix, enabling vectorized coverage
counting via `dot(mask, alive)`. Each rule selects the condition (or
two-condition conjunction from top-20 × top-30 candidates) that minimizes
mean regret over uncovered records.

The 15 semantic reroll verbs decompose into three families: face-specific
(Keep Face 1–6, 6 actions), pattern-based (Keep Pair, Keep Two Pairs,
Keep Triple, Keep Quad, Keep Triple+Highest, Keep Pair+Kicker, 6 actions),
and structural (Reroll All, Keep Straight Draw, Keep All, 3 actions). Projection
from 32-column bitmask regret to 15-column semantic regret is lossy: when no
semantic action matches the oracle's optimal bitmask, all 15 columns
receive a penalty value of 999.0. The regret-minimizing rule selector then
picks the least-bad semantic option.

:::

:::depth-3

The four-phase pipeline runs in approximately 12 minutes on an M1 Max:

```bash
# Phase 1-2: Export regret data (200K games, 8 threads)
YATZY_BASE_PATH=. solver/target/release/yatzy-regret-export \
  --games 200000                           # 8.9s, produces ~4 GB

# Phase 3: Greedy rule induction (Python)
yatzy-analyze induce-rules \
  --subsample 300000 --max-rules 100       # 673s for 3 x 100 rules

# Phase 4: Evaluate category-only (1M games)
YATZY_BASE_PATH=. solver/target/release/yatzy-eval-policy \
  --rules outputs/rosetta/rules_cat.json \
  --mode category-only --games 1000000     # 21.6s, EV = 227.5

# Evaluate full semantic rules (1M games)
YATZY_BASE_PATH=. solver/target/release/yatzy-eval-policy \
  --rules outputs/rosetta/rules_all.json \
  --mode full --games 1000000              # 2.5s, EV = 184.4
```

The Rust evaluator compiles semantic actions to position-specific bitmasks at
runtime. The `compile_to_mask(dice, action)` function maps each of
the 15 `SemanticRerollAction` variants to the appropriate 5-bit mask
given sorted dice. For example, `KeepTriple` with dice
`[2,5,5,5,6]` produces mask `0b10001` (reroll positions
0 and 4, keep positions 1–3). The compilation is O(1) per action and
handles edge cases (no valid triple, no pair, etc.) by falling back to
reroll-all.

The top 5 induced category rules illustrate the zero-regret foundation:

```
1. IF has_large_straight         THEN Large Straight   (regret 0.0000)
2. IF face_count_1 == 5          THEN Yatzy             (regret 0.0000)
3. IF categories_left <= 1      THEN Four of a Kind    (regret 0.0000)
4. IF face_count_1 == 4          THEN Ones              (regret 0.0000)
5. IF !cat_avail_fives AND
      face_count_5 >= 5         THEN Yatzy             (regret 0.0000)
```

:::

:::
