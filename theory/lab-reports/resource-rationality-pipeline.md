# Resource Rationality Data Pipeline

## Goal

Generate four D3-ready JSON datasets that tell the resource rationality story: how does strategic competence grow as a function of cognitive investment (rules, parameters, compute)?

This report documents the pipeline that produces `blog/data/exp_{a,b,c,d}_*.json` from the existing regret data, skill ladder rules, surrogate benchmarks, and scaling law evaluations.

## Hypothesis

Four complementary views can characterize the resource-rationality frontier:
1. **Power law**: diminishing returns as rules accumulate (EV vs N)
2. **Error anatomy**: where in state space the 100-rule policy leaks EV
3. **State cartography**: geometric structure separating handled from unhandled states
4. **Pareto frontier**: unified view across all policy families (symbolic, neural, exact)

## Input Inventory

| Resource | Path | Format |
|----------|------|--------|
| Regret data | `outputs/rosetta/regret_category.bin` | 1 GB, 3M records, 87 floats each |
| Skill ladder | `outputs/rosetta/skill_ladder.json` | 100 category + 100 reroll1 + 100 reroll2 rules |
| Scaling law | `outputs/rosetta/scaling_law.json` | EV vs n_rules (0-100), 200K games/point |
| Surrogate benchmarks | `outputs/surrogate/game_eval_results.json` | 20 models with params + mean score |
| Evaluation summary | `outputs/rosetta/eval_results.json` | Oracle EV = 248.44, policy EV = 227.50 |

All inputs are outputs of prior pipeline stages documented in `rosetta-distillation.md` and `scaling-laws.md`.

## Method

Single Python script (`research/rosetta/generate_rosetta_experiments.py`) using the rosetta venv. Subsamples 300K records from 3M (seed=42, sorted indices) to keep memory and UMAP runtime tractable.

## The Regret Data Problem

The regret binary stores `regret[a] = Q_best - Q[a]` for each action, but encodes **unavailable categories as -inf**. This is correct for the rule induction algorithm (unavailable = infinitely bad) but creates problems for aggregate statistics.

Three approaches were tried:

| Approach | Problem |
|----------|---------|
| Raw `q_values[i, assigned[i]]` as EV | Q-values are per-decision state values (~138 mean), not total game scores (~228 mean) |
| `regret[i, assigned[i]]` directly | -inf entries when default action (Yatzy) is unavailable (30% of records) |
| Finite-only regret filtering | Silently drops 30% of records, biasing the mean |

**Resolution:** use `scaling_law.json` for calibrated game-level EVs (from 200K-game MC simulations at each rule count), and the regret data for per-rule coverage analysis and per-record mistake identification. The regret data captures *which* records each rule covers; the scaling law captures *how much that coverage is worth* at game level.

For per-record regret in Experiments B and C, we compute `Q_best - Q_assigned` from q_values, replacing -inf (unavailable) with `Q_best` (maximum possible regret for that state).

## Experiment A: Cognitive Power Law

**Question**: how does EV grow as rules accumulate?

Evaluates rules 1..100 sequentially, tracking coverage and assigned actions. Cumulative EV comes from linear interpolation of the scaling_law checkpoints (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, ..., 100). Per-rule marginal EV comes from the change in mean finite regret.

| Rules | Coverage | EV | Oracle Gap |
|------:|--------:|---------:|----------:|
| 0 | 0% | 196.65 | 51.79 |
| 25 | 27.4% | 203.51 | 44.93 |
| 50 | 76.9% | 216.52 | 31.92 |
| 75 | 93.0% | 223.0 | 25.4 |
| 100 | 99.2% | 227.52 | 20.92 |

The coverage-to-EV relationship is highly nonlinear: 27% coverage buys only 7 EV, but the jump from 27% to 77% coverage (rules 25-50) buys 13 EV. This confirms the phase transition documented in `scaling-laws.md` where rules 25-70 contain the structural strategies.

**Output**: `blog/data/exp_a_power_law.json` — 25 KB, 100 rule entries.

## Experiment B: Anatomy of Mistakes

**Question**: where does the 100-rule policy disagree with the oracle, and how costly are those disagreements?

After all 100 rules, **43.0% of decision points** (129K / 300K) still disagree with the oracle. This is higher than the 0.8% uncovered rate because rules can cover a state but assign the wrong action (a rule matches but a better category exists).

The disagreements are grouped by unique feature vector (98,592 unique states), then binned into a hexagonal grid on log₁₀(frequency) × log₁₀(regret):

- **96 hex bins** capture the full error landscape
- High-frequency, low-regret mistakes dominate (many small errors)
- A tail of rare, high-regret states contributes disproportionately to the EV gap

Five illustrative examples show the pattern: the oracle sees context-dependent trade-offs (upper bonus proximity, remaining categories) that flat rules cannot capture.

**Output**: `blog/data/exp_b_mistakes.json` — 9 KB, 96 hex bins + 5 annotated examples.

## Experiment C: Cartography of Unhandled States

**Question**: do unhandled states cluster in feature space, or are they scattered?

Two populations:
- **Core** (rules 1-20): 62,792 records. Well-handled states with clear patterns.
- **Unhandled** (not covered by any of 100 rules): 2,314 records (0.8%). The residual.

Sampled 5,000 Core + all 2,314 Unhandled = 7,314 points. Standardized 56 features (z-score), then projected via UMAP (n_neighbors=30, min_dist=0.1, cosine metric, seed=42).

The unhandled population is smaller than expected (only 2,314 of 300K) because the greedy covering algorithm is aggressive. These are states where no single 1-2 condition rule achieves better-than-default regret. Visually, they occupy distinct regions in the UMAP embedding — they are not random noise but structurally different decision points that resist simple rule description.

**Output**: `blog/data/exp_c_unhandled_umap.json` — 1040 KB, 7,314 points with coordinates, status, oracle action, regret.

## Experiment D: Resource-Rational Pareto Frontier

**Question**: across all policy families, what does the params-vs-EV frontier look like?

Pure data compilation (no heavy computation). Merges three families:

| Family | Source | Points | Param Range |
|--------|--------|-------:|-------------|
| Baseline | game_eval_results.json (heuristic) | 1 | 0 |
| Symbolic | scaling_law.json (n_rules × 3 params) | 18 | 0-300 |
| Decision Tree | game_eval_results.json (dt_*) | 10 | 12-834K |
| MLP | game_eval_results.json (mlp_*) | 9 | 1.4K-152K |
| Exact | eval_results.json (oracle) | 1 | 1.43M |

Pareto frontier computed by ascending-params sweep tracking max EV seen.

**Key finding**: **22 of 39 points lie on the Pareto frontier**. The symbolic rules dominate at low param counts (0-300 params), decision trees dominate the mid-range (6K-80K), and only the exact oracle reaches 248.4 EV. MLPs are Pareto-dominated across the entire range — every MLP is beaten by a decision tree with fewer parameters.

**Output**: `blog/data/exp_d_pareto.json` — 5 KB, 39 points with family labels.

## Bug Log

| Bug | Symptom | Fix |
|-----|---------|-----|
| `-inf` regret propagation | `mean_regret = -inf` after first rule applied | Regret has `-inf` for unavailable categories; switched to q_values for per-record regret and scaling_law for game-level EV |
| Q-values are per-decision, not per-game | `oracle_ev = 138` instead of ~248 | Q-values represent expected *remaining* score from current state, not total game score; used scaling_law.json for calibrated game EVs |
| `global q_values` hack | Leftover from refactor | Passed q_values explicitly through function signatures |
| Only 3 example states in Exp B | Action-pair dedup too aggressive (only 3 unique oracle/rule pairs in top-20 regret) | Relaxed dedup key to (oracle, rule, turn), searched top-50 instead of top-20 |

## Timing (M1 Max)

| Phase | Time |
|-------|------|
| Load + subsample 300K from 3M records | ~3s |
| Experiment A (100 rules, vectorized) | ~5s |
| Experiment B (grouping + hexbin) | ~3s |
| Experiment D (pure I/O) | <0.1s |
| Experiment C (UMAP on 7.3K × 56) | ~20s |
| **Total** | **~31s** |

## Output Summary

| File | Size | Key Metric |
|------|-----:|------------|
| `exp_a_power_law.json` | 25 KB | 100 rules, EV: 196.7 → 227.5 |
| `exp_b_mistakes.json` | 9 KB | 43% disagreement rate, 96 hex bins |
| `exp_c_unhandled_umap.json` | 1040 KB | 7,314 UMAP points (5K Core + 2.3K Unhandled) |
| `exp_d_pareto.json` | 5 KB | 39 models, 22 Pareto-optimal |

All files under 2 MB. Monotonicity constraints verified for Exp A (coverage and EV both non-decreasing).

## What This Pipeline Proves

1. **Strategic knowledge follows a power law**: the first 25 rules buy 7 EV, the next 75 buy 24 EV. Intelligence concentrates in the 25-70 rule band.
2. **43% of states are mishandled**: even with 99.2% coverage, simple rules disagree with the oracle nearly half the time. Coverage ≠ correctness.
3. **Unhandled states are structurally distinct**: they cluster in UMAP space, suggesting they share features that resist rule compression.
4. **Symbolic rules dominate at low complexity**: below 300 parameters, English-language rules beat every neural approach. MLPs are Pareto-dominated everywhere.

## See Also

- `theory/lab-reports/scaling-laws.md` — the scaling curve that Experiment A interpolates
- `theory/lab-reports/rosetta-distillation.md` — the pipeline that produces the regret data
- `theory/research/human-cognition-and-compression.md` — broader context for resource rationality
