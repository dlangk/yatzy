# Fast-Exp LSE Bias: The θ = 0.15/0.16 Seam Bug

**Date:** 2026-07-09. **Status:** root-caused, fixed, verified.

A silent numerical bug in the NEON fast-exp kernel biased every log-domain
(LSE) solve low. All strategy tables with |θ| > 0.15 computed on Apple Silicon
before 2026-07-09 undervalued L(S) by ≈ 4 in log domain (≈ 25 CE points at the
seam) and carried mildly distorted policies. The utility-domain solver
(|θ| ≤ 0.15) was exact throughout. The bug surfaced as a visible discontinuity
in the mean-variance frontier between θ = 0.15 and θ = 0.16, symmetric in sign.

## 1. Symptom

After fixing an unrelated data-staleness issue in the treatise pipeline, the
exact-density sweep still showed a kink at the ±0.15/±0.16 boundary:

| θ | mean (exact density) | | θ | mean |
|---:|---:|---|---:|---:|
| 0.14 | 230.44 | | −0.14 | 233.39 |
| 0.15 | 229.23 | | −0.15 | 231.89 |
| 0.16 | 230.41 (reversal) | | −0.16 | 226.27 (5.6-pt cliff) |
| 0.17 | 228.89 | | −0.17 | 225.52 |

mean(θ) cannot locally increase on the risk-seeking branch under correct
optimization; the boundary is exactly the solver's regime switch
(utility-domain for |θ| ≤ 0.15, log-domain LSE above), so the two code paths
disagreed.

## 2. Hypotheses

- **H1 (stale data):** tables at the seam were from different solver
  versions. Rejected: fresh recomputes reproduced the discontinuity.
  (A red herring made H1 look confirmed at first: `yatzy-precompute`
  silently loads an existing table instead of recomputing, so early
  "recompute" controls were no-ops. See §6.)
- **H2 (utility solver wrong):** rejected by ground truth (§3).
- **H3 (LSE solver wrong):** confirmed.

## 3. Method: exact PMF as ground truth

The forward density evolution produces the exact score PMF of the policy
stored in a table. From the PMF, ln E[e^{θX}] is computable independently of
the solver's recursion, so each solver can be audited against the true value
of its own policy:

| Solver | claimed L₀ | exact PMF L₀ | verdict |
|---|---:|---:|---|
| utility, θ = 0.15 | 46.1914 | 46.1913 | exact |
| LSE, θ = 0.16 | 45.5810 | 49.4772 | undervalues own policy by 3.90 |

Head-to-head at θ = 0.16, the utility solver's θ = 0.15 policy (ln MGF
49.5227) beat the LSE solver's own policy (49.4772): the LSE argmax was also
mildly wrong, not just its reported value. A second independent proof:
stored L₀ values were non-monotone in θ across the seam
(L₀(0.140) = 42.87 > L₀(0.146) = 40.79), impossible for correct solvers since
dL₀/dθ equals the exponentially tilted mean, which is positive.

## 4. Root cause

`neon_fast_exp_f32x4` (solver/src/simd.rs), used only by
`neon_weighted_exp_sum_64` in the LSE chance-node kernels, mixed two
incompatible fast-exp variants:

- Range reduction computed the natural-log remainder r = x − n·ln2 ∈ [0, ln2)
  (expf style, expecting an e^r polynomial).
- The polynomial was the Cephes exp2f set (ln2, ln2²/2!, …), which computes
  2^f for the base-2 fraction f = t − floor(t) ∈ [0, 1).

The kernel therefore returned 2^n·2^r instead of 2^n·e^r:

```
returned / true = 2^(−f(1−ln2)) = e^(−0.2127·f),  f ∈ [0, 1)
```

Up to −19.2% per exp call, ≈ −10% on average, and exactly 0 at f = 0. The
max element of every LSE has diff = 0, so the dominant term was always exact:
values stayed plausible, argmax mostly survived, and the unit test of the day
(exp(0) = 1) sat precisely on the bug's fixed point. Per-node LSE
underestimate ≈ −0.09; compounded over ~45 chance nodes per game ≈ −4.0 in
L₀, matching the measured −3.90. The scalar non-aarch64 fallback used exact
`.exp()`, so only Apple Silicon builds were affected.

## 5. Fix and verification

Fix (solver/src/simd.rs): feed the polynomial its designed input, the base-2
fraction f = t − n. One vsub replaces two vfms; the ln2 hi/lo constants are
gone. Accuracy after fix: 1.8e-5 max relative error (the polynomial's design
accuracy at f → 1), enforced by a new regression test sweeping [−22, +1]
(`test_fast_exp_accuracy_range`).

Verification, in order:

1. Forced-LSE build at θ = 0.15: L₀ = 46.1917 vs utility 46.1914 (the two
   solvers agree; seam closed).
2. θ = 0.16 recompute: L₀ = 49.5244 ≥ 49.5227 (better than the best
   previously-known feasible policy, as an optimal solver must be).
3. Utility path untouched: θ = 0.05 and θ = 0.15 tables bit-identical to
   pre-fix. EV (θ = 0) and oracle unaffected (no fast-exp in those paths).
4. All 189+ solver tests pass; L₀ strictly increasing across the full
   recomputed θ grid, both signs, through both seams.
5. Bench-check: DP and simulation benches pass except `simulate_lockstep_10k`
   (37.9 → 39.8 ms A-B, +5%), a code-layout effect on the smallest benchmark:
   the θ = 0 lockstep path contains no exp calls, and the 100k/1M benches run
   the same inner loop with no proportional slowdown. Baseline re-record
   recommended.

## 6. Process trap: precompute silently loads existing tables

`compute_all_state_values` returns cached values from disk when the table
file exists. Any "recompute and compare" experiment that does not delete the
file first is a no-op that confirms whatever is on disk. This masked the bug
during the first control experiment and produced a false "both solvers
agree" result. Always `rm` the target table before a verification recompute.

## 7. Blast radius and regeneration

Wrong before the fix (values ~4 low in L₀, policies slightly suboptimal):

- All |θ| > 0.15 strategy tables computed on Apple Silicon: recomputed
  (±0.16..±0.40 step 0.01, ±0.5, ±0.75, ±1, ±1.5, ±2, ±3).
- All |θ| > 0.15 exact densities and downstream treatise data
  (sweep_summary, kde_curves, tail_exact): regenerated.
- Monte Carlo simulations under |θ| > 0.15 policies
  (data/simulations/theta/): stale until `just sweep` reruns; doc tables
  citing θ = ±0.2, ±1, ±3 statistics (risk-sensitive-strategy.md §2,
  applications-and-appendices.md Appendix C) carry pre-fix numbers.

Unaffected: θ = 0 (EV), oracle.bin, all |θ| ≤ 0.15 tables/densities, all
frontend/profiler data derived from θ = 0.

## 8. Follow-up (same day): hot-path audit suite implemented

A full audit of the hot path was planned (two code-mapping passes + an
adversarial design review) and the test plan implemented. The suite runs as
two manual tiers: `cargo test` (default, every build) and `just audit`
(`--include-ignored` + `YATZY_REQUIRE_DATA=1`, which turns silent data-skips
into failures). 217 tests total; all pass.

New guards, each designed to fail under a systematic bias (not just an
exact-identity break):

- **NaN canary** (tests/test_audit_dp.rs): poison all slots, solve in all
  four modes, require reachable-finite / unreachable-still-NaN. Kills
  reachability, padding, dispatch, and uninitialized-read bugs in one test.
  Empirical byproduct: padding slots for masks where up = 63 is unreachable
  hold garbage BY CONSTRUCTION and are provably never read.
- **θ-boundary continuity + pins**: utility-vs-LSE CE jump at ±0.149/±0.151
  bounded (the shipped bug was a ~26-point jump here); EV(start) pinned to
  248.4400677; CE(θ) monotone over a grid through both seams.
- **Independent f64 mini-solver differential**: a from-definitions solver
  (no CSR, no SIMD, no padding, no fast exp, explicit min(up+scr, 63))
  agrees with production at levels 13-14 in all five mode variants.
- **Keep-table full-content enumeration**: every CSR row's (column,
  probability) content vs exhaustive 6^n enumeration — catches the
  wrong-column class that row-sum checks are blind to. Plus mask round-trips
  and the used_kids child-before-parent lattice invariant.
- **Scoring golden test**: an independent rule-based scorer over all 252×15
  cells + edge pins (the old full-domain check compared the table against
  the function that filled it).
- **Kernel property tests**: all 14 NEON kernels + argmax vs test-local
  scalar references, bitwise, randomized inputs with ±inf seeds and exact
  ties — including the six previously untested kernels (the whole θ < 0
  surface). FMA reference uses mul_add (fused, like vfmaq).
- **ln-MGF adjudicator**: stored L₀ vs exact-PMF ln Σ p·e^{θx} for 8 θ
  spanning both regimes and signs — the ground-truth check that adjudicated
  this bug, automated.
- **Duplicate-copy differential**: engine.rs vs lockstep.rs compute_group6
  bitwise on synthetic inputs (the "intentional duplication" guard).
- **DP determinism**: 1-thread vs 8-thread solves bit-identical.
- **Monotonicity suite** (direction-matrixed): V nondecreasing in upper
  score for EV/max/θ>0, stored-L nonincreasing for θ<0; step ≤ bonus bound.
- **PRNG census**: the 12-bit multiply-high face census pinned exactly
  (683,683,682,683,683,682 of 4096; bias +0.049%/−0.098%) instead of the old
  ±3% statistical band; window layout and roll_die verified by shadow
  extraction.
- **1M-game engine equivalence** (audit tier): mean cross-check tightened to
  ~0.06% sensitivity.

Hardening shipped alongside: `yatzy-precompute --force` + a loud
LOADED-CACHED banner (kills the §6 trap); the storage loader now refuses
wrong-θ and zero-filled tables; `UTILITY_THETA_LIMIT` single-sourced in
constants.rs; CI removed by owner decision (testing is manual, on the Apple
Silicon dev machine — the only place the NEON kernels compile anyway).

**MC data refreshed (same day):** all stored |θ| > 0.15 simulations rerun
against fixed tables (1M games each), plus previously-unstored ±0.5, ±0.75,
±1.5, ±2, ±3. The θ-sweep doc tables were updated. The pattern is
instructive: mid-range policies moved most (θ = −0.20 mean 223.6 → 227.5 —
the buggy risk-averse policy was measurably suboptimal; θ = +0.20
225.3 → 223.8, correctly trading more mean for tail), while every |θ| ≥ 0.5
row came out bit-identical — fully-degenerate policies make decisions by
gaps too large for a ≤19% exp bias to flip. The rerun also caught a false
positive in the new load guard: the CE sanity band was set near-EV
(100, 400), but CE(−3) ≈ 70 is legitimate — the correct band is the physical
score range (5, 380). Fixed.

**Density mass conservation (H4) — resolved.** `KeepTable` now carries an
exact f64 mirror `vals_f64` (the multinomial value was already computed in
f64 at build time and thrown away by the f32 cast; we just keep it). The
density forward-DP reads `vals_f64` at its five mass-propagation sites; the
hot-path solver still reads f32 `vals` and is byte-for-byte unchanged
(verified: L₀ identical after `--force`). Total probability went from
1 + 3.1e-7 back to **1 − 2.5e-12** — exactly the figure
theta-sweep-architecture reported before the regression, so this closes
accuracy-review-2026-07 §6.5: `vals` had been f64, was silently narrowed to
f32, and is now split by consumer. Cost: +34 KB resident, zero solver/perf
impact. The conservation test is re-tightened to 1e-10; decision (argmax)
math in the density stays f32 to reproduce the solver's exact policy.

**Open follow-ups:** (1) Oracle value-based decision audit (T9).
(2) Per-node Jensen sampling (T12; largely subsumed by the mini-solver
differential). (3) The §7 disagreement-rate and per-category analyses at
high θ still cite pre-fix runs (need full-recording reruns, not just
scores.bin).

## See Also

- `neon-intrinsics.md`: the kernel family this bug lived in
- `../foundations/risk-parameter-theta.md`: LSE recurrence and log-domain math
- `../strategy/risk-sensitive-strategy.md`: θ sweep results (§2 tables stale
  for |θ| > 0.15 until MC rerun)
- `../reviews/accuracy-review-2026-07.md`: the review pass that preceded this
