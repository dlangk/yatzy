# Contradictions and Discrepancies

Cross-referencing all quantitative claims in `theory/`. Most are consistent; discrepancies documented here with resolution.

---

## 1. Precompute Time: Multiple Values Reported

| Source | Value | Context |
|--------|-------|---------|
| `foundations/algorithm-and-dp.md` L177 | ~2.3 s (θ=0), ~7 s (θ≠0) | Pre-NEON baseline |
| `lab-reports/optimization-log.md` L286 | 1.04 s baseline EV | Post-NEON, pre-oracle |
| `foundations/risk-parameter-theta.md` L286 | ~1.1 s EV | Post-NEON (MEMORY.md canonical) |
| `lab-reports/neon-intrinsics.md` | ~0.48 s (utility domain) | Post-NEON, |θ|≤0.15 |

**Resolution**: Not contradictory — these are measurements at different optimization stages. The measured progression per the lab reports is:
- Original C: ~20 min → Step 7 (Rayon): ~2.3 s → batched SoA: ~1.40 s → pre-NEON baseline: ~1.30 s → NEON: ~1.10 s end-to-end EV precompute (0.48 s DP compute) → utility-domain θ-class: ~0.49 s

**Canonical values** (M1 Max, post all optimizations; M5 Max 2026-07 values in `solver/CLAUDE.md`):
- EV (θ=0): **1.10 s** (M5 Max: 113 ms DP compute)
- Utility (|θ|≤0.15): **~0.49 s** (M5 Max: ~0.12 s)
- LSE (|θ|>0.15): **~2.7 s** (M5 Max: ~0.60 s)

---

## 2. Oracle MC Throughput: 1.57M vs 5.6M games/s

| Source | Value |
|--------|-------|
| `lab-reports/oracle-policy.md` L95 | 1,571,817 games/s |
| `lab-reports/density-condensation.md` L80 | 5,599,845 games/s |

**Resolution**: Different measurement points. The oracle-policy report was written before the NEON argmax optimization (6.5× speedup on oracle precompute). The density-condensation report includes the final NEON pipeline. Both are correct for their respective code versions.

**Canonical value**: **5.6M games/s** (final optimized)

---

## 3. Oracle Precompute Overhead: 3.77× vs 1.4×

| Source | Value |
|--------|-------|
| `lab-reports/oracle-policy.md` L82 | 3.77× (3.92 s vs 1.04 s) |
| `lab-reports/density-condensation.md` L71 | 1.4× (1.54 s vs 1.10 s) |

**Resolution**: Pre-optimization vs post-optimization. The NEON argmax kernel reduced oracle-build overhead from 3.77× to 1.4×. Both values are historically accurate.

**Canonical value**: **1.4×** (final optimized)

---

## 4. Density Evolution Full Sweep Time: 4.2 h vs 3.7 h

| Source | Value |
|--------|-------|
| `lab-reports/optimization-log.md` L253 | 4.2 h (37 thetas) |
| `strategy/theta-sweep-architecture.md` L255 | 3.7 h (37 thetas) |

**Resolution** (corrected 2026-07): not different hardware runs. Both numbers
sit in the SAME table in theta-sweep-architecture.md: 3.7 h is the
simulate/evolve stage row, 4.2 h is the total row of the same run.
optimization-log.md quotes the total.

**Status**: Not a discrepancy. Use **4.2 h total** (3.7 h of it density
evolution) for the 37-θ sweep on M1 Max.

---

## 5. Bonus Hit Rate: 87% vs 90% vs 83.6%

| Source | Value | Context |
|--------|-------|---------|
| `strategy/strategy-guide.md` L7 | 87% | "Solver upper bonus rate" |
| `foundations/optimal-play-and-distributions.md` L65 | ~90% | "Bonus hit rate (θ=0)" |
| `strategy/risk-sensitive-strategy.md` L235 | 83.6% | "Bonus achievement rate" |

**Resolution** (settled 2026-07 with fresh data): the θ=0 optimal-policy
end-of-game bonus rate is **89.77% ± 0.04%** (1M-game MC, corroborated by the
exact mixture decomposition 55.1% + 34.7% = 89.8%). So:
- **~90% is correct** (canonical value).
- **87% is wrong** (≈30 standard errors from the measurement; it cannot be
  sampling noise and cannot round to ~90%). It stems from an older or flawed
  run and has been corrected where it appeared.
- **83.6% is a different statistic**, not a contradiction: it is the fraction
  of games that have already secured the bonus when *entering* the final turn
  (start-of-turn-15 state; exact forward-DP gives 0.8378). The remaining
  ~6 points of games secure the bonus on the last fill.

**Status**: Resolved. Canonical: **89.8%** end-of-game, **83.8%** entering
the final turn.

---

## 6. Solver EV: 248.3 vs 248.4

| Source | Value |
|--------|-------|
| Most sources | 248.4 |
| `strategy/applications-and-appendices.md` L43 (also L140, L143) | 248.3 |
| `lab-reports/hardware-and-hot-path.md` L86 | f64: 248.439987, f32: 248.440140 |

**Resolution** (corrected 2026-07): the 248.3 occurrences in applications-and-appendices.md are OBSERVED Monte Carlo means from finite runs (a 10K-game profiling run, SE ~0.39, and 200K-game sweep rows at theta = -0.01/+0.005), not statements of the exact EV. They are within sampling error of the exact 248.4400677 and need no correction; only prose that states the canonical EV must say 248.4.

**Canonical value**: **248.4** (rounded to 1 decimal)

---

## 7. Total Speedup: 500× vs 522×

| Source | Value |
|--------|-------|
| `lab-reports/cache-hierarchy-targeting.md` L31 | ~500× |
| `lab-reports/optimization-log.md` L74 | 522× |

**Resolution**: 500× is the rounded headline figure; 522× is the precise calculation (20 min / 2.3 s). Both are correct.

---

## Summary

Revised 2026-07 after a full adversarial re-audit (see
`accuracy-review-2026-07.md`). The original bottom line of "0 genuine
contradictions" was overstated:

- The bonus rate contained **one genuine contradiction in a canonical value**
  (87% was simply wrong; ~90% correct; 83.6% a different, correct statistic).
- The 248.3 occurrences turned out to be legitimate finite-sample Monte Carlo
  observations, not typos (section 6), but the original "rounding convention"
  explanation was still wrong.
- Section 4's "different hardware runs" explanation was wrong: 3.7 h vs 4.2 h
  are the stage and total rows of the SAME run's table.
- Section 1's optimization chronology skipped stages; the measured progression
  is 2.3 s (Rayon) → 1.40 s (batched SoA) → 1.30 s (pre-NEON) → ~1.10 s
  end-to-end EV (0.48 s compute) → 0.49 s utility-domain.

The registry-scale re-audit found ~200 confirmed issues across theory/ and
treatise/ (mostly stale line references and drifted numbers); the corrected
state is recorded in `accuracy-review-2026-07.md`.
