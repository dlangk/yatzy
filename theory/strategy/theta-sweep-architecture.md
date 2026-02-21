# Theta Sweep Architecture

This document describes the computational pipeline for sweeping over risk
parameter theta: precomputing strategy tables, generating score distributions,
and the optimizations that make it practical.

---

## Pipeline Overview

A theta sweep answers the question: *how does the score distribution change
as a function of the risk parameter theta?* The pipeline has three stages:

1. **Precompute** a strategy table for each theta (~8 MB per table)
2. **Generate** the score distribution under each strategy
3. **Analyze** the resulting distributions (percentiles, means, Pareto frontier)

Two methods exist for stage 2: Monte Carlo simulation (fast, finite variance)
and exact density evolution (slow, zero variance).

---

## Strategy Table Precomputation

The backward induction DP fills a table of 2,097,152 state values
(64 upper scores x 32,768 scored-category bitmasks). For theta = 0 this stores
expected values; for theta != 0 it stores log-domain exponential utilities
L(S) = ln(E[e^(theta * total) | S]).

### Three Solver Modes

| Mode | Theta range | Stochastic nodes | Decision nodes | Runtime |
|------|-------------|-----------------|----------------|---------|
| EV | theta = 0 | Weighted sum | argmax | ~1.3s |
| Utility domain | 0 < \|theta\| <= 0.15 | Weighted sum | argmax/argmin | ~1.3s |
| Log domain (LSE) | \|theta\| > 0.15 | Log-sum-exp | argmax/argmin | ~7.4s |

Measured on Apple M1 Max, 8 threads. The EV and utility-domain solvers share
identical inner loops for stochastic nodes (plain SpMM weighted sums). The
log-domain solver pays ~5.7x overhead from exp/ln calls at every chance node.

### Why Three Modes?

The log-domain solver is the general-purpose solution for all theta != 0, but
the exp/ln operations at stochastic nodes are expensive. The key insight is that
for small theta, we can work directly in the utility domain U(S) = E[e^(theta *
remaining) | S] without numerical issues:

- Stochastic nodes become plain weighted sums: E[U] = sum(p_i * U_i). This is
  identical to the EV solver -- same SIMD-friendly SpMM inner loop, no
  transcendentals.
- Decision nodes use argmax/argmin on U values directly. Since e^(theta*x) is
  monotonic, comparing U values gives the same decisions as comparing L values.
- Scoring becomes multiplication: U(S) = e^(theta * score) * U(successor),
  using a precomputed ExpScores table.

After the DP completes in utility domain, a single pass converts back to log
domain: L(S) = ln(U(S)). This costs ~3ms for 2M values and keeps all downstream
consumers (simulation, analytics, profiling) unchanged.

### Numerical Safety Limits

The utility domain stores e^(theta * remaining_score). Two failure modes:

1. **f32 overflow** (exponent > 88): Maximum remaining score is ~374 (turn 0).
   e^(theta * 374) overflows f32 when theta > 88/374 = 0.235.

2. **Mantissa erasure** (ratio > 2^24): The max score spread at a single chance
   node is ~100 points (Yatzy 50 + bonus 50). e^(theta * 100) > 2^24 when
   theta > ln(2^24)/100 = 0.166.

The cutoff is set at |theta| <= 0.15, providing margin against both limits. This
covers the entire "interesting" range where risk preferences produce meaningfully
different strategies (|theta| < 0.2, per the regime map in
`risk_parameter_theta.md`).

For |theta| > 0.15, the existing log-domain solver runs at ~7.4s. These extreme
theta values produce degenerate strategies (LSE collapses to max/min), so speed
is unimportant.

---

## Score Distribution: Monte Carlo vs Exact Density

### Monte Carlo Simulation

Simulate N games under the optimal policy for each theta. Store scores as
i16 arrays in binary files (~1.91 MB per million games per theta).

Measured wall-clock times (Apple M1 Max, 8 threads, theta = 0):

| Games | Wall time | Throughput | Storage |
|------:|----------:|-----------:|--------:|
| 10K | 0.2s | 50K games/s | 20 KB |
| 100K | 1.9s | 53K games/s | 200 KB |
| 1M | 19.6s | 51K games/s | 1.91 MB |
| 10M | 3m 20s | 50K games/s | 19.1 MB |

Throughput is linear -- no overhead from batching or I/O at any scale.

### Exact Density Evolution

Forward DP that propagates P(state, accumulated_score) through 15 turns.
Starting from state (0, 0) with probability 1.0, enumerate all possible paths
through each turn:

1. Initial roll: 252 dice sets with known probabilities
2. Optimal keep decision (from strategy table)
3. All reachable dice after reroll
4. Optimal keep decision
5. All reachable dice after second reroll
6. Optimal category assignment: points earned + successor state

Each turn transforms the distribution by computing per-state transitions (all
(successor, points, probability) triples) and propagating score distributions
through them. After 15 turns, apply the upper bonus (50 points if upper score
>= 63) to get the final PMF.

- **Runtime**: ~6 min per theta (8 threads, Apple Silicon)
- **Variance**: Zero. The PMF is mathematically exact.
- **Output**: JSON with full PMF, mean, variance, std_dev, percentiles
- **Verification**: Mean matches sv[state_index(0, 0)] within 0.00015 pts.
  Probability sums to 1.0 within 2.5e-12.
- **Peak memory**: ~480 MB at turns 7-8

---

## How Many Games? A Statistical Analysis

The score distribution has std dev sigma ~ 38.5 points. This determines how
many MC games are needed for various quantities to converge.

### Standard Error of the Mean

```
SE(mean) = sigma / sqrt(N)
```

| Games N | SE(mean) | 95% CI width |
|--------:|---------:|-------------:|
| 10K | 0.39 pts | +/- 0.76 pts |
| 100K | 0.12 pts | +/- 0.24 pts |
| 1M | 0.039 pts | +/- 0.076 pts |
| 10M | 0.012 pts | +/- 0.024 pts |

For mean estimation, **100K games** gives sub-0.25-point precision. More than
sufficient given that the theta grid step causes ~1-2 point mean differences.

### Percentile Estimation

Tail percentiles are harder. The standard error of the p-th quantile from
order statistics is approximately:

```
SE(Q_p) ~ sigma * sqrt(p * (1-p) / N) / phi(z_p)
```

where phi is the normal PDF and z_p is the normal quantile. For the tails:

| Percentile | phi(z_p) | SE at 100K | SE at 1M | SE at 10M |
|-----------:|---------:|-----------:|---------:|----------:|
| p1 | 0.027 | 4.2 pts | 1.3 pts | 0.42 pts |
| p5 | 0.103 | 2.6 pts | 0.81 pts | 0.26 pts |
| p10 | 0.176 | 1.9 pts | 0.59 pts | 0.19 pts |
| p25 | 0.318 | 1.3 pts | 0.42 pts | 0.13 pts |
| p50 | 0.399 | 1.2 pts | 0.38 pts | 0.12 pts |
| p75 | 0.318 | 1.3 pts | 0.42 pts | 0.13 pts |
| p90 | 0.176 | 1.9 pts | 0.59 pts | 0.19 pts |
| p95 | 0.103 | 2.6 pts | 0.81 pts | 0.26 pts |
| p99 | 0.027 | 4.2 pts | 1.3 pts | 0.42 pts |

Key takeaways:

- **p50** (median): 100K games gives +-1.2 pts. Adequate.
- **p5/p95**: 1M games gives +-0.8 pts. Good for identifying optimal theta.
- **p1/p99**: Even 10M games only gives +-0.4 pts. For sub-point precision
  at the tails, MC is fundamentally limited.

This is exactly why density evolution exists: it produces exact percentiles
with zero variance, regardless of how deep in the tail.

### Practical Recommendations

| Use case | Games | Time (37 thetas) | Rationale |
|----------|------:|------------------:|-----------|
| Quick exploration | 10K | 7s + precompute | SE(mean) = 0.4 pts, enough to see trends |
| Standard sweep | 100K | 70s + precompute | SE(mean) = 0.12 pts, p50 +-1.2 pts |
| Publication means | 1M | 12 min + precompute | SE(mean) = 0.04 pts, p5/p95 +-0.8 pts |
| MC tail analysis | 10M | 2h + precompute | p1/p99 +-0.4 pts; still noisy |
| Exact distributions | density | 3.7h + precompute | Zero variance everywhere |

**The sweet spot is 1M games for MC sweep + density evolution for key thetas.**
1M games resolves means to 0.04 pts and percentiles to ~1 pt, which is enough
to identify the optimal theta for each metric. Density evolution then provides
exact values for the thetas that matter.

The current sweep uses 1M games per theta, which is well-calibrated: the SE on
p5/p95 (~0.8 pts) is smaller than the effect of stepping theta by 0.01 in the
dense region (~1-3 pts change in percentiles).

---

## Sweep Grid Design

The theta grid uses non-uniform spacing that reflects the structure of the
risk-sensitive solver. The canonical grid (37 values from -3.0 to +3.0) is
defined in `configs/theta_grid.toml`.

### Three Named Grids

| Grid | Thetas | Coverage | Purpose |
|------|-------:|----------|---------|
| `dense` | 19 | [-0.1, +0.1], step 0.005-0.03 | Useful range only |
| `sparse` | 17 | [-3.0, +3.0], step 0.05-1.0 | Broad coverage |
| `all` | 37 | [-3.0, +3.0], dense center | Complete picture |

### Why Non-Uniform Spacing?

The product |theta| * sigma_EV is a dimensionless control parameter that
determines the solver's regime (see `risk_parameter_theta.md`). With
sigma_EV ~ 10 at a typical chance node:

- **|theta| < 0.03** (|theta| * sigma < 0.3): Indistinguishable from EV.
  Policy barely changes. Dense sampling here is wasteful.
- **0.03 < |theta| < 0.1** (0.3 < |theta| * sigma < 1.0): The Pareto frontier
  bends sharply. Moving theta by 0.01 produces measurably different percentiles.
  Dense sampling is essential.
- **0.1 < |theta| < 0.5** (1 < |theta| * sigma < 5): Diminishing returns.
  Strategies are strongly biased but changes are gradual.
- **|theta| >= 1** (|theta| * sigma >= 10): Fully degenerate. LSE has collapsed
  to pure max/min. A few samples confirm the plateau.

### Grid + Solver Mode Alignment

All 19 dense-grid thetas fall within the utility-domain solver range
(|theta| <= 0.15), so the dense region precomputes at full speed (~1.3s each).
The log-domain solver (~7.4s) only runs for the 18 sparse/degenerate thetas
where speed doesn't matter.

| Grid | Utility-domain thetas | Log-domain thetas | Precompute time |
|------|---------:|-------:|------:|
| dense (19) | 19 | 0 | 25s |
| sparse (17) | 7 | 10 | 83s |
| all (37) | 19 | 18 | 2.5 min |

---

## End-to-End Runtime

### All 37 thetas, by game count

| Stage | 10K games | 100K games | 1M games | 10M games | Density |
|-------|----------:|-----------:|---------:|----------:|--------:|
| Precompute | 2.5 min | 2.5 min | 2.5 min | 2.5 min | 2.5 min |
| Simulate/evolve | 7s | 70s | 12 min | 2.1 h | 3.7 h |
| **Total** | **2.6 min** | **3.7 min** | **15 min** | **2.3 h** | **4.2 h** |
| SE(mean) | 0.39 pts | 0.12 pts | 0.039 pts | 0.012 pts | 0 |
| SE(p5) | 2.6 pts | 0.81 pts | 0.26 pts | 0.081 pts | 0 |
| SE(p1) | 4.2 pts | 1.3 pts | 0.42 pts | 0.13 pts | 0 |
| Storage/theta | 20 KB | 200 KB | 1.91 MB | 19.1 MB | ~50 KB JSON |
| Total storage | 740 KB | 7.2 MB | 69 MB | 690 MB | ~1.8 MB |

### Dense grid only (19 thetas)

| Stage | 10K games | 100K games | 1M games | 10M games | Density |
|-------|----------:|-----------:|---------:|----------:|--------:|
| Precompute | 25s | 25s | 25s | 25s | 25s |
| Simulate/evolve | 4s | 36s | 6.2 min | 1.1 h | 1.9 h |
| **Total** | **29s** | **61s** | **6.6 min** | **1.1 h** | **1.9 h** |

### Decision framework

Precomputation dominates at low game counts. At 10K games, simulation is 3%
of total time -- there is no reason to use fewer games than 100K.

At 1M games, simulation time overtakes precomputation. Going to 10M makes
sense only if you need sub-point tail percentiles and don't want to run density
evolution.

Density evolution is 18x slower than 1M MC but produces exact results. It is
most valuable for:
- Resolving theta-optimal percentiles to exact integers
- Providing ground truth to validate MC confidence intervals
- Computing the full PMF shape (multimodality, skewness, exact support)

---

## Plan vs Reality: Comparing Predictions to Measurements

The original plan (authored before implementation) made specific quantitative
predictions. Here we compare each to the measured outcome.

### Utility-Domain Solver

| Metric | Plan prediction | Measured | Assessment |
|--------|----------------|----------|------------|
| Runtime (utility) | ~1.4s | 1.3s | Accurate. Plan based on θ=0 EV solver baseline. |
| Runtime (log domain) | ~7s | 7.4s | Accurate. |
| Speedup | 5x | 5.7x | Slightly better than predicted. |
| f32 safety cutoff | \|θ\| ≤ 0.15 | \|θ\| ≤ 0.15 | Exact match. Derived analytically from mantissa erasure at e^(θ×100) > 2^24. |
| Max diff vs log-domain | ~0.001 pts | 5.3e-6 pts | 190x better than predicted. The plan used a conservative bound; actual cancellation errors are much smaller because the DP averages over many stochastic nodes. |
| Code reuse (Groups 5/3/1) | "IDENTICAL to EV solver, reuse!" | Groups 5/3 needed a separate function (min/max decision logic differs) but stochastic inner loop is identical. Group 1 fully reused. | Mostly correct. The plan slightly underestimated the surface area of the min/max branching in decision nodes. |

**Verdict**: The utility-domain analysis was almost exactly right. The numerical
limits were derived from first principles (f32 mantissa width, max score spread
per chance node) and held precisely. The runtime prediction was within 10%.

The one surprise was accuracy: the plan predicted ~0.001 pt max diff based on
the f32 epsilon (~6e-8) times the score range (~374), but the actual DP
accumulates errors through averaging, not multiplication, so errors partially
cancel rather than compound. The 5.3e-6 max diff means the utility-domain solver
is effectively lossless.

### Density Evolution Runtime

| Metric | Plan prediction | Measured | Assessment |
|--------|----------------|----------|------------|
| Per-state transition | 1-5 ms | Not individually measured, but consistent with total | Plausible. |
| Total active states | ~750K across 15 turns | ~750K (estimated from turn logs) | Accurate. |
| Sequential time | ~37 min | Not measured (always ran parallel) | — |
| Parallel time (8 threads) | 4-5 min/theta | **6.1 min/theta** | **22-50% slower than predicted.** |
| Full grid (37 thetas) | ~3 hours | ~3.7 hours (extrapolated) | Consistent with per-theta overestimate. |
| Useful range (19 thetas) | ~1.5 hours | ~1.9 hours (extrapolated) | Same factor. |

**Why 6 min instead of 4-5?** The plan estimated transitions at 1-5 ms per state
and divided by 8 threads. The likely sources of the gap:

1. **Sequential merge bottleneck.** Transition computation is parallel (rayon),
   but propagating score distributions through transitions is sequential (HashMap
   merging). This is the Amdahl's law penalty — the plan assumed fully parallel
   execution.

2. **Score bin count underestimated.** The plan assumed 50-100 score bins per
   state. At mid-game turns (7-8), states accumulate 100-200+ bins as the score
   distribution spreads. More bins means more work in the sequential merge phase.

3. **HashMap overhead.** The plan's throughput estimate was based on the
   transition computation alone. The HashMap insertions and lookups during
   distribution propagation add constant-factor overhead not accounted for.

The 6 min measured time is 1.2-1.5x the plan's prediction — a reasonable miss
for a back-of-envelope estimate of a complex pipeline with sequential phases.

### Memory

| Metric | Plan prediction | Measured | Assessment |
|--------|----------------|----------|------------|
| Peak active states | ~300K at turns 7-8 | ~100K states, ~5-6M score bins | Different decomposition but same total. |
| Peak memory | ~480 MB | ~480 MB | Accurate (estimated from process monitoring). |

The plan predicted 300K states × 100 bins × 16B = 480 MB. Reality had fewer
states but more bins per state, arriving at the same total. The plan got the
right answer for slightly wrong reasons — a common pattern in capacity planning.

### Verification Criteria

| Criterion | Plan target | Measured | Status |
|-----------|------------|----------|--------|
| Utility solver max diff | < 0.001 pts | 5.3e-6 pts | **PASS** (190x margin) |
| Density mean vs EV table | 245.87 ± 0.01 | 248.44 (diff = 0.00015 from sv) | **PASS** (plan had wrong EV value but correct test logic) |
| Probability conservation | Σ = 1.0 in f64 | 1.0 - 2.5e-12 | **PASS** |
| MC comparison (θ=0.05) | Exact pcts within MC CIs | Not yet run | Pending |

The plan cited sv[state_index(0,0)] = 245.87, which was incorrect — the actual
value is 248.44. This appears to have been a stale number from before the
Scandinavian Yatzy scoring rules were finalized (50-point bonus instead of 35
changes the game-start EV by ~2.5 points). The verification logic was correct:
compare density mean against the actual table value, not a hardcoded constant.

### Overall Assessment

The plan was remarkably well-calibrated:

- **Utility-domain solver**: All predictions confirmed. Runtime, numerical
  limits, and code structure were exactly as designed. The only deviation
  (accuracy 190x better) was a pleasant surprise.

- **Density evolution**: Runtime was 22-50% slower than predicted due to
  Amdahl's law on the sequential merge phase — a systematic bias in the plan's
  throughput model, which assumed full parallelism. Memory prediction was
  spot-on despite using a different state/bin decomposition.

- **Architecture decisions**: All four triage decisions were correct. The
  utility-domain solver delivered the predicted 5x speedup for the interesting
  theta range. Density evolution produces exact results. Skipping the Policy
  Oracle and Lockstep MC was justified — decision tracking works as function
  returns (no 1GB file needed) and density evolution supersedes MC for sweep
  statistics.

The main lesson: **Amdahl's law matters for mixed parallel/sequential pipelines.**
The plan modeled density evolution as embarrassingly parallel (divide total work
by thread count), but the sequential HashMap merge phase accounts for ~20-30% of
wall time. A chunked or sharded merge strategy could close this gap.

---

## Implementation

### Files

| File | Purpose |
|------|---------|
| `solver/src/batched_solver.rs` | `solve_widget_batched_utility`, `precompute_exp_scores`, `ExpScores` type |
| `solver/src/state_computation.rs` | Dispatch logic (EV / utility / LSE), U-to-L conversion |
| `solver/src/phase0_tables.rs` | Terminal state initialization for utility domain |
| `solver/src/density/transitions.rs` | Per-state transition computation with decision tracking |
| `solver/src/density/forward.rs` | 15-turn forward density evolution |
| `solver/src/bin/density_sweep.rs` | CLI entry point for exact density computation |

### CLI

```bash
# Precompute (auto-selects utility or log domain)
YATZY_BASE_PATH=. solver/target/release/yatzy-precompute --theta 0.05

# Monte Carlo sweep (resumable)
YATZY_BASE_PATH=. solver/target/release/yatzy-sweep --grid all --games 1000000

# Exact density evolution
YATZY_BASE_PATH=. solver/target/release/yatzy-density --theta 0
YATZY_BASE_PATH=. solver/target/release/yatzy-density --grid dense
YATZY_BASE_PATH=. solver/target/release/yatzy-density --thetas 0,0.05,0.1

# Justfile shortcuts
just sweep                    # MC sweep, all 37 thetas
just density --theta 0        # Single exact PMF
just density --grid dense     # All 19 dense-grid thetas
```

### Output Formats

**Monte Carlo**: Binary `scores.bin` (32-byte header + i16[N]) in
`data/simulations/theta/theta_{value}/`.

**Density evolution**: JSON in `outputs/density/`:
```json
{
  "theta": 0.0,
  "mean": 248.439989,
  "variance": 1489.234,
  "std_dev": 38.59,
  "percentiles": {"p1": 148, "p5": 183, "p10": 195, ...},
  "pmf": [[10, 0.000000000000123], [11, 0.000000000000456], ...]
}
```
