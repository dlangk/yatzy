# Quantitative Facts Registry

Every numerical claim in the treatise, with source file and line reference.

## Conventions
- **Source**: path relative to `theory/`
- **Section**: treatise part where claim appears
- **Verified**: ✓ = cross-referenced against ≥2 sources or code

---

## Part I — The Geometry of a Game

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 1 | Total state slots (upper × categories) | 2,097,152 | foundations/algorithm-and-dp.md | 13 | I | ✓ |
| 2 | Upper score range | 0–63 | foundations/algorithm-and-dp.md | 10 | I | ✓ |
| 3 | Category bitmask width | 15 bits | foundations/algorithm-and-dp.md | 11 | I | ✓ |
| 4 | Reachable states (after pruning) | ~1,430,000 | foundations/algorithm-and-dp.md | 13 | I | ✓ |
| 5 | Pruning fraction | 31.8% | lab-reports/cache-hierarchy-targeting.md | 24 | I | ✓ |
| 6 | Distinct dice outcomes (5d6) | 252 | foundations/algorithm-and-dp.md | 18 | I | ✓ |
| 7 | Unique keep-multisets per outcome | 462 | foundations/algorithm-and-dp.md | 20 | I | ✓ |
| 8 | Non-zero keep entries (deduped) | 4,368 | lab-reports/cache-hierarchy-targeting.md | 27 | I | ✓ |
| 9 | Raw keep masks per dice set | 31 | lab-reports/cache-hierarchy-targeting.md | 26 | I | |
| 10 | Average unique keeps per dice set | 16.3 | lab-reports/cache-hierarchy-targeting.md | 26 | I | |
| 11 | Categories in Scandinavian Yatzy | 15 | foundations/algorithm-and-dp.md | 5 | I | ✓ |
| 12 | Upper bonus threshold | 63 points | foundations/algorithm-and-dp.md | 10 | I | ✓ |
| 13 | Upper bonus value | 50 points | foundations/algorithm-and-dp.md | 10 | I | ✓ |

## Part II — Engineering the Solver

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 14 | STATE_STRIDE (topological padding) | 128 | foundations/algorithm-and-dp.md | 15 | II | ✓ |
| 15 | State array size (f32) | ~16 MB | lab-reports/cache-hierarchy-targeting.md | 49 | II | ✓ |
| 16 | KeepTable working set | 51 KB | lab-reports/cache-hierarchy-targeting.md | 31 | II | |
| 17 | Dense transition table | 16.3 MB (79% zeros) | lab-reports/cache-hierarchy-targeting.md | 39,44 | II | |
| 18 | Sparse CSR format | 5.1 MB | lab-reports/cache-hierarchy-targeting.md | 44 | II | |
| 19 | Original C solver time | ~20 min | lab-reports/optimization-log.md | 9 | II | ✓ |
| 20 | Final optimized time (EV, θ=0) | ~1.10 s | lab-reports/optimization-log.md | 286 | II | ✓ |
| 21 | Total speedup | ~500× | lab-reports/cache-hierarchy-targeting.md | 31 | II | ✓ |
| 22 | Step 1 speedup (upper-section merge) | 5× | lab-reports/optimization-log.md | 14 | II | |
| 23 | Step 7 speedup (Rayon parallel) | 40× | lab-reports/optimization-log.md | 45 | II | |
| 24 | NEON registers | 32 × 128-bit | lab-reports/neon-intrinsics.md | 109 | II | |
| 25 | NEON f32 lanes | 4 per register | lab-reports/neon-intrinsics.md | 110 | II | |
| 26 | NEON FMA throughput (fp32) | 25.8 GFLOPS/core | lab-reports/neon-intrinsics.md | 119 | II | |
| 27 | Firestorm P-core clock | 3.228 GHz | lab-reports/neon-intrinsics.md | 139 | II | |
| 28 | L1d cache per core | 128 KB | lab-reports/cache-hierarchy-targeting.md | 12 | II | ✓ |
| 29 | L2 cache (shared) | 12 MB | lab-reports/cache-hierarchy-targeting.md | 12 | II | |
| 30 | SLC cache | ~48 MB | lab-reports/cache-hierarchy-targeting.md | 12 | II | |
| 31 | Per-widget cycles | ~45,000 | lab-reports/cache-hierarchy-targeting.md | 241 | II | |
| 32 | f32 vs f64 max absolute diff | 0.000458 pts | lab-reports/hardware-and-hot-path.md | 84 | II | |
| 33 | f32 ordering flips | 22 out of 1.43M | lab-reports/hardware-and-hot-path.md | 288 | II | |
| 34 | Mmap load time | <1 ms | foundations/algorithm-and-dp.md | 16 | II | |

## Part III — Luck, Variance, and the Shape of Outcomes

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 35 | EV-optimal mean score | 248.4 | foundations/optimal-play-and-distributions.md | 37 | III | ✓ |
| 36 | Standard deviation | 38.5 | foundations/optimal-play-and-distributions.md | 38 | III | ✓ |
| 37 | Max-policy precomputed EV | 374 | foundations/optimal-play-and-distributions.md | 47 | III | |
| 38 | Max-policy simulated mean | 118.7 | foundations/optimal-play-and-distributions.md | 47 | III | |
| 39 | Bonus hit rate (θ=0) | ~90% | foundations/optimal-play-and-distributions.md | 65 | III | |
| 40 | Yatzy probability (θ=0) | ~39% | foundations/optimal-play-and-distributions.md | 65 | III | |
| 41 | Mixture component 1 (no bonus, no Yatzy) mean | 164 | foundations/optimal-play-and-distributions.md | 74 | III | |
| 42 | Mixture component 2 (no bonus, Yatzy) mean | 213 | foundations/optimal-play-and-distributions.md | 75 | III | |
| 43 | Mixture component 3 (bonus, no Yatzy) mean | 237 | foundations/optimal-play-and-distributions.md | 76 | III | |
| 44 | Mixture component 4 (bonus + Yatzy) mean | 288 | foundations/optimal-play-and-distributions.md | 77 | III | |
| 45 | Bonus hit vs miss gap | +72 pts (50 bonus + 22 correlated) | foundations/optimal-play-and-distributions.md | 93 | III | |
| 46 | Heuristic vs optimal gap | ~82 pts | research/human-cognition-and-compression.md | 11 | III | ✓ |
| 47 | Heuristic mean | 166 | strategy/strategy-guide.md | 3 | III | |
| 48 | Category error loss | 38.4 pts/game | strategy/strategy-guide.md | 3 | III | |
| 49 | Reroll error loss (1st + 2nd) | 44.2 pts/game | strategy/strategy-guide.md | 3 | III | |
| 50 | Heuristic upper bonus rate | 1.2% | strategy/strategy-guide.md | 7 | III | |
| 51 | Solver upper bonus rate | 87% | strategy/strategy-guide.md | 7 | III | |

## Part IV — Thermodynamics of Risk

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 52 | EV solver runtime (θ=0) | ~1.10 s | foundations/risk-parameter-theta.md | 286 | IV | ✓ |
| 53 | Utility domain runtime (|θ|≤0.15) | ~0.49 s | foundations/risk-parameter-theta.md | 286 | IV | |
| 54 | LSE log-domain runtime (|θ|>0.15) | ~2.7 s | foundations/risk-parameter-theta.md | 286 | IV | |
| 55 | f32 safety cutoff | |θ| ≤ 0.15 | strategy/theta-sweep-architecture.md | 72 | IV | ✓ |
| 56 | f32 overflow threshold | θ > 0.235 | strategy/theta-sweep-architecture.md | 66 | IV | |
| 57 | Mantissa erasure threshold | θ > 0.166 | strategy/theta-sweep-architecture.md | 70 | IV | |
| 58 | θ grid values (total) | 37 | strategy/theta-sweep-architecture.md | 215 | IV | ✓ |
| 59 | Dense grid count | 19 θ values | strategy/theta-sweep-architecture.md | 213 | IV | |
| 60 | Sparse grid count | 17 θ values (+ 1 overlap) | strategy/theta-sweep-architecture.md | 213 | IV | |
| 61 | Best θ for p5 | −0.03 → 183 | strategy/risk-sensitive-strategy.md | 35 | IV | |
| 62 | Best θ for p50 | −0.005 → 249 | strategy/risk-sensitive-strategy.md | 36 | IV | |
| 63 | Best θ for p95 | +0.07 → 313 | strategy/risk-sensitive-strategy.md | 38 | IV | |
| 64 | Best θ for p99 | +0.10 → 329 | strategy/risk-sensitive-strategy.md | 39 | IV | |
| 65 | Mean loss quadratic fit | 247.0 − 0.9θ − 618θ² | strategy/theta-sweep-architecture.md | 58 | IV | |
| 66 | Std gain quadratic fit | 39.5 + 53.5θ − 17θ² | strategy/theta-sweep-architecture.md | 59 | IV | |
| 67 | θ=−1.0 mean | 198.2 | strategy/risk-sensitive-strategy.md | 21 | IV | |
| 68 | θ=+1.0 mean | 194.5 | strategy/risk-sensitive-strategy.md | 29 | IV | |

## Part V — Multiplayer and Adaptive Play

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 69 | Head-to-head draw rate (EV vs EV) | 0.76% | strategy/applications-and-appendices.md | 52 | V | |
| 70 | Head-to-head winning margin | 43.5 pts | strategy/applications-and-appendices.md | 52 | V | |
| 71 | Disagreement rate (θ=0.07 vs θ=0) reroll | 12.5% | strategy/risk-sensitive-strategy.md | 125 | V | |
| 72 | Disagreement rate (θ=0.07 vs θ=0) category | 12.4% | strategy/risk-sensitive-strategy.md | 125 | V | |
| 73 | Disagreements per game (θ=0.07) | 5.6 of 45 decisions | strategy/risk-sensitive-strategy.md | 125 | V | |
| 74 | Yatzy preservation at θ≈0.05–0.07 | 42.8% hit rate | strategy/risk-sensitive-strategy.md | 158 | V | |
| 75 | Yatzy preservation at θ=0 | 38.8% hit rate | strategy/risk-sensitive-strategy.md | 158 | V | |
| 76 | Full House hit rate (θ=0) | 92% | strategy/risk-sensitive-strategy.md | 162 | V | |
| 77 | Full House hit rate (θ=3) | 54% | strategy/risk-sensitive-strategy.md | 162 | V | |
| 78 | Bonus achievement rate | 83.6% | strategy/risk-sensitive-strategy.md | 235 | V | |

## Part VI — Compressing Genius

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 79 | Oracle EV | 248.4 | lab-reports/rosetta-distillation.md | 114 | VI | ✓ |
| 80 | Category-only rules EV (100 rules) | 227.5 | lab-reports/rosetta-distillation.md | 93 | VI | ✓ |
| 81 | Full rules (category + reroll bitmask) EV | 168.7 | lab-reports/rosetta-distillation.md | 94 | VI | |
| 82 | DT depth-20 (413K params) | 245 mean | lab-reports/rosetta-distillation.md | 115 | VI | ✓ |
| 83 | DT depth-15 (81K params) | 239 mean | lab-reports/rosetta-distillation.md | 116 | VI | |
| 84 | DT depth-10 (6K params) | 216 mean | lab-reports/rosetta-distillation.md | 119 | VI | |
| 85 | MLP [64,32] (5K params) | 221 mean | lab-reports/rosetta-distillation.md | 118 | VI | ✓ |
| 86 | Human-level score range | 220–230 | research/human-cognition-and-compression.md | 13 | VI | |
| 87 | Training data export | 200K games | research/human-cognition-and-compression.md | 14 | VI | |
| 88 | Export throughput | 22K games/sec | lab-reports/rosetta-distillation.md | 44 | VI | |
| 89 | Best RL score (literature) | ~236 | research/rl-and-ml-approaches.md | 9 | VI | |

## Part VII — The Rosetta Stone

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 90 | Candidate conditions generated | 370 | lab-reports/rosetta-distillation.md | 61 | VII | |
| 91 | Category rules extracted | 100 | lab-reports/rosetta-distillation.md | 82 | VII | |
| 92 | Category uncovered rate | 0.8% | lab-reports/rosetta-distillation.md | 83 | VII | |
| 93 | Reroll 1 uncovered rate | 7.5% | lab-reports/rosetta-distillation.md | 83 | VII | |
| 94 | Reroll 2 uncovered rate | 3.0% | lab-reports/rosetta-distillation.md | 83 | VII | |
| 95 | Semantic reroll actions | 15 | lab-reports/semantic-reroll-actions.md | 13 | VII | |
| 96 | Semantic reroll EV | 184.4 | lab-reports/semantic-reroll-actions.md | 54 | VII | |
| 97 | Bitmask reroll EV | 168.7 | lab-reports/semantic-reroll-actions.md | 55 | VII | |
| 98 | Semantic improvement over bitmask | +15.7 | lab-reports/semantic-reroll-actions.md | 112 | VII | |
| 99 | Scaling: 0 rules EV | 196.7 | lab-reports/scaling-laws.md | 37 | VII | |
| 100 | Scaling: 50 rules EV | 216.5 | lab-reports/scaling-laws.md | 46 | VII | |
| 101 | Scaling: 100 rules EV | 227.5 | lab-reports/scaling-laws.md | 51 | VII | ✓ |
| 102 | Oracle gap closed by 100 rules | 59.7% | lab-reports/scaling-laws.md | 90 | VII | |
| 103 | Incompressible residual | 40.3% (20.9 EV) | lab-reports/scaling-laws.md | 95 | VII | |
| 104 | Peak marginal value (rules 26–40) | 0.74 EV/rule | lab-reports/scaling-laws.md | 114 | VII | |

## Part VIII — Interpretability

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 105 | Cognitive profiling parameters | 4 (θ, β, γ, d) | research/human-cognition-and-compression.md | 11 | VIII | ✓ |
| 106 | Quiz scenarios | 30 | research/human-cognition-and-compression.md | 11 | VIII | |
| 107 | Q-grid combinations | 108 (6θ × 6γ × 3d) | research/human-cognition-and-compression.md | 12 | VIII | |
| 108 | Player card grid | 648 combos × 10K games | strategy/applications-and-appendices.md | 17 | VIII | ✓ |
| 109 | Grid precomputation time | ~2.5 min | strategy/applications-and-appendices.md | 17 | VIII | |

## Performance Benchmarks (Cross-cutting)

| # | Claim | Value | Source | Line | Section | Verified |
|---|-------|-------|--------|------|---------|----------|
| 110 | Oracle precompute overhead | 1.4× (1.54 s) | lab-reports/density-condensation.md | 71 | II | ✓ |
| 111 | Oracle total on-disk | ~3.17 GB | lab-reports/oracle-policy.md | 47 | II | |
| 112 | Oracle MC throughput | 5.6M games/s | lab-reports/density-condensation.md | 80 | II | ✓ |
| 113 | Lockstep MC throughput | 232K games/s | lab-reports/oracle-policy.md | 94 | II | |
| 114 | Density evolution (non-oracle) | ~381 s | lab-reports/optimization-log.md | 287 | II | |
| 115 | Density evolution (oracle) | 3.0 s | lab-reports/density-condensation.md | 113 | II | ✓ |
| 116 | Oracle density speedup | 126× | lab-reports/density-condensation.md | 19 | II | ✓ |
| 117 | American Yahtzee EV (literature) | 254.59 | research/exact-solver-survey.md | 13 | I | |
