# Code Smells

## Files Over Thresholds

### Rust (threshold: 500 LOC)

| File | LOC | Concern |
|------|-----|---------|
| `profiling/scenarios.rs` | 1,640 | 4 massive functions doing candidate pool, quiz assembly, Q-grid |
| `rosetta/policy.rs` | 1,387 | Monolithic policy distillation |
| `bin/decision_gaps.rs` | 1,256 | Large analysis binary |
| `bin/decision_sensitivity.rs` | 1,214 | Large analysis binary |
| `simulation/engine.rs` | 1,049 | Sequential sim + recording in one file |
| `bin/difficult_scenarios.rs` | 1,016 | Large analysis binary |
| `simulation/adaptive.rs` | 1,006 | Noisy policies |
| `widget_solver.rs` | 1,001 | 16 public functions (core hot path) |
| `batched_solver.rs` | 908 | 7 solver variants |
| `bin/scenarios.rs` | 850 | Scenario pipeline |
| `bin/pivotal_scenarios.rs` | 825 | Critical decisions |
| `bin/scenario_sensitivity.rs` | 797 | Multi-θ eval |
| `simulation/multiplayer.rs` | 758 | 2-player sim |
| `simulation/strategy.rs` | 743 | Adaptive multi-θ |
| `scenarios/select.rs` | 685 | Difficulty selection |
| `bin/export_training_data.rs` | 641 | Data export |
| `simd.rs` | 597 | 14 NEON kernels (justified by nature) |
| `bin/winrate.rs` | 585 | Head-to-head |
| `phase0_tables.rs` | 564 | Lookup tables |
| `bin/regret_export.rs` | 552 | Per-decision regret |
| `bin/heuristic_gap.rs` | 552 | Gap analysis |
| `simulation/heuristic.rs` | 544 | Heuristic baselines |
| `simulation/statistics.rs` | 522 | Aggregate stats |
| `bin/generate_profile_scenarios.rs` | 531 | Quiz generation |
| `density/transitions.rs` | 514 | Per-state transitions |
| `constants.rs` | 510 | Constants + enums |
| `simulation/lockstep.rs` | 499 | Horizontal SIMD |
| `game_mechanics.rs` | 490 | Scoring rules |
| `bin/frontier_test.rs` | 478 | Frontier test |
| `rosetta/dsl.rs` | 478 | Rule DSL |
| `bin/percentile_sweep.rs` | 493 | Percentile sweep |
| `scenarios/enrich.rs` | 458 | Enrichment |
| `density/forward.rs` | 450 | Forward DP |

**28+ Rust files over 500 LOC.** Many binary targets have large monolithic `main()` functions.

### Frontend / Blog (threshold: 300 LOC)

| File | LOC | Concern |
|------|-----|---------|
| `reducer.test.ts` | 384 | Large test file (acceptable) |
| `reducer.ts` | 355 | Largest non-test file |
| `blog/js/profile/components/scenario-card.js` | 361 | Complex quiz component |
| `blog/js/profile/estimator.js` | 347 | MLE algorithm |
| `blog/js/profile/components/parameter-chart.js` | 308 | D3 convergence chart |

### Analytics (threshold: 300 LOC)

| File | LOC | Concern |
|------|-----|---------|
| `cli.py` | 1,501 | God file: 40+ commands in one file |
| `surrogate.py` | 927 | Training + diagnosis + scaling in one file |
| `skill_ladder.py` | 822 | Rule induction |
| `surrogate_eval.py` | 808 | Full-game simulation |
| `theta_estimator.py` | 697 | MLE algorithm |
| `plots/categories.py` | 604 | Per-category plots |
| `plots/percentile_sweep.py` | 605 | Percentile plots |
| `plots/surrogate.py` | 552 | Surrogate plots |
| `plots/state_flow.py` | 523 | State flow plots |
| `plots/scenario_cards.py` | 510 | Scenario rendering |
| `plots/density.py` | 435 | Density plots |
| `plots/percentile_frontiers.py` | 411 | Frontier plots |
| `plots/modality.py` | 361 | Modality plots |
| `plots/efficiency.py` | 341 | Efficiency plots |
| `plots/difficulty_cards.py` | 335 | Difficulty cards |
| `theta_validation.py` | 326 | Validation |
| `plots/yatzy_hypothesis.py` | 327 | Hypothesis tests |
| `plots/sensitivity.py` | 308 | Sensitivity plots |
| `density.py` | 305 | Forward-DP (Python) |
| `compute.py` | 305 | Statistical kernel |

**20 Python files over 300 LOC.**

## Functions Over 50 Lines

### Rust — Extremely Large Functions

| File | Function | Lines | Issue |
|------|----------|-------|-------|
| `profiling/scenarios.rs` | `compute_q_grid()` | ~1,561 | Should be decomposed into phases |
| `profiling/scenarios.rs` | `assemble_quiz()` | ~1,065 | Selection + diversity + scoring all in one |
| `profiling/scenarios.rs` | `build_master_pool()` | ~911 | Pool + classification + clustering |
| `api_computations.rs` | `compute_roll_response()` | ~212 | Long but fairly linear pipeline |
| `phase0_tables.rs` | `precompute_lookup_tables()` | ~203 | Orchestrator (calls 8 functions) |
| `game_mechanics.rs` | `update_upper_score()` | ~165 | Many match arms |
| `simulation/engine.rs` | `simulate_game_with_recording()` | ~150 | Recording adds boilerplate |
| Most `bin/*.rs` | `main()` | 300-1200 | CLI parsing + orchestration + output |

## Duplicated Logic

### Critical: Frontend ↔ Blog Game (identical reimplementation)

The blog game (`blog/js/game/`) is a complete vanilla JS reimplementation of the React frontend:

| Module | Frontend | Blog | Overlap |
|--------|----------|------|---------|
| Reducer | `reducer.ts` (355 LOC) | `reducer.js` (290 LOC) | ~90% |
| API | `api.ts` (64 LOC) | `api.js` (~50 LOC) | ~100% |
| Mask translation | `mask.ts` (67 LOC) | `mask.js` (~60 LOC) | ~100% |
| Constants | `constants.ts` (27 LOC) | `constants.js` (~25 LOC) | ~100% |
| Components | 9 React components | 8 vanilla JS components | Parallel UI |

**Total duplicated game logic: ~500 LOC** maintained in two places.

### Constants Duplication (3 codebases)

Game constants (CATEGORY_COUNT=15, BONUS_THRESHOLD=63, BONUS_SCORE=50, category names) are defined independently in:
1. `solver/src/constants.rs`
2. `frontend/src/constants.ts`
3. `blog/js/game/constants.js`

No shared source of truth.

### Binary Format Specs (Rust ↔ Python)

Binary file format specs (magic numbers, header sizes, record layouts) are defined independently in:
1. `solver/src/storage.rs` + `simulation/raw_storage.rs`
2. `analytics/src/yatzy_analysis/io.py` + `config.py`

## Dead Code

### Potentially Unused Modules

| File | Concern |
|------|---------|
| `simulation/strategy.rs` (743 LOC) | Adaptive multi-θ — used by which binary? |
| `analytics/adaptive.py` (91 LOC) | Small analysis script, may be superseded |
| `analytics/density.py` (305 LOC) | Python reimplementation of Rust density — redundant? |

### Unused Imports / Dead Branches

Not analyzed in depth (requires compilation analysis / `cargo clippy` / `ruff`).

## Inconsistent Patterns

### State Management
| Component | Pattern |
|-----------|---------|
| Frontend | `useReducer` (React hooks) |
| Blog game | Custom Flux store (`store.js`) |
| Blog quiz | Custom Flux store (`store.js`) |

Three different state management approaches for similar functionality.

### Visualization Technology
| Component | Rendering |
|-----------|-----------|
| Frontend TrajectoryChart | HTML5 Canvas (hand-written) |
| Blog article charts | SVG via D3.js v7 |
| Blog quiz parameter chart | SVG via D3.js v7 |

Two different rendering approaches for visualizations.

### CLI Architecture (analytics)
`cli.py` is a 1,501-line monolith with 40+ Click commands. Should be split into command groups (e.g., `cli/compute.py`, `cli/plot.py`, `cli/surrogate.py`, etc.).

### Binary Target Bloat (Rust)
28 binary targets, many with 500-1200 line `main()` functions that duplicate:
- CLI argument parsing patterns
- Context initialization (load tables, oracle, etc.)
- Output formatting and file writing

A shared CLI framework or command pattern would reduce this.

## Unnecessary Dependencies

### Analytics
| Package | Usage | Concern |
|---------|-------|---------|
| `pandas` | Minimal — mostly `.to_pandas()` boundary with seaborn | Could use polars throughout |
| `seaborn` | Thin layer over matplotlib | Could use matplotlib directly |
| `plotly` | Only for 1 interactive 3D density plot | Heavy dependency for single use |
| `torch` | Optional surrogate training | Only needed for MLP surrogates |

### Frontend
- No unnecessary dependencies detected. The frontend is intentionally lean (React + Vite only).

### Blog
- D3.js loaded via CDN (not a dependency concern)
- `vitest` in blog `package.json` for testing mask/reducer logic

## Scripts Bloat

| Script | LOC | Concern |
|--------|-----|---------|
| `scripts/generate_strategy_guide.py` | 11,395 | Extremely large for a script |
| `scripts/scaling_sweep.py` | 3,778 | Could be a CLI command |
| `scripts/cfg_sweep.py` | 2,471 | Could be a CLI command |

These scripts total ~17,600 LOC and are not integrated into the analytics package.

## Summary of Top Concerns

1. **Frontend/Blog duplication** (~500 LOC of game logic maintained twice)
2. **cli.py monolith** (1,501 LOC, 40+ commands in one file)
3. **profiling/scenarios.rs** (1,640 LOC, 4 functions each 900+ lines)
4. **28 binary targets** with duplicated init/CLI patterns
5. **Constants duplication** across 3 codebases
6. **Binary format specs** duplicated between Rust and Python
7. **Scripts/** not integrated into analytics package
