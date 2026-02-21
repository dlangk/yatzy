# Architecture

## System Overview

Delta Yatzy is an optimal-play Scandinavian Yatzy system with four components:

1. **Rust Solver** — backward-induction DP engine, REST API, simulation, density evolution
2. **React Frontend** — game-playing UI with real-time optimal-action hints
3. **Python Analytics** — statistical analysis, visualization, risk-sweep pipelines
4. **Blog** — static site with D3 article charts, profiling quiz, embedded game

All game intelligence lives in the solver. The frontend and blog are thin clients that either call the API at runtime or consume pre-computed JSON. Analytics reads binary simulation files from disk.

## End-to-End Data Flow

```
Phase 1: Precomputation (offline, ~1s per θ)
─────────────────────────────────────────────
  Backward induction DP
  ├── Enumerate 2,097,152 game states (upper_score × scored_categories)
  ├── For each state: compute E[final_score | optimal play]
  ├── Store: data/strategy_tables/all_states_theta_*.bin (16 MB each)
  └── Optional: build PolicyOracle (3.17 GB flat Vec<u8>)

Phase 2: Simulation (offline, ~1M games/θ)
──────────────────────────────────────────
  Monte Carlo with optimal policy
  ├── Sequential (52K g/s), lockstep (232K g/s), or oracle (5.6M g/s)
  ├── Store: data/simulations/theta/theta_*/scores.bin (~2 MB each)
  └── Optional: full recording (simulation_raw.bin, 289 B/game)

Phase 3: Analysis (offline, Python)
────────────────────────────────────
  yatzy-analyze compute
  ├── Read scores.bin (numpy vectorized)
  ├── Compute: KDE, percentiles, CVaR, MER, SDVA, entropy
  ├── Store: outputs/aggregates/parquet/{summary,kde,mer,sdva}.parquet
  └── Export: outputs/aggregates/csv/sweep_summary.csv

  yatzy-analyze plot
  ├── Read parquet aggregates
  ├── Generate: 50+ PNG/SVG/HTML visualizations
  └── Store: outputs/plots/

Phase 4: Serving (runtime)
──────────────────────────
  yatzy (axum server, port 9000)
  ├── Mmap-loads all_states.bin (<1ms, zero-copy)
  ├── Serves: /evaluate, /state_value, /density, /health
  ├── Stateless lookups + on-demand density computation
  └── CORS: all origins

Phase 5: Client interaction
───────────────────────────
  Frontend (React, port 5173) ←→ Solver API
  Blog game (Vanilla JS)      ←→ Solver API
  Blog quiz (Vanilla JS)      ←  Pre-computed JSON (no API at runtime)
  Blog article (D3 charts)    ←  Pre-computed JSON (no API at runtime)
```

## Rust Backend Architecture

### Computation Pipeline

```
Phase 0: Lookup Tables (~0.3s)
├── 252 sorted 5-dice multisets (R_{5,6})
├── 15×252 category scores s(r,c)
├── Sparse CSR KeepTable (462 keeps, 4,368 non-zero)
├── 252 dice-set probabilities P(⊥→r)
├── Reachability mask (prunes ~31.8% of states)
└── Scored-category popcount cache

Phase 2: Backward Induction (rayon parallel, ~1.1s for θ=0)
├── For num_scored = 14 → 0:
│   ├── Collect scored_masks with popcount == num_scored
│   ├── scored_masks.par_iter() → one rayon task per mask
│   └── Per mask: batched SpMM over 64 upper scores
│       ├── Group 6: best category     e[0][ds] = max_c [s(r,c) + E(succ)]
│       ├── Group 5: best keep (1 rr)  e[1][ds] = max_k [Σ P(k→r')·e[0][r']]
│       ├── Group 3: best keep (2 rr)  e[2][ds] = max_k [Σ P(k→r')·e[1][r']]
│       └── Group 1: expected value    E(S) = Σ P(⊥→r)·e[2][r]
└── Write: all_states.bin (16-byte header + f32[4,194,304])
```

### Server Architecture

```
tokio runtime (async)
├── axum::Router
│   ├── GET  /health        → {"status":"OK"}
│   ├── GET  /state_value   → E[score | state]
│   ├── GET  /score_histogram → binned distribution
│   ├── GET  /statistics    → aggregate sim stats
│   ├── POST /evaluate      → full roll evaluation (masks, categories, EVs)
│   └── POST /density       → spawn_blocking(density_evolution)
├── State: Arc<ServerState>
│   ├── ctx: Arc<YatzyContext> (mmap-backed)
│   └── oracle: Option<Arc<PolicyOracle>>
└── CorsLayer: allow all
```

## Frontend Architecture (React + TypeScript)

### Component Hierarchy

```
App (orchestrator)
├── ActionBar          Roll / Reroll / Reset buttons
├── DiceBar            5 dice with optimal-mask hints
│   └── Die (×5)      Value display, ±controls, hold toggle
├── DiceLegend         Color legend (keep vs reroll)
├── EvalPanel          Grid: state EV, mask EVs, delta, best category
├── TrajectoryChart    Canvas: score evolution + percentile bands
├── Scorecard          Fixed-layout table (table-layout: fixed)
│   └── ScorecardRow (×15)  Category + score + EV + action button
└── DebugPanel         JSON state dump (toggleable)
```

### State Machine

```
Flux-like: useReducer(gameReducer, initialState)
14 actions: ROLL, TOGGLE_DIE, REROLL, SCORE_CATEGORY,
            SET_EVAL_RESPONSE, SET_DIE_VALUE, SET_REROLLS,
            SET_CATEGORY_SCORE, UNSET_CATEGORY, RESET_GAME,
            SET_INITIAL_EV, SET_DENSITY_RESULT, TOGGLE_DEBUG

Turn phases: idle → rolled → (reroll cycles) → score → idle
             After 15 categories → game_over

Persistence: localStorage (excludes transient eval response)
```

### API Integration

```
api.ts → 4 fetch calls:
  POST /evaluate   — on roll/reroll (triggers SET_EVAL_RESPONSE)
  GET  /health     — connectivity check
  GET  /state_value — initial EV seed
  POST /density    — percentile bands for trajectory
```

## Analytics Pipeline Architecture

### CLI Pipeline (`yatzy-analyze`)

```
Binary simulation files
    ↓ io.read_all_scores() [numpy vectorized, auto-detect format]
{theta: sorted_scores}
    ↓ compute.compute_all()
    ├── Summary: 37 columns (percentiles, moments, CVaR, entropy, …)
    ├── KDE: 1000-point density + CDF per θ
    ├── MER: marginal exchange rates
    └── SDVA: stochastic dominance violation area
    ↓ store.save_*()
outputs/aggregates/parquet/{summary,kde,mer,sdva}.parquet
    ↓ plots/*
outputs/plots/ (50+ PNG/SVG/HTML)
```

### Key Subcommands

| Stage | Commands | Input | Output |
|-------|----------|-------|--------|
| Compute | `compute`, `compute --csv` | scores.bin | parquet, CSV |
| Plot | `plot`, `categories`, `efficiency` | parquet | PNG/SVG |
| Model | `surrogate-train`, `surrogate-eval` | training data | models, eval |
| Profile | `theta-validate`, `profile-validate` | scenarios.json | recovery stats |
| Rules | `skill-ladder` | regret export | ranked rules |

## Shared Types and Contracts

### API Contract (solver ↔ frontend/blog)

```typescript
// POST /evaluate
Request:  { dice: number[], upper_score: number, scored_categories: number, rerolls_remaining: number }
Response: {
  mask_evs: number[] | null,     // 32 reroll mask EVs (null if rerolls=0)
  optimal_mask: number | null,
  optimal_mask_ev: number | null,
  categories: CategoryInfo[],     // 15 entries
  optimal_category: number,
  optimal_category_ev: number,
  state_ev: number
}

// GET /state_value?upper_score=N&scored_categories=M
Response: { expected_final_score: number }

// POST /density
Request:  { upper_score: number, scored_categories: number, accumulated_score: number }
Response: { mean: number, std_dev: number, percentiles: Record<string, number> }
```

### Binary Contracts (solver → analytics)

| Format | Magic | Header | Record |
|--------|-------|--------|--------|
| Strategy table | `0x59545A53` | 16B (magic, version, states, θ) | f32[4,194,304] |
| Scores (compact) | `0x59545353` | 32B (magic, unused, games) | i16[N] |
| Full recording | `0x59545352` | 32B | GameRecord(289B)×N |
| Multiplayer | `0x594C504D` | 32B | MultiplayerRecord(64B)×N |
| Oracle | `0x4C43524F` | 16B | u8[1.05B] × 3 arrays |

### Shared Constants

| Constant | Rust | TypeScript | Blog JS |
|----------|------|------------|---------|
| CATEGORY_COUNT | 15 | 15 | 15 |
| UPPER_CATEGORIES | 6 | 6 | 6 |
| BONUS_THRESHOLD | 63 | 63 | 63 |
| BONUS_SCORE | 50 | 50 | 50 |
| NUM_DICE | 5 | 5 | 5 |

These constants are duplicated across all three codebases with no shared source of truth.
