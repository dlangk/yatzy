# Rust Backend Deep Dive

## Crate Structure

Single crate (not a workspace). `solver/Cargo.toml`:

**Library:** `src/lib.rs` (re-exports all modules)

**28 Binary Targets** in `src/bin/`:

| Binary | File | Purpose |
|--------|------|---------|
| `yatzy` | server.rs | axum HTTP API server |
| `yatzy-precompute` | precompute.rs | Backward induction DP |
| `yatzy-simulate` | simulate.rs | Monte Carlo simulation |
| `yatzy-sweep` | sweep.rs | θ sweep (resumable) |
| `yatzy-density` | density.rs | Exact forward-DP PMF |
| `yatzy-scenarios` | scenarios.rs | Master scenario pipeline |
| `yatzy-profile-scenarios` | generate_profile_scenarios.rs | Quiz generation |
| `yatzy-player-card-grid` | player_card_grid.rs | 648×10K grid pre-compute |
| `yatzy-export-training-data` | export_training_data.rs | Surrogate training data |
| `yatzy-decision-sensitivity` | decision_sensitivity.rs | Board state sensitivity |
| `yatzy-decision-gaps` | decision_gaps.rs | Runner-up analysis |
| `yatzy-regret-export` | regret_export.rs | Per-decision regret |
| `yatzy-eval-policy` | eval_policy.rs | Policy evaluation |
| `pivotal-scenarios` | pivotal_scenarios.rs | Critical decision points |
| `difficult-scenarios` | difficult_scenarios.rs | High EV-gap decisions |
| `scenario-sensitivity` | scenario_sensitivity.rs | Multi-θ scenario eval |
| `heuristic-gap` | heuristic_gap.rs | Per-decision gap vs heuristic |
| `yatzy-winrate` | winrate.rs | Head-to-head comparison |
| `yatzy-multiplayer` | multiplayer.rs | 2-player simulation |
| `yatzy-underdog-sweep` | underdog_sweep.rs | Underdog policy eval |
| `yatzy-frontier-test` | frontier_test.rs | State-dependent vs fixed θ |
| `yatzy-percentile-sweep` | percentile_sweep.rs | Adaptive θ per percentile |
| `yatzy-human-baseline` | human_baseline.rs | Heuristic baselines |
| `yatzy-category-sweep` | category_sweep.rs | Per-category stats |
| `yatzy-conditional` | conditional.rs | Conditional Yatzy rates |
| `export-greedy-game` | export_greedy_game.rs | Blog data export |
| `export-state-heatmap` | export_state_heatmap.rs | Blog heatmap data |
| `yatzy-rosetta` | rosetta.rs | Policy distillation |

**Key Dependencies:**
- `axum` 0.8 + `tokio` 1.x (async HTTP)
- `rayon` 1.x (parallel DP)
- `memmap2` (zero-copy file I/O)
- `serde` + `serde_json` (serialization)
- `rand` + `rand_chacha` (RNG)
- `toml` (config parsing)
- `tower-http` (CORS middleware)

**Release Profile:** opt-level 3, fat LTO, codegen-units 1, panic=abort

## Source File Inventory

### Core Modules (~8,500 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `widget_solver.rs` | 1,001 | SOLVE_WIDGET: Groups 1→6 DP (16 public functions) |
| `batched_solver.rs` | 908 | Batched SpMM: 64 upper scores simultaneously (7 public) |
| `simd.rs` | 597 | 14 NEON intrinsic kernels + fast exp |
| `phase0_tables.rs` | 564 | Precompute all lookup tables (9 functions) |
| `constants.rs` | 510 | STATE_STRIDE=128, Category enum, state_index() |
| `game_mechanics.rs` | 490 | Yatzy scoring rules: s(S, r, c) |
| `state_computation.rs` | 386 | Phase 2 DP with rayon par_iter |
| `storage.rs` | 377 | Binary I/O, zero-copy mmap (v5/v6) |
| `dice_mechanics.rs` | 350 | Dice operations, probabilities |
| `api_computations.rs` | 261 | HTTP handler computation wrappers |
| `server.rs` | 232 | axum router, CORS, 6 handlers |
| `types.rs` | 232 | YatzyContext, KeepTable, StateValues |
| `lib.rs` | ~50 | Module re-exports |

### Simulation Modules (~5,500 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `simulation/engine.rs` | 1,049 | Sequential simulate_game, with/without recording |
| `simulation/adaptive.rs` | 1,006 | Noisy/softmax policies |
| `simulation/strategy.rs` | 743 | Adaptive multi-θ policies |
| `simulation/multiplayer.rs` | 758 | 2-player head-to-head |
| `simulation/heuristic.rs` | 544 | Heuristic baselines |
| `simulation/statistics.rs` | 522 | Aggregate stats: mean, std, percentiles, CVaR |
| `simulation/lockstep.rs` | 499 | Horizontal SIMD lockstep + oracle variant |
| `simulation/raw_storage.rs` | 433 | Binary simulation I/O (mmap) |
| `simulation/sweep.rs` | 197 | θ sweep inventory, grid resolution |
| `simulation/radix_sort.rs` | 155 | 2-pass counting sort for lockstep |
| `simulation/fast_prng.rs` | 121 | SplitMix64: 5 dice from single u64 |

### Profiling & Scenarios (~5,200 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `profiling/scenarios.rs` | 1,640 | Master pool, quiz assembly, Q-grid |
| `scenarios/select.rs` | 685 | Difficulty-based selection |
| `scenarios/enrich.rs` | 458 | θ-sensitivity enrichment |
| `scenarios/classify.rs` | 419 | Decision type/phase/tension |
| `scenarios/actions.rs` | 406 | Semantic action representation |
| `scenarios/collect.rs` | 320 | Candidate pool from simulations |
| `profiling/qvalues.rs` | 335 | Q-value with (θ,γ,d) perturbation |
| `scenarios/types.rs` | 214 | DecisionType, GamePhase, etc. |
| `profiling/player_card.rs` | 151 | Noisy sim for player card |

### Rosetta (~1,900 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `rosetta/policy.rs` | 1,387 | Distill oracle → human-readable rules |
| `rosetta/dsl.rs` | 478 | Rule DSL representation |

### Density Evolution (~960 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `density/forward.rs` | 450 | Forward density evolution (dense Vec<f64>) |
| `density/transitions.rs` | 514 | Per-state transition computation |

### Binary Targets (~13,500 LOC across 28 files)

Largest: `decision_gaps` (1,256), `decision_sensitivity` (1,214), `difficult_scenarios` (1,016), `scenarios` (850), `pivotal_scenarios` (825), `scenario_sensitivity` (797).

**Total: ~37,700 LOC across ~70 source files.**

## Hot Paths

### Group 6 → 5 → 3 → 1 Pipeline (widget_solver.rs)

This is the innermost computation. For a single state (upper_score, scored_categories):

```
Group 6: For each of 252 dice sets, find best category
  e_ds[0][r] = max_c { s(r,c) + E[successor state] }
  → Iterates 15 categories × 252 dice = 3,780 operations
  → Uses NEON add-max kernels for batched upper-score processing

Group 5: For each of 252 dice sets, find best keep (1 reroll left)
  e_ds[1][r] = max_k { Σ P(k→r') · e_ds[0][r'] }  for each keep k
  → 462 unique keeps × sparse CSR dot product (avg ~9 non-zeros)
  → Uses NEON FMA + max kernels

Group 3: Same as Group 5 but for 2 rerolls
  e_ds[2][r] = max_k { Σ P(k→r') · e_ds[1][r'] }

Group 1: Weighted sum over initial roll probabilities
  E(S) = Σ P(⊥→r) · e_ds[2][r]
  → 252 FMA operations
```

### Batched SpMM (batched_solver.rs)

Processes all 64 upper-score variants of one scored_mask simultaneously:

```
Memory per rayon thread:
  e[0]: 252×64 = 129 KB   (ping buffer)
  e[1]: 252×64 = 129 KB   (pong buffer)
  keep_ev: 462×64 = 118 KB
  Total: ~247 KB (fits in L2 cache)

7 public functions:
  solve_widget_batched()          — EV mode (θ=0)
  solve_widget_batched_risk()     — log-domain LSE (|θ|>0.15)
  solve_widget_batched_utility()  — utility domain (|θ|≤0.15)
  solve_widget_batched_max()      — max-policy mode
  batched_group1_pub()            — reusable Group 1
  build_oracle_for_scored_mask()  — argmax tracking for oracle
  precompute_exp_scores()         — e^(θ·score) table
```

### NEON SIMD Kernels (simd.rs)

14 hand-written ARM NEON kernels operating on 64×f32 (256-byte) blocks:

| Kernel | Operation | Used For |
|--------|-----------|----------|
| `neon_fma_64` | dst += scalar × src | Group 5/3 Step 1 (keep accumulation) |
| `neon_max_64` | dst = max(dst, src) | Group 5/3 Step 2 (best keep) |
| `neon_min_64` | dst = min(dst, src) | Risk-averse variant |
| `neon_add_max_64` | dst = max(dst, scalar + src) | Group 6 lower categories |
| `neon_add_min_64` | Risk-averse variant | |
| `neon_add_max_offset_64` | dst = max(dst, scalar + sv[base+offset+i]) | Group 6 upper categories |
| `neon_add_min_offset_64` | Risk-averse variant | |
| `neon_mul_max_64` | dst = max(dst, scalar × src) | Group 6 utility domain |
| `neon_mul_min_64` | Risk-averse variant | |
| `neon_mul_max_offset_64` | Utility upper categories | |
| `neon_mul_min_offset_64` | Risk-averse variant | |
| `neon_max_64_argmax` | Track best keep + index | Oracle builder (VBSL blend) |
| `neon_fast_exp_f32x4` | Degree-5 Cephes polynomial | Utility solver (~2 ULP) |
| `neon_weighted_sum_64` | Alias of FMA | Group 1 final sum |

All process 16 iterations of 4-lane SIMD (vfmaq_f32, vmaxq_f32, vbslq_f32).

## Web Server: Route Map

### axum Router (server.rs)

```rust
Router::new()
    .route("/health", get(handle_health))
    .route("/state_value", get(handle_state_value))
    .route("/score_histogram", get(handle_score_histogram))
    .route("/statistics", get(handle_statistics))
    .route("/evaluate", post(handle_evaluate))
    .route("/density", post(handle_density))
    .layer(CorsLayer::very_permissive())
    .with_state(Arc::new(server_state))
```

### Endpoint Details

| Endpoint | Handler | Computation | Response Time |
|----------|---------|-------------|---------------|
| `GET /health` | Trivial | None | <1ms |
| `GET /state_value?upper_score&scored_categories` | Array lookup | `sv[state_index(up, scored)]` | <1ms |
| `GET /score_histogram` | Binning | Precomputed histogram bins | <1ms |
| `GET /statistics` | Stats | Aggregated sim statistics | <1ms |
| `POST /evaluate` | Full pipeline | Groups 6→5→3→1 for 252 dice sets | ~10-50ms |
| `POST /density` | `spawn_blocking` | Forward density evolution | ~3s (oracle) |

### Request/Response Types

```rust
// POST /evaluate
#[derive(Deserialize)]
struct EvaluateRequest {
    dice: Vec<i32>,           // 5 dice values [1-6]
    upper_score: i32,         // 0-63
    scored_categories: i32,   // 15-bit bitmask
    rerolls_remaining: i32,   // 0-2
}

#[derive(Serialize)]
struct RollResponse {
    mask_evs: Option<[f64; 32]>,        // All reroll mask EVs
    optimal_mask: Option<i32>,
    optimal_mask_ev: Option<f64>,
    categories: [CategoryInfo; 15],
    optimal_category: i32,
    optimal_category_ev: f64,
    state_ev: f64,
}

struct CategoryInfo {
    id: usize,
    name: &'static str,
    score: i32,
    available: bool,
    ev_if_scored: f64,
}
```

## Key Data Structures

### YatzyContext (types.rs)

Central immutable context shared via `Arc` across threads:

```rust
YatzyContext {
    all_dice_sets: [[i32; 5]; 252],              // All sorted 5-dice multisets
    num_combinations: usize,                      // Always 252
    index_lookup: [[[[[i32; 6]; 6]; 6]; 6]; 6],  // dice → index reverse lookup
    precomputed_scores: [[i32; 15]; 252],        // s(r, c) for all (r, c)
    scored_category_count_cache: Vec<i32>,        // popcount(C) for all 32,768 masks
    factorial: [i32; 6],                          // 0!..5!
    state_values: StateValues,                    // E-table: Owned or Mmap
    dice_set_probabilities: [f64; 252],          // P(⊥→r) for each dice set
    keep_table: KeepTable,                        // Sparse CSR transitions
    reachable: [[bool; 64]; 64],                 // Phase 1 reachability
    theta: f32,                                   // Risk parameter θ
    max_policy: bool,                             // Max instead of EV
}
```

### KeepTable (Sparse CSR)

462 keep-multisets, 4,368 non-zero entries (~47% compression from 252×31 masks):

```rust
KeepTable {
    vals: Vec<f32>,                  // Transition probabilities P(k→ds')
    cols: Vec<i32>,                  // Destination dice-set indices
    row_start: [i32; 463],          // CSR row boundaries
    unique_count: [i32; 252],       // # unique keeps per dice set
    unique_keep_ids: [[i32; 31]; 252], // Keep IDs (dedup masks)
    mask_to_keep: Vec<i32>,         // Mask 0-31 → keep index
    keep_to_mask: Vec<i32>,         // Keep index → first mask
}
```

### StateValues

```rust
enum StateValues {
    Owned(Vec<f32>),                     // During precomputation
    Mmap { mmap: memmap2::Mmap },       // From disk (load in <1ms)
}
// as_slice() skips 16-byte header for Mmap variant
```

### PolicyOracle (3.17 GB)

```rust
PolicyOracle {
    oracle_cat: Vec<u8>,     // [si*252+ds] = best category (0-14)
    oracle_keep1: Vec<u8>,   // [si*252+ds] = best keep with 1 reroll left
    oracle_keep2: Vec<u8>,   // [si*252+ds] = best keep with 2 rerolls left
}
// Total entries: 3 × NUM_STATES × 252 = 3 × 1,056,964,608
// Encoding: keep=0 → keep all, keep=j+1 → unique_keep_ids[ds][j]
```

## Memory Layout

### STATE_STRIDE=128 (Topological Padding)

```
state_index(upper_score, scored_categories) = scored_categories * 128 + upper_score

Per scored_mask block (512 bytes):
  [0..63]     Actual upper scores 0-63 (256 bytes = 4 L1 cache lines)
  [64..127]   Copies of index 63 (capped value)

Total: 32,768 masks × 128 stride = 4,194,304 entries × 4 bytes = 16 MB

Purpose: Enables sv[base + min(up+scr, 63)] → sv[base + up + scr] without min()
  because indices 64-127 all duplicate the value at index 63.
```

### Binary File Format

**State file (v5/v6):**
```
Header (16 bytes):
  magic: u32 = 0x59545A53 ("STZY")
  version: u32 = 6
  total_states: u32 = 4,194,304
  theta_bits: u32 = f32::to_bits(θ)
Data:
  float32[4,194,304] in state_index order (16 MB)
```

**Oracle file:**
```
Header (16 bytes):
  magic: u32 = 0x4C43524F ("ORCL")
  version: u32 = 1
  (8 bytes unused)
Data:
  oracle_cat[1,056,964,608]: Vec<u8>     (1.05 GB)
  oracle_keep1[1,056,964,608]: Vec<u8>   (1.05 GB)
  oracle_keep2[1,056,964,608]: Vec<u8>   (1.05 GB)
```

## Concurrency Model

### Precomputation: Rayon Only

```rust
// state_computation.rs
scored_masks.par_iter().for_each_init(
    || BatchedBuffers::new(),
    |bufs, &scored| {
        let results = solve_widget_batched(ctx, sv, scored, bufs);
        // Unsafe write: each state_index is unique → no data races
        unsafe {
            let ptr = sv_ptr.load(Ordering::Relaxed);
            let base = scored as usize * STATE_STRIDE;
            for up in 0..64 {
                *ptr.add(base + up) = results[up];
            }
        }
    }
);
```

### Server: Tokio Async

- All handlers are async (stateless lookups)
- Density computation uses `tokio::task::spawn_blocking()` to avoid blocking the async runtime
- No rayon in the server process

### Simulation: Rayon or Sequential

- Sequential engine: single-threaded, uses `rand::thread_rng()`
- Lockstep: processes batch of games with radix sort grouping (SplitMix64 PRNG)
- No mixed tokio+rayon in simulation binaries

## Density Evolution (density/)

### Forward DP Algorithm

Three phases per turn (15 turns total):

```
Phase 1 — Compute Transitions (parallel):
  For each active state S:
    For each valid category c:
      For each of 252 dice outcomes r:
        Compute transition: (next_state, points, probability)

Phase 2 — Build Destination Index:
  HashMap: dest_state_index → [(src_idx, transition_idx), ...]

Phase 3 — Parallel Merge:
  For each destination state:
    P(dest, score) += Σ P(src, score-points) × P(transition)
```

**Data structures:**
- `StateEntry = (state_index: u32, dense_dist: Vec<f64>)` — 384 score bins
- Non-oracle: ~6 min/θ (transition computation is 85% of time)
- Oracle: ~3s/θ (126x speedup via probability-array propagation)

## Functions Over 50 Lines

| File | Function | Lines |
|------|----------|-------|
| `profiling/scenarios.rs` | `build_master_pool()` | ~911 |
| `profiling/scenarios.rs` | `assemble_quiz()` | ~1,065 |
| `profiling/scenarios.rs` | `compute_q_grid()` | ~1,561 |
| `api_computations.rs` | `compute_roll_response()` | ~212 |
| `phase0_tables.rs` | `precompute_lookup_tables()` | ~203 |
| `game_mechanics.rs` | `update_upper_score()` | ~165 |
| `simulation/engine.rs` | `simulate_game_with_recording()` | ~150 |
| `widget_solver.rs` | `compute_expected_state_value()` | ~96 |
| `game_mechanics.rs` | `calculate_category_score()` | ~93 |
| `storage.rs` | `save_all_state_values()` | ~70 |
| `widget_solver.rs` | `choose_best_reroll_mask_risk()` | ~67 |
| `widget_solver.rs` | `compute_opt_lse_for_n_rerolls()` | ~65 |
| `storage.rs` | `load_all_state_values()` | ~63 |

Note: Many binary targets have large `main()` functions (500-1200 lines).

## Performance Benchmarks (M1 Max, 8 P-cores)

| Operation | Time | Throughput |
|-----------|------|------------|
| Phase 0 (lookup tables) | ~0.3s | — |
| Phase 2 EV (θ=0) | ~1.1s | 1.9M states/s |
| Phase 2 + oracle | ~1.5s | 1.4x overhead |
| Phase 2 utility (|θ|≤0.15) | ~0.49s | 4.3M states/s |
| Phase 2 LSE (|θ|>0.15) | ~2.7s | 780K states/s |
| MC simulation (sequential) | — | 52K games/s |
| MC simulation (lockstep) | — | 232K games/s |
| MC simulation (oracle) | — | 5.6M games/s |
| Density evolution (non-oracle) | ~381s | — |
| Density evolution (oracle) | ~3.0s | 126x speedup |
| API /evaluate | ~10-50ms | — |
| API /state_value | <1ms | — |
