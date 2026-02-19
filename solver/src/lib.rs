//! # Yatzy — Optimal Scandinavian Yatzy Solver
//!
//! Computes the optimal expected score for every reachable game state using
//! **backward induction** (retrograde analysis) over a DAG of ~1.4M states.
//!
//! ## Algorithm overview
//!
//! The algorithm is described in `theory/optimal_yahtzee_pseudocode.md`.
//! It has three phases:
//!
//! | Phase | Pseudocode function | Rust module | Description |
//! |-------|---------------------|-------------|-------------|
//! | 0 | `PRECOMPUTE_ROLLS_AND_PROBABILITIES` | [`phase0_tables`] | Build static lookup tables: dice sets R_{5,6}, scores, keep-multiset transition table, probabilities, reachability |
//! | 1 | `COMPUTE_REACHABILITY` | [`phase0_tables::precompute_reachability`] | Prune unreachable (upper_mask, upper_score) pairs (~31.8% eliminated) |
//! | 2 | `COMPUTE_OPTIMAL_STRATEGY` | [`state_computation`] | Backward induction from \|C\|=15 down to \|C\|=0, calling SOLVE_WIDGET per state |
//!
//! Each turn-start state is solved by [`widget_solver::compute_expected_state_value`]
//! (pseudocode `SOLVE_WIDGET`), which evaluates 6 groups bottom-up using ping-pong
//! buffers.
//!
//! ## State representation
//!
//! A game state S = (m, C) where:
//! - `m` ∈ [0, 63]: upper-section score (capped)
//! - `C`: 15-bit bitmask of scored categories (Ones=bit 0 .. Yatzy=bit 14)
//!
//! Flat index: `state_index(m, C) = C * STATE_STRIDE + m` (STATE_STRIDE=128),
//! giving 4,194,304 total slots (~16 MB). Indices 64..127 per scored mask
//! are padded with the capped value (index 63) for branchless upper-category access.
//!
//! ## Key differences from the pseudocode
//!
//! - **Scandinavian Yatzy**: 15 categories (adds One Pair, Two Pairs, Small/Large
//!   Straight as separate categories), 50-point upper bonus (not 35).
//! - **No Yahtzee bonus flag**: the pseudocode's `f` flag is not used.
//! - **Keep-multiset dedup**: the inner loop iterates deduplicated keep-multisets
//!   (avg 16.3 per dice set vs 31 raw masks), eliminating ~47% of redundant work.
//! - **Sparse CSR storage** for transition probabilities P(r'→r) — only 4,368
//!   non-zero entries across 462 keep rows.
//! - **f32 throughout**: all internal computation and storage uses f32. Empirical
//!   testing shows max 0.00046 point difference vs f64 accumulation, with zero
//!   impact on optimal play decisions.

#![allow(clippy::needless_range_loop)]

pub mod api_computations;
pub mod batched_solver;
pub mod constants;
pub mod density;
pub mod dice_mechanics;
pub mod game_mechanics;
pub mod phase0_tables;
pub mod profiling;
pub mod server;
pub mod simd;
pub mod simulation;
pub mod state_computation;
pub mod storage;
pub mod types;
pub mod widget_solver;
