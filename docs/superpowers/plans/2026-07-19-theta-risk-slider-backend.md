# Theta Risk Slider Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/evaluate` return the optimal-play feedback for all five risk detents in one response, with each non-neutral detent's numbers as certainty-equivalent (CE) points, so the frontend slider can re-center instantly with no extra calls.

**Architecture:** Neutral (θ=0) keeps the existing EV path unchanged. Each non-neutral detent runs a new certainty-equivalent recursion that mirrors the solver's utility-domain math (`exp(θ·score)·U(successor)`, min/max flipped for θ<0), reusing the existing linear reroll-propagation helpers. The server loads the four non-neutral tables at startup alongside the neutral context. The response keeps its current flat fields (equal to the Neutral detent, so the live frontend keeps working) and adds a `detents` array.

**Tech Stack:** Rust, axum, serde_json, memmap2. Solver crate at `solver/`.

**Spec:** `docs/superpowers/specs/2026-07-19-theta-risk-slider-design.md`

## Global Constraints

- **No emdashes** in prose or comments. Use periods, commas, colons, semicolons, parentheses.
- **The web server MUST NEVER panic on user input** (`.claude/rules/rust-api.md`): validate and return HTTP 400/500. The HPC core may `unwrap`/`expect` on invariants.
- **`/evaluate` stays latency-critical:** no blocking, no unnecessary allocation in the per-request path. Load all tables at startup via mmap.
- **Backward compatible:** the response must keep its existing top-level fields (`categories`, `optimal_category`, `optimal_category_ev`, `state_ev`, and `mask_evs`/`optimal_mask`/`optimal_mask_ev` when rerolls remain), equal to the Neutral detent, so the currently-deployed frontend keeps working after this ships. The new `detents` array is additive.
- **Neutral is identical to today:** detent index 2 (θ=0) uses the existing `compute_roll_response`, byte-for-byte.
- **Detents (index order):** θ = −0.08, −0.04, 0.00, +0.04, +0.08. Tables: `all_states_theta_-0.080.bin`, `all_states_theta_-0.040.bin`, `all_states.bin`, `all_states_theta_0.040.bin`, `all_states_theta_0.080.bin` (all present on disk, 16,777,232 bytes each).
- **CE definition:** `U(S) = E[e^{θ·remaining}]`, `CE = (1/θ)·ln U`, in points. The tables store `L(S) = ln U(S)`, so `U = exp(L)`. Optimal action = argmax CE (equivalently argmax U for θ>0, argmin U for θ<0).
- **Utility domain only:** all four non-neutral detents satisfy |θ| ≤ 0.08 < 0.15, so the multiplicative utility recursion applies (no LSE). Max total score is 374; `exp(0.08·374) ≈ 1.8e16` fits f32.
- Run `cargo fmt`, `cargo clippy`, `cargo test` after changes. Run `just bench-check` from repo root if any hot path is touched (this plan does not touch the precompute hot paths).

---

### Task 1: Certainty-equivalent roll response

The core math. A new `compute_roll_response_theta` mirroring `compute_roll_response` but in utility/CE domain, plus a utility-propagation helper. Reuses the linear mask helpers unchanged.

**Files:**
- Modify: `solver/src/api_computations.rs` (add two functions + tests)

**Interfaces:**
- Consumes: `compute_expected_value_for_reroll_mask` (unchanged, from `widget_solver.rs`), `precompute_exp_scores` + `ExpScores` (from `batched_solver.rs`), `state_index`, `update_upper_score`, `is_category_scored`, `CATEGORY_NAMES`, `CATEGORY_COUNT`, `NUM_KEEP_MULTISETS`.
- Produces:
  - `pub fn compute_roll_response_theta(ctx: &YatzyContext, sv_theta: &[f32], exp_scores: &ExpScores, theta: f32, upper_score: i32, scored_categories: i32, dice: &[i32; 5], rerolls_remaining: i32) -> RollResponse` (same `RollResponse` struct); all value fields are CE.
  - `fn propagate_reroll_level_utility(ctx: &YatzyContext, e_prev: &[f32; 252], e_current: &mut [f32; 252], minimize: bool)`, the utility analog of `compute_expected_values_for_n_rerolls`, with min/max by sign.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `solver/src/api_computations.rs` (after the existing tests). These use `make_ctx()` (phase-0 tables only; `state_values` is zero-filled), so with `L(S)=0` everywhere `U=1` and the CE of scoring category c reduces to its immediate score. That is enough to exercise the recursion's structure and the exp/ln/argmax plumbing without loading real tables.

```rust
    #[test]
    fn test_ce_zero_table_reduces_to_scores() {
        // With a zero state table (L=0 -> U=1), CE of scoring category c is its
        // immediate score, exactly like the EV path with a zero table.
        let ctx = make_ctx();
        let exp_scores = crate::batched_solver::precompute_exp_scores(&ctx, 0.04);
        let sv = ctx.state_values.as_slice();
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response_theta(&ctx, sv, &exp_scores, 0.04, 0, 0, &dice, 0);

        // Small Straight scores 15; with U=1 successors its CE equals 15.
        let ss = &resp.categories[CATEGORY_SMALL_STRAIGHT];
        assert!(ss.available);
        assert!((ss.ev_if_scored - 15.0).abs() < 1e-3, "CE was {}", ss.ev_if_scored);
        assert_eq!(resp.optimal_category, CATEGORY_SMALL_STRAIGHT as i32);
    }

    #[test]
    fn test_ce_positive_theta_argmax_is_finite() {
        // Structural: with rerolls, all 32 mask CEs are finite and the optimal
        // mask CE is >= keep-all (mask 0) CE.
        let ctx = make_ctx();
        let exp_scores = crate::batched_solver::precompute_exp_scores(&ctx, 0.08);
        let sv = ctx.state_values.as_slice();
        let dice = [3, 3, 4, 5, 6];
        let resp = compute_roll_response_theta(&ctx, sv, &exp_scores, 0.08, 0, 0, &dice, 2);
        let mask_evs = resp.mask_evs.expect("rerolls remain");
        for m in mask_evs.iter() {
            assert!(m.is_finite());
        }
        assert!(resp.optimal_mask_ev.unwrap() >= mask_evs[0] - 1e-6);
    }

    #[test]
    fn test_ce_negative_theta_reduces_to_scores() {
        // Same zero-table reduction must hold for theta < 0 (exercises the
        // minimize path and the (1/theta) sign).
        let ctx = make_ctx();
        let exp_scores = crate::batched_solver::precompute_exp_scores(&ctx, -0.08);
        let sv = ctx.state_values.as_slice();
        let dice = [1, 2, 3, 4, 5];
        let resp = compute_roll_response_theta(&ctx, sv, &exp_scores, -0.08, 0, 0, &dice, 0);
        let chance = &resp.categories[CATEGORY_CHANCE];
        assert!((chance.ev_if_scored - 15.0).abs() < 1e-3, "CE was {}", chance.ev_if_scored);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd solver && cargo test --features full compute_roll_response_theta 2>&1 | head -30`
Expected: FAIL to compile (`compute_roll_response_theta` and `propagate_reroll_level_utility` do not exist yet).

- [ ] **Step 3: Implement the recursion and helper**

Add to `solver/src/api_computations.rs`. First extend the imports at the top of the file:

```rust
use crate::batched_solver::ExpScores;
```

(`NUM_KEEP_MULTISETS` is already in scope via the existing `use crate::constants::*;`, so do not re-import it. `ExpScores` is the only new import.)

Then add these two functions (place them after `compute_roll_response`):

```rust
/// Utility analog of `compute_expected_values_for_n_rerolls`: propagate one
/// reroll level over utilities. Stochastic node is a plain weighted sum (same as
/// EV); the keep-decision node takes the extremum that maximizes the certainty
/// equivalent, which is max utility for theta > 0 and min utility for theta < 0.
fn propagate_reroll_level_utility(
    ctx: &YatzyContext,
    e_prev: &[f32; 252],
    e_current: &mut [f32; 252],
    minimize: bool,
) {
    let kt = &ctx.keep_table;
    // Expected utility for each unique keep-multiset (linear weighted sum).
    let mut keep_u = [0.0f32; NUM_KEEP_MULTISETS];
    for kid in 0..NUM_KEEP_MULTISETS {
        let start = kt.row_start[kid] as usize;
        let end = kt.row_start[kid + 1] as usize;
        let mut u = 0.0f32;
        for k in start..end {
            u += kt.vals[k] * e_prev[kt.cols[k] as usize];
        }
        keep_u[kid] = u;
    }
    // For each dice set, take the sign-correct extremum over its keeps.
    for ds_i in 0..252 {
        let mut best = e_prev[ds_i]; // mask 0 = keep all
        for j in 0..kt.unique_count[ds_i] as usize {
            let kid = kt.unique_keep_ids[ds_i][j] as usize;
            let u = keep_u[kid];
            let better = if minimize { u < best } else { u > best };
            if better {
                best = u;
            }
        }
        e_current[ds_i] = best;
    }
}

/// Certainty-equivalent variant of `compute_roll_response` for a nonzero theta.
///
/// `sv_theta` is this detent's log-domain table (stores L(S) = ln U(S)).
/// `exp_scores[ds][c] = e^{theta*score(ds,c)}`. All value fields in the returned
/// `RollResponse` are certainty equivalents CE = (1/theta) ln U, in points.
/// Optimal actions are argmax CE (max utility for theta > 0, min for theta < 0).
pub fn compute_roll_response_theta(
    ctx: &YatzyContext,
    sv_theta: &[f32],
    exp_scores: &ExpScores,
    theta: f32,
    upper_score: i32,
    scored_categories: i32,
    dice: &[i32; 5],
    rerolls_remaining: i32,
) -> RollResponse {
    let inv_theta = 1.0f64 / theta as f64;
    let minimize = theta < 0.0;
    let worst = if minimize { f32::INFINITY } else { f32::NEG_INFINITY };

    // Sort dice and track original indices for mask remapping (same as EV path).
    let mut indexed: [(i32, usize); 5] = [
        (dice[0], 0), (dice[1], 1), (dice[2], 2), (dice[3], 3), (dice[4], 4),
    ];
    indexed.sort_by_key(|&(v, _)| v);
    let sort_map: [usize; 5] = [
        indexed[0].1, indexed[1].1, indexed[2].1, indexed[3].1, indexed[4].1,
    ];
    let sorted_dice: [i32; 5] = [
        indexed[0].0, indexed[1].0, indexed[2].0, indexed[3].0, indexed[4].0,
    ];
    let ds_index = find_dice_set_index(ctx, &sorted_dice);

    // Utility of scoring category c from a dice set = exp(theta*score) * U(succ),
    // where U(succ) = exp(L(succ)). CE = (1/theta) ln(utility).
    let util_of_score = |ds_i: usize, c: usize| -> f64 {
        let scr = ctx.precomputed_scores[ds_i][c];
        let new_up = update_upper_score(upper_score, c, scr);
        let new_scored = scored_categories | (1 << c);
        let l_succ = sv_theta[state_index(new_up as usize, new_scored as usize)] as f64;
        exp_scores[ds_i][c] as f64 * l_succ.exp()
    };

    // --- Categories: CE of scoring each available category now ---
    let categories: [CategoryInfo; CATEGORY_COUNT] = std::array::from_fn(|c| {
        let scr = ctx.precomputed_scores[ds_index][c];
        let available = !is_category_scored(scored_categories, c);
        let ev_if_scored = if available {
            inv_theta * util_of_score(ds_index, c).ln()
        } else {
            f64::NEG_INFINITY
        };
        CategoryInfo {
            id: c,
            name: CATEGORY_NAMES[c],
            score: scr,
            available,
            ev_if_scored,
        }
    });

    // optimal_category = argmax CE among available (CE is already sign-correct).
    let mut optimal_category = -1i32;
    let mut optimal_category_ev = f64::NEG_INFINITY;
    for cat in &categories {
        if cat.available && cat.ev_if_scored > optimal_category_ev {
            optimal_category_ev = cat.ev_if_scored;
            optimal_category = cat.id as i32;
        }
    }

    // --- Group 6 base: utility of the best "score now" action per dice set ---
    let mut e_ds_0 = [worst; 252];
    for ds_i in 0..252 {
        let mut best = worst;
        for c in 0..CATEGORY_COUNT {
            if !is_category_scored(scored_categories, c) {
                let u = util_of_score(ds_i, c) as f32;
                let better = if minimize { u < best } else { u > best };
                if better {
                    best = u;
                }
            }
        }
        e_ds_0[ds_i] = best;
    }

    if rerolls_remaining > 0 {
        let e_ds_for_masks = if rerolls_remaining == 1 {
            e_ds_0
        } else {
            let mut e_ds_1 = [worst; 252];
            propagate_reroll_level_utility(ctx, &e_ds_0, &mut e_ds_1, minimize);
            e_ds_1
        };

        // 32 mask utilities (pure weighted sum; helper reused unchanged), in sorted order.
        let mut sorted_mask_util = [0.0f64; 32];
        for mask in 0..32 {
            sorted_mask_util[mask] =
                compute_expected_value_for_reroll_mask(ctx, ds_index, &e_ds_for_masks, mask as i32);
        }

        // Remap sorted -> original dice order, converting utility to CE.
        let mut mask_evs = [0.0f64; 32];
        for orig_mask in 0u32..32 {
            let mut sorted_mask = 0u32;
            for si in 0..5 {
                if orig_mask & (1 << sort_map[si]) != 0 {
                    sorted_mask |= 1 << si;
                }
            }
            mask_evs[orig_mask as usize] = inv_theta * sorted_mask_util[sorted_mask as usize].ln();
        }

        // optimal_mask = argmax CE (already sign-correct after the (1/theta) factor).
        let mut best_mask = 0i32;
        let mut best_mask_ev = mask_evs[0];
        for mask in 1..32 {
            if mask_evs[mask] > best_mask_ev {
                best_mask_ev = mask_evs[mask];
                best_mask = mask as i32;
            }
        }

        let effective_state_ev = best_mask_ev.max(optimal_category_ev);

        RollResponse {
            mask_evs: Some(mask_evs),
            optimal_mask: Some(best_mask),
            optimal_mask_ev: Some(best_mask_ev),
            categories,
            optimal_category,
            optimal_category_ev,
            state_ev: effective_state_ev,
        }
    } else {
        RollResponse {
            mask_evs: None,
            optimal_mask: None,
            optimal_mask_ev: None,
            categories,
            optimal_category,
            optimal_category_ev,
            state_ev: optimal_category_ev,
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd solver && cargo test --features full ce_ 2>&1 | tail -20`
Expected: PASS (the three `test_ce_*` tests).

- [ ] **Step 5: Format, lint, full test**

Run: `cd solver && cargo fmt && cargo clippy --features full 2>&1 | tail -5 && cargo test --features full 2>&1 | tail -5`
Expected: no fmt diff, no clippy errors, all tests pass (existing count + 3 new).

- [ ] **Step 6: Commit**

```bash
git add solver/src/api_computations.rs
git commit -m "Add certainty-equivalent roll response for risk detents

compute_roll_response_theta mirrors the EV path in utility domain
(exp(theta*score)*U(successor), min/max flipped for theta<0), reusing the
linear reroll-propagation helper and reporting CE = (1/theta) ln U."
```

---

### Task 2: Load the four non-neutral detent tables at startup

**Files:**
- Modify: `solver/src/server.rs` (extend `ServerState`, add a constructor field, build the detents)
- Modify: `solver/src/bin/server.rs` (load the four tables and pass them in)

**Interfaces:**
- Consumes: `storage::load_state_values_standalone` (returns `Option<StateValues>`, does NOT validate theta), `batched_solver::precompute_exp_scores`, `types::StateValues`, `types::ExpScores` via `batched_solver::ExpScores`.
- Produces:
  - A `pub struct Detent { pub theta: f32, pub sv: StateValues, pub exp_scores: Box<ExpScores> }` in `server.rs`.
  - `ServerState.detents: Vec<Detent>` (the four non-neutral detents; Neutral is served from `ctx`).
  - `create_router_full` gains a `detents: Vec<Detent>` parameter.

- [ ] **Step 1: Add the `Detent` struct and `ServerState` field**

In `solver/src/server.rs`, add near the top after the imports:

```rust
use crate::batched_solver::{precompute_exp_scores, ExpScores};
use crate::types::StateValues;

/// A non-neutral risk detent: its table (log-domain L(S)) plus precomputed
/// exp(theta*score). Neutral (theta=0) is served from the shared context.
pub struct Detent {
    pub theta: f32,
    pub sv: StateValues,
    pub exp_scores: Box<ExpScores>,
}
```

Add the field to `ServerState`:

```rust
pub struct ServerState {
    pub ctx: Arc<YatzyContext>,
    pub oracle: Option<Arc<PolicyOracle>>,
    pub percentile_table: Option<HashMap<u32, PercentileEntry>>,
    /// The four non-neutral risk detents (theta = -0.08, -0.04, +0.04, +0.08).
    pub detents: Vec<Detent>,
    /// Limits concurrent MC density simulations to 1 at a time.
    pub density_semaphore: Arc<Semaphore>,
}
```

- [ ] **Step 2: Thread `detents` through the router constructors**

Update `create_router` (no detents; pass empty) and `create_router_full` (accept detents) in `solver/src/server.rs`:

```rust
pub fn create_router(ctx: Arc<YatzyContext>) -> Router {
    let state = Arc::new(ServerState {
        ctx,
        oracle: None,
        percentile_table: None,
        detents: Vec::new(),
        density_semaphore: Arc::new(Semaphore::new(1)),
    });
    create_router_with_state(state)
}

pub fn create_router_full(
    ctx: Arc<YatzyContext>,
    oracle: Option<PolicyOracle>,
    percentile_table: Option<HashMap<u32, PercentileEntry>>,
    detents: Vec<Detent>,
) -> Router {
    let state = Arc::new(ServerState {
        ctx,
        oracle: oracle.map(Arc::new),
        percentile_table,
        detents,
        density_semaphore: Arc::new(Semaphore::new(1)),
    });
    create_router_with_state(state)
}
```

- [ ] **Step 3: Add a helper that builds the four detents, with a failing test**

Add to `solver/src/server.rs` a pure builder plus a test. The builder loads a table via the standalone loader and precomputes exp-scores; it logs and skips a detent whose file is missing (the endpoint degrades to fewer detents rather than failing to boot).

```rust
/// Load the four non-neutral detent tables from disk. Missing files are logged
/// and skipped (the endpoint then returns fewer detents). Neutral is not here.
pub fn load_detents(ctx: &YatzyContext) -> Vec<Detent> {
    const NON_NEUTRAL_THETAS: [f32; 4] = [-0.08, -0.04, 0.04, 0.08];
    let mut detents = Vec::new();
    for &theta in &NON_NEUTRAL_THETAS {
        let path = crate::storage::state_file_path(theta);
        match crate::storage::load_state_values_standalone(&path) {
            Some(sv) => {
                let exp_scores = precompute_exp_scores(ctx, theta);
                detents.push(Detent { theta, sv, exp_scores });
                println!("Loaded risk detent theta={} from {}", theta, path);
            }
            None => {
                eprintln!("Risk detent theta={} not loaded (missing {})", theta, path);
            }
        }
    }
    detents
}
```

Add a test (data-gated: needs the real tables, so `#[ignore]` like `test_precomputed.rs`):

```rust
#[cfg(test)]
mod detent_tests {
    use super::*;
    use crate::phase0_tables;

    #[test]
    #[ignore] // needs data/strategy_tables/all_states_theta_*.bin
    fn test_load_detents_finds_all_four() {
        let mut ctx = YatzyContext::new_boxed();
        phase0_tables::precompute_lookup_tables(&mut ctx);
        let detents = load_detents(&ctx);
        assert_eq!(detents.len(), 4, "expected all four detent tables present");
        let thetas: Vec<f32> = detents.iter().map(|d| d.theta).collect();
        assert_eq!(thetas, vec![-0.08, -0.04, 0.04, 0.08]);
    }
}
```

- [ ] **Step 4: Run the (ignored) test explicitly to verify it passes with data**

Run: `cd solver && cargo test --features full test_load_detents_finds_all_four -- --ignored --nocapture 2>&1 | tail -15`
Expected: PASS, printing "Loaded risk detent theta=..." four times. (If it fails to find files, confirm the working directory: run from repo root with `YATZY_BASE_PATH=.`, or `cd` to the repo root and use `cargo test --manifest-path solver/Cargo.toml`. The paths are relative to the process working directory, same as the server.)

- [ ] **Step 5: Load detents in the server binary**

In `solver/src/bin/server.rs`, after the percentile-table load and before building the `Arc`, add:

```rust
    // Load the four non-neutral risk detent tables for the /evaluate slider.
    let detents = yatzy::server::load_detents(&ctx);
    println!("{} of 4 risk detents loaded", detents.len());
```

Then change the router construction to pass them:

```rust
    let ctx = Arc::new(*ctx);
    let app = create_router_full(ctx, oracle, percentile_table, detents);
```

Note `load_detents(&ctx)` must be called BEFORE `let ctx = Arc::new(*ctx);` (it borrows the boxed ctx). Keep that order.

- [ ] **Step 6: Build and run the full suite**

Run: `cd solver && cargo fmt && cargo clippy --features full 2>&1 | tail -5 && cargo test --features full 2>&1 | tail -5`
Expected: builds clean; all non-ignored tests pass. There is exactly one other caller of `create_router_full`: `tests/test_api.rs:46` in `fn app_with_density()` (`create_router_full(get_ctx(), None, None)`). Add a fourth argument `Vec::new()` there:

```rust
fn app_with_density() -> axum::Router {
    create_router_full(get_ctx(), None, None, Vec::new())
}
```

- [ ] **Step 7: Commit**

```bash
git add solver/src/server.rs solver/src/bin/server.rs
git commit -m "Load the four non-neutral risk detent tables at startup

ServerState gains a detents vec (theta = -0.08/-0.04/+0.04/+0.08), each with
its log-domain table and precomputed exp(theta*score). Missing files are
skipped so the server still boots."
```

---

### Task 3: Return all five detents from `/evaluate`

**Files:**
- Modify: `solver/src/server.rs` (`handle_evaluate`: build the neutral response as today, plus a `detents` array)

**Interfaces:**
- Consumes: `compute_roll_response` (neutral), `compute_roll_response_theta` (Task 1), `ServerState.detents` (Task 2).
- Produces: the JSON response gains a `detents` array of five objects (index order θ = −0.08, −0.04, 0, +0.04, +0.08), each with the same fields as the flat response. The flat top-level fields remain and equal the Neutral entry.

- [ ] **Step 1: Write a failing integration test**

Add to `solver/tests/test_api.rs` (data-gated `#[ignore]`; it needs the real detent tables). It reuses the file's existing helpers: `get_ctx()` (computes the neutral table inline), `evaluate_request(body)`, `body_json(...)`, and the `oneshot` pattern. `load_detents` takes `&YatzyContext`; `&ctx` where `ctx: Arc<YatzyContext>` deref-coerces. Add at the end of the file:

```rust
#[tokio::test]
#[ignore] // needs data/strategy_tables detent files (all_states_theta_*.bin)
async fn evaluate_returns_five_detents_neutral_matches_flat() {
    let ctx = get_ctx(); // neutral table computed inline by setup_ctx
    let detents = yatzy::server::load_detents(&ctx);
    assert_eq!(detents.len(), 4, "all four detent tables must be present");
    let app = create_router_full(ctx, None, None, detents);

    let body = serde_json::json!({
        "dice": [3, 3, 4, 5, 6],
        "upper_score": 0,
        "scored_categories": 0,
        "rerolls_remaining": 2,
    });
    let resp = app.oneshot(evaluate_request(body)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let json = body_json(resp.into_body()).await;

    // Five detents in theta order.
    let ds = json["detents"].as_array().expect("detents array");
    assert_eq!(ds.len(), 5);
    let thetas: Vec<f64> = ds.iter().map(|d| d["theta"].as_f64().unwrap()).collect();
    assert_eq!(thetas, vec![-0.08, -0.04, 0.0, 0.04, 0.08]);

    // Neutral detent (index 2) equals the flat top-level fields (backward compat).
    assert_eq!(ds[2]["optimal_mask"], json["optimal_mask"]);
    assert_eq!(ds[2]["optimal_category"], json["optimal_category"]);

    // Documented decision flip for [3,3,4,5,6]: neutral keeps a pair of 3s;
    // aggressive (+0.04, index 3) keeps 5,6 for the large straight. Decode the
    // reroll mask (bit set = reroll) into the kept dice values.
    let kept = |mask: u64| -> std::collections::BTreeSet<i32> {
        let d = [3, 3, 4, 5, 6];
        (0..5usize).filter(|i| mask & (1 << i) == 0).map(|i| d[i]).collect()
    };
    let neutral_kept = kept(ds[2]["optimal_mask"].as_u64().unwrap());
    let aggr_kept = kept(ds[3]["optimal_mask"].as_u64().unwrap());
    assert_ne!(neutral_kept, aggr_kept, "risk should change the kept dice");
    assert!(
        aggr_kept.contains(&5) && aggr_kept.contains(&6),
        "aggressive keeps 5,6 for the straight; got {:?}",
        aggr_kept
    );
}
```

This is the correctness gate for the CE recursion against the real policy. The theory (`theory/strategy/risk-sensitive-strategy.md`) documents this specific flip. If the observed aggressive keep is not `{5,6}` when run against the real table, print the masks, reconcile against the theory doc, and adjust the assertion to the documented optimal, but the two detents MUST differ (that is the load-bearing check).

- [ ] **Step 2: Run it to verify it fails**

Run: `cd solver && cargo test --features full evaluate_returns_five_detents -- --ignored --nocapture 2>&1 | tail -20`
Expected: FAIL. It compiles (all helpers already exist in `test_api.rs`) but the assertion `json["detents"].as_array().expect(...)` panics because the handler does not emit a `detents` field yet.

- [ ] **Step 3: Build the `detents` array in the handler**

In `solver/src/server.rs`, refactor `handle_evaluate`. Extract the "response object from a `RollResponse`" logic into a local closure, compute the five detents in θ order, keep the flat fields from the neutral one. Replace the body after input validation with:

```rust
    // Serialize one RollResponse into the flat JSON object shape.
    let to_obj = |resp: &crate::api_computations::RollResponse, theta: f32| -> serde_json::Value {
        let categories: Vec<serde_json::Value> = resp
            .categories
            .iter()
            .map(|cat| {
                serde_json::json!({
                    "id": cat.id,
                    "name": cat.name,
                    "score": cat.score,
                    "available": cat.available,
                    "ev_if_scored": if cat.ev_if_scored.is_finite() { cat.ev_if_scored } else { 0.0 },
                })
            })
            .collect();
        let mut obj = serde_json::json!({
            "theta": theta,
            "categories": categories,
            "optimal_category": resp.optimal_category,
            "optimal_category_ev": resp.optimal_category_ev,
            "state_ev": resp.state_ev,
        });
        if let Some(mask_evs) = resp.mask_evs {
            obj["mask_evs"] = serde_json::json!(mask_evs);
            obj["optimal_mask"] = serde_json::json!(resp.optimal_mask.unwrap());
            obj["optimal_mask_ev"] = serde_json::json!(resp.optimal_mask_ev.unwrap());
        }
        obj
    };

    // Neutral (theta = 0): the existing EV path, unchanged.
    let neutral = compute_roll_response(
        &state.ctx,
        req.upper_score,
        req.scored_categories,
        &req.dice,
        req.rerolls_remaining,
    );

    // Build EXACTLY five detents in fixed theta order. A missing detent table
    // falls back to the neutral response (that risk level then behaves as
    // Neutral), so the array is always length 5. The frontend relies on that.
    const DETENT_THETAS: [f32; 5] = [-0.08, -0.04, 0.0, 0.04, 0.08];
    let detent_objs: Vec<serde_json::Value> = DETENT_THETAS
        .iter()
        .map(|&theta| {
            if theta == 0.0 {
                return to_obj(&neutral, 0.0);
            }
            match state.detents.iter().find(|d| (d.theta - theta).abs() < 1e-6) {
                Some(d) => {
                    let r = crate::api_computations::compute_roll_response_theta(
                        &state.ctx, d.sv.as_slice(), &d.exp_scores, d.theta,
                        req.upper_score, req.scored_categories, &req.dice, req.rerolls_remaining,
                    );
                    to_obj(&r, d.theta)
                }
                // Missing table: emit neutral values under this detent's theta,
                // so the array stays length 5 and the level degrades to Neutral.
                None => to_obj(&neutral, theta),
            }
        })
        .collect();

    // Flat fields = neutral (backward compatible), plus the detents array.
    let mut result = to_obj(&neutral, 0.0);
    result["detents"] = serde_json::Value::Array(detent_objs);

    Ok(Json(result))
```

Keep the input-validation block above this unchanged. Remove the old response-building lines it replaces. The `to_obj` closure adds a `theta` field to the flat object too; that is harmless for the old frontend (it ignores unknown fields).

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `cd solver && cargo test --features full evaluate_returns_five_detents -- --ignored --nocapture 2>&1 | tail -25`
Expected: PASS. If the flip assertion fails, inspect the printed optimal masks, confirm against the theory doc, and correct the expected kept-sets (the two detents must differ; that is the real assertion).

- [ ] **Step 5: Regression, existing API tests still pass**

Run: `cd solver && cargo test --features full --test test_api 2>&1 | tail -10`
Expected: all existing (non-ignored) API tests pass. The flat fields are unchanged, so tests asserting `optimal_mask`/`mask_evs`/`categories` still hold.

- [ ] **Step 6: fmt, clippy, commit**

Run: `cd solver && cargo fmt && cargo clippy --features full 2>&1 | tail -5`

```bash
git add solver/src/server.rs solver/tests/test_api.rs
git commit -m "Return all five risk detents from /evaluate

Response keeps its flat fields (equal to Neutral, backward compatible) and
adds a detents array in theta order (-0.08..+0.08), neutral via the existing
EV path and the four others via the CE recursion."
```

---

### Task 4: Deploy assembly + docs

**Files:**
- Modify: `deploy/deploy.sh` (copy the four detent tables into the backend image context)
- Modify: `solver/CLAUDE.md` (API Reference: note the `detents` array)

**Interfaces:** none produced; this task makes the feature shippable and documented.

- [ ] **Step 1: Ship the detent tables in the backend image**

In `deploy/deploy.sh`, in the "Backend context" assembly (right after the line that copies `all_states.bin`), add:

```bash
# Risk detent tables for the /evaluate slider (theta = -0.08/-0.04/+0.04/+0.08).
for T in -0.080 -0.040 0.040 0.080; do
    SRC="$PROJECT_ROOT/data/strategy_tables/all_states_theta_${T}.bin"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$BACKEND_CTX/data/strategy_tables/"
    else
        echo "WARNING: detent table $SRC missing; that risk level will be unavailable"
    fi
done
echo "✓ Risk detent tables included"
```

- [ ] **Step 2: Verify the deploy script is still valid bash**

Run: `bash -n deploy/deploy.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 3: Update the API reference**

In `solver/CLAUDE.md`, in the API Reference table row for `/evaluate`, and the surrounding notes, document that the response now includes a `detents` array (five entries, θ = −0.08, −0.04, 0, +0.04, +0.08) whose value fields are certainty-equivalent points, and that the flat top-level fields remain and equal the Neutral detent. Keep it to two or three sentences. No emdashes.

- [ ] **Step 4: Commit**

```bash
git add deploy/deploy.sh solver/CLAUDE.md
git commit -m "Ship risk detent tables in the backend image; document detents response"
```

---

## Definition of Done

- [ ] `cd solver && cargo test --features full` passes (existing tests + the three `test_ce_*`).
- [ ] `cargo test --features full -- --ignored` passes the data-gated detent tests (`test_load_detents_finds_all_four`, `evaluate_returns_five_detents_neutral_matches_flat`) when the tables are present.
- [ ] `cargo fmt --check` and `cargo clippy --features full` are clean.
- [ ] A local `just serve` + `curl -s -XPOST localhost:9000/evaluate -d '{"dice":[3,3,4,5,6],"upper_score":0,"scored_categories":0,"rerolls_remaining":2}'` returns a `detents` array of five, the flat fields equal `detents[2]`, and the aggressive detent's `optimal_mask` differs from neutral's.
- [ ] `bash -n deploy/deploy.sh` is clean and the four tables are copied into the backend context.
- [ ] `solver/CLAUDE.md` API Reference documents the `detents` array.
- [ ] The Neutral detent's numbers are unchanged from before this plan (the live frontend keeps working).

## Contract with the frontend plan

The frontend plan (`docs/superpowers/plans/2026-07-19-theta-risk-slider-frontend.md`) consumes this exact shape:

- Response is backward compatible: the flat top-level fields equal the Neutral detent, so the currently-live frontend keeps working and the new frontend can ship independently.
- `detents` is **always length 5**, fixed index order θ = −0.08, −0.04, 0, +0.04, +0.08 (index 2 = Neutral). A missing table degrades that level to Neutral values but keeps the array length 5.
- Each detent entry has the same field names as the flat response (`mask_evs`, `optimal_mask`, `optimal_mask_ev`, `categories[].ev_if_scored`, `optimal_category`, `optimal_category_ev`, `state_ev`) plus `theta`. At non-neutral detents these numbers are certainty-equivalent points. So a detent entry is structurally an `EvaluateResponse`, and the frontend selects `detents[riskLevel]` as `lastEvalResponse` with no field remapping.
