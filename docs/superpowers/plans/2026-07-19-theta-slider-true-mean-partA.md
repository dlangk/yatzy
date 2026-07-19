# Theta Slider Part A: True Mean Everywhere — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the risk slider's certainty-equivalent (CE) value numbers with the true **mean under the selected detent's policy** `V_πθ(S)`, so "Projected value" peaks at Neutral and dips at both extremes (matching the theory), while the θ-policy's decisions/advice stay exactly as shipped.

**Architecture:** A new offline precompute builds a companion mean table per non-neutral detent (`all_states_mean_theta_{θ}.bin`) by backward induction that follows the θ-policy (reusing `density::transitions::compute_transitions`) while accumulating expected score. The server loads these into each `Detent` and computes `/evaluate` value fields as means (decisions unchanged). The frontend relabels the now-honest numbers and handles the signed mismatch cost. Response shape is unchanged.

**Tech Stack:** Rust (solver: rayon backward induction, axum API), vanilla TypeScript (frontend), Docker multi-stage deploy.

## Global Constraints

- **Neutral (θ=0) is byte-identical to today.** `V_π0 = V*`, so Neutral reuses `all_states.bin`; no new θ=0 table. The mean path at θ=0 must equal the existing EV path exactly (regression gate).
- **Decisions are unchanged.** `optimal_mask`, `optimal_category`, advice, and mismatch marks remain the utility argmax/argmin from the θ-table, computed by the shipped `compute_roll_response_theta`. Only the displayed **value numbers** change (CE → mean).
- **Detent set is fixed:** θ ∈ {−0.08, −0.04, 0.00, +0.04, +0.08}; the four non-neutral thetas each get a mean table. File name format: `all_states_mean_theta_{θ:.3}.bin` (e.g. `all_states_mean_theta_-0.080.bin`), same 16-byte header + `f32[4,194,304]` format as `all_states.bin`.
- **Hot-path rules apply** (`.claude/rules/rust-hot-paths.md`): run `just bench-check` before and after any solver change; revert on regression. The new precompute is offline (not a request hot path); `/evaluate` stays microsecond-scale.
- **`V_πθ(S) ≤ V*(S)` at every state.** The Projected value must never exceed the Neutral projection for the same (state, dice). This is a correctness invariant, not just a display goal.
- **No emdashes** in prose/comments/docs (CLAUDE.md writing style).
- After changing solver code, the CLAUDE.md staleness rule applies to `all_states_theta_*.bin`; the new mean tables are derived from those, so rebuild mean tables whenever the θ tables are rebuilt.

---

## File Structure

- `solver/src/density/transitions.rs` — make `compute_group6_with_category` and `compute_reroll_with_decisions` `pub(crate)`; add `expected_score_under_policy`.
- `solver/src/policy_mean.rs` (new) — `compute_policy_mean_table(ctx, sv_theta, theta) -> Vec<f32>`: backward mean induction. Registered in `lib.rs`.
- `solver/src/storage.rs` — add `save_state_values_standalone(sv, theta, filename)` and `mean_state_file_path(theta)` helper (mirrors existing writer + `state_file_path`).
- `solver/src/bin/policy_mean.rs` (new) — CLI: build mean tables for the four detents (or `--theta θ`), skip existing unless `--force`.
- `solver/src/api_computations.rs` — add `compute_roll_response_mean(...)`.
- `solver/src/server.rs` — extend `Detent` with `mean_sv: StateValues`; load mean tables in `load_detents`; switch the detents array in `handle_evaluate` to `compute_roll_response_mean` (Neutral still via the plain EV `compute_roll_response`).
- `justfile` (root) — `precompute-means` recipe.
- `deploy/deploy-full.sh`, `deploy/Dockerfile.backend` — copy the four mean tables into the backend image.
- `solver/CLAUDE.md`, `frontend/CLAUDE.md` — doc updates.
- Frontend: `components/RiskSlider.ts`, `components/EvalPanel.ts`, `components/Scorecard.ts`, `components/DeltaBox.ts`, `components/DecisionLog.ts`, plus `reducer.ts`/`types.ts` if cost-sign handling needs a helper. `mask.ts` if the cost line needs sign handling.

---

## Task 1: `expected_score_under_policy` helper

**Files:**
- Modify: `solver/src/density/transitions.rs` (make two fns `pub(crate)`, add one fn + test)

**Interfaces:**
- Produces: `pub(crate) fn expected_score_under_policy(ctx: &YatzyContext, sv: &[f32], mean_sv: &[f32], up_score: i32, scored: i32, theta: f32) -> f32`
- Produces (visibility change): `pub(crate) fn compute_group6_with_category(...) -> ([f32;252],[u8;252])`, `pub(crate) fn compute_reroll_with_decisions(...) -> ([f32;252],[u16;252])`
- Consumes: existing `compute_transitions(ctx, sv, up, scored, theta) -> Vec<StateTransition>`

- [ ] **Step 1: Change visibility.** Change `fn compute_group6_with_category` (line ~34) and `fn compute_reroll_with_decisions` (line ~132) to `pub(crate) fn`. Also export the `KEEP_ALL` const as `pub(crate) const KEEP_ALL: u16`.

- [ ] **Step 2: Write the failing test** (append to the `#[cfg(test)] mod tests` block):

```rust
#[test]
fn test_expected_score_under_policy_terminal_adjacent() {
    // With one category left (Yatzy) and a zero continuation implied by terminal,
    // the EV-policy (theta=0) expected score from a state equals sv[state] (V*),
    // because V_pi0 == V*. We build a tiny mean_sv covering only the terminal level.
    let mut ctx = YatzyContext::new_boxed();
    crate::phase0_tables::precompute_lookup_tables(&mut ctx);
    crate::state_computation::compute_all_state_values(&mut ctx);
    let sv = ctx.state_values.as_slice();

    // mean_sv: terminal states = bonus(up), everything else 0 (only terminal is read
    // as a successor when exactly one category remains).
    let mut mean_sv = vec![0.0f32; NUM_STATES];
    let all_scored = (1u32 << CATEGORY_COUNT) - 1;
    for up in 0..=63usize {
        mean_sv[state_index(up, all_scored as usize)] = if up >= 63 { 50.0 } else { 0.0 };
    }

    // State with only Yatzy remaining, up=0.
    let scored = (all_scored ^ (1 << CATEGORY_YATZY)) as i32;
    let got = expected_score_under_policy(&ctx, sv, &mean_sv, 0, scored, 0.0);
    let want = sv[state_index(0, scored as usize)];
    assert!((got - want).abs() < 0.05, "policy-eval mean {} vs V* {}", got, want);
}
```

- [ ] **Step 3: Run it to verify it fails** (function not defined):
Run: `cd solver && cargo test density::transitions::tests::test_expected_score_under_policy_terminal_adjacent`
Expected: FAIL (unresolved name).

- [ ] **Step 4: Implement** (add near the bottom of the module, before `#[cfg(test)]`):

```rust
/// Expected final score from (up_score, scored) under the theta-policy defined by
/// `sv` (log-utility L(S) for theta!=0, EV table V*(S) for theta==0), reading
/// continuation means from `mean_sv` (the V_pi_theta table being built).
///
/// Reuses `compute_transitions`, which already encodes the theta-policy's within-turn
/// decisions and returns deduplicated (next_state, points, prob) with sum(prob)=1.
/// Then folds the mean: V_pi_theta(S) = sum_t prob_t * (points_t + mean_sv[next_t]).
pub(crate) fn expected_score_under_policy(
    ctx: &YatzyContext,
    sv: &[f32],
    mean_sv: &[f32],
    up_score: i32,
    scored: i32,
    theta: f32,
) -> f32 {
    let trans = compute_transitions(ctx, sv, up_score, scored, theta);
    let mut ev = 0.0f64;
    for t in &trans {
        ev += t.prob * (t.points as f64 + mean_sv[t.next_state as usize] as f64);
    }
    ev as f32
}
```

- [ ] **Step 5: Run test to verify it passes:**
Run: `cd solver && cargo test density::transitions::tests::test_expected_score_under_policy_terminal_adjacent`
Expected: PASS.

- [ ] **Step 6: Commit** — `git add solver/src/density/transitions.rs && git commit -m "feat(solver): expected_score_under_policy for policy-eval mean"`

---

## Task 2: Backward mean-induction table build

**Files:**
- Create: `solver/src/policy_mean.rs`
- Modify: `solver/src/lib.rs` (add `pub mod policy_mean;`)
- Modify: `solver/src/storage.rs` (add `mean_state_file_path` + `save_state_values_standalone`)

**Interfaces:**
- Produces: `pub fn compute_policy_mean_table(ctx: &YatzyContext, sv_theta: &[f32], theta: f32) -> Vec<f32>` (length `NUM_STATES`, `V_πθ` for all states).
- Produces: `pub fn mean_state_file_path(theta: f32) -> String` → `data/strategy_tables/all_states_mean_theta_{theta:.3}.bin`.
- Produces: `pub fn save_state_values_standalone(sv: &[f32], theta: f32, filename: &str)` (writes the standard 16-byte header with `theta` + `f32[NUM_STATES]`; mirror the body of the existing `save_all_state_values`).
- Consumes: `expected_score_under_policy` (Task 1), `ctx.reachable`, `state_index`, `state_computation.rs` iteration pattern (mirror its level loop + `AtomicPtr` writes).

- [ ] **Step 1: Add storage helpers.** In `solver/src/storage.rs`, read the existing `save_all_state_values` and `state_file_path`. Add `mean_state_file_path(theta)` (same as `state_file_path` but with the `all_states_mean_theta_` prefix) and `save_state_values_standalone(sv, theta, filename)` that writes the identical header (magic `0x59545A53`, current version, `theta` bits) followed by the `f32` slice. Do not change existing functions.

- [ ] **Step 2: Write the failing test** (in `solver/src/policy_mean.rs`, `#[cfg(test)]`):

```rust
#[test]
#[ignore] // ~minutes: full backward mean induction at theta=0
fn test_policy_mean_theta0_matches_v_star() {
    // V_pi0 == V*, so building the mean table with the EV table as the decision
    // table must reproduce all_states.bin at every reachable state.
    let mut ctx = crate::types::YatzyContext::new_boxed();
    crate::phase0_tables::precompute_lookup_tables(&mut ctx);
    crate::state_computation::compute_all_state_values(&mut ctx);
    let v_star = ctx.state_values.as_slice().to_vec();

    let mean = compute_policy_mean_table(&ctx, &v_star, 0.0);

    let all_scored = (1u32 << CATEGORY_COUNT) - 1;
    let mut max_abs = 0.0f32;
    for scored in 0..(1u32 << CATEGORY_COUNT) {
        let upper_mask = (scored & 0x3F) as usize;
        for up in 0..=63usize {
            if ctx.reachable[upper_mask][up] {
                let i = state_index(up, scored as usize);
                max_abs = max_abs.max((mean[i] - v_star[i]).abs());
            }
        }
        let _ = all_scored;
    }
    assert!(max_abs < 0.05, "max |V_pi0 - V*| = {} over reachable states", max_abs);
}
```

- [ ] **Step 3: Run to verify it fails:**
Run: `cd solver && cargo test policy_mean::tests::test_policy_mean_theta0_matches_v_star -- --ignored`
Expected: FAIL (unresolved `compute_policy_mean_table`).

- [ ] **Step 4: Implement `compute_policy_mean_table`.** Mirror `state_computation.rs::compute_all_state_values_impl`'s level loop and `AtomicPtr` scatter exactly, but the per-state value is `expected_score_under_policy(ctx, sv_theta, &mean_being_built, up, scored, theta)`.

```rust
//! Backward policy-evaluation: expected final score under the theta-policy.
//!
//! Companion to `state_computation.rs`. Where that maximizes (EV) or computes
//! log-utility, this follows a fixed theta-policy (decisions read from `sv_theta`)
//! and accumulates the expected SCORE, giving V_pi_theta(S) = mean final score if
//! you play the theta-optimal policy from S. At theta=0, V_pi0 == V*.

use std::sync::atomic::{AtomicPtr, Ordering};
use rayon::prelude::*;

use crate::constants::*;
use crate::density::transitions::expected_score_under_policy;
use crate::types::YatzyContext;

pub fn compute_policy_mean_table(ctx: &YatzyContext, sv_theta: &[f32], theta: f32) -> Vec<f32> {
    let mut mean = vec![0.0f32; NUM_STATES];
    let all_scored = ((1u32 << CATEGORY_COUNT) - 1) as usize;

    // Terminal level (all 15 scored): value = upper bonus only.
    for up in 0..=63usize {
        mean[state_index(up, all_scored)] = if up >= 63 { 50.0 } else { 0.0 };
    }
    // Topological padding (up 64..127) = capped value at up=63.
    {
        let capped = mean[state_index(63, all_scored)];
        let base = state_index(0, all_scored);
        for pad in 64..STATE_STRIDE {
            mean[base + pad] = capped;
        }
    }

    // Levels 14..0: value = expected_score_under_policy reading mean at level+1.
    for num_scored in (0..=14).rev() {
        // Collect reachable scored masks at this level.
        let mut scored_masks: Vec<i32> = Vec::new();
        for scored in 0..(1u32 << CATEGORY_COUNT) {
            if scored.count_ones() == num_scored as u32 {
                let upper_mask = (scored & 0x3F) as usize;
                if (0..=63usize).any(|up| ctx.reachable[upper_mask][up]) {
                    scored_masks.push(scored as i32);
                }
            }
        }

        let ptr = AtomicPtr::new(mean.as_mut_ptr());
        // SAFETY of reads: expected_score_under_policy only reads mean at successor
        // states (level num_scored+1), already written. Writes go to distinct
        // (up, scored) indices per mask; each mask handled by one worker.
        scored_masks.par_iter().for_each(|&scored| {
            let upper_mask = (scored & 0x3F) as usize;
            let mean_slice = unsafe { std::slice::from_raw_parts(ptr.load(Ordering::Relaxed), NUM_STATES) };
            let base = state_index(0, scored as usize);
            let p = ptr.load(Ordering::Relaxed);
            let mut capped = 0.0f32;
            for up in 0..=63usize {
                if ctx.reachable[upper_mask][up] {
                    let v = expected_score_under_policy(ctx, sv_theta, mean_slice, up as i32, scored, theta);
                    unsafe { *p.add(state_index(up, scored as usize)) = v; }
                    if up == 63 { capped = v; }
                }
            }
            // Fill padding for this mask with the up=63 value (or the highest reachable up if 63 unreachable).
            if capped == 0.0 {
                // find highest reachable up as the capped value
                for up in (0..=63usize).rev() {
                    if ctx.reachable[upper_mask][up] {
                        capped = unsafe { *p.add(state_index(up, scored as usize)) };
                        break;
                    }
                }
            }
            for pad in 64..STATE_STRIDE {
                unsafe { *p.add(base + pad) = capped; }
            }
        });
    }

    mean
}
```

Note for the implementer: the aliasing of `mean` via both a `&raw` read slice and `AtomicPtr` writes within one `par_iter` mirrors `state_computation.rs` (reads at level+1, writes at level, disjoint). Keep the `SAFETY` comments. If `cargo clippy` objects to the read-slice/pointer aliasing, follow the exact pattern `state_computation.rs` uses (it extracts `sv_slice = ctx.state_values.as_slice()` for reads and an `AtomicPtr` for writes; here reads and writes are the same array but disjoint index sets, so build the read slice once from the raw pointer as shown).

- [ ] **Step 5: Run the ignored test to verify it passes:**
Run: `cd solver && cargo test policy_mean::tests::test_policy_mean_theta0_matches_v_star -- --ignored --nocapture`
Expected: PASS, `max |V_pi0 - V*|` well under 0.05.

- [ ] **Step 6: Add a fast (non-ignored) sanity test** so CI-tier `cargo test` covers the leaf math without a full build:

```rust
#[test]
fn test_policy_mean_last_two_levels_theta0() {
    // Build the mean table but only exercise cheaply: compare a handful of
    // level-13/14 states against V* (these depend only on the top two levels,
    // but compute_policy_mean_table builds all levels; keep this test's asserts
    // to a few states so runtime is dominated by the build. Marked non-ignored
    // only if the full build is < ~a few seconds on the dev machine; otherwise
    // move to #[ignore] and rely on the theta0 regression above.)
}
```
If the full build is too slow for the default suite, delete this stub and keep only the `#[ignore]` regression from Step 2 (the audit tier runs ignored tests via `just audit`). Decide based on measured build time; document the choice in a comment.

- [ ] **Step 7: Commit** — `git add solver/src/policy_mean.rs solver/src/lib.rs solver/src/storage.rs && git commit -m "feat(solver): backward policy-eval mean table (V_pi_theta)"`

---

## Task 3: `policy_mean` binary + justfile recipe

**Files:**
- Create: `solver/src/bin/policy_mean.rs`
- Modify: `justfile` (root)

**Interfaces:**
- Consumes: `compute_policy_mean_table`, `mean_state_file_path`, `save_state_values_standalone`, `storage::{load_state_values_standalone, state_file_path}`, `env_config`.

- [ ] **Step 1: Implement the binary.** Loosely mirror `solver/src/bin/precompute.rs` startup (read it for the ctx-init + `YATZY_BASE_PATH` pattern). Logic:

```rust
// For each theta in [-0.08, -0.04, 0.04, 0.08] (or a single --theta arg):
//   - if mean_state_file_path(theta) exists and no --force: skip with a notice.
//   - load the decision table state_file_path(theta) via load_state_values_standalone;
//     if missing, print a clear error naming the file and continue.
//   - mean = compute_policy_mean_table(&ctx, sv_theta.as_slice(), theta);
//   - save_state_values_standalone(&mean, theta, &mean_state_file_path(theta));
//   - print V_pi_theta(start) = mean[state_index(0,0)] and confirm <= V*(start).
```
Include `--force` and optional `--theta θ`. Print each detent's start-state mean so the operator can eyeball that it is below the Neutral 247 and dips at both extremes.

- [ ] **Step 2: Add the justfile recipe.** Read the existing `precompute`/`sweep` recipes for the env pattern, then add:

```make
# Build V_pi_theta mean tables for the four risk-slider detents
precompute-means *ARGS:
    cd solver && YATZY_BASE_PATH=.. cargo run --release --bin policy_mean -- {{ARGS}}
```

- [ ] **Step 3: Build + run to generate the four tables** (requires the four `all_states_theta_*.bin` to exist; build them first with `just sweep` or `yatzy-precompute --theta` if absent):
Run: `just precompute-means`
Expected: four `data/strategy_tables/all_states_mean_theta_{-0.080,-0.040,0.040,0.080}.bin` written; printed start means all `< 247` and `-0.08`/`+0.08` both below `-0.04`/`+0.04` respectively (quadratic dip).

- [ ] **Step 4: Commit** — `git add solver/src/bin/policy_mean.rs justfile && git commit -m "feat(solver): policy_mean binary + precompute-means recipe"`

---

## Task 4: `compute_roll_response_mean` in the API layer

**Files:**
- Modify: `solver/src/api_computations.rs` (add fn + tests)

**Interfaces:**
- Produces: `pub fn compute_roll_response_mean(ctx, sv_theta: &[f32], mean_sv: &[f32], exp_scores: &ExpScores, theta: f32, upper_score: i32, scored_categories: i32, dice: &[i32;5], rerolls_remaining: i32) -> RollResponse`
- Consumes: `compute_roll_response_theta` (decisions), `compute_group6_with_category` + `compute_reroll_with_decisions` + `KEEP_ALL` from `density::transitions` (mean-track continuation policy), `compute_expected_value_for_reroll_mask` (widget_solver), `update_upper_score`, `find_dice_set_index`.

**Design (decisions from CE path; values are means):**
1. `let ce = compute_roll_response_theta(ctx, sv_theta, exp_scores, theta, up, scored, dice, rerolls);` — gives `optimal_mask`, `optimal_category`, and the sorted/remapped structure. Keep `ce.optimal_mask` and `ce.optimal_category` verbatim.
2. Sort dice → `ds_index` and `sort_map` (same block as `compute_roll_response_theta`).
3. Mean track:
   - `let (e_ds_0_util, best_cat) = compute_group6_with_category(ctx, sv_theta, up, scored, theta, theta < 0.0);`
   - `m_ds_0[ds] = score(ds, best_cat[ds]) + mean_sv[succ(ds, best_cat[ds])]` for all 252.
   - Per-category display mean (for the current `ds_index`): `ev_if_scored = score(ds_index, c) + mean_sv[succ(ds_index, c)]` for available `c`, else `NEG_INFINITY`. (Direct action mean; no decision.)
   - `optimal_category_ev` (mean) = `score(ds_index, ce.optimal_category) + mean_sv[succ]` (use the CE path's chosen category so the reported optimal mean matches the recommendation exactly).
   - If `rerolls == 2`: `let (_e1, best_keep_1) = compute_reroll_with_decisions(ctx, &e_ds_0_util, theta, theta<0.0);` then `m_ds_1[ds] = if best_keep_1[ds]==KEEP_ALL { m_ds_0[ds] } else { sum over keep row: vals[k]*m_ds_0[cols[k]] }`. `m_ds_for_masks = m_ds_1`. Else (`rerolls==1`) `m_ds_for_masks = m_ds_0`.
   - Mask means (sorted): `sorted_mask_means[mask] = compute_expected_value_for_reroll_mask(ctx, ds_index, &m_ds_for_masks, mask)`. Remap sorted→original via `sort_map` into `mask_evs`.
   - `optimal_mask_ev` (mean) = `mask_evs[ce.optimal_mask]` (mask means, original order).
   - `state_ev` (mean): if `rerolls>0 && ce.optimal_mask_ev >= ce.optimal_category_ev` → `optimal_mask_ev` else `optimal_category_ev`. (Top-level winner decided by CE; value reported as its mean.)
4. Build the returned `RollResponse` with **decision fields from `ce`** (`optimal_mask`, `optimal_category`) and **all value fields from the mean track** (`mask_evs`, `optimal_mask_ev`, `categories[].ev_if_scored`, `optimal_category_ev`, `state_ev`). `mask_evs`/`optimal_mask`/`optimal_mask_ev` are `None` when `rerolls==0` (as in `compute_roll_response`).

- [ ] **Step 1: Write the failing tests** (append to the `#[cfg(test)]` block):

```rust
#[test]
fn test_mean_equals_ev_at_theta0() {
    // At theta=0, mean_sv == V* and decisions == EV-optimal, so compute_roll_response_mean
    // must equal compute_roll_response field-for-field.
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    crate::state_computation::compute_all_state_values(&mut ctx);
    let sv = ctx.state_values.as_slice();
    let exp_scores = crate::batched_solver::precompute_exp_scores(&ctx, 0.0);
    let dice = [3, 3, 4, 5, 6];
    let ev = compute_roll_response(&ctx, 7, 0, &dice, 2);
    // theta must be nonzero for compute_roll_response_theta's inv_theta; use the
    // documented theta->0 handling: compute_roll_response_mean must special-case
    // theta==0 to delegate to compute_roll_response (see Step 2). Assert that delegation:
    let mean = compute_roll_response_mean(&ctx, sv, sv, &exp_scores, 0.0, 7, 0, &dice, 2);
    assert_eq!(mean.optimal_mask, ev.optimal_mask);
    assert_eq!(mean.optimal_category, ev.optimal_category);
    assert!((mean.state_ev - ev.state_ev).abs() < 1e-4);
    let (me, ee) = (mean.mask_evs.unwrap(), ev.mask_evs.unwrap());
    for m in 0..32 { assert!((me[m] - ee[m]).abs() < 1e-4, "mask {} {} vs {}", m, me[m], ee[m]); }
}

#[test]
fn test_mean_decisions_match_ce_and_value_le_neutral() {
    // Non-neutral: decisions identical to the CE path; mean state_ev <= EV-optimal state_ev.
    let mut ctx = YatzyContext::new_boxed();
    phase0_tables::precompute_lookup_tables(&mut ctx);
    crate::state_computation::compute_all_state_values(&mut ctx);
    let v_star = ctx.state_values.as_slice().to_vec();
    // For a real check we need the theta decision table + its mean table. Absent data,
    // assert the structural contract with sv_theta = V* and mean_sv = V* (a valid
    // stand-in that keeps decisions = EV and means = V*): state_ev must be finite and
    // equal for both. The DATA-backed cross-check lives in tests/test_api.rs (Task 5).
    let exp_scores = crate::batched_solver::precompute_exp_scores(&ctx, 0.04);
    let dice = [2, 2, 5, 5, 6];
    let ce = compute_roll_response_theta(&ctx, &v_star, &exp_scores, 0.04, 10, 0, &dice, 2);
    let mean = compute_roll_response_mean(&ctx, &v_star, &v_star, &exp_scores, 0.04, 10, 0, &dice, 2);
    assert_eq!(mean.optimal_mask, ce.optimal_mask);
    assert_eq!(mean.optimal_category, ce.optimal_category);
    assert!(mean.state_ev.is_finite());
}
```

- [ ] **Step 2: Implement `compute_roll_response_mean`** per the Design above. Special-case `theta == 0.0` to `return compute_roll_response(ctx, upper_score, scored_categories, dice, rerolls_remaining);` (avoids the `1/theta` in the CE call and guarantees Neutral byte-identity). For `theta != 0.0`, follow steps 1-4.

- [ ] **Step 3: Run the tests:**
Run: `cd solver && cargo test api_computations::tests::test_mean`
Expected: both PASS.

- [ ] **Step 4: `just bench-check`** (the API path changed; confirm no `/evaluate` regression — the mean function is only called for the four non-neutral detents, Neutral unchanged):
Run: `just bench-check`
Expected: PASS.

- [ ] **Step 5: Commit** — `git add solver/src/api_computations.rs && git commit -m "feat(solver): compute_roll_response_mean (true mean values, CE decisions)"`

---

## Task 5: Wire mean tables into `Detent` + `/evaluate`

**Files:**
- Modify: `solver/src/server.rs` (`Detent`, `load_detents`, `handle_evaluate`)
- Modify: `solver/tests/test_api.rs` (data-gated cross-check)
- Modify: `solver/CLAUDE.md` (API reference note)

**Interfaces:**
- Consumes: `compute_roll_response_mean`, `mean_state_file_path`, `load_state_values_standalone`.

- [ ] **Step 1: Extend `Detent`.** Add `pub mean_sv: StateValues` to the `Detent` struct (`server.rs:36`). In `load_detents` (`server.rs:84`), for each non-neutral theta load `mean_state_file_path(theta)` via `load_state_values_standalone` alongside the existing `sv`. If the mean table is missing, log a clear warning and **skip that detent** (same missing-table policy as today), so a partial deploy degrades to Neutral for that detent rather than serving CE.

- [ ] **Step 2: Switch the detents computation.** In `handle_evaluate` (`server.rs:~296`), the four non-neutral detent entries must call `compute_roll_response_mean(ctx, &d.sv, &d.mean_sv, &d.exp_scores, d.theta, ...)` instead of `compute_roll_response_theta`. The Neutral entry (index 2) stays on the plain EV `compute_roll_response` (unchanged), and the flat top-level fields still equal `detents[2]`.

- [ ] **Step 3: Data-gated cross-check test** in `tests/test_api.rs` (guarded by the existing `YATZY_REQUIRE_DATA` / data-presence pattern used by other tests there):

```rust
// With real detent tables present: for a nonzero detent, the mean state_ev must be
// <= the Neutral state_ev for the same (state, dice), and decisions must match the
// CE path. Skips cleanly when tables are absent.
```
Assert `mean_detent.state_ev <= neutral.state_ev + 1e-3` and `mean_detent.optimal_mask == ce_theta.optimal_mask`.

- [ ] **Step 4: Run API tests (with data if available):**
Run: `cd solver && cargo test --test test_api` and, if tables exist, `YATZY_REQUIRE_DATA=1 cargo test --test test_api`
Expected: PASS (or clean skip without data).

- [ ] **Step 5: Manual curl smoke test.** Start `just dev-backend`, then:
Run: `curl -s localhost:9000/evaluate -H 'content-type: application/json' -d '{"dice":[1,2,3,4,5],"upper_score":0,"scored_categories":0,"rerolls_remaining":2}' | jq '.detents[].state_ev'`
Expected: five numbers, index 2 (Neutral) is the max, and both ends are below it (e.g. roughly `~243, ~246, ~247.6, ~246, ~243`). Confirm no value exceeds Neutral.

- [ ] **Step 6: Update `solver/CLAUDE.md`.** In the API reference, change the `/evaluate` detents note: non-neutral detents now hold **mean-under-that-detent's-policy** values (expected scores), not certainty equivalents; `state_ev` peaks at Neutral. Note the four `all_states_mean_theta_*.bin` tables and the `precompute-means` recipe.

- [ ] **Step 7: Commit** — `git add solver/src/server.rs solver/tests/test_api.rs solver/CLAUDE.md && git commit -m "feat(solver): /evaluate detents return true mean, not CE"`

---

## Task 6: Frontend relabel + signed-cost handling

**Files:**
- Modify: `frontend/src/components/RiskSlider.ts` (off-neutral note copy)
- Modify: `frontend/src/components/EvalPanel.ts`, `frontend/src/components/Scorecard.ts` (value labels)
- Modify: `frontend/src/components/DeltaBox.ts`, `frontend/src/components/DecisionLog.ts` (signed cost)
- Modify: `frontend/src/mask.ts` if the dice-bar cost line needs sign handling
- Modify: `frontend/CLAUDE.md` (Risk slider paragraph)
- Test: `frontend/src/reducer.test.ts` / `mask.test.ts` as touched

**Context:** The `/evaluate` response **shape is unchanged**; only the numeric meaning of the value fields changed from CE to mean. So no store/reducer structural change is required. The work is copy + sign handling.

- [ ] **Step 1: Update the RiskSlider off-neutral note.** Current copy says values are "risk-adjusted (certainty-equivalent) points, not expected scores." Replace with copy stating the numbers are now the **expected score under the selected risk level's play**, which is highest at Neutral; higher-risk levels trade expected points for a wider outcome spread (the finish-range visualization of that spread is coming in a later update). Keep it to one or two short sentences. No emdashes.

- [ ] **Step 2: Relabel value fields.** In `EvalPanel.ts` and `Scorecard.ts`, off-neutral value labels should read as an expected-score term (e.g. "Exp. score (<level>)" / "Projected value"), not a certainty-equivalent term. At Neutral, labels are exactly as today. Grep for the strings introduced by the CE version ("certainty-equivalent", "risk-adj", "Value" off-neutral) and update them.

- [ ] **Step 3: Handle signed cost.** Off-neutral, the θ-policy's recommended action can have a **lower mean** than a non-recommended action, so `optimal_value - chosen_value` (the mismatch cost) can be `<= 0` when the user picks a higher-mean action than the risk advice. Decide the rendering (recommended): clamp the red "giving up points" bar/number at 0 (never show negative as red), and when the raw delta is negative, render it as `0.0` cost (the user did not lose expected points relative to the advice). Apply the same clamp in `DeltaBox.ts`, `DecisionLog.ts`, and the dice-bar cost line (`mask.ts`). At Neutral the delta is always `>= 0`, so behavior there is unchanged. Add/adjust a `mask.ts` unit test asserting the cost line clamps a negative raw delta to 0.

- [ ] **Step 4: Run frontend tests + typecheck:**
Run: `cd frontend && npm test && npm run lint && npx tsc --noEmit`
Expected: PASS.

- [ ] **Step 5: Update `frontend/CLAUDE.md`** Risk-slider paragraph: value fields are now expected scores under the selected detent's policy (mean), highest at Neutral; the trajectory projection still stays pinned to Neutral until Part B adds risk-shaped bands. Remove the "certainty-equivalent" wording.

- [ ] **Step 6: Commit** — `git add frontend/ && git commit -m "feat(frontend): risk slider shows true mean, clamps signed cost"`

---

## Task 7: Deploy assembly + docs

**Files:**
- Modify: `deploy/deploy-full.sh` (copy the four mean tables into the backend context)
- Modify: `deploy/Dockerfile.backend` (no change if it already `COPY data/ /app/data/`; verify the mean tables land under `data/strategy_tables/`)

- [ ] **Step 1: Copy mean tables into the build context.** In `deploy/deploy-full.sh`, extend the detent-table copy loop (currently copies `all_states_theta_{-0.080,-0.040,0.040,0.080}.bin`) to also copy `all_states_mean_theta_{-0.080,-0.040,0.040,0.080}.bin`, with the same "WARNING: missing" guard. These are ~16 MB each (~64 MB added).

- [ ] **Step 2: Verify the Dockerfile bakes them.** `Dockerfile.backend` copies `data/ /app/data/`; confirm the assembled `$BCTX/data/strategy_tables/` now contains the mean tables so they ship. No Dockerfile edit expected; if the copy is selective, add the mean tables.

- [ ] **Step 3: Dry check the assembly locally** (do not deploy yet): run the context-assembly portion or `bash -n deploy/deploy-full.sh` and confirm the file list includes the mean tables.

- [ ] **Step 4: Commit** — `git add deploy/ && git commit -m "chore(deploy): ship V_pi_theta mean tables in backend image"`

---

## Final verification (whole-branch)

- [ ] `just check` (lint + typecheck + test + bench) passes.
- [ ] `just precompute-means` regenerates all four tables cleanly; start means dip below 247 at both extremes.
- [ ] Backend curl: `/evaluate` detent `state_ev` peaks at Neutral, no value exceeds it.
- [ ] Browser (localhost): drag the slider; "Projected value" is highest at Neutral and lower at both ends; decisions/advice unchanged from before; no negative red cost; Neutral visually identical to today.
- [ ] Deploy via `deploy/deploy-full.sh` (backend rebuild). Verify live `/evaluate` returns mean-based detents (index 2 max).

## See Also

- `docs/superpowers/specs/2026-07-19-theta-risk-slider-design.md` (Revision section: CE → true mean)
- `theory/strategy/risk-sensitive-strategy.md` (`mean(θ)` peaks at 0; `std(θ)` rises with θ)
- Part B (later): risk-shaped finish range via θ-aware `/density`
