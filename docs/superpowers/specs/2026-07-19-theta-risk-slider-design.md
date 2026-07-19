# Theta Risk Slider (Play UI): Design

**Date:** 2026-07-19
**Status:** Approved (design), pending implementation plan

## Summary

Add a five-detent "risk appetite" slider to the Play UI. Moving it re-centers the
whole turn-by-turn feedback (advice, mismatch, cost, the Delta-vs-optimal skill
total, and the per-option value numbers) onto the chosen risk level, expressed in
**certainty-equivalent points**. At the Neutral detent the UI is byte-identical to
today. The long-run projection (trajectory chart + finish range) stays a labeled
neutral reference.

The slider is instant: one `/evaluate` call per roll returns the feedback for all
five detents at once, the frontend caches it, and moving the slider is pure
client-side selection with zero network round-trips.

## Goals

- Let a player see, in real time, how optimal play shifts with risk preference.
- Keep Neutral identical to the current game, so the default experience is unchanged.
- Sub-frame latency when dragging the slider (no per-drag network calls).
- Keep every displayed number truthful: no "expected score" label on a risk-adjusted
  quantity.

## Non-Goals

- No continuous θ. The slider snaps to five precomputed detents; arbitrary θ would
  need on-the-fly DP (113-600 ms) and is out of scope.
- No risk-specific long-run distribution. The trajectory chart and finish range stay
  the neutral (EV-optimal) projection, labeled as such. Per-risk distributions are a
  possible later phase; they need a new density/simulation engine that does not exist.
- No change to the density/oracle path or the `/density` endpoint.

## Background: why certainty-equivalent

Investigated before design (findings in the θ-semantics report). The solver's
risk-sensitive objective is exponential utility:

```
U(X) = E[e^{θ·X}]           CE(X) = (1/θ)·ln U(X)
```

For θ=0 the state-value table stores the expected final score E[S]. For θ≠0 the
table (`all_states_theta_*.bin`) stores `L(S) = ln U(S)`, a log-utility, **not** an
expected score. The current `/evaluate` (`api_computations.rs`) hardcodes the θ=0 EV
recursion (`score + state_value`, plain max, probability-weighted mask sums), which
is wrong for a θ≠0 table.

**The resolution is to show the certainty-equivalent (CE) everywhere.** CE is in
score units ("the guaranteed score a risk-X player would trade this gamble for"), is
the correct risk-mode value, and is computable from the existing tables via the
risk-sensitive recursion. Crucially, `lim(θ→0) CE = EV`, so the Neutral detent uses
the existing EV path and looks exactly like today; only off-center detents diverge.

## Decisions (from brainstorming)

1. **Scope: everything re-centers.** Advice, best-keep, best-category, mismatch
   marks, cost line, the Delta total, and the per-option value numbers all follow the
   slider, in CE points. Numbers relabel from "E[final]" to a risk-adjusted value at
   non-neutral detents.
2. **Projection stays neutral.** Trajectory chart and finish range keep showing the
   EV-optimal projection with a small "projected under EV-optimal play" label.
3. **Five symmetric detents**, all in the utility domain (|θ| ≤ 0.15, same solver
   speed as θ=0), all tables present on disk:

   | Index | Label | θ | table |
   |------|-------|---:|-------|
   | 0 | Very safe | −0.08 | `all_states_theta_-0.080.bin` |
   | 1 | Safe | −0.04 | `all_states_theta_-0.040.bin` |
   | 2 | Neutral | 0.00 | `all_states.bin` |
   | 3 | Aggressive | +0.04 | `all_states_theta_0.040.bin` |
   | 4 | Very aggressive | +0.08 | `all_states_theta_0.080.bin` |

4. **Slider UI:** a detented horizontal slider with all five labels, a handle on the
   current stop, the current level named above, draggable and click-to-jump. Placed
   directly beneath the mode-toggle row (Optimal hints / Intro guide / Override mode),
   above the dice caption. Monochrome (ink + brand accent handle); no safe/aggressive
   colour gradient, to avoid overloading the green/red/orange meanings. Persists as a
   pref; defaults to Neutral.

## Architecture

### Backend

**Response shape.** `/evaluate` returns an array of five per-detent results in one
response. Each entry has the same shape as today's response, but the value fields are
CE (which at Neutral equal EV):

```
EvaluateResponse {
  detents: [DetentResult; 5]   // index 0..4, matching the table above
}
DetentResult {
  theta: f32,
  mask_ces: [f32; 32],          // CE of each keep-mask (was mask_evs)
  optimal_mask: u32,
  optimal_mask_ce: f32,
  categories: [{ id, name, score, available, ce_if_scored }; N],
  optimal_category: i32,
  optimal_category_ce: f32,
  state_ce: f32,
}
```

The Neutral entry (index 2) is exactly today's EV response with `_ev`→`_ce` renamed;
its numbers are unchanged. The request is unchanged (`{dice, upper_score,
scored_categories, rerolls_remaining}`); no θ field; the server always returns all
five detents.

**CE recursion.** Add a risk-sensitive variant of `compute_roll_response`
(`api_computations.rs`) / the `widget_solver` helpers, working in the utility domain
(all detents satisfy |θ| ≤ 0.15). For a detent with parameter θ and its state-value
table storing `L(S)`:

- Recover utility `U(S) = exp(L(S))`.
- Scoring category c from roll r in state S contributes `e^{θ·score(r,c)} · U(next S)`.
- Reroll-mask utility = probability-weighted sum of the resulting rolled-state
  utilities (reuse the θ-independent `keep_table`).
- **Optimal action:** for θ > 0, argmax U; for **θ < 0, argmin U** (because
  CE = (1/θ)·ln U and (1/θ) < 0). This sign flip is the known subtlety; it mirrors
  `batched_solver.rs`'s min/max handling and MUST be verified per detent.
- **Reported value:** `CE = (1/θ)·ln U`, in points.
- The Neutral detent (θ=0) uses the existing EV path unchanged (CE is the θ→0 limit;
  do not divide by zero).

**Table loading.** At startup, load the five detent tables into a small structure
alongside the single shared `YatzyContext`, using the existing
`storage::load_state_values_standalone` (the proven "one context + N StateValues"
pattern already used in `simulation/strategy.rs` and several `bin/*.rs` tools). Only
the state-values slice differs per detent; every other context table is θ-independent
and shared. Neutral reuses the already-loaded `all_states.bin`.

**Cost / memory / latency.** Five detents × the current 2-9 μs = ~10-45 μs per
request, still negligible. Five tables ≈ 80 MB of mmap (demand-paged), comfortable
even on the current 2 GB box. Payload grows ~5× (five × 32 mask floats + 15 category
floats ≈ a few KB JSON). Fine.

### Frontend

**Pref.** Add `riskLevel: number` to `Prefs` (index 0-4, default 2 = Neutral),
persisted with the other prefs. New action `SET_RISK_LEVEL`.

**Caching + selection (the instant-slider mechanism).** On roll/reroll, `/evaluate`
returns all five detents; store the full array. Derive the existing
`lastEvalResponse` as **the selected detent's slice** (`detents[riskLevel]` with
`mask_ces`→`mask_evs` etc. field mapping). Because every existing consumer (DiceBar
via `mask.ts`, EvalPanel, Scorecard, the delta computation) reads `lastEvalResponse`,
they all re-center automatically when the selected detent changes, with no per-consumer
changes. Moving the slider dispatches `SET_RISK_LEVEL`, which re-derives the slice
from the cached array and re-renders: **no network call**.

**Delta history re-centering.** Each decision's cost is recorded at all five detents
(available in the same all-detents response at decision time). Extend the trajectory
point's `delta` to a per-detent `delta: number[5]` (or a parallel array). DeltaBox and
DecisionLog read `delta[riskLevel]`. Sliding re-totals instantly from cached values;
no re-evaluation of past turns. The stored per-detent deltas are excluded from
persistence heavy-field rules as today.

**Relabeling.** At a non-neutral detent, EvalPanel's value labels and the Scorecard
E[final] header read a risk-adjusted term (e.g. "Value (risk-adj.)") instead of
"Expected final"; the DeltaBox subtitle reads "behind <level>-optimal play" instead
of "behind perfect play". At Neutral, all labels are exactly as today.

**Projection stays neutral.** The trajectory chart and the "80% finish range" always
use the Neutral detent's data (always present in the response), with a caption note
"projected under EV-optimal play". They do not change with the slider.

**Slider component.** New `components/RiskSlider.ts` following the standard component
pattern: a detented range control (native `<input type=range min=0 max=4 step=1>`
styled with visible detents and labels, or an equivalent accessible custom control),
dispatching `SET_RISK_LEVEL`. Keyboard-accessible (arrow keys move detents). Named
current level above. Placed in a new strip under the action bar's mode row.

## Latency

Dragging the slider is client-side only (cache selection + re-render): sub-frame, no
network. The only network cost is the existing one `/evaluate` per roll, now ~5× the
microsecond compute (still tens of μs server-side; network round-trip unchanged and
dominates as before).

## Deployment / ops

- **Backend rebuild required.** This changes `/evaluate` and adds table loading, so
  it is not a frontend-only ship. The backend image must be rebuilt and redeployed,
  which hits the ARM cross-compile blocker in `deploy-env-gotchas`. Resolve by
  building the backend image on the server, or fixing cross-compile, before shipping.
- **Ship five detent tables** (~80 MB) into the backend image (currently only
  `all_states.bin`). Adjust the deploy assembly to copy the five files.
- **RAM:** ~80 MB extra mmap, demand-paged. Fits the current 2 GB e2-small; the
  optional e2-medium bump is not required for this feature.
- Update `solver/CLAUDE.md` API Reference and `frontend/CLAUDE.md` / `api.ts` per the
  repo's API-change rule.

## Testing

- **Backend (Rust):** for the Neutral detent, the CE response must equal today's EV
  response exactly (regression). For each non-neutral detent, `optimal_mask` and
  `optimal_category` must match the precomputed θ-policy. Cross-check against the
  `batched_solver` argmax or a short simulation at that θ. Unit-test the CE math
  including the **θ < 0 argmin sign** on a hand-checked small case. Verify `CE = EV`
  at θ=0 limit handling (no divide-by-zero).
- **Frontend (node, no DOM):** extend `mask.ts` view-model tests so that selecting a
  detent slice produces the right advice/cost/ticks; assert Neutral matches the
  current fixtures. Reducer test for `SET_RISK_LEVEL`. Delta-history re-centering:
  given per-detent deltas, DeltaBox total = sum at the selected level.
- **Manual (browser):** slider drags re-center advice/cost/delta instantly with no
  network; Neutral is visually identical to today; labels relabel off-center;
  trajectory/finish-range stay put with the neutral caption; keyboard arrows move
  detents; persists across reload; defaults to Neutral on first visit.

## Risks

- **θ < 0 sign conventions.** The theory accuracy review flagged sign-convention
  wording for θ<0. The argmin-for-θ<0 rule above must be verified against the
  precomputed policy before shipping (the backend cross-check test is the gate).
- **CE interpretability at the extremes.** −0.08 / +0.08 are well inside the utility
  domain, but confirm the CE numbers read sensibly (monotone, non-negative cost) in
  the browser pass.
- **Two-language change.** Backend (Rust) + frontend (TS) + deploy. Decompose into
  separate plans (below) so each ships and verifies independently.

## Decomposition

Two implementation plans, backend first:

1. **Backend:** all-detents `/evaluate` with the CE recursion, five-table loading,
   response shape, tests, and the deploy-assembly change. Verifiable in isolation via
   `curl`/tests (Neutral == today; non-neutral policy matches precomputed).
2. **Frontend:** `riskLevel` pref + action, all-detents caching + selected-slice
   derivation, per-detent delta history, relabeling, neutral-pinned projection, and
   the `RiskSlider` component. Verifiable against the new backend locally.

## Open Questions

None.

## See Also

- θ-semantics investigation (this session)
- `theory/foundations/risk-parameter-theta.md`, `theory/strategy/risk-sensitive-strategy.md`
- `solver/CLAUDE.md` (API reference, θ parameter), `deploy-env-gotchas` memory
