# Theta Risk Slider Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a five-detent risk slider to the Play UI that re-centers the turn-by-turn feedback (advice, cost, per-option values, and the Delta total) instantly and client-side, while the long-run trajectory projection stays a neutral reference.

**Architecture:** The backend `/evaluate` returns a `detents` array (five `EvaluateResponse`-shaped entries plus `theta`, index order θ = −0.08, −0.04, 0, +0.04, +0.08) with flat fields equal to the Neutral entry. The frontend caches the array, derives `lastEvalResponse = detents[riskLevel]`, and records each decision's cost at all five detents so the Delta total re-totals from cache when the slider moves. No network call on slider change.

**Tech Stack:** Vanilla TypeScript strict, Vite, vitest (node, no DOM), CSS custom properties.

**Spec:** `docs/superpowers/specs/2026-07-19-theta-risk-slider-design.md`
**Depends on:** the backend plan (`docs/superpowers/plans/2026-07-19-theta-risk-slider-backend.md`) being deployed/running, but the response is backward compatible, so the frontend can be developed against the new backend locally and shipped independently.

## Global Constraints

- **No emdashes** in prose, comments, or UI copy.
- **No layout shifts** (`.claude/rules/frontend.md`): fixed sizes, reserve heights, never swap element types on state change.
- **No new npm dependencies.** Tests run in node with no DOM; all logic lives in pure functions.
- **Neutral (riskLevel 2) is identical to today.** A detent slice at Neutral equals today's response, so at Neutral every consumer behaves exactly as before.
- **The slider is client-side instant.** Moving it dispatches `SET_RISK_LEVEL` only. Never triggers a network call.
- **The trajectory chart and finish range stay neutral,** labeled. Their inputs (`expectedFinal`, `stateEv` on trajectory points) always come from the Neutral detent, regardless of `riskLevel`.
- **Detent index order:** 0 = Very safe (θ−0.08), 1 = Safe (−0.04), 2 = Neutral (0), 3 = Aggressive (+0.04), 4 = Very aggressive (+0.08). `riskLevel` defaults to 2.
- **Colour discipline:** the slider is monochrome (ink track, brand-accent handle). No safe/aggressive colour gradient (would overload green/red/orange).
- Run `cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint .` after changes. Verify in the browser before calling done.

---

### Task 1: State plumbing (types, persistence, reducer selection)

The core: cache detents, select a slice by `riskLevel`, keep the projection neutral. Per-detent delta *values* are wired in Task 2; this task establishes the storage and the selection.

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/persistence.ts`
- Modify: `frontend/src/reducer.ts`
- Modify: `frontend/src/reducer.test.ts` (fixtures + new tests)
- Modify: `frontend/src/mask.test.ts` (fixtures)

**Interfaces:**
- Produces:
  - `EvaluateResponse` gains optional `theta?: number` and `detents?: EvaluateResponse[]`.
  - `Prefs` gains `riskLevel: number` (0-4, default 2).
  - `TrajectoryPoint` gains `deltas?: number[]` (per-detent; `delta` stays as the neutral fallback for old persisted points).
  - `GameStateSnapshot` gains `detents: EvaluateResponse[] | null`.
  - `GameAction` gains `{ type: 'SET_RISK_LEVEL'; level: number }` and `SET_EVAL_RESPONSE` swaps `rerollDelta?: number` for `rerollDeltas?: number[]`.
  - Reducer helper `applyEvalToCategories(categories, response)` and `pointDelta(p, level)` util.

- [ ] **Step 1: Extend the types**

In `frontend/src/types.ts`:

Add to `EvaluateResponse` (after `state_ev`):
```ts
  /** Risk parameter for this response (0 = neutral). Present per detent. */
  theta?: number;
  /** All five risk detents (index order theta -0.08..+0.08). Flat fields above
   *  equal detents[2] (Neutral). Absent if the backend predates the slider. */
  detents?: EvaluateResponse[];
```

Add to `Prefs` (after `override`):
```ts
  /** Selected risk detent, index 0-4 (2 = Neutral). Drives which detent slice
   *  the whole turn-by-turn feedback re-centers on. */
  riskLevel: number;
```

Add to `TrajectoryPoint` (after `delta?: number;`):
```ts
  /** Per-detent cost of this decision (index 0-4). deltas[riskLevel] is shown;
   *  delta above stays as the neutral fallback for points saved before the slider. */
  deltas?: number[];
```

Add to `GameStateSnapshot` (after `lastEvalResponse`):
```ts
  /** Cached five-detent eval for the current roll (null between rolls). Derived,
   *  excluded from persistence like lastEvalResponse. */
  detents: EvaluateResponse[] | null;
```

In `GameAction`, replace the `SET_EVAL_RESPONSE` line and add `SET_RISK_LEVEL`:
```ts
  | { type: 'SET_EVAL_RESPONSE'; response: EvaluateResponse; trajectoryEvent?: 'roll' | 'reroll'; rerollLabel?: string; rerollDeltas?: number[] }
  | { type: 'SET_RISK_LEVEL'; level: number }
```

Add this exported helper at the end of `types.ts`:
```ts
/** Read a trajectory point's cost at a given detent, falling back to the single
 *  neutral delta for points saved before per-detent deltas existed. */
export function pointDelta(p: TrajectoryPoint, level: number): number | null {
  return p.deltas?.[level] ?? p.delta ?? null;
}
```

- [ ] **Step 2: Extend persistence**

In `frontend/src/persistence.ts`:

Add `riskLevel: 2` to `DEFAULT_PREFS`:
```ts
export const DEFAULT_PREFS: Prefs = {
  showHints: false,
  showDebug: false,
  guideOpen: true,
  autoplayDelay: 500,
  override: false,
  riskLevel: 2,
};
```

Exclude `detents` from persisted game state. Change `PersistedGame`:
```ts
export type PersistedGame = Omit<GameStateSnapshot, 'lastEvalResponse' | 'detents'>;
```
and the `saveGame` destructure:
```ts
export function saveGame(state: GameState): void {
  const {
    lastEvalResponse: _l,
    detents: _d,
    undoStack: _u,
    redoStack: _r,
    prefs: _p,
    ...game
  } = state;
  void _l; void _d; void _u; void _r; void _p;
  write(GAME_KEY, game);
}
```

- [ ] **Step 3: Write failing reducer tests**

In `frontend/src/reducer.test.ts`, first add `riskLevel: 2` to the two inline `prefs` literals (at `freshState()` and the RESET_GAME preferences test). Then, in `freshState()`, add `detents: null,` to the returned snapshot (next to `lastEvalResponse: null`).

Add these tests (a helper builds a 5-detent response where each detent has a distinct `optimal_category_ev` so selection is observable):

```ts
  describe('risk detents', () => {
    function detentResponse(): EvaluateResponse {
      // Five detents; category 0's ev_if_scored increases with index so the
      // selected slice is observable. Flat fields equal detents[2] (Neutral).
      const mk = (ev: number): EvaluateResponse => ({
        categories: [{ id: 0, name: 'Ones', score: 3, available: true, ev_if_scored: ev }],
        optimal_category: 0,
        optimal_category_ev: ev,
        state_ev: ev,
      });
      const detents = [mk(200), mk(210), mk(220), mk(230), mk(240)];
      return { ...detents[2], detents };
    }

    it('SET_EVAL_RESPONSE selects the slice at the current riskLevel', () => {
      const s0 = { ...freshState(), prefs: { ...freshState().prefs, riskLevel: 4 } };
      const s1 = gameReducer(s0, { type: 'SET_EVAL_RESPONSE', response: detentResponse() });
      expect(s1.lastEvalResponse?.optimal_category_ev).toBe(240); // detent 4
      expect(s1.detents).not.toBeNull();
      expect(s1.categories[0].evIfScored).toBe(240); // scorecard re-centered
    });

    it('SET_RISK_LEVEL re-selects the slice and re-maps categories, no refetch', () => {
      const withEval = gameReducer(freshState(), {
        type: 'SET_EVAL_RESPONSE', response: detentResponse(),
      });
      expect(withEval.lastEvalResponse?.optimal_category_ev).toBe(220); // neutral default
      const safe = gameReducer(withEval, { type: 'SET_RISK_LEVEL', level: 0 });
      expect(safe.prefs.riskLevel).toBe(0);
      expect(safe.lastEvalResponse?.optimal_category_ev).toBe(200); // detent 0
      expect(safe.categories[0].evIfScored).toBe(200);
    });

    it('SET_RISK_LEVEL clamps out-of-range levels', () => {
      const withEval = gameReducer(freshState(), {
        type: 'SET_EVAL_RESPONSE', response: detentResponse(),
      });
      expect(gameReducer(withEval, { type: 'SET_RISK_LEVEL', level: 9 }).prefs.riskLevel).toBe(4);
      expect(gameReducer(withEval, { type: 'SET_RISK_LEVEL', level: -3 }).prefs.riskLevel).toBe(0);
    });

    it('a backend response without detents falls back to the flat response', () => {
      const flat: EvaluateResponse = {
        categories: [{ id: 0, name: 'Ones', score: 3, available: true, ev_if_scored: 199 }],
        optimal_category: 0, optimal_category_ev: 199, state_ev: 199,
      };
      const s = gameReducer({ ...freshState(), prefs: { ...freshState().prefs, riskLevel: 3 } },
        { type: 'SET_EVAL_RESPONSE', response: flat });
      expect(s.lastEvalResponse?.optimal_category_ev).toBe(199);
      expect(s.detents).toBeNull();
    });
  });
```

- [ ] **Step 4: Run to verify they fail**

Run: `cd frontend && npx vitest run reducer.test 2>&1 | tail -20`
Expected: FAIL (type errors on the new fields / `SET_RISK_LEVEL` not handled).

- [ ] **Step 5: Implement the reducer**

In `frontend/src/reducer.ts`:

Add a category-mapping helper near the top (module scope), extracted from the existing `SET_EVAL_RESPONSE` body:
```ts
/** Apply a detent slice's per-category scores/EVs to the scorecard categories. */
function applyEvalToCategories(
  categories: GameState['categories'],
  response: EvaluateResponse,
): GameState['categories'] {
  return categories.map((cat) => {
    const apiCat = response.categories.find((c) => c.id === cat.id);
    if (!apiCat) return cat;
    return { ...cat, suggestedScore: apiCat.score, evIfScored: apiCat.ev_if_scored, available: apiCat.available };
  });
}

/** Clamp a slider index into [0, 4]. */
function clampLevel(level: number): number {
  return Math.max(0, Math.min(4, Math.round(level)));
}
```

Replace the `SET_EVAL_RESPONSE` case body with (selects the slice; keeps the projection neutral):
```ts
    case 'SET_EVAL_RESPONSE': {
      const level = state.prefs.riskLevel;
      const detents = action.response.detents ?? null;
      const selected = detents?.[level] ?? action.response;
      const neutral = detents?.[2] ?? action.response; // projection stays neutral
      const categories = applyEvalToCategories(state.categories, selected);

      let trajectory = state.trajectory;
      if (action.trajectoryEvent) {
        const accScore = computeAccumulatedScore(state);
        const turn = state.categories.filter((c) => c.isScored).length;
        const deltas = action.rerollDeltas;
        const point: TrajectoryPoint = {
          index: trajectory.length,
          turn,
          event: action.trajectoryEvent,
          expectedFinal: state.rawScoredSum + neutral.state_ev, // neutral projection
          accumulatedScore: accScore,
          stateEv: neutral.state_ev,
          label: action.rerollLabel,
          delta: deltas ? deltas[2] : undefined,
          deltas,
        };
        trajectory = [...trajectory, point];
      }
      return { ...state, categories, detents, lastEvalResponse: selected, trajectory };
    }

    case 'SET_RISK_LEVEL': {
      const level = clampLevel(action.level);
      const selected = state.detents?.[level] ?? state.lastEvalResponse;
      const categories = selected
        ? applyEvalToCategories(state.categories, selected)
        : state.categories;
      return {
        ...state,
        prefs: { ...state.prefs, riskLevel: level },
        lastEvalResponse: selected,
        categories,
      };
    }
```

Ensure `EvaluateResponse` and `TrajectoryPoint` are imported in `reducer.ts` (they are already used). Ensure any place that constructs a fresh snapshot (`freshSnapshot()`/initial state) includes `detents: null`. Find it: `grep -n "lastEvalResponse: null" frontend/src/reducer.ts frontend/src/persistence.ts` and add `detents: null` alongside each (including `initialState()`'s reset spread, which must also null `detents` on load like it nulls `lastEvalResponse`).

- [ ] **Step 6: Update mask.test.ts fixtures**

In `frontend/src/mask.test.ts`, add `riskLevel: 2` to the two inline `prefs` literals (the `state()` helper and the hints-off test), and add `detents: null,` to the `state()` snapshot next to `lastEvalResponse`.

- [ ] **Step 7: Run all frontend tests, typecheck, lint**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint .`
Expected: all pass (existing + new risk-detent tests), no type errors, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/types.ts frontend/src/persistence.ts frontend/src/reducer.ts frontend/src/reducer.test.ts frontend/src/mask.test.ts
git commit -m "Cache risk detents and select a slice by riskLevel

SET_EVAL_RESPONSE stores the five-detent array and derives lastEvalResponse
+ scorecard categories from detents[riskLevel]; the trajectory projection
stays pinned to the neutral detent. SET_RISK_LEVEL re-selects with no refetch."
```

---

### Task 2: Per-detent delta history

Record each decision's cost at all five detents so the Delta total re-totals from cache when the slider moves.

**Files:**
- Modify: `frontend/src/main.ts` (reroll deltas across detents; pass full response through)
- Modify: `frontend/src/reducer.ts` (`SCORE_CATEGORY`: per-detent score deltas; neutral projection)
- Modify: `frontend/src/reducer.test.ts` (a test)

**Interfaces:**
- Consumes: `maskCostVsOptimal` (`mask.ts`, unchanged), `rerollMaskFromDice`, `GameStateSnapshot.detents` (Task 1).
- Produces: trajectory points carry `deltas: number[5]`.

- [ ] **Step 1: Write a failing test for per-detent score deltas**

In `frontend/src/reducer.test.ts`, inside the `risk detents` describe, add:

```ts
    it('SCORE_CATEGORY records the cost at all five detents', () => {
      // Build a rolled state with a 5-detent eval where scoring category 0 is
      // suboptimal by a different amount at each detent.
      const mk = (chosenEv: number, bestEv: number): EvaluateResponse => ({
        categories: [
          { id: 0, name: 'Ones', score: 3, available: true, ev_if_scored: chosenEv },
          { id: 1, name: 'Twos', score: 0, available: true, ev_if_scored: bestEv },
        ],
        optimal_category: 1,
        optimal_category_ev: bestEv,
        state_ev: bestEv,
      });
      // detent d: chosen - best = -(d+1), so deltas should be [-1,-2,-3,-4,-5].
      const detents = [mk(99, 100), mk(98, 100), mk(97, 100), mk(96, 100), mk(95, 100)];
      const rolled = {
        ...rolledState(),
        detents,
        lastEvalResponse: detents[2],
        categories: freshState().categories.map((c) =>
          c.id === 0 ? { ...c, evIfScored: 97, available: true, suggestedScore: 3 } : c,
        ),
      };
      const scored = gameReducer(rolled, { type: 'SCORE_CATEGORY', categoryId: 0 });
      const pt = scored.trajectory[scored.trajectory.length - 1];
      expect(pt.deltas).toEqual([-1, -2, -3, -4, -5]);
      expect(pt.delta).toBe(-3); // neutral (index 2)
    });
```

(If `rolledState()` does not expose `detents`, set it inline as above. Confirm `rolledState()` returns a `rolled` phase snapshot; adapt field names to the existing helper.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run reducer.test 2>&1 | tail -15`
Expected: FAIL (`pt.deltas` is undefined; SCORE_CATEGORY does not compute per-detent deltas yet).

- [ ] **Step 3: Compute per-detent score deltas in the reducer**

In `frontend/src/reducer.ts`, in the `SCORE_CATEGORY` case, replace the single-delta computation and the trajectory point's `delta`/`expectedFinal`/`stateEv` with per-detent deltas and a neutral projection. Locate the existing block:

```ts
      const optimalCatEv = state.lastEvalResponse?.optimal_category_ev ?? null;
      let scoreDelta: number | undefined;
      if (optimalCatEv !== null) {
        scoreDelta = evIfScored - optimalCatEv;
      }
```

Replace it with:

```ts
      // Per-detent cost of scoring this category: at each detent d, the chosen
      // category's CE minus that detent's best category CE.
      const perDetentScoreDeltas = (state.detents ?? []).map((d) => {
        const chosen = d.categories.find((c) => c.id === catId);
        if (!chosen) return 0;
        return chosen.ev_if_scored - d.optimal_category_ev;
      });
      const scoreDeltas = perDetentScoreDeltas.length === 5 ? perDetentScoreDeltas : undefined;
      // Neutral (index 2) single delta fallback, also used for old readers.
      const neutralOptEv = state.detents?.[2]?.optimal_category_ev
        ?? state.lastEvalResponse?.optimal_category_ev ?? null;
      const neutralChosenEv = state.detents?.[2]?.categories.find((c) => c.id === catId)?.ev_if_scored
        ?? evIfScored;
      const scoreDelta = neutralOptEv !== null ? neutralChosenEv - neutralOptEv : undefined;
```

Then, where the trajectory `point` is built, keep the projection neutral and add `deltas`. The `successorEv`/`expectedFinal` currently use `evIfScored` (the selected detent). Change them to use the neutral detent's category EV so the chart stays neutral:

```ts
      // Projection stays neutral: use the Neutral detent's ev_if_scored.
      const neutralEvIfScored = state.detents?.[2]?.categories.find((c) => c.id === catId)?.ev_if_scored
        ?? evIfScored;
      const successorEv = neutralEvIfScored - cat.suggestedScore;
      const point: TrajectoryPoint = {
        index: trajectory.length,
        turn: derived.scoredCount,
        event: 'score',
        expectedFinal: turnPhase === 'game_over' ? accScore : derived.rawScoredSum + successorEv,
        accumulatedScore: accScore,
        stateEv: turnPhase === 'game_over' ? 0 : successorEv,
        upperScore: derived.upperScore,
        scoredCategories: derived.scoredCategories,
        label: scoreLabel,
        delta: scoreDelta,
        deltas: scoreDeltas,
      };
```

(Remove the now-duplicated `const successorEv = evIfScored - cat.suggestedScore;` line it replaces.)

- [ ] **Step 4: Compute per-detent reroll deltas in main.ts**

In `frontend/src/main.ts`, change the reroll-delta computation to compute across all detents of the pre-reroll state, and rename the pending variable. Replace:

```ts
let pendingRerollDelta: number | undefined;
```
with:
```ts
let pendingRerollDeltas: number[] | undefined;
```

In the `REROLL` branch, replace:
```ts
    pendingRerollDelta = undefined;
    const cost = maskCostVsOptimal(prev.lastEvalResponse, rerollMaskFromDice(prev.dice));
    if (cost !== null) pendingRerollDelta = cost;
```
with:
```ts
    pendingRerollDeltas = undefined;
    const mask = rerollMaskFromDice(prev.dice);
    if (prev.detents && prev.detents.length === 5) {
      pendingRerollDeltas = prev.detents.map((d) => maskCostVsOptimal(d, mask) ?? 0);
    } else {
      const cost = maskCostVsOptimal(prev.lastEvalResponse, mask);
      if (cost !== null) pendingRerollDeltas = Array(5).fill(cost);
    }
```

In `evaluateIfNeeded`, rename `trajDelta` accordingly and pass `rerollDeltas`:
```ts
  const trajDeltas = pendingRerollDeltas;
  pendingRerollDeltas = undefined;
  ...
    .then(response => dispatch({
      type: 'SET_EVAL_RESPONSE',
      response,
      trajectoryEvent: trajEvent,
      rerollLabel: trajLabel,
      rerollDeltas: trajDeltas,
    }))
```
(remove the old `const trajDelta = pendingRerollDelta;` and `rerollDelta: trajDelta,` lines).

The `evaluate()` response already carries `detents` (it is `res.json()` untyped-through), so no change to `api.ts` is needed; `response.detents` flows into the reducer.

- [ ] **Step 5: Run tests, typecheck, lint**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint .`
Expected: all pass, including the new SCORE_CATEGORY per-detent test.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/main.ts frontend/src/reducer.ts frontend/src/reducer.test.ts
git commit -m "Record each decision's cost at all five risk detents

Reroll deltas (main.ts) and score deltas (reducer) are computed across the
cached detents, so the Delta total can re-total from cache when the slider
moves. Trajectory projection fields stay pinned to the neutral detent."
```

---

### Task 3: Delta total and decision log re-center on the slider

**Files:**
- Modify: `frontend/src/components/DeltaBox.ts`
- Modify: `frontend/src/components/DecisionLog.ts`

**Interfaces:**
- Consumes: `pointDelta(p, level)` (Task 1), `state.prefs.riskLevel`.

- [ ] **Step 1: Re-center the Delta total**

In `frontend/src/components/DeltaBox.ts`, change `cumulativeDelta` to take a level and use `pointDelta`:
```ts
import { pointDelta } from '../types.ts';

function cumulativeDelta(trajectory: TrajectoryPoint[], level: number): number {
  let sum = 0;
  for (const p of trajectory) {
    const d = pointDelta(p, level);
    if (d != null) sum += d;
  }
  return sum;
}
```
Update the `render()` call site to pass `getState().prefs.riskLevel`, and ensure the subscribe fires on `riskLevel` change (add `state.prefs.riskLevel !== prev.prefs.riskLevel` to the change guard, or drop the early-return guard for prefs). Confirm the existing subscribe guard; add the riskLevel comparison so the box re-renders when the slider moves.

- [ ] **Step 2: Re-center the decision log**

In `frontend/src/components/DecisionLog.ts`, find where it reads `p.delta` (grep `delta` in the file) and replace with `pointDelta(p, getState().prefs.riskLevel)`. Add `state.prefs.riskLevel` to its subscribe change guard so the log re-renders on slider moves.

- [ ] **Step 3: Verify no layout shift and correct sums (unit-level)**

There is no DOM test env, so verify by reasoning + the browser pass in Task 6. Run `cd frontend && npx tsc -b --noEmit && npx eslint .` to confirm types/lint.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DeltaBox.ts frontend/src/components/DecisionLog.ts
git commit -m "Re-total Delta and decision-log costs at the selected risk level"
```

---

### Task 4: The RiskSlider component

**Files:**
- Create: `frontend/src/components/RiskSlider.ts`
- Modify: `frontend/src/App.ts` (insert the slider strip after the action bar)
- Modify: `frontend/src/style.css` (slider styling)
- (No store/side-effect changes needed: the component dispatches `SET_RISK_LEVEL` directly.)

**Interfaces:**
- Consumes: `dispatch`, `subscribe`, `getState`, `SET_RISK_LEVEL`.
- Produces: `initRiskSlider(container: HTMLElement): void`.

- [ ] **Step 1: Create the component**

Create `frontend/src/components/RiskSlider.ts`:
```ts
import { getState, dispatch, subscribe } from '../store.ts';

const LABELS = ['Very safe', 'Safe', 'Neutral', 'Aggressive', 'Very aggressive'];

/**
 * Five-detent risk slider. A native range input (min 0, max 4, step 1) so
 * keyboard arrows move detents for free. Dispatches SET_RISK_LEVEL, which
 * re-centers the whole screen client-side (no network). The current level is
 * named above; below-neutral vs above is not colour-coded (monochrome), to
 * avoid overloading the green/red/orange meanings.
 */
export function initRiskSlider(container: HTMLElement): void {
  container.className = 'risk-slider';

  const heading = document.createElement('div');
  heading.className = 'risk-slider-heading';
  const title = document.createElement('span');
  title.textContent = 'Risk appetite:';
  const current = document.createElement('span');
  current.className = 'risk-slider-current';
  heading.appendChild(title);
  heading.appendChild(current);
  container.appendChild(heading);

  const input = document.createElement('input');
  input.type = 'range';
  input.min = '0';
  input.max = '4';
  input.step = '1';
  input.className = 'risk-slider-input';
  input.setAttribute('aria-label', 'Risk appetite');
  input.addEventListener('input', () => {
    dispatch({ type: 'SET_RISK_LEVEL', level: Number(input.value) });
  });
  container.appendChild(input);

  const ticks = document.createElement('div');
  ticks.className = 'risk-slider-ticks';
  for (const label of LABELS) {
    const t = document.createElement('span');
    t.className = 'risk-slider-tick';
    t.textContent = label;
    ticks.appendChild(t);
  }
  container.appendChild(ticks);

  // A one-line note shown only off-neutral: the values are risk-adjusted.
  const note = document.createElement('div');
  note.className = 'risk-slider-note';
  container.appendChild(note);

  function render() {
    const level = getState().prefs.riskLevel;
    if (input.value !== String(level)) input.value = String(level);
    current.textContent = LABELS[level];
    input.setAttribute('aria-valuetext', LABELS[level]);
    const neutral = level === 2;
    note.textContent = neutral
      ? ''
      : 'Values shown are risk-adjusted (certainty-equivalent) points, not expected scores.';
    note.style.visibility = neutral ? 'hidden' : 'visible';
  }

  render();
  subscribe((state, prev) => {
    if (state.prefs.riskLevel === prev.prefs.riskLevel) return;
    render();
  });
}
```

- [ ] **Step 2: Wire it into App.ts**

In `frontend/src/App.ts`, import and insert the slider strip right after `initActionBar(actionBarEl)` and before the panel caption:
```ts
import { initRiskSlider } from './components/RiskSlider.ts';
```
```ts
  const actionBarEl = document.createElement('div');
  diceSection.appendChild(actionBarEl);
  initActionBar(actionBarEl);

  const riskSliderEl = document.createElement('div');
  diceSection.appendChild(riskSliderEl);
  initRiskSlider(riskSliderEl);
```
(leave the existing caption + dice code after it).

- [ ] **Step 3: Style the slider**

In `frontend/src/style.css`, add (monochrome; reserve the note height so toggling neutral does not shift):
```css
/* Risk appetite slider: five detents, monochrome (no risk colour gradient). */
.risk-slider {
  margin: 8px auto 4px;
  max-width: 360px;
  text-align: center;
}
.risk-slider-heading {
  font-size: 12px;
  color: var(--text-muted);
}
.risk-slider-current {
  margin-left: 6px;
  font-weight: 700;
  color: var(--text);
}
.risk-slider-input {
  width: 100%;
  margin: 6px 0 2px;
  accent-color: var(--color-orange);
  cursor: pointer;
}
.risk-slider-ticks {
  display: flex;
  justify-content: space-between;
  font-size: 9px;
  color: var(--text-muted);
}
.risk-slider-tick {
  flex: 1 1 0;
  text-align: center;
}
.risk-slider-tick:first-child { text-align: left; }
.risk-slider-tick:last-child { text-align: right; }
/* Reserved height so showing/hiding the note never shifts the dice below. */
.risk-slider-note {
  min-height: 28px;
  margin-top: 4px;
  font-size: 11px;
  line-height: 1.3;
  color: var(--text-muted);
}
```

- [ ] **Step 4: Typecheck, lint, build**

Run: `cd frontend && npx tsc -b --noEmit && npx eslint . && npx vite build 2>&1 | tail -2`
Expected: clean; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/RiskSlider.ts frontend/src/App.ts frontend/src/style.css
git commit -m "Add the risk appetite slider (five detents, client-side re-center)"
```

---

### Task 5: Relabel numbers off-neutral and label the neutral projection

At a non-neutral detent, the EV-worded labels must not claim "expected score". The trajectory projection gets a neutral caption.

**Files:**
- Modify: `frontend/src/components/EvalPanel.ts`
- Modify: `frontend/src/components/Scorecard.ts`
- Modify: `frontend/src/components/DeltaBox.ts`
- Modify: `frontend/src/App.ts` (trajectory caption) or the chart caption source

**Interfaces:**
- Consumes: `state.prefs.riskLevel`.

- [ ] **Step 1: Relabel EvalPanel value labels off-neutral**

In `frontend/src/components/EvalPanel.ts`, the value-column labels "Keep EV", "Your keep EV", "Score EV", and "Expected final" are EV-specific. Make the label text depend on `riskLevel`: at Neutral keep today's text; off-neutral, swap the EV wording. Because `buildColumn` sets labels once at construction, add a small update in `render()` that rewrites those label cells' text based on `getState().prefs.riskLevel === 2`. Concretely, keep references to the label spans and in `render()` set:
- "Keep EV" -> "Keep value", "Your keep EV" -> "Your keep value", "Score EV" -> "Score value", "Expected final" -> "Projected value" when off-neutral; restore the originals at Neutral.
Add `state.prefs.riskLevel` to EvalPanel's subscribe change guard so it re-renders on slider moves. Keep the tooltips as-is (they still explain the concept).

- [ ] **Step 2: Relabel the Scorecard E[final] header off-neutral**

In `frontend/src/components/Scorecard.ts`, the header cell text is `'E[final]'` with tooltip `TIP.scEfinal`. Keep a reference to that header span and, in the component's `render()` (or add a subscribe), set its text to `'E[final]'` at Neutral and `'Value'` off-neutral. Add `state.prefs.riskLevel` to the change guard.

- [ ] **Step 3: Relabel the DeltaBox subtitle off-neutral**

In `frontend/src/components/DeltaBox.ts`, the subtitle reads "expected points behind perfect play" / "perfect play so far". Off-neutral, say "behind <level>-optimal play" using the level name. Add a `LABELS` array (or import from RiskSlider if exported) and set:
- Neutral: unchanged text.
- Off-neutral (not perfect): `expected points behind ${label.toLowerCase()}-optimal play`; (perfect): `optimal for ${label.toLowerCase()} play so far`.
The subtitle already re-renders on trajectory change; ensure it also re-renders on `riskLevel` change (added in Task 3 Step 1).

- [ ] **Step 4: Caption the trajectory projection as neutral**

The trajectory chart shows the neutral projection regardless of risk. Add a short note to its caption so that is truthful. In `frontend/src/App.ts`, find the trajectory chart's `panelCaption(...)` (grep `panelCaption` and `projected`), and append " This projection assumes neutral (EV-optimal) play." to that caption string. This caption is static (does not depend on riskLevel), so no reactive change is needed.

- [ ] **Step 5: Typecheck, lint, build, full test**

Run: `cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint . && npx vite build 2>&1 | tail -2`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/EvalPanel.ts frontend/src/components/Scorecard.ts frontend/src/components/DeltaBox.ts frontend/src/App.ts
git commit -m "Relabel value fields as risk-adjusted off-neutral; caption the neutral projection"
```

---

### Task 6: Browser verification + docs

**Files:**
- Modify: `frontend/CLAUDE.md`

- [ ] **Step 1: Run both servers against the new backend**

The backend plan must be built and running locally (`just dev-backend`) so `/evaluate` returns `detents`. Then `just dev-frontend`, open `http://localhost:5173/yatzy/play/`, and clear localStorage for a fresh state.

- [ ] **Step 2: Manual verification checklist**

- [ ] At Neutral (default), the UI is visually identical to before this feature; the slider sits under the mode-toggle row centered on Neutral.
- [ ] Roll, turn on Optimal hints, then drag the slider: the advice row, mismatch marks, and cost line change **instantly** with no network request (watch the Network tab: no `/evaluate` call on drag).
- [ ] The Delta total and the decision log re-total as you slide, and return to their Neutral values at Neutral.
- [ ] Off-neutral, the EvalPanel labels, the scorecard E[final] header, and the DeltaBox subtitle read the risk-adjusted wording; at Neutral they read the original wording.
- [ ] The trajectory chart and finish range do **not** change as you slide; the caption notes the neutral projection.
- [ ] Keyboard: focus the slider, arrow keys move detents and the whole screen re-centers.
- [ ] Reload: the selected risk level persists (it is a pref); the detents cache is re-fetched on the next roll.
- [ ] Toggling the slider shifts nothing else on the page (no layout shift): the note height is reserved.

- [ ] **Step 3: Update the frontend docs**

In `frontend/CLAUDE.md`, add `components/RiskSlider.ts` to the source-layout table (one line), and note in the architecture section that `/evaluate` returns a `detents` array cached in state, with `lastEvalResponse` derived as `detents[riskLevel]`. Update the test count if it changed. No emdashes.

- [ ] **Step 4: Commit**

```bash
git add frontend/CLAUDE.md
git commit -m "Document the risk slider in the frontend guide"
```

---

## Definition of Done

- [ ] `cd frontend && npx vitest run` passes (existing + new risk-detent and per-detent-delta tests).
- [ ] `npx tsc -b --noEmit`, `npx eslint .`, and `npx vite build` are clean.
- [ ] Against the new backend: dragging the slider re-centers advice/cost/Delta/values instantly with zero `/evaluate` calls; Neutral is identical to before; the trajectory chart stays put with a neutral caption; off-neutral labels read risk-adjusted; the level persists across reload; no layout shift on slider moves.
- [ ] `frontend/CLAUDE.md` documents the slider and the detents-in-response.

## Notes / open risks

- **Deploy order:** ship the backend first (it is backward compatible), then this frontend. Until the backend is deployed, `response.detents` is absent and the reducer falls back to the flat neutral response (the slider still renders but every level shows Neutral). Do not deploy this frontend to production before the backend that returns `detents` is live, or the slider will appear to do nothing.
- **DecisionLog `delta` read:** Task 3 Step 2 assumes DecisionLog reads `p.delta`. Confirm by grepping; if it reads the delta differently, adapt to `pointDelta(p, riskLevel)` at the actual read site.
