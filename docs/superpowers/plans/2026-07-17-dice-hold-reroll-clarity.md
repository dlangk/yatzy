# Dice Hold/Reroll Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the player's hold/reroll selection from the solver's advice in the Play UI dice bar, so both are readable at a glance and each colour means exactly one thing.

**Architecture:** All display semantics move into a pure `diceBarViewModel(state)` function in `frontend/src/mask.ts`, tested in node with no DOM. `DiceBar.ts` becomes a dumb renderer of that view model. Each die and its `HOLD`/`REROLL` label merge into a single `<button>`, serving mouse, keyboard, and touch from one control. `DiceLegend.ts` is deleted.

**Tech Stack:** Vanilla TypeScript (strict), Vite, vitest (node environment, no DOM), CSS custom properties.

**Spec:** `docs/superpowers/specs/2026-07-17-dice-hold-reroll-clarity-design.md`

## Global Constraints

- **No emdashes.** Never use `--`, `&mdash;`, or `—` as emdashes in prose, captions, comments, or docs. Use periods, commas, colons, semicolons, or parentheses. (Project rule, `CLAUDE.md`.)
- **No layout shifts.** The UI must be pixel-stable. No element may change size, move, or reflow when state changes. Never return `null`; always render containers at fixed size with placeholders. (`.claude/rules/frontend.md`.)
- **No new npm dependencies.** Specifically no jsdom, happy-dom, or testing-library. Tests run in plain node.
- **Do NOT modify `shared/dice.js`.** It is shared by the treatise, play UI, and profiler. Its `optimal-keep` and `optimal-reroll` states look dead after this change but are still used by `profiler/js/profile/components/scenario-card.js:99`. Deleting them breaks the profiler.
- **Green and red never mean hold/reroll.** They mean quality, as in `Scorecard.ts:164-170`. Hold is ink (`--text`), reroll is orange, red (`--color-danger`) means "giving up points" only.
- **Orange has two tokens.** `--color-orange` (`#F37021`) is the brand orange, for fills and accents. `--color-orange-text` is for text and focus rings: `#B54A0C` in light (4.87:1), `#F37021` in dark (5.93:1). The brand orange measures 2.69:1 on the light background, under the WCAG AA 4.5:1 text floor and the 1.4.11 3:1 indicator floor. Never use `--color-orange` for small text.
- **Advice is stable.** The solver advice shown for a die must depend only on `optimal_mask`, never on the player's `kept` flags. This is the bug being fixed; do not reintroduce it.
- **Masks are in original dice order.** Bit `i` set in `optimal_mask` means "reroll die `i`". Do not sort.
- **Touch targets:** every interactive element is at least 24x24 CSS px (WCAG 2.2 SC 2.5.8, Level AA).
- **Baseline:** 35 tests in 3 files pass today (`cd frontend && npx vitest run`). Never let this number drop.

---

### Task 1: Pure view-model in `mask.ts`

This is the backbone. Every rule in the spec becomes a pure assertion here, so the DOM layer needs no tests.

**Files:**
- Create: `frontend/src/mask.ts`
- Test: `frontend/src/mask.test.ts`

**Interfaces:**
- Consumes: `GameState`, `DieState`, `EvaluateResponse` from `frontend/src/types.ts` (type-only imports; `mask.ts` must have no runtime dependency on the store).
- Produces:
  - `rerollMaskFromDice(dice: readonly DieState[]): number`
  - `maskCostVsOptimal(ev: EvaluateResponse | null, mask: number): number | null`
  - `diceBarViewModel(s: GameState): DiceBarViewModel`
  - `COST_EPS: number` (0.005)
  - types `DieAdvice`, `DieViewModel`, `DiceBarViewModel`

**Note on a deliberate deviation from the spec:** the spec sketched a `visible: { advice, ticks, cost }` field. It is dropped here as redundant duplicate state: `advice === null` already means hidden, and `cost === null` already means no cost line. One source of truth per fact.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/mask.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { rerollMaskFromDice, maskCostVsOptimal, diceBarViewModel, COST_EPS } from './mask.ts';
import type { GameState, DieState, EvaluateResponse } from './types.ts';

function dice(spec: [number, boolean][]): DieState[] {
  return spec.map(([value, kept]) => ({ value, kept }));
}

/** mask_evs has 32 entries (one per 5-bit reroll mask). Optimal mask is 0b00110 (reroll dice 1 and 2). */
function evalResponse(over: Partial<EvaluateResponse> = {}): EvaluateResponse {
  const mask_evs = Array.from({ length: 32 }, () => 100);
  mask_evs[0b00110] = 110; // the optimal mask scores best
  return {
    mask_evs,
    optimal_mask: 0b00110,
    optimal_mask_ev: 110,
    categories: [],
    optimal_category: 0,
    optimal_category_ev: 50,
    state_ev: 100,
    ...over,
  };
}

function state(over: Partial<GameState> = {}): GameState {
  return {
    dice: dice([[1, true], [2, false], [3, false], [4, true], [5, true]]),
    upperScore: 0,
    scoredCategories: 0,
    rerollsRemaining: 2,
    categories: [],
    totalScore: 0,
    bonus: 0,
    rawScoredSum: 0,
    lastEvalResponse: evalResponse(),
    turnPhase: 'rolled',
    trajectory: [],
    undoStack: [],
    redoStack: [],
    prefs: { showHints: true, showDebug: false, guideOpen: false, autoplayDelay: 300 },
    ...over,
  } as GameState;
}

describe('rerollMaskFromDice', () => {
  it('returns 0 when every die is held', () => {
    expect(rerollMaskFromDice(dice([[1, true], [2, true], [3, true], [4, true], [5, true]]))).toBe(0);
  });

  it('returns 31 when no die is held', () => {
    expect(rerollMaskFromDice(dice([[1, false], [2, false], [3, false], [4, false], [5, false]]))).toBe(31);
  });

  it('sets bit i for each die that will be rerolled, in original dice order', () => {
    // dice 1 and 2 are not held -> 0b00110 == 6
    expect(rerollMaskFromDice(dice([[1, true], [2, false], [3, false], [4, true], [5, true]]))).toBe(0b00110);
  });
});

describe('maskCostVsOptimal', () => {
  it('returns 0 when the player mask is the optimal mask', () => {
    expect(maskCostVsOptimal(evalResponse(), 0b00110)).toBe(0);
  });

  it('returns a negative number for a worse mask', () => {
    expect(maskCostVsOptimal(evalResponse(), 0b00000)).toBe(-10);
  });

  it('returns null when mask_evs is absent', () => {
    expect(maskCostVsOptimal(evalResponse({ mask_evs: undefined }), 0)).toBeNull();
  });

  it('returns null when optimal_mask_ev is absent', () => {
    expect(maskCostVsOptimal(evalResponse({ optimal_mask_ev: undefined }), 0)).toBeNull();
  });

  it('returns null when the eval response is null', () => {
    expect(maskCostVsOptimal(null, 0)).toBeNull();
  });
});

describe('diceBarViewModel advice', () => {
  it('says REROLL where optimal_mask has the bit set and HOLD where it does not', () => {
    // optimal_mask 0b00110 -> reroll dice 1 and 2
    expect(diceBarViewModel(state()).advice)
      .toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
  });

  // REGRESSION for the shipped bug: the legend said green meant "hold" while the code
  // made it mean "you agree with the solver", so advice moved when the player toggled.
  it('is identical for two states differing only in kept flags', () => {
    const allHeld = state({ dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]) });
    const noneHeld = state({ dice: dice([[1, false], [2, false], [3, false], [4, false], [5, false]]) });
    expect(diceBarViewModel(allHeld).advice).toEqual(diceBarViewModel(noneHeld).advice);
  });

  it('does not change when a single die is toggled', () => {
    const before = diceBarViewModel(state()).advice;
    const toggled = state({ dice: dice([[1, false], [2, false], [3, false], [4, true], [5, true]]) });
    expect(diceBarViewModel(toggled).advice).toEqual(before);
  });
});

describe('diceBarViewModel ticks and cost', () => {
  it('shows no ticks and no cost when the player matches the optimal mask', () => {
    const vm = diceBarViewModel(state());
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  // A costless tie must never be flagged red. Red means "giving up points".
  it('shows no ticks for a different mask that ties the optimal mask on EV', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b11000] = 110; // different mask, identical EV
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, false], [5, false]]), // mask 0b11000
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  it('ticks exactly the dice whose choice differs from the advice when the mask is costly', () => {
    // Player holds everything (mask 0), optimal is 0b00110, cost -10.
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
    }));
    expect(vm.advice).toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
    expect(vm.ticks).toEqual([false, true, true, false, false]);
    expect(vm.cost).toBe(-10);
  });

  it('does not tick when the cost is inside EPS', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b00000] = 110 - (COST_EPS / 2); // -0.0025, below the threshold
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.cost).toBeNull();
  });

  it('ticks when the cost is just past EPS', () => {
    const evs = Array.from({ length: 32 }, () => 0);
    evs[0b00110] = 110;
    evs[0b00000] = 110 - (COST_EPS * 2); // -0.01, past the threshold
    const vm = diceBarViewModel(state({
      dice: dice([[1, true], [2, true], [3, true], [4, true], [5, true]]),
      lastEvalResponse: evalResponse({ mask_evs: evs }),
    }));
    expect(vm.ticks).toEqual([false, true, true, false, false]);
    expect(vm.cost).toBeCloseTo(-0.01, 6);
  });
});

describe('diceBarViewModel hidden states', () => {
  it('hides advice when hints are off but still populates the dice', () => {
    const vm = diceBarViewModel(state({
      prefs: { showHints: false, showDebug: false, guideOpen: false, autoplayDelay: 300 },
    }));
    expect(vm.advice).toBeNull();
    expect(vm.cost).toBeNull();
    expect(vm.ticks).toEqual([false, false, false, false, false]);
    expect(vm.dice).toHaveLength(5);
  });

  it('hides advice when no rerolls remain', () => {
    expect(diceBarViewModel(state({ rerollsRemaining: 0 })).advice).toBeNull();
  });

  it('hides advice when the turn is not in the rolled phase', () => {
    expect(diceBarViewModel(state({ turnPhase: 'idle' })).advice).toBeNull();
  });

  it('hides advice when there is no eval response', () => {
    expect(diceBarViewModel(state({ lastEvalResponse: null })).advice).toBeNull();
  });
});

describe('diceBarViewModel dice', () => {
  it('labels held dice HOLD and unheld dice REROLL', () => {
    expect(diceBarViewModel(state()).dice.map(d => d.label))
      .toEqual(['HOLD', 'REROLL', 'REROLL', 'HOLD', 'HOLD']);
  });

  it('blanks values and labels and fades every die when the turn is not rolled', () => {
    const vm = diceBarViewModel(state({ turnPhase: 'idle' }));
    expect(vm.dice.map(d => d.value)).toEqual([0, 0, 0, 0, 0]);
    expect(vm.dice.map(d => d.label)).toEqual(['', '', '', '', '']);
    expect(vm.dice.every(d => d.faded)).toBe(true);
    expect(vm.dice.every(d => d.disabled)).toBe(true);
  });

  it('disables the dice when no rerolls remain but keeps them unfaded', () => {
    const vm = diceBarViewModel(state({ rerollsRemaining: 0 }));
    expect(vm.dice.every(d => d.disabled)).toBe(true);
    expect(vm.dice.every(d => d.faded)).toBe(false);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd frontend && npx vitest run src/mask.test.ts
```

Expected: FAIL. Vitest cannot resolve `./mask.ts` (the module does not exist yet).

- [ ] **Step 3: Write the implementation**

Create `frontend/src/mask.ts`:

```typescript
/**
 * Pure dice-mask helpers and the DiceBar view model.
 *
 * This module is the single seam where the dice bar's display semantics live, so
 * they can be tested in node without a DOM. DiceBar.ts renders this and nothing
 * else. Keep this file free of runtime imports (types only).
 *
 * Mask convention: bit i set means "reroll die i". Masks are in original dice
 * order and are never sorted.
 */
import type { DieState, EvaluateResponse, GameState } from './types.ts';

/** What the solver would do with a die. */
export type DieAdvice = 'HOLD' | 'REROLL';

export interface DieViewModel {
  /** Die face, 1-6. 0 renders the blank "?" placeholder. */
  value: number;
  held: boolean;
  /** Button text. Empty string before the first roll of a turn. */
  label: DieAdvice | '';
  disabled: boolean;
  faded: boolean;
}

export interface DiceBarViewModel {
  dice: DieViewModel[];
  /** Solver advice per die, or null when the advice row is hidden. */
  advice: DieAdvice[] | null;
  /** Per-die mismatch markers. All false unless the player's mask is costly. */
  ticks: boolean[];
  /** Expected points given up by the player's current mask, or null when there is
   *  nothing to report. Always negative when present. */
  cost: number | null;
}

/**
 * Half the displayed precision. A mask whose cost is inside this band is treated
 * as free: two distinct masks can tie on EV, and a costless tie must not be
 * flagged red, because red means "giving up points".
 */
export const COST_EPS = 0.005;

/** Build the reroll mask for the player's current selection. Bit i set = reroll die i. */
export function rerollMaskFromDice(dice: readonly DieState[]): number {
  let mask = 0;
  for (let i = 0; i < dice.length; i++) {
    if (!dice[i].kept) mask |= 1 << i;
  }
  return mask;
}

/**
 * Expected points given up by `mask` versus the solver's best mask.
 * Returns 0 for the optimal mask, a negative number for a worse one, and null
 * when the response lacks the data to say.
 */
export function maskCostVsOptimal(ev: EvaluateResponse | null, mask: number): number | null {
  const playerEv = ev?.mask_evs?.[mask];
  const optimalEv = ev?.optimal_mask_ev;
  if (playerEv === undefined || optimalEv === undefined) return null;
  return playerEv - optimalEv;
}

/** Project game state into everything the dice bar needs to render. */
export function diceBarViewModel(s: GameState): DiceBarViewModel {
  const active = s.turnPhase === 'rolled';
  const hasEval = s.lastEvalResponse !== null;
  const canToggle = active && hasEval && s.rerollsRemaining > 0;
  const optimalMask = s.lastEvalResponse?.optimal_mask ?? null;

  const dice: DieViewModel[] = s.dice.map(d => ({
    value: active ? d.value : 0,
    held: active ? d.kept : true,
    label: active ? (d.kept ? 'HOLD' : 'REROLL') : '',
    disabled: !canToggle,
    faded: !active,
  }));

  const noTicks = s.dice.map(() => false);

  const showAdvice = s.prefs.showHints && active && hasEval
    && optimalMask !== null && s.rerollsRemaining > 0;
  if (!showAdvice) {
    return { dice, advice: null, ticks: noTicks, cost: null };
  }

  const advice: DieAdvice[] = s.dice.map(
    (_, i) => ((optimalMask! & (1 << i)) ? 'REROLL' : 'HOLD'),
  );

  const rawCost = maskCostVsOptimal(s.lastEvalResponse, rerollMaskFromDice(s.dice));
  const costly = rawCost !== null && rawCost < -COST_EPS;
  if (!costly) {
    return { dice, advice, ticks: noTicks, cost: null };
  }

  const ticks = s.dice.map((d, i) => d.kept !== (advice[i] === 'HOLD'));
  return { dice, advice, ticks, cost: rawCost };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd frontend && npx vitest run src/mask.test.ts
```

Expected: PASS, 23 tests.

- [ ] **Step 5: Confirm the full suite still passes and types are clean**

```bash
cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint src/mask.ts src/mask.test.ts
```

Expected: 58 tests passing (35 baseline + 23 new), no type errors, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/mask.ts frontend/src/mask.test.ts
git commit -m "Add pure dice-mask helpers and DiceBar view model

Advice derives from optimal_mask alone, so it cannot move when the player
toggles dice. Ticks gate on cost < -0.005 rather than mask difference, so a
costless tie is never flagged red."
```

---

### Task 2: Route `main.ts` and `EvalPanel.ts` through the shared helper

Both files already compute the reroll mask and the cost-vs-optimal by hand. Task 1 added a third potential copy. Collapse all three onto one implementation before building on it.

**Files:**
- Modify: `frontend/src/main.ts:41-49`
- Modify: `frontend/src/components/EvalPanel.ts:122-128`

**Interfaces:**
- Consumes: `rerollMaskFromDice`, `maskCostVsOptimal` from Task 1.
- Produces: nothing new. This task is behaviour-preserving by design.

- [ ] **Step 1: Confirm the current behaviour is green before touching it**

```bash
cd frontend && npx vitest run
```

Expected: 58 tests passing. If not, stop and fix before continuing.

- [ ] **Step 2: Replace the hand-rolled mask logic in `main.ts`**

In `frontend/src/main.ts`, add to the imports at the top of the file:

```typescript
import { rerollMaskFromDice, maskCostVsOptimal } from './mask.ts';
```

Then replace this block (currently `main.ts:41-49`):

```typescript
    if (prev.lastEvalResponse?.mask_evs) {
      let playerMask = 0;
      for (let i = 0; i < prev.dice.length; i++) {
        if (!prev.dice[i].kept) playerMask |= 1 << i;
      }
      const playerMaskEv = prev.lastEvalResponse.mask_evs[playerMask] ?? null;
      const optimalMaskEv = prev.lastEvalResponse.optimal_mask_ev ?? null;
      if (playerMaskEv !== null && optimalMaskEv !== null) {
        pendingRerollDelta = playerMaskEv - optimalMaskEv;
      }
    }
```

with:

```typescript
    const cost = maskCostVsOptimal(prev.lastEvalResponse, rerollMaskFromDice(prev.dice));
    if (cost !== null) pendingRerollDelta = cost;
```

- [ ] **Step 3: Replace the hand-rolled mask logic in `EvalPanel.ts`**

In `frontend/src/components/EvalPanel.ts`, add to the imports at the top of the file:

```typescript
import { rerollMaskFromDice } from '../mask.ts';
```

Then replace this block (currently `EvalPanel.ts:121-128`):

```typescript
    // Your keep EV (masks are in original dice order)
    let currentMaskEv: number | null = null;
    if (s.lastEvalResponse?.mask_evs) {
      let rerollMask = 0;
      for (let i = 0; i < s.dice.length; i++) {
        if (!s.dice[i].kept) rerollMask |= 1 << i;
      }
      currentMaskEv = s.lastEvalResponse.mask_evs[rerollMask] ?? null;
    }
```

with:

```typescript
    // Your keep EV (masks are in original dice order)
    const currentMaskEv = s.lastEvalResponse?.mask_evs?.[rerollMaskFromDice(s.dice)] ?? null;
```

- [ ] **Step 4: Verify nothing changed**

```bash
cd frontend && npx vitest run && npx tsc -b --noEmit && npx eslint src/main.ts src/components/EvalPanel.ts
```

Expected: 58 tests still passing, no type errors, no lint errors. This task must not change any test outcome; it is a pure dedupe.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/main.ts frontend/src/components/EvalPanel.ts
git commit -m "Route main.ts and EvalPanel.ts through the shared mask helpers

Collapses three copies of the reroll-mask and cost-vs-optimal logic onto one
tested implementation. Behaviour unchanged."
```

---

### Task 3: Merge each die and its label into one button

The die is currently a `<div>` with a click handler: unreachable by keyboard, invisible to screen readers, with no pressed state. Merging the die and its label into one `<button>` fixes that and yields a single 56x74 touch target.

**Files:**
- Modify: `frontend/src/components/Die.ts` (whole file)
- Modify: `frontend/src/style.css:292-316` (the `.die-container`, `.die-arrow`, `.die-wrap` rules)

**Interfaces:**
- Consumes: `createDieSVG` from `/yatzy/shared/dice.js` (unchanged; **do not edit that file**), `DieAdvice` from `./mask.ts`.
- Produces: `createDie(onToggle, onIncrement, onDecrement): DieElements`, where `DieElements.update` now takes `{ value, held, label, disabled, faded }`. The `kept`, `isOptimalReroll`, and `isOptimalKeep` options are gone. Task 4 depends on this exact shape.

- [ ] **Step 1: Rewrite `Die.ts`**

Replace the entire contents of `frontend/src/components/Die.ts` with:

```typescript
// @ts-expect-error - shared module served at runtime
import { createDieSVG } from '/yatzy/shared/dice.js';
import type { DieAdvice } from '../mask.ts';

/** DOM handles and update function returned by {@link createDie}. */
export interface DieElements {
  container: HTMLDivElement;
  update: (opts: {
    value: number;
    held: boolean;
    label: DieAdvice | '';
    disabled: boolean;
    faded: boolean;
  }) => void;
}

/**
 * Creates a single die: an up arrow, one hold button containing the die SVG and
 * its HOLD/REROLL label, and a down arrow.
 *
 * The die and its label are deliberately ONE button, not a die with a button
 * beneath it. That gives one focus stop, one 56x74 touch target, and one control
 * serving mouse, keyboard, and touch. A separate 56x18 label button would fail
 * WCAG 2.2 SC 2.5.8 (24x24 minimum).
 *
 * This component carries no hint logic. The solver's advice lives in DiceBar's
 * advice row, so that it stays readable without toggling dice.
 *
 * Callbacks: onToggle (hold/reroll), onIncrement (+1), onDecrement (-1).
 */
export function createDie(
  onToggle: () => void,
  onIncrement: () => void,
  onDecrement: () => void,
): DieElements {
  const container = document.createElement('div');
  container.className = 'die-container';

  const upBtn = document.createElement('button');
  upBtn.type = 'button';
  upBtn.className = 'die-arrow';
  upBtn.innerHTML = '&#9650;';
  upBtn.setAttribute('aria-label', 'Increase die value');
  upBtn.addEventListener('click', onIncrement);

  const holdBtn = document.createElement('button');
  holdBtn.type = 'button';
  holdBtn.className = 'die-hold';
  holdBtn.addEventListener('click', onToggle);

  const dieWrap = document.createElement('span');
  dieWrap.className = 'die-wrap';
  holdBtn.appendChild(dieWrap);

  const labelEl = document.createElement('span');
  labelEl.className = 'die-label';
  holdBtn.appendChild(labelEl);

  const downBtn = document.createElement('button');
  downBtn.type = 'button';
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.setAttribute('aria-label', 'Decrease die value');
  downBtn.addEventListener('click', onDecrement);

  container.appendChild(upBtn);
  container.appendChild(holdBtn);
  container.appendChild(downBtn);

  function update(opts: {
    value: number;
    held: boolean;
    label: DieAdvice | '';
    disabled: boolean;
    faded: boolean;
  }) {
    // Border is always neutral. Green/red outlines used to encode agreement with
    // the solver here, which contradicted the legend and moved as the player
    // toggled. Fill and pips echo held state as a redundant cue only.
    let fill: string, pipOp: number;
    if (opts.faded) {
      fill = 'var(--bg-alt)';
      pipOp = 0.5;
    } else if (opts.held) {
      fill = 'var(--bg)';
      pipOp = 1;
    } else {
      fill = 'var(--bg-alt)';
      pipOp = 0.45;
    }

    const svg = createDieSVG(opts.value || 0, {
      size: 56,
      state: 'normal',
      clickable: !opts.disabled,
      fill,
      stroke: 'var(--border)',
      sw: 2,
      pipOp,
    });

    dieWrap.innerHTML = '';
    dieWrap.appendChild(svg);

    labelEl.textContent = opts.label;
    holdBtn.disabled = opts.disabled;
    holdBtn.setAttribute('aria-pressed', String(opts.held));
    holdBtn.classList.toggle('die-hold--reroll', !opts.held && !opts.faded);
    holdBtn.setAttribute(
      'aria-label',
      opts.value === 0
        ? 'Die, not yet rolled'
        : `Die showing ${opts.value}, ${opts.held ? 'held' : 'will be rerolled'}`,
    );
  }

  return { container, update };
}
```

- [ ] **Step 2: Update the die CSS**

In `frontend/src/style.css`, replace the `.die-container`, `.die-arrow`, and `.die-wrap` rules (currently lines 292-316) with:

```css
.die-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

/* 24px, not 16px: WCAG 2.2 SC 2.5.8 target size minimum. */
.die-arrow {
  width: 56px;
  height: 24px;
  font-size: 10px;
  padding: 0;
  border: 1px solid var(--border);
  background: var(--bg-alt);
  cursor: pointer;
  border-radius: 3px;
  line-height: 22px;
  color: var(--text);
  touch-action: manipulation;
}

.die-arrow:disabled {
  cursor: default;
  opacity: 0.4;
}

/* The die and its label are one control: one focus stop, one 56x74 tap target. */
.die-hold {
  width: 56px;
  padding: 0;
  border: none;
  background: none;
  color: var(--text);
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  border-radius: 4px;
  touch-action: manipulation;
}

.die-hold:disabled {
  cursor: default;
}

/* Uses the text ramp, not the brand orange: a focus indicator needs 3:1 per
   WCAG 1.4.11, and #F37021 measures only 2.69:1 on the light background. */
.die-hold:focus-visible {
  outline: 2px solid var(--color-orange-text);
  outline-offset: 2px;
}

.die-wrap {
  width: 56px;
  height: 56px;
  display: block;
}

/* Fixed height so a blank label cannot collapse the row. */
.die-label {
  height: 14px;
  line-height: 14px;
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--text);
}

/* Orange means "in motion", not "wrong". Green/red are reserved for quality.
   Uses the text ramp: the brand orange is only 2.69:1 on the light background. */
.die-hold--reroll .die-label {
  color: var(--color-orange-text);
}
```

Also add the text-orange token to both theme blocks. In `:root`, after `--color-orange: #F37021;`:

```css
  /* Orange for text on paper. The brand orange (#F37021) measures 2.69:1 on
     --bg, under the WCAG AA 4.5:1 floor. This darker ramp measures 4.87:1.
     Use --color-orange for fills and accents, this for text. */
  --color-orange-text: #B54A0C;
```

And in `.dark`, after its `--color-orange: #F37021;`:

```css
  /* #F37021 measures 5.93:1 on the dark --bg, so it needs no darkening. */
  --color-orange-text: #F37021;
```

- [ ] **Step 3: Verify types and lint**

```bash
cd frontend && npx tsc -b --noEmit
```

Expected: FAIL, with errors in `src/components/DiceBar.ts`. It still passes `kept`, `isOptimalReroll`, and `isOptimalKeep` to `update()`, which no longer accepts them. This is expected and Task 4 fixes it. Do not patch `DiceBar.ts` here.

- [ ] **Step 4: Commit**

Committing with `DiceBar.ts` temporarily broken is intentional: this task's deliverable is the die control, and Task 4's is the bar. They are reviewed separately.

```bash
git add frontend/src/components/Die.ts frontend/src/style.css
git commit -m "Make each die a single hold button with a HOLD/REROLL label

The die was a click-handler div: no keyboard access, no role, no pressed
state. It is now one button wrapping the die SVG and its label, giving one
focus stop and one 56x74 touch target. Border is permanently neutral; the
green/red agreement outline is gone. Value arrows grow 16px to 24px to meet
WCAG 2.2 SC 2.5.8.

DiceBar.ts does not compile until the next commit."
```

---

### Task 4: Render the view model, advice row, ticks, and cost line

**Files:**
- Modify: `frontend/src/components/DiceBar.ts` (whole file)
- Modify: `frontend/src/style.css` (append the dice-advice rules)

**Interfaces:**
- Consumes: `diceBarViewModel`, `DiceBarViewModel` from Task 1; `createDie`, `DieElements` from Task 3.
- Produces: `initDiceBar(container: HTMLElement): void` (unchanged signature).

- [ ] **Step 1: Rewrite `DiceBar.ts`**

Replace the entire contents of `frontend/src/components/DiceBar.ts` with:

```typescript
import { getState, dispatch, subscribe } from '../store.ts';
import { createDie, type DieElements } from './Die.ts';
import { diceBarViewModel, type DiceBarViewModel } from '../mask.ts';
import { TOTAL_DICE } from '../constants.ts';

/**
 * The five dice, plus the solver's advice row beneath them.
 *
 * The player's choice and the solver's advice are two separate rows sharing the
 * same columns. They used to be one object: the die's fill carried the choice and
 * its border carried the solver's verdict, which meant the advice was only
 * readable by toggling dice until the border turned green.
 *
 * All display logic lives in diceBarViewModel(). This function renders it.
 */
export function initDiceBar(container: HTMLElement): void {
  container.className = 'dice-bar-group';

  const diceRow = document.createElement('div');
  diceRow.className = 'dice-bar';
  container.appendChild(diceRow);

  const dice: DieElements[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const idx = i;
    const die = createDie(
      () => dispatch({ type: 'TOGGLE_DIE', index: idx }),
      () => {
        const s = getState();
        const v = s.turnPhase === 'rolled' ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v >= 6 || v === 0 ? 1 : v + 1 });
      },
      () => {
        const s = getState();
        const v = s.turnPhase === 'rolled' ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v <= 1 || v === 0 ? 6 : v - 1 });
      },
    );
    dice.push(die);
    diceRow.appendChild(die.container);
  }

  // Mismatch ticks sit between the two rows so the eye lands on the disagreement.
  const tickRow = document.createElement('div');
  tickRow.className = 'dice-ticks';
  container.appendChild(tickRow);
  const tickCells: HTMLSpanElement[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const cell = document.createElement('span');
    cell.className = 'dice-tick';
    tickRow.appendChild(cell);
    tickCells.push(cell);
  }

  const adviceHeader = document.createElement('div');
  adviceHeader.className = 'dice-advice-header';
  adviceHeader.textContent = 'OPTIMAL PLAY';
  container.appendChild(adviceHeader);

  const adviceRow = document.createElement('div');
  adviceRow.className = 'dice-advice';
  container.appendChild(adviceRow);
  const adviceCells: HTMLSpanElement[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const cell = document.createElement('span');
    cell.className = 'dice-advice-cell';
    adviceRow.appendChild(cell);
    adviceCells.push(cell);
  }

  const costLine = document.createElement('div');
  costLine.className = 'dice-cost';
  container.appendChild(costLine);

  function render(vm: DiceBarViewModel) {
    for (let i = 0; i < TOTAL_DICE; i++) {
      dice[i].update(vm.dice[i]);
    }

    // Heights stay reserved in every branch below: contents change, nothing moves.
    const showAdvice = vm.advice !== null;
    adviceHeader.style.visibility = showAdvice ? 'visible' : 'hidden';
    for (let i = 0; i < TOTAL_DICE; i++) {
      const advice = vm.advice?.[i] ?? '';
      adviceCells[i].textContent = advice;
      adviceCells[i].classList.toggle('dice-advice-cell--reroll', advice === 'REROLL');
      tickCells[i].classList.toggle('dice-tick--on', vm.ticks[i]);
    }

    costLine.textContent = vm.cost === null
      ? ''
      : `Costs you ${Math.abs(vm.cost).toFixed(1)} pts vs optimal`;
  }

  render(diceBarViewModel(getState()));
  subscribe((state, prev) => {
    if (state.dice === prev.dice &&
        state.lastEvalResponse === prev.lastEvalResponse &&
        state.prefs.showHints === prev.prefs.showHints &&
        state.rerollsRemaining === prev.rerollsRemaining &&
        state.turnPhase === prev.turnPhase) return;
    render(diceBarViewModel(state));
  });
}
```

- [ ] **Step 2: Append the advice-row CSS**

In `frontend/src/style.css`, replace the entire `/* Dice legend */` block (currently lines 318-333, covering `.dice-legend` and `.dice-legend .swatch`) with:

```css
/* Solver advice, in the same five columns as the dice. */
.dice-bar-group {
  margin: 12px 0 8px;
}

.dice-ticks {
  display: flex;
  gap: 8px;
  justify-content: center;
  height: 4px;
  margin-top: 4px;
}

.dice-tick {
  width: 56px;
  height: 4px;
  border-radius: 2px;
  background: transparent;
}

/* Red only ever means "giving up points". See mask.ts COST_EPS. */
.dice-tick--on {
  background: var(--color-danger);
}

.dice-advice-header {
  text-align: center;
  font-family: var(--font-mono);
  font-size: 8px;
  letter-spacing: 0.12em;
  color: var(--text-muted);
  height: 12px;
  line-height: 12px;
  margin-top: 2px;
}

.dice-advice {
  display: flex;
  gap: 8px;
  justify-content: center;
  min-height: 14px;
}

.dice-advice-cell {
  width: 56px;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.04em;
  line-height: 14px;
  color: var(--text);
}

.dice-advice-cell--reroll {
  color: var(--color-orange-text);
}

.dice-cost {
  text-align: center;
  min-height: 14px;
  line-height: 14px;
  font-size: 11px;
  color: var(--color-danger);
}
```

- [ ] **Step 3: Verify the suite and types**

```bash
cd frontend && npx vitest run && npx eslint src/components/DiceBar.ts
```

Expected: 58 tests passing, no lint errors. `npx tsc -b --noEmit` still fails on `DiceLegend.ts` and `App.ts` until Task 5 removes them.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/DiceBar.ts frontend/src/style.css
git commit -m "Split the dice bar into a choice row and a solver advice row

The advice row is a pure function of optimal_mask, so it no longer moves when
the player toggles dice. Red ticks flag only mask choices that actually cost
points; a costless tie is not flagged."
```

---

### Task 5: Delete `DiceLegend`, rewrite the tooltips, fix the caption

A legend exists to decode colours. Once the UI says `HOLD` and `REROLL` in words, there is nothing to decode. Deleting the file also removes the four contradictory tooltips at their source rather than rewording them.

**Files:**
- Delete: `frontend/src/components/DiceLegend.ts`
- Modify: `frontend/src/App.ts:5`, `:16-17`, `:75-79`
- Modify: `frontend/src/tooltips.ts:31-34`, `:57-62`

**Interfaces:**
- Consumes: `initDiceBar` from Task 4.
- Produces: nothing. `initDiceLegend` ceases to exist.

- [ ] **Step 1: Delete the legend component**

```bash
git rm frontend/src/components/DiceLegend.ts
```

- [ ] **Step 2: Unwire it from `App.ts`**

In `frontend/src/App.ts`, delete this import line (currently line 17):

```typescript
import { initDiceLegend } from './components/DiceLegend.ts';
```

Delete this block (currently lines 75-77):

```typescript
  const diceLegendEl = document.createElement('div');
  diceSection.appendChild(diceLegendEl);
  initDiceLegend(diceLegendEl);
```

Update the layout comment on line 5 from:

```typescript
 *   1. Right   — Dice (ActionBar + DiceBar + DiceLegend) + Analysis (EvalPanel + TrajectoryChart)
```

to (note: no emdash, per the project rule):

```typescript
 *   1. Right:  Dice (ActionBar + DiceBar) + Analysis (EvalPanel + TrajectoryChart)
```

Replace the caption on line 79, which still says "keep" and predates the labels:

```typescript
  diceSection.appendChild(panelCaption('Your five dice. Click a die to keep it; the arrows set values.'));
```

with:

```typescript
  diceSection.appendChild(panelCaption(
    'Your five dice. Click a die to hold or release it; the arrows set values. '
    + 'With hints on, the row beneath shows what optimal play would do.',
  ));
```

- [ ] **Step 3: Rewrite the tooltip copy**

In `frontend/src/tooltips.ts`, delete these four entries (currently lines 31-34):

```typescript
  // Dice legend
  legendKept: 'A die you are holding. It will not be rerolled.',
  legendReroll: 'A die that will be rerolled.',
  legendOptimal: 'Green outline: the optimal play is to hold this die.',
  legendSuboptimal: 'Red outline: the optimal play is to reroll this die; holding it gives up points.',
```

Then replace these five entries (currently lines 57-62):

```typescript
  // Dice (composed dynamically from the current state, see DiceBar)
  dieKept: 'You are holding this die.',
  dieReroll: 'This die will be rerolled.',
  dieAgree: ' Optimal play agrees.',
  dieKeepButReroll: 'You are holding this die, but optimal play is to reroll it.',
  dieRerollButKeep: 'This die will be rerolled, but optimal play is to hold it.',
```

with:

```typescript
  // Dice bar
  adviceRow:
    'What the optimal strategy would do with each die. It does not change as you ' +
    'hold or release dice, so you can read it straight off a fresh roll.',
  diceCost:
    'Expected points given up by your current hold versus the best one. ' +
    'A hold that ties the best one costs nothing and is not flagged.',
```

- [ ] **Step 4: Attach the two surviving tooltips**

In `frontend/src/components/DiceBar.ts`, add to the imports:

```typescript
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';
```

Then, immediately after `adviceHeader.textContent = 'OPTIMAL PLAY';`, add:

```typescript
  attachTooltip(adviceHeader, TIP.adviceRow);
```

And immediately after `container.appendChild(costLine);`, add:

```typescript
  attachTooltip(costLine, TIP.diceCost);
```

- [ ] **Step 5: Verify the whole thing compiles and passes**

```bash
cd frontend && npx tsc -b --noEmit && npx vitest run && npx eslint . && npx vite build
```

Expected: no type errors (nothing references `DiceLegend` or the deleted tooltips), 58 tests passing, no lint errors, clean build.

- [ ] **Step 6: Verify no dangling references remain**

```bash
cd /Users/langkilde/dev/public/yatzy && grep -rn "DiceLegend\|legendKept\|legendReroll\|legendOptimal\|legendSuboptimal\|dieKept\|dieReroll\|dieAgree\|dieKeepButReroll\|dieRerollButKeep\|dice-legend" frontend/ --exclude-dir=node_modules --exclude-dir=dist
```

Expected: no output. If `frontend/CLAUDE.md` matches, leave it: Task 6 fixes the docs.

- [ ] **Step 7: Commit**

```bash
git add -A frontend/src
git commit -m "Delete DiceLegend and the tooltips that contradicted the code

The legend told users green meant 'hold this die' while Die.ts made green mean
'you agree with the solver', so a die correctly marked for reroll showed green.
Deleting the legend removes the contradictory copy at its source: a UI that
says HOLD and REROLL in words has nothing left to decode."
```

---

### Task 6: Update the docs and run the manual verification

`frontend/CLAUDE.md` is already wrong before this change: it lists a `mask.ts` and `mask.test.ts` that did not exist, and claims 46 tests when 35 ran. Task 1 recreated `mask.ts`, so the entry becomes true again; the counts still need fixing.

**Files:**
- Modify: `frontend/CLAUDE.md`

**Interfaces:**
- Consumes: everything above. Produces: nothing.

- [ ] **Step 1: Get the real test count**

```bash
cd frontend && npx vitest run 2>&1 | grep -E "Test Files|Tests "
```

Expected: `Test Files  4 passed (4)` and `Tests  58 passed (58)`. Use the actual numbers below rather than trusting this plan's arithmetic.

- [ ] **Step 2: Fix the test line in `frontend/CLAUDE.md`**

Replace:

```markdown
- **Tests**: vitest — `mask.test.ts` (11), `reducer.test.ts` (23), `store.test.ts` (6), `api.test.ts` (6)
```

with (substituting the real per-file counts from Step 1):

```markdown
- **Tests**: vitest, node environment, no DOM. `mask.test.ts` (23), `reducer.test.ts` (23), `store.test.ts` (6), `api.test.ts` (6)
```

Also update the `npm test` comment near the top of the file from `# Run vitest (46 tests)` to `# Run vitest (58 tests)`.

- [ ] **Step 3: Fix the source layout table in `frontend/CLAUDE.md`**

Replace the `mask.ts` row:

```markdown
| `mask.ts` | Dice sort-mapping utilities |
```

with:

```markdown
| `mask.ts` | Pure dice-mask helpers and the DiceBar view model. All dice-bar display logic lives here so it is testable without a DOM |
```

Delete this row entirely:

```markdown
| `components/DiceLegend.ts` | Optimal keep/reroll color legend |
```

Replace the `DiceBar.ts` and `Die.ts` rows:

```markdown
| `components/DiceBar.ts` | Five-die display with keep toggles |
| `components/Die.ts` | Single die SVG with pip layout |
```

with:

```markdown
| `components/DiceBar.ts` | Renders `diceBarViewModel()`: five dice, the solver advice row, mismatch ticks, cost line |
| `components/Die.ts` | Single die: value arrows plus one button wrapping the die SVG and its HOLD/REROLL label |
```

- [ ] **Step 4: Document the colour contract in `frontend/CLAUDE.md`**

Add this section immediately before the `## No Layout Shifts (Hard Requirement)` heading:

```markdown
## Colour Contract

Each colour means exactly one thing across the Play UI. Do not overload them.

| Colour | Meaning | Example |
|--------|---------|---------|
| ink (`--text`) | Hold: settled | Die label, advice row |
| orange (`--color-orange`) | Reroll: in motion | Die label, advice row |
| green (`--color-success`) | Good, ahead of par | Scorecard bonus |
| red (`--color-danger`) | Giving up points | Mismatch ticks, cost line, scorecard bonus |

Green and red previously encoded hold/reroll on the dice while also meaning
quality in the scorecard. That collision is the bug fixed in
`docs/superpowers/specs/2026-07-17-dice-hold-reroll-clarity-design.md`. Do not
reintroduce it.
```

- [ ] **Step 5: Run the full quality gate**

```bash
cd /Users/langkilde/dev/public/yatzy && just typecheck && cd frontend && npx vitest run && npx eslint . && npx vite build
```

Expected: all green.

- [ ] **Step 6: Manual verification in a real browser**

Not automated: there is no DOM test environment, by decision. Desktop browser with a mouse is the prioritized platform, so the first four items are the gate and the rest are best-effort.

```bash
# Terminal 1
just dev-backend
# Terminal 2
just dev-frontend
# Then open http://localhost:5173/yatzy/play/
```

Mouse (the priority path):

- [ ] Click Roll, then click a die: its label flips HOLD to REROLL and the fill dims
- [ ] With hints on, the advice row reads left to right and does **not** change when you toggle dice
- [ ] Hold a die the solver wants rerolled: a red tick appears under it and the cost line states the points
- [ ] Match the solver exactly: no ticks, no cost line
- [ ] Toggling Hints on and off shifts nothing else on the page
- [ ] Dark mode: labels, ticks, and cost line are all legible

Keyboard:

- [ ] Tab reaches every die's hold button; Space and Enter both toggle it
- [ ] Focus ring is visible on the hold button in both light and dark themes
- [ ] `aria-pressed` flips with held state (inspect the accessibility tree in devtools)
- [ ] A screen reader announces the die's value and its held state (macOS: Cmd+F5 for VoiceOver)

Touch (best-effort, per platform priority):

- [ ] On a phone or devtools device emulation at 390px width: the dice bar does not scroll horizontally
- [ ] Tapping a die toggles hold, with no double-tap zoom delay
- [ ] Arrows are tappable without hitting the hold button by mistake
- [ ] Measure in devtools: the hold button is at least 24x24 CSS px, the arrows are 56x24

- [ ] **Step 7: Commit**

```bash
git add frontend/CLAUDE.md
git commit -m "Update frontend docs for the dice bar redesign

Corrects two pre-existing errors: mask.ts and mask.test.ts were listed but did
not exist, and the test count said 46 when 35 ran. Drops DiceLegend, and adds
the colour contract so the green/red overload is not reintroduced."
```

---

## Definition of Done

- [ ] `cd frontend && npx vitest run` reports 58 tests in 4 files, all passing
- [ ] `just typecheck` is clean
- [ ] `cd frontend && npx eslint .` is clean
- [ ] `cd frontend && npx vite build` succeeds
- [ ] No file under `frontend/` mentions `DiceLegend` or the deleted tooltip keys
- [ ] `shared/dice.js` is untouched (`git diff --stat shared/` is empty)
- [ ] The mouse and keyboard items in Task 6 Step 6 are confirmed in a real browser
- [ ] `frontend/CLAUDE.md` test counts match actual output
