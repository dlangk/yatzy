# Dice Hold/Reroll Clarity: Design

**Date:** 2026-07-17
**Status:** Approved (design), pending implementation plan

## Summary

The Play UI dice bar encodes two different meanings (the player's own hold/reroll
selection, and the solver's judgment of it) inside a single 56px die, using
overlapping visual channels. Users cannot tell which dice are being rerolled, and
cannot read the solver's advice without toggling dice and watching colors move.

This change separates the two meanings into two vertically aligned rows sharing the
same five columns. Each die becomes a single labeled `HOLD` / `REROLL` button stating
**your** choice, and a read-only **advice row** beneath states what the solver would
do. Colour stops carrying two jobs: hold/reroll becomes ink vs orange, and green/red
is reserved for its existing meaning of good vs bad.

`DiceLegend` is deleted. Once the UI states `HOLD` and `REROLL` in words, there is
nothing left for a legend to decode.

## Motivation: Defects in Current Behaviour

Investigated before design. All four are in `frontend/src/`.

### 1. The legend states the opposite of what the code does (functional bug)

`tooltips.ts:33-34` documents the outline colours as actions:

> `legendOptimal`: "Green outline: the optimal play is to hold this die."
> `legendSuboptimal`: "Red outline: the optimal play is to reroll this die; holding
> it gives up points."

`Die.ts:71-83` implements them as *agreement*:

| player | solver | outline |
|--------|--------|---------|
| kept   | keep   | green   |
| kept   | reroll | red     |
| reroll | reroll | green   |
| reroll | keep   | red     |

A die marked for reroll that the solver also wants rerolled shows a **green**
outline, while the legend says green means hold. Any user who reads the legend and
then reads the dice receives a direct contradiction.

### 2. The legend contradicts itself

`DiceLegend.ts:11-12` labels the swatches "Optimal" and "Suboptimal" (a *correctness*
model) while their tooltips describe *actions*. Two mental models in one four-item
legend.

### 3. The hint is feedback, not advice

Because the outline keys off agreement, colours rearrange as the player toggles. The
solver's recommendation is not readable from a fresh roll; it can only be found by
toggling until green appears. Meanwhile `EvalPanel` presents the same information as
a stable list ("Best keep"), so one screen states the same fact two incompatible ways.

### 4. Two meanings, overlapping channels, inverted priority

The die carries selection (fill `--bg` vs `--bg-alt`, plus pip opacity) and judgment
(saturated border). Selection, which the player controls and most needs to see, gets
the weakest channel available: a subtle gray shift. Judgment gets the loudest.
Red/green additionally fights the convention that green means go, which is the wrong
intuition here.

### 5. Keep toggle is not accessible (found while investigating)

`Die.ts:42` attaches `click` to a bare `<div>`. It is not focusable, has no role, and
has no pressed state. Keyboard and screen-reader users cannot hold a die.

### Non-defect checked and cleared

`optimal_mask` bit indexing was audited for a sort-mapping bug (hints pointing at the
wrong dice). It is correct: masks are in original dice order, used consistently in
`DiceBar.ts:42`, `EvalPanel.ts:104`, `autoplay.ts:56`.

## Goals

- A player can tell at a glance which dice will be rerolled, without hovering.
- A player can read the solver's recommendation from a fresh roll, without toggling.
- Disagreement with the solver is visible, and its cost is stated once.
- Each colour means exactly one thing across the whole Play UI.
- Hold/reroll works under mouse, keyboard, and touch from the same markup, with no
  input mode treated as a second-class path.
- Keyboard and screen-reader users can hold and release dice.
- Every interactive target meets WCAG 2.2 SC 2.5.8 (Target Size Minimum, AA, 24x24).

## Non-Goals

- No change to the solver, to `/evaluate`, or to any API contract.
- No per-die EV attribution. EV is defined per keep-mask (`mask_evs` is indexed by
  the whole 5-bit mask), so "this die costs you 2.3 points" is not a well-defined
  quantity. Cost is reported once per roll.
- No change to `EvalPanel`, `DeltaBox`, or `DecisionLog`.
- No drag-to-tray interaction. `.claude/rules/frontend.md` forbids elements moving on
  state change.
- No DOM/component test environment. See Testing for what this leaves uncovered.
- Value arrows are resized but not otherwise redesigned. They stay always-visible and
  in tab order.

## Design

### Layout

```
 ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
 │ ▲  │ │ ▲  │ │ ▲  │ │ ▲  │ │ ▲  │    value arrow, 56x24 (was 56x16)
 ├────┤ ├────┤ ├────┤ ├────┤ ├────┤
 │ ●● │ │ ●  │ │ ●● │ │ ●● │ │ ●  │    ─┐
 │ ●● │ │    │ │ ●  │ │ ●● │ │  ● │     │ ONE <button aria-pressed>
 │HOLD│ │RRLL│ │HOLD│ │RRLL│ │HOLD│    ─┘ 56x74. die + label, one target
 ├────┤ ├────┤ ├────┤ ├────┤ ├────┤
 │ ▼  │ │ ▼  │ │ ▼  │ │ ▼  │ │ ▼  │    value arrow, 56x24
 └────┘ └────┘ └────┘ └────┘ └────┘
              ▃             ▃          mismatch strip (4px, always reserved)
   HOLD  REROLL REROLL REROLL HOLD     advice row: read-only, never recolors
        Costs you 2.3 pts vs optimal   cost line
```

`RRLL` above is an artifact of the 4-character ASCII cell. The real label is
`REROLL`, spelled out.

The advice row reuses the dice columns, so the component gains no width. At the
existing 56px die and 8px gap, five columns occupy 312px and still fit a 390px
phone viewport.

Vertical cost: the die container grows from 92px to roughly 126px (arrows 16px to
24px each, plus the ~18px label), and the advice block replaces the 16px legend for
about +16px. Net roughly +50px.

### One control, three input modes

The die and its `HOLD` / `REROLL` label are a **single `<button>`**, not a die with a
button beneath it. This is what makes the same markup serve all three input modes:

| Mode | Interaction |
|------|-------------|
| Mouse | Click the die, as today |
| Keyboard | Tab to the die, Space/Enter toggles. This is the only keyboard gap today: the arrows are already real `<button>`s and reachable, but the toggle is a `<div>` (defect #5) |
| Touch | One 56x74 target, comfortably past the 24x24 minimum and past Apple's 44x44pt |

A separate 56x18 label-button was considered and rejected: 18px tall fails WCAG 2.2
SC 2.5.8 (Target Size Minimum, Level AA, 24x24 CSS px), and it would put two adjacent
controls where one suffices.

### Colour semantics

| Meaning | Channel | Rationale |
|---------|---------|-----------|
| Hold | ink (`--text`), solid | Settled, locked |
| Reroll | orange (`--accent`), tinted | In motion, about to change. Already the app's accent |
| Giving up points | red (`--color-danger`) | Matches `Scorecard.ts:170`, where red already means behind par |

Green and red are **not** used for hold/reroll. They already mean quality in
`Scorecard.ts:164-170` (upper-bonus par). Reusing them for action would recreate
defect #4 in a new location, and would paint a legitimate reroll as an error.

### Component responsibilities

| Component | Change |
|-----------|--------|
| `Die.ts` | The `<div>` wrapper becomes a `<button>` containing the die SVG and its label. Border permanently neutral. Drop `isOptimalKeep` / `isOptimalReroll` from the update options. Keep fill and pip opacity as a redundant echo of hold. Arrows grow to 24px |
| `DiceBar.ts` | Renders `diceBarViewModel(state)`. Owns the advice row, mismatch strip, and cost line |
| `DiceLegend.ts` | **Deleted** |
| `tooltips.ts` | Remove `legendKept`, `legendReroll`, `legendOptimal`, `legendSuboptimal`. Rework the four `die*` entries: the labels now state the action, so tooltips explain consequence, not decode colour |
| `mask.ts` | **Recreated**. Exports `rerollMaskFromDice(dice)`, `maskCostVsOptimal(evalResponse, mask)`, and `diceBarViewModel(state)` |
| `App.ts` | Drop the `initDiceLegend` call |
| `style.css` | `.die-arrow` height 16px to 24px. Remove `.dice-legend` rules. Add advice row, tick strip, cost line |

### The hold button

- Text is spelled out: `HOLD` / `REROLL`. Six characters at 10px mono is ~36px, fits
  in 56px. Icons alone are not unambiguous and the goal is explicitness.
- `aria-pressed` reflects held state; `aria-label` names the die and its value.
- Disabled when `turnPhase !== 'rolled'`, when there is no eval response, or when
  `rerollsRemaining === 0`, matching today's `canToggle` in `DiceBar.ts:40`.
- `touch-action: manipulation` to suppress the 300ms double-tap zoom delay.

### Advice row

A pure function of `optimal_mask`. It never changes as the player toggles, which is
what makes a fresh roll readable as a sentence.

Shown when `prefs.showHints && turnPhase === 'rolled' && hasEval && rerollsRemaining > 0`,
which is today's `showMaskHints` condition (`DiceBar.ts:43`), unchanged.

### Mismatch strip and cost line

A 4px strip between the rows. The cost of the player's current hold is:

```
cost = mask_evs[rerollMaskFromDice(dice)] - optimal_mask_ev     // <= 0
```

This is the identical formula already used in `main.ts:41-49` and `EvalPanel.ts:122-128`.
Extracting it into `mask.ts` removes a third copy rather than adding one.

**Ticks are gated on cost, not on mask difference.** Red means "giving up points",
so it must not appear when no points are given up. Two distinct keep-masks can have
equal EV; in that case the player's hold differs from `optimal_mask` yet costs
nothing, and flagging it red would contradict the colour semantics above and call a
free alternative a mistake.

Rule:

- `cost < -EPS` (`EPS = 0.005`, half the displayed precision): tick every column
  where the choice differs from `optimal_mask`, and show the cost line.
- otherwise: no ticks, no cost line. This covers both an exact match and a costless
  tie.

Consequence, and it is intended: the advice row keeps showing the solver's mask while
a costless alternative goes unflagged. The advice remains "what the solver would do",
not "the only acceptable play".

Heights stay reserved in all cases.

### Hidden states

Per `.claude/rules/frontend.md`, nothing may move on state change. The advice row,
mismatch strip, and cost line reserve their height via `min-height` and hide their
contents, following the existing `DiceLegend` pattern (`style.css:318-327`,
`min-height: 16px`). Applies when hints are off, when `rerollsRemaining === 0`, and
when the turn is not in the `rolled` phase.

Net vertical change is roughly +50px. See the Layout section for the breakdown.

## Testing

Current frontend baseline is **35 tests in 3 files** (`api`, `reducer`, `store`),
verified by `npx vitest run`. There is no DOM environment: `vite.config.ts` has no
`test` block, and jsdom, happy-dom, and testing-library are all absent. The existing
tests run in plain node and stub globals (`store.test.ts:3-5`).

**Decision: no new test dependency.** All semantics move into pure functions tested
in node, matching the existing pattern. The DOM layer stays a dumb projection with no
logic of its own.

### Testability strategy

`diceBarViewModel(state)` is the seam. It is pure, takes store state, and returns
everything the DOM needs:

```
{ dice:    [{ value, held, disabled, faded }, ...],
  advice:  ['HOLD' | 'REROLL', ...] | null,   // null when hidden
  ticks:   [boolean, ...],
  cost:    number | null,
  visible: { advice, ticks, cost } }
```

`DiceBar.ts` renders this and nothing else. Every rule in this spec then becomes a
pure assertion.

### `mask.test.ts` (new)

- `rerollMaskFromDice`: all held (0), none held (31), mixed, bit order matches the
  `optimal_mask` convention (bit set = reroll).
- `maskCostVsOptimal`: 0 when the player mask equals the optimal mask; negative for a
  genuinely worse mask; null when `mask_evs` or `optimal_mask_ev` is absent.
- **Regression for defect #1** (the shipped bug): advice derives from `optimal_mask`
  alone. Assert the advice array is identical across two states differing only in
  `kept` flags. This is the test that would have caught the legend/code contradiction.
- **Regression for defect #3**: toggling any die never mutates `advice`.
- **Costless tie**: a mask differing from `optimal_mask` but equal in EV yields no
  ticks and no cost line. Most likely rule to regress.
- Cost gating at the `EPS = 0.005` boundary: `-0.004` yields no ticks, `-0.006` does.
- Hidden states: hints off, `rerollsRemaining === 0`, and `turnPhase !== 'rolled'`
  each yield `visible.advice === false` while the dice array stays fully populated.

`main.ts` and `EvalPanel.ts` must produce identical values after switching to the
shared helper; their behaviour is covered by `reducer.test.ts`.

### Manual verification (not automated)

Accepted gap: with no DOM environment, the accessibility and touch work below is
**not covered by any automated test**. This repo has no CI and tests are run by hand,
so these are checklist items for the implementer, confirmed in a real browser before
this is called done:

- [ ] Tab reaches every die's hold button; Space and Enter both toggle it
- [ ] `aria-pressed` flips with held state (inspect the accessibility tree)
- [ ] A screen reader announces the die's value and its held state
- [ ] Focus ring is visible on the hold button in light and dark themes
- [ ] Hold button measures at least 24x24 CSS px; arrows measure 56x24
- [ ] On a real phone: tapping the die toggles hold, and no arrow mis-tap silently
      changes a die's value
- [ ] No double-tap zoom delay on the hold button
- [ ] Toggling hints on and off shifts nothing on the page (no-layout-shift rule)

### Known residual risk

The arrows now meet WCAG AA (24px) but remain below Apple HIG (44pt) and Material
(48dp), and they stay flush against a much larger hold target. A mis-tap on touch can
still change a die's **value** rather than its held state, which corrupts game state
rather than merely missing. Accepted for this change; revisit if it shows up in use.

## Documentation Updates

`frontend/CLAUDE.md` is currently inaccurate and must be corrected as part of this
change:

- It lists `mask.ts` and `mask.test.ts` (11 tests) in the source layout and test
  list. Neither exists. The claimed total of 46 tests is 35 actual plus those 11,
  confirming both were deleted without a doc update. This change recreates `mask.ts`,
  so the entry becomes true again; the test count must be restated to the real total.
- The `components/DiceLegend.ts` row must be removed from the component table.
- The `components/DiceBar.ts` and `components/Die.ts` descriptions must be updated.

## Open Questions

None.

## See Also

- `.claude/rules/frontend.md`: no-layout-shift constraint
- `frontend/CLAUDE.md`: component table, test counts
