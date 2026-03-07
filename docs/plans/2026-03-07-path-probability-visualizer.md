# Path Probability Visualizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** An interactive teaching tool in Section 3 (The Solver / Transition Probabilities) that lets users construct a three-roll Yatzy path and see how path probability compounds multiplicatively.

**Architecture:** A single chart module (`path-probability.js`) renders two side-by-side visualizations: a probability chain (numeric) and a probability funnel (D3 SVG). The user picks dice values and keep/reroll decisions across three rolls using clickable dice. All probability math is computed client-side using the same multinomial logic already in `transition-matrix.js`. No data files, no API calls.

**Tech Stack:** Vanilla JS, D3.js (SVG for funnel), CSS variables for theming. Reuses `renderDiceSelectable` from `js/utils/dice-interactive.js` and helpers from `js/yatzy-viz.js`.

---

## Conventions Reference

Before starting, read these files for context:
- `treatise/CLAUDE.md` — build system, chart pattern, theme rules, no-emdash rule
- `treatise/js/charts/transition-matrix.js` — multiset enumeration, multinomial probability math (reuse these functions)
- `treatise/js/utils/dice-interactive.js` — `renderDiceSelectable()` API
- `treatise/js/yatzy-viz.js` — `COLORS`, `resolveColor()`, `getTextColor()`, `getMutedColor()`, `createChart()`
- `treatise/css/charts.css` — existing chart CSS classes (`.chart-container`, `.chart-caption`, `.dice-row`, `.die`, `.chart-stats-panel`, `.chart-stat`, `.chart-btn`)
- `treatise/index.html:24-139` — how chart modules are imported and registered in `chartInits`
- `treatise/sections/03-solver.md:53-65` — current "Transition Probabilities" subsection where we insert the new chart

## Key Design Decisions

1. **Scandinavian Yatzy** (not American Yahtzee): 5 dice, 3 rolls per turn, 15 categories, 50-point upper bonus. The PRD says "Yahtzee" in its title but we always say "Yatzy".
2. **No animation** per PRD. Static updates on interaction.
3. **Layout**: The two visualizations (chain + funnel) stack vertically inside one `.chart-container`. Chain on top (compact), funnel below (SVG).
4. **Dice interaction**: Each roll step has 5 clickable dice. Values cycle 1-6 on click. After setting a roll, the user clicks dice to toggle keep/reroll. A "Next" button advances to the next step.
5. **Funnel visualization**: A horizontal SVG with 4 columns (start, roll 1, roll 2, roll 3). The user's path is a highlighted band whose width is proportional to its probability. The "rest" of the probability mass is shown as a gray band.
6. **Probability display**: Fractions (e.g. `6/216`) and "1 in N" format. Never raw decimals. Final product gets a human-scale anchor sentence.

---

### Task 1: Create the chart module with probability math

**Files:**
- Create: `treatise/js/charts/path-probability.js`

This is the largest task. The module exports `initPathProbability()` which:
1. Builds the DOM: three roll-step panels (each with 5 dice + keep toggles), the probability chain display, and the funnel SVG.
2. Computes probabilities using multinomial math (extracted from `transition-matrix.js` pattern).
3. Updates both visualizations on every user interaction.

**Step 1: Create the chart module**

Write the complete `path-probability.js` file. Key internal structure:

```
// Imports: COLORS, resolveColor, getTextColor, getMutedColor from yatzy-viz.js
// Imports: PIPS from utils/dice-interactive.js (for rendering dice SVGs)

// ── Probability math ──
// multinomialProb(dice, k): probability of a specific 5-die or k-die outcome
//   For initial roll (5 dice): count arrangements via multinomial coefficient / 6^5
//   For reroll (free dice): multinomial coefficient of free dice values / 6^free
//   "dice" is a sorted array of face values, "k" is how many dice were kept

// ── State ──
// steps[0]: { dice: [1,1,1,1,1], kept: Set() }  — roll 1 + keeps
// steps[1]: { dice: [1,1,1,1,1], kept: Set() }  — roll 2 + keeps
// steps[2]: { dice: [1,1,1,1,1] }                — roll 3 (final)
// activeStep: 0|1|2 — which step the user is editing

// ── DOM structure (built in init) ──
// <div class="path-prob-widget">
//   <div class="path-steps">          ← three roll panels side by side
//     <div class="path-step">         ← one per roll
//       <div class="path-step-header">Roll 1</div>
//       <div class="path-step-dice">  ← 5 clickable dice (SVG, cycle value on click)
//       <div class="path-step-keep">  ← "Click dice to keep" instruction + toggle state
//       <div class="path-step-prob">  ← probability of this step
//     </div>
//     ...
//   </div>
//   <div class="path-chain">          ← the multiplication chain
//     P₁ × P₂ × P₃ = P(path)
//     + human-scale anchor sentence
//   </div>
//   <div class="path-funnel-svg">     ← D3 SVG funnel
//   </div>
// </div>

// ── Dice rendering ──
// Use inline SVG dice (same pip layout as PIPS in dice-interactive.js)
// Click on a die: cycle its value 1→2→...→6→1
// For rolls 2 and 3: only the rerolled dice are editable; kept dice are locked
// Keep toggle: after setting dice values, click to mark as kept (accent border)
// Kept dice carry forward to the next roll (shown grayed in next panel)

// ── Probability chain rendering ──
// Three cards showing:
//   label: "Roll 1", "Reroll (3 free dice)", "Reroll (2 free dice)"
//   fraction: e.g. "6 / 7,776"
//   oneInN: e.g. "1 in 1,296"
//   opacity: scale with running product (dimmer = smaller)

// Final product:
//   fraction, "1 in N", anchor: "If N people played..."

// ── Funnel rendering (D3 SVG) ──
// Horizontal layout: 4 columns (Start, After Roll 1, After Roll 2, After Roll 3)
// Each column is a vertical bar of height H representing probability 1
// The user's path is a highlighted band (accent color) whose height = P(path so far)
// The rest is gray
// Highlighted bands are connected left-to-right by a trapezoid showing the narrowing
// Labels on the highlighted band show the running probability

// On the first column (Start), the full bar is highlighted (P=1)
// On column 2, highlighted height = P(roll 1)
// On column 3, highlighted height = P(roll 1) × P(roll 2 | kept)
// On column 4, highlighted height = P(roll 1) × P(roll 2) × P(roll 3)
// Since probabilities get very small, use a log-scale or minimum pixel height
//   (e.g. min 2px) so the path remains visible even when P is tiny
```

The probability math for each step:

- **Roll 1 (5 dice, no keeps):** `multinomial(5, freqs) / 6^5` where `freqs` are the frequency counts of each face value. Example: dice [1,2,3,4,5] has freqs [1,1,1,1,1,0], multinomial = 5! / (1!×1!×1!×1!×1!×0!) = 120, P = 120/7776.
- **Roll 2 (reroll after keeps):** Only the free dice matter. If `k` dice are kept, `free = 5 - k` dice are rerolled. The rerolled dice must produce specific values. P = `multinomial(free, freeFreqs) / 6^free`. The `freeFreqs` are computed by subtracting kept-dice frequencies from the full roll-2 frequencies.
- **Roll 3:** Same as roll 2, using keeps from step 2.

**Step 2: Verify it builds**

Run: `cd treatise && npm run build`
Expected: No errors (the module isn't wired up yet, but it should be valid JS)

**Step 3: Commit**

```bash
git add treatise/js/charts/path-probability.js
git commit -m "Add path-probability chart module for transition probability visualizer"
```

---

### Task 2: Add the chart container to Section 3 markdown

**Files:**
- Modify: `treatise/sections/03-solver.md:61-65`

**Step 1: Insert the chart container**

After the existing transition-matrix chart container (line 65, after the `:::`), add a new chart container. The insertion goes between the closing `:::` of the transition-matrix block and the `### The Keep Shortcut` heading.

Find this text in `03-solver.md`:
```
:::html
<div class="chart-container" id="chart-transition-matrix">
  <p class="chart-caption">The 462 &times; 252 keep-to-outcome probability matrix. Each colored cell is a non-zero entry. Columns grouped by number of dice kept (k).</p>
</div>
:::

### The Keep Shortcut
```

Replace with:
```
:::html
<div class="chart-container" id="chart-transition-matrix">
  <p class="chart-caption">The 462 &times; 252 keep-to-outcome probability matrix. Each colored cell is a non-zero entry. Columns grouped by number of dice kept (k).</p>
</div>
:::

These transition probabilities compound multiplicatively across a turn. Each roll narrows the set of reachable outcomes, and the probability of any specific three-roll path is the product of three conditional probabilities. The tool below lets you construct a path, roll by roll, and watch the joint probability shrink.

:::html
<div class="chart-container" id="chart-path-probability">
  <p class="chart-caption">Construct a three-roll path by setting dice values and choosing which to keep. The chain shows how each step's probability multiplies into the path total; the funnel shows the path narrowing geometrically.</p>
</div>
:::

### The Keep Shortcut
```

**Step 2: Rebuild treatise**

Run: `cd treatise && npm run build`
Expected: Builds successfully. The new `<div id="chart-path-probability">` appears in `sections/03-solver.html`.

**Step 3: Commit**

```bash
git add treatise/sections/03-solver.md
git commit -m "Add path-probability chart container to Section 3"
```

---

### Task 3: Wire up the chart in index.html

**Files:**
- Modify: `treatise/index.html:37-38` (import) and `treatise/index.html:116-117` (chartInits registration)

**Step 1: Add the import**

Find this block near line 37:
```js
import { initTransitionMatrix } from './js/charts/transition-matrix.js';
// Chart modules — Solver
```

Add the import immediately after `initTransitionMatrix`:
```js
import { initTransitionMatrix } from './js/charts/transition-matrix.js';
import { initPathProbability } from './js/charts/path-probability.js';
// Chart modules — Solver
```

**Step 2: Register in chartInits**

Find this block near line 116:
```js
'chart-transition-matrix': { init: initTransitionMatrix, loaded: false },
```

Add the new entry immediately after:
```js
'chart-transition-matrix': { init: initTransitionMatrix, loaded: false },
'chart-path-probability': { init: initPathProbability, loaded: false },
```

**Step 3: Rebuild and verify**

Run: `cd treatise && npm run build`
Expected: Builds with no errors. The chart container is in the HTML and registered for lazy init.

**Step 4: Commit**

```bash
git add treatise/index.html
git commit -m "Wire path-probability chart into treatise lazy loading"
```

---

### Task 4: Add CSS for the path probability widget

**Files:**
- Modify: `treatise/css/charts.css` (append before the responsive media query at line 499)

**Step 1: Add the styles**

Insert before the `@media (max-width: 720px)` block at line 499:

```css
/* Path probability visualizer */
.path-prob-widget {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.path-steps {
  display: flex;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.path-step {
  flex: 1;
  padding: 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.path-step + .path-step {
  border-left: 1px solid var(--border);
}

.path-step-header {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--text);
}

.path-step-subheader {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.path-step-dice {
  display: flex;
  gap: 4px;
  justify-content: center;
}

.path-step-dice .die-svg {
  width: 40px;
  height: 40px;
  cursor: pointer;
  transition: transform 0.1s;
}

.path-step-dice .die-svg:hover {
  transform: scale(1.08);
}

.path-step-dice .die-svg.die-kept {
  opacity: 0.4;
  cursor: default;
}

.path-step-dice .die-svg.die-kept:hover {
  transform: none;
}

.path-step-keep {
  display: flex;
  gap: 4px;
  justify-content: center;
  min-height: 28px;
  align-items: center;
}

.path-step-keep-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-align: center;
}

.path-step-prob {
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.path-step-prob-fraction {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 600;
}

.path-step-prob-onein {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* Probability chain */
.path-chain {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding: 0.75rem;
  background: var(--bg-alt);
  border-radius: 8px;
}

.path-chain-factor {
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.path-chain-factor-fraction {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  font-weight: 600;
}

.path-chain-factor-label {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.path-chain-op {
  font-size: 1.2rem;
  color: var(--text-muted);
  font-weight: 300;
}

.path-chain-result {
  text-align: center;
  padding: 0.5rem 1rem;
  border-left: 2px solid var(--accent);
}

.path-chain-result-value {
  font-family: var(--font-mono);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
}

.path-chain-result-label {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.path-chain-anchor {
  width: 100%;
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-muted);
  font-style: italic;
  margin-top: 0.25rem;
}

/* Funnel SVG */
.path-funnel-svg svg {
  width: 100%;
  height: auto;
  display: block;
}
```

Also add inside the existing `@media (max-width: 720px)` block:

```css
.path-steps { flex-direction: column; }
.path-step + .path-step { border-left: none; border-top: 1px solid var(--border); }
.path-chain { font-size: 0.85rem; }
```

**Step 2: Rebuild and verify**

Run: `cd treatise && npm run build`
Expected: No errors.

**Step 3: Commit**

```bash
git add treatise/css/charts.css
git commit -m "Add CSS for path probability visualizer"
```

---

### Task 5: Visual testing and polish

**Step 1: Start dev servers**

Run in two terminals:
```bash
just dev-backend    # Terminal 1
just dev-frontend   # Terminal 2
```

**Step 2: Open the page**

Navigate to `http://localhost:5173/yatzy/#solver` and scroll to the "Transition Probabilities" subsection. The path probability widget should appear below the transition matrix.

**Step 3: Verify interactions**

- Click dice to cycle values (1 through 6)
- Click dice in the keep row to toggle keep/reroll
- Kept dice carry forward to next roll (shown grayed out)
- Probability chain updates with each change
- Funnel SVG shows path narrowing
- All probabilities display as fractions and "1 in N" (never raw decimals)
- Final product includes human-scale anchor sentence
- Dark mode: toggle theme, verify all colors update correctly
- Responsive: narrow browser window, verify stacking works

**Step 4: Fix any visual issues found in step 3**

Iterate on the chart module and CSS until the widget looks right.

**Step 5: Final commit**

```bash
git add -u
git commit -m "Polish path probability visualizer"
```

---

## File Change Summary

| Action | File | What |
|--------|------|------|
| Create | `treatise/js/charts/path-probability.js` | Chart module (~400 lines) |
| Modify | `treatise/sections/03-solver.md` | Add intro paragraph + chart container div |
| Modify | `treatise/index.html` | Import + chartInits registration (2 lines) |
| Modify | `treatise/css/charts.css` | Widget styles (~100 lines) |

## Verification

```bash
cd treatise && npm run build          # Must succeed
just dev-frontend                      # Visual check at localhost:5173/yatzy/#solver
```
