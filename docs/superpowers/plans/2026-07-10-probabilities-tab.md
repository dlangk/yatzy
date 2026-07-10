# Probabilities Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone client-side "Probabilities" UI at `/yatzy/prob/` that shows, across two rerolls, the probability of reaching a chosen target under the user's hold vs. optimal play.

**Architecture:** A static UI mirroring `profiler/` (plain ES-module JS, no build step). A pure `engine.js` computes target-reachability probabilities by exact enumeration; `targets.js` supplies category/exact-hand predicates; `render.js` builds a three-row live simulation trace using the shared dice renderer. Registered in `shared/nav.js`, served via one `serveDir` line in dev and an nginx `location` block + deploy copy step in prod.

**Tech Stack:** Vanilla ES-module JavaScript, `shared/dice.js` SVG renderer, Newsreader/theme CSS tokens copied from profiler, `node --test` for unit tests (no new deps).

## Global Constraints

- No emdashes in any prose/UI copy. Use periods, commas, colons, semicolons, parentheses.
- Plain ES-module JavaScript with JSDoc types. No build step. `prob/package.json` is `{ "private": true, "type": "module" }`.
- All dice are sorted-ascending `number[]` in 1..6 throughout the engine.
- "Optimal" means target-probability-maximizing, NOT the score-maximizing game solver. State this in the UI.
- Reuse `shared/dice.js` / `shared/dice.css` for dice; do not re-implement pip layout.
- Category target set is the 8 binary-achievement patterns (see Task 2). Upper-section categories (ones..sixes) and chance are excluded because binary achievement is undefined for them.
- Rows and rerolls-left: Row 0 = 2 rerolls left, Row 1 = 1, Row 2 = 0 (final, pass/fail).

---

### Task 1: Probability engine (`prob/js/engine.js`)

**Files:**
- Create: `prob/js/engine.js`
- Create: `prob/package.json`
- Test: `prob/js/engine.test.js`

**Interfaces:**
- Produces:
  - `rerollOutcomes(k: number) -> Array<{ dice: number[], p: number }>` — all sorted multiset outcomes of rolling `k` dice (k in 0..5), `p` = multinomial probability, `Σp == 1`. `k===0` returns `[{ dice: [], p: 1 }]`.
  - `keepSets(hand: number[]) -> number[][]` — distinct sorted sub-multisets of a 5-die hand (includes `[]` and the full hand).
  - `makeSolver(target: (hand:number[])=>boolean) -> { pOpt, pYou, bestKeep }` where
    - `pOpt(hand: number[], r: number) -> number` — best achievable P(target) with `r` rerolls left.
    - `pYou(hand: number[], r: number, keep: number[]) -> number` — P(target) if you keep exactly `keep` now then play optimally.
    - `bestKeep(hand: number[], r: number) -> number[]` — the sorted keep-set achieving `pOpt`.

- [ ] **Step 1: Write `prob/package.json`**

```json
{
  "private": true,
  "type": "module"
}
```

- [ ] **Step 2: Write the failing test** — `prob/js/engine.test.js`

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rerollOutcomes, keepSets, makeSolver } from './engine.js';

const sorted = (a) => a.slice().sort((x, y) => x - y);

test('rerollOutcomes: k=0 is the empty certain outcome', () => {
  assert.deepEqual(rerollOutcomes(0), [{ dice: [], p: 1 }]);
});

test('rerollOutcomes: k=1 is six equally likely faces', () => {
  const out = rerollOutcomes(1);
  assert.equal(out.length, 6);
  for (const o of out) assert.ok(Math.abs(o.p - 1 / 6) < 1e-12);
});

test('rerollOutcomes: probabilities sum to 1 for k=0..5', () => {
  for (let k = 0; k <= 5; k++) {
    const s = rerollOutcomes(k).reduce((acc, o) => acc + o.p, 0);
    assert.ok(Math.abs(s - 1) < 1e-9, `k=${k} sum=${s}`);
  }
});

test('keepSets: distinct sub-multisets include empty and full', () => {
  const ks = keepSets([1, 1, 2, 3, 4]).map((k) => k.join(''));
  assert.ok(ks.includes(''));
  assert.ok(ks.includes('11234'));
  // no duplicate sub-multisets
  assert.equal(new Set(ks).size, ks.length);
});

const smallStraight = (h) => sorted(h).join('') === '12345';

test('pOpt: r=0 is 1 when target met, 0 otherwise', () => {
  const s = makeSolver(smallStraight);
  assert.equal(s.pOpt([1, 2, 3, 4, 5], 0), 1);
  assert.equal(s.pOpt([1, 2, 3, 4, 6], 0), 0);
});

test('pYou == 1/6: keep 1234, reroll one die, need a 5 (one reroll)', () => {
  const s = makeSolver(smallStraight);
  const p = s.pYou([1, 2, 3, 4, 6], 1, [1, 2, 3, 4]);
  assert.ok(Math.abs(p - 1 / 6) < 1e-12, `p=${p}`);
});

test('pOpt with one reroll from [1,2,3,4,6] equals best hold (1/6)', () => {
  const s = makeSolver(smallStraight);
  assert.ok(Math.abs(s.pOpt([1, 2, 3, 4, 6], 1) - 1 / 6) < 1e-12);
  assert.deepEqual(s.bestKeep([1, 2, 3, 4, 6], 1), [1, 2, 3, 4]);
});

test('pYou equals pOpt when your keep is the optimal keep', () => {
  const s = makeSolver(smallStraight);
  const hand = [1, 2, 3, 4, 6];
  const best = s.bestKeep(hand, 2);
  assert.ok(Math.abs(s.pYou(hand, 2, best) - s.pOpt(hand, 2)) < 1e-12);
});

test('more rerolls never hurt: pOpt(hand,2) >= pOpt(hand,1)', () => {
  const s = makeSolver(smallStraight);
  const hand = [1, 2, 3, 4, 6];
  assert.ok(s.pOpt(hand, 2) >= s.pOpt(hand, 1) - 1e-12);
});

test('yatzy in two rerolls from a fresh hand ~ 4.60% (known odds)', () => {
  const yatzy = (h) => new Set(h).size === 1;
  const s = makeSolver(yatzy);
  // Expected P(5-of-a-kind in 3 rolls, optimal) ≈ 0.04603.
  const p = s.pOpt([1, 2, 3, 4, 5], 2);
  assert.ok(Math.abs(p - 0.04603) < 0.002, `p=${p}`);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd prob && node --test js/engine.test.js`
Expected: FAIL — `engine.js` does not exist / exports undefined.

- [ ] **Step 4: Write `prob/js/engine.js`**

```javascript
/**
 * Pure probability engine for target reachability across dice rerolls.
 * A `hand` is a sorted-ascending array of five dice values in 1..6.
 * @module engine
 */

/** @type {Map<number, Array<{dice:number[], p:number}>>} */
const _rerollCache = new Map();

/** k! for k in 0..5. */
function factorial(n) {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

/**
 * All sorted multiset outcomes of rolling `k` fair dice, with probabilities.
 * @param {number} k dice to roll (0..5)
 * @returns {Array<{dice:number[], p:number}>}
 */
export function rerollOutcomes(k) {
  if (_rerollCache.has(k)) return _rerollCache.get(k);
  /** @type {Array<{dice:number[], p:number}>} */
  const out = [];
  if (k === 0) {
    out.push({ dice: [], p: 1 });
  } else {
    const total = Math.pow(6, k);
    const rec = (start, current) => {
      if (current.length === k) {
        // Number of ordered arrangements of this multiset.
        const counts = {};
        for (const v of current) counts[v] = (counts[v] || 0) + 1;
        let perms = factorial(k);
        for (const v in counts) perms /= factorial(counts[v]);
        out.push({ dice: current.slice(), p: perms / total });
        return;
      }
      for (let v = start; v <= 6; v++) {
        current.push(v);
        rec(v, current);
        current.pop();
      }
    };
    rec(1, []);
  }
  _rerollCache.set(k, out);
  return out;
}

/**
 * Distinct sorted sub-multisets of a 5-die hand (which dice to keep).
 * @param {number[]} hand sorted 5-die hand
 * @returns {number[][]}
 */
export function keepSets(hand) {
  const seen = new Set();
  /** @type {number[][]} */
  const res = [];
  for (let mask = 0; mask < 32; mask++) {
    const keep = [];
    for (let i = 0; i < 5; i++) if (mask & (1 << i)) keep.push(hand[i]);
    keep.sort((a, b) => a - b);
    const key = keep.join('');
    if (!seen.has(key)) {
      seen.add(key);
      res.push(keep);
    }
  }
  return res;
}

/**
 * Build a memoized solver for a fixed target predicate.
 * @param {(hand:number[])=>boolean} target
 */
export function makeSolver(target) {
  /** @type {Map<string, number>} */
  const memo = new Map();

  /** Expected pOpt(next, r-1) over rerolling the non-kept dice. */
  function applyKeep(keep, r) {
    const k = 5 - keep.length;
    let sum = 0;
    for (const { dice, p } of rerollOutcomes(k)) {
      const next = keep.concat(dice).sort((a, b) => a - b);
      sum += p * pOpt(next, r - 1);
    }
    return sum;
  }

  /** Best achievable P(target) from `hand` with `r` rerolls left. */
  function pOpt(hand, r) {
    if (r <= 0) return target(hand) ? 1 : 0;
    const key = r + ':' + hand.join('');
    const cached = memo.get(key);
    if (cached !== undefined) return cached;
    let best = 0;
    for (const keep of keepSets(hand)) {
      const v = applyKeep(keep, r);
      if (v > best) best = v;
    }
    memo.set(key, best);
    return best;
  }

  /** P(target) if you keep exactly `keep` now, then play optimally. */
  function pYou(hand, r, keep) {
    if (r <= 0) return target(hand) ? 1 : 0;
    return applyKeep(keep.slice().sort((a, b) => a - b), r);
  }

  /** The sorted keep-set achieving pOpt(hand, r). */
  function bestKeep(hand, r) {
    let best = -1;
    let bestK = hand.slice();
    for (const keep of keepSets(hand)) {
      const v = applyKeep(keep, r);
      if (v > best) {
        best = v;
        bestK = keep;
      }
    }
    return bestK;
  }

  return { pOpt, pYou, bestKeep };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd prob && node --test js/engine.test.js`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add prob/package.json prob/js/engine.js prob/js/engine.test.js
git commit -m "feat(prob): target-reachability probability engine"
```

---

### Task 2: Target predicates (`prob/js/targets.js`)

**Files:**
- Create: `prob/js/targets.js`
- Test: `prob/js/targets.test.js`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CATEGORIES: Array<{ id: string, label: string, test: (hand:number[])=>boolean }>` — the 8 binary pattern categories.
  - `categoryById(id: string) -> {id,label,test} | undefined`
  - `exactTarget(targetHand: number[]) -> (hand:number[])=>boolean` — true iff sorted hands are equal.

- [ ] **Step 1: Write the failing test** — `prob/js/targets.test.js`

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CATEGORIES, categoryById, exactTarget } from './targets.js';

const cat = (id) => categoryById(id).test;

test('one pair', () => {
  assert.equal(cat('one_pair')([1, 1, 3, 4, 6]), true);
  assert.equal(cat('one_pair')([1, 2, 3, 4, 6]), false);
});

test('two pairs (yatzy is not two pairs)', () => {
  assert.equal(cat('two_pairs')([1, 1, 4, 4, 6]), true);
  assert.equal(cat('two_pairs')([1, 1, 4, 6, 6]), true);
  assert.equal(cat('two_pairs')([1, 1, 1, 4, 4]), true); // trips + pair contains two pairs
  assert.equal(cat('two_pairs')([1, 1, 3, 4, 6]), false);
  assert.equal(cat('two_pairs')([5, 5, 5, 5, 5]), false);
});

test('three and four of a kind', () => {
  assert.equal(cat('three_kind')([2, 2, 2, 4, 6]), true);
  assert.equal(cat('three_kind')([2, 2, 4, 4, 6]), false);
  assert.equal(cat('four_kind')([2, 2, 2, 2, 6]), true);
  assert.equal(cat('four_kind')([2, 2, 2, 4, 6]), false);
});

test('full house is exactly 3+2 of distinct values', () => {
  assert.equal(cat('full_house')([2, 2, 2, 5, 5]), true);
  assert.equal(cat('full_house')([2, 2, 2, 2, 5]), false);
  assert.equal(cat('full_house')([5, 5, 5, 5, 5]), false);
  assert.equal(cat('full_house')([2, 2, 3, 5, 5]), false);
});

test('small and large straight are the fixed patterns', () => {
  assert.equal(cat('small_straight')([1, 2, 3, 4, 5]), true);
  assert.equal(cat('small_straight')([2, 3, 4, 5, 6]), false);
  assert.equal(cat('large_straight')([2, 3, 4, 5, 6]), true);
  assert.equal(cat('large_straight')([1, 2, 3, 4, 5]), false);
});

test('yatzy is five of a kind', () => {
  assert.equal(cat('yatzy')([4, 4, 4, 4, 4]), true);
  assert.equal(cat('yatzy')([4, 4, 4, 4, 6]), false);
});

test('CATEGORIES has 8 entries with unique ids and labels', () => {
  assert.equal(CATEGORIES.length, 8);
  assert.equal(new Set(CATEGORIES.map((c) => c.id)).size, 8);
  for (const c of CATEGORIES) assert.ok(c.label && !c.label.includes('--'));
});

test('exactTarget matches regardless of order', () => {
  const t = exactTarget([6, 4, 3, 2, 1]);
  assert.equal(t([1, 2, 3, 4, 6]), true);
  assert.equal(t([1, 2, 3, 4, 5]), false);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd prob && node --test js/targets.test.js`
Expected: FAIL — `targets.js` does not exist.

- [ ] **Step 3: Write `prob/js/targets.js`**

```javascript
/**
 * Target predicates for the probabilities tool.
 * Categories are binary-achievement patterns (achieved or not), unlike the
 * upper-section scoring categories which are score amounts, not patterns.
 * @module targets
 */

/** Tally of value -> count for a hand. */
function counts(hand) {
  const c = {};
  for (const v of hand) c[v] = (c[v] || 0) + 1;
  return c;
}

/** Sorted array of the count values, e.g. full house -> [2,3]. */
function shape(hand) {
  return Object.values(counts(hand)).sort((a, b) => a - b);
}

function hasNOfAKind(hand, n) {
  return Object.values(counts(hand)).some((c) => c >= n);
}

function keyOf(hand) {
  return hand.slice().sort((a, b) => a - b).join('');
}

/** @type {Array<{id:string,label:string,test:(hand:number[])=>boolean}>} */
export const CATEGORIES = [
  { id: 'one_pair', label: 'One pair', test: (h) => hasNOfAKind(h, 2) },
  {
    id: 'two_pairs',
    label: 'Two pairs',
    test: (h) => Object.values(counts(h)).filter((c) => c >= 2).length >= 2,
  },
  { id: 'three_kind', label: 'Three of a kind', test: (h) => hasNOfAKind(h, 3) },
  { id: 'four_kind', label: 'Four of a kind', test: (h) => hasNOfAKind(h, 4) },
  {
    id: 'full_house',
    label: 'Full house',
    test: (h) => {
      const s = shape(h);
      return s.length === 2 && s[0] === 2 && s[1] === 3;
    },
  },
  { id: 'small_straight', label: 'Small straight', test: (h) => keyOf(h) === '12345' },
  { id: 'large_straight', label: 'Large straight', test: (h) => keyOf(h) === '23456' },
  { id: 'yatzy', label: 'Yatzy', test: (h) => new Set(h).size === 1 },
];

/** @param {string} id */
export function categoryById(id) {
  return CATEGORIES.find((c) => c.id === id);
}

/**
 * Predicate that matches a specific unordered five-die hand.
 * @param {number[]} targetHand
 * @returns {(hand:number[])=>boolean}
 */
export function exactTarget(targetHand) {
  const target = keyOf(targetHand);
  return (hand) => keyOf(hand) === target;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd prob && node --test js/targets.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prob/js/targets.js prob/js/targets.test.js
git commit -m "feat(prob): category and exact-hand target predicates"
```

---

### Task 3: Static UI shell + nav + dev serving

**Files:**
- Create: `prob/index.html`, `prob/css/style.css`, `prob/css/prob.css`, `prob/js/theme.js`, `prob/README.md`
- Modify: `shared/nav.js` (PAGES array + `detectActive`)
- Modify: `frontend/vite.config.ts` (add `serveDir` for `/yatzy/prob/`)

**Interfaces:**
- Consumes: `shared/nav.js`, `shared/dice.css`.
- Produces: a served page at `http://localhost:5173/yatzy/prob/` with the nav bar, a heading, and an empty `#prob-root` container; `initTheme()` default export in `prob/js/theme.js`.

- [ ] **Step 1: Add the nav entry** — edit `shared/nav.js`

In the `PAGES` array, add after the `profile` entry:

```javascript
  { id: 'prob',     label: 'Probabilities', href: '/yatzy/prob/', tooltip: 'Dice probabilities across two rerolls' },
```

In `detectActive()`, add before the `return 'treatise'` line:

```javascript
  if (path.includes('/prob')) return 'prob';
```

- [ ] **Step 2: Copy the theme helper** — create `prob/js/theme.js`

```javascript
export function initTheme() {
  const toggle = document.getElementById('theme-toggle');
  const icon = document.getElementById('theme-icon');
  if (!toggle || !icon) return;

  function setTheme(dark) {
    document.documentElement.classList.toggle('dark', dark);
    icon.textContent = dark ? '☀' : '☾';
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  }

  const saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    setTheme(true);
  }

  toggle.addEventListener('click', () => {
    setTheme(!document.documentElement.classList.contains('dark'));
  });

  return { setTheme };
}
```

- [ ] **Step 3: Create `prob/css/style.css`** (page chrome — copied tokens from profiler)

```css
:root {
  --bg: #f6f5ef;
  --bg-alt: #eae9e3;
  --text: #050505;
  --text-muted: #555;
  --accent: rgba(243, 112, 33, 0.8);
  --accent-light: rgba(243, 112, 33, 0.15);
  --border: #d4d3cd;
  --font-serif: 'Newsreader', Georgia, 'Times New Roman', serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --content-width: 720px;
  --color-success: #2ca02c;
  --color-danger: #b40426;
}

.dark {
  --bg: #1a1a1a;
  --bg-alt: #252525;
  --text: #e0ddd5;
  --text-muted: #999;
  --accent: rgba(243, 112, 33, 0.9);
  --accent-light: rgba(243, 112, 33, 0.15);
  --border: #3a3a3a;
  --color-success: #4caf50;
  --color-danger: #e05252;
}

*, *::before, *::after { box-sizing: border-box; }

html { font-size: 18px; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-serif);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  transition: background 0.3s, color 0.3s;
}

.container {
  max-width: var(--content-width);
  margin: 0 auto;
  padding: 2rem 1.25rem 4rem;
}

.page-header h1 { margin: 0 0 0.25rem; font-weight: 700; }
.page-header p { margin: 0 0 1.5rem; color: var(--text-muted); }

.theme-toggle {
  position: fixed;
  top: 44px;
  right: 16px;
  background: none;
  border: 1px solid var(--border);
  border-radius: 50%;
  width: 32px;
  height: 32px;
  cursor: pointer;
  color: var(--text-muted);
  z-index: 50;
}
```

- [ ] **Step 4: Create `prob/css/prob.css`** (placeholder — filled in Task 4)

```css
/* Tool-specific layout — populated in Task 4. */
.prob-tool { margin-top: 1rem; }
```

- [ ] **Step 5: Create `prob/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Yatzy Probabilities</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="css/style.css">
  <link rel="stylesheet" href="css/prob.css">
  <link rel="stylesheet" href="/yatzy/shared/dice.css">
</head>
<body>
<script type="module" src="/yatzy/shared/nav.js"></script>

<button class="theme-toggle" id="theme-toggle" aria-label="Toggle dark mode">
  <span id="theme-icon">&#9790;</span>
</button>

<div class="container">
  <div class="page-header">
    <h1>Dice Probabilities</h1>
    <p>How likely are you to reach a target across two rerolls? "Best" assumes optimal holds that maximize the chance of the target (not the score-maximizing game solver).</p>
  </div>
  <div id="prob-root" class="prob-tool"></div>
</div>

<script type="module" src="js/main.js"></script>
<script type="module">
  import { initTheme } from './js/theme.js';
  initTheme();
</script>

</body>
</html>
```

- [ ] **Step 6: Create a stub `prob/js/main.js`** so the module loads

```javascript
// Entry point — full rendering wired in Task 4.
const root = document.getElementById('prob-root');
if (root) root.textContent = 'Loading...';
```

- [ ] **Step 7: Create `prob/README.md`**

```markdown
# Probabilities UI

Standalone client-side tool at `/yatzy/prob/`. Given an opening hand and a
target, shows the probability of reaching the target across two rerolls under
your hold vs. optimal (target-maximizing) play.

- `js/engine.js` — pure probability engine (exact enumeration). Unit-tested.
- `js/targets.js` — category + exact-hand predicates. Unit-tested.
- `js/render.js` — three-row live simulation trace.
- `js/main.js` — entry point.

No build step. Run tests: `node --test js/`.
```

- [ ] **Step 8: Register dev serving** — edit `frontend/vite.config.ts`

In the `plugins` array, add BEFORE the treatise entry (so `/yatzy/prob/` matches before the `/yatzy/` catch-all):

```typescript
    serveDir('/yatzy/prob/', path.join(root, 'prob')),
```

Resulting order:

```typescript
  plugins: [
    serveDir('/yatzy/shared/', path.join(root, 'shared')),
    serveDir('/yatzy/profile/', path.join(root, 'profiler')),
    serveDir('/yatzy/prob/', path.join(root, 'prob')),
    serveDir('/yatzy/', path.join(root, 'treatise')),
  ],
```

- [ ] **Step 9: Verify the shell serves** (dev server assumed running via `just dev-frontend`)

Run: `curl -s http://localhost:5173/yatzy/prob/ | grep -o '<title>[^<]*</title>'`
Expected: `<title>Yatzy Probabilities</title>`

Run: `curl -s http://localhost:5173/yatzy/prob/js/main.js | head -1`
Expected: the first comment line of `main.js`.

- [ ] **Step 10: Commit**

```bash
git add prob/index.html prob/css prob/js/theme.js prob/js/main.js prob/README.md shared/nav.js frontend/vite.config.ts
git commit -m "feat(prob): static UI shell, nav entry, dev serving"
```

---

### Task 4: Render the live simulation trace (`prob/js/render.js`)

**Files:**
- Create: `prob/js/render.js`
- Modify: `prob/js/main.js` (call `initProbTool`)
- Modify: `prob/css/prob.css` (full styles)

**Interfaces:**
- Consumes: `makeSolver` from `engine.js`; `CATEGORIES`, `categoryById`, `exactTarget` from `targets.js`; `createDieSVG` from `/yatzy/shared/dice.js`.
- Produces: `initProbTool(root: HTMLElement) -> void` — builds the entire interactive tool into `root`.

State model (module-local, in `render.js`):

```
state = {
  mode: 'category' | 'exact',
  categoryId: string,          // active category in category mode
  exactHand: number[5],        // target hand in exact mode
  rows: [Row, Row, Row],       // Row 0 always has dice; 1 and 2 may be null
}
Row = { dice: number[5] | null, keep: boolean[5] }
```

Row `r` has `rerollsLeft = 2 - index`. Rolling row `i`'s randomizer replaces
its non-kept dice with random faces to produce row `i+1` (kept dice carry down;
`keep` on the new row resets to all-true). Editing a die value or toggling a
hold on row `i` clears rows `i+1..2`.

- [ ] **Step 1: Write `prob/js/render.js`**

```javascript
/**
 * Live simulation trace UI: three rows of dice, per-row holds and randomizers,
 * live "you" vs "best" target probabilities.
 * @module render
 */
import { makeSolver } from './engine.js';
import { CATEGORIES, categoryById, exactTarget } from './targets.js';
import { createDieSVG } from '/yatzy/shared/dice.js';

const REROLLS = [2, 1, 0]; // rerolls left per row index

/** Deterministic default opening hand from the user's example. */
function defaultHand() {
  return [1, 2, 3, 4, 6];
}

function randomDie() {
  return 1 + Math.floor(Math.random() * 6);
}

function pct(p) {
  return (p * 100).toFixed(1) + '%';
}

export function initProbTool(root) {
  const state = {
    mode: 'category',
    categoryId: 'small_straight',
    exactHand: [1, 2, 3, 4, 5],
    rows: [
      { dice: defaultHand(), keep: [true, true, true, true, false] },
      { dice: null, keep: [true, true, true, true, true] },
      { dice: null, keep: [true, true, true, true, true] },
    ],
  };

  root.innerHTML = '';

  // ── Target selector ───────────────────────────────────────────────
  const targetBar = document.createElement('div');
  targetBar.className = 'target-bar';
  root.appendChild(targetBar);

  const modeSel = document.createElement('select');
  modeSel.className = 'mode-select';
  for (const [val, label] of [['category', 'Category'], ['exact', 'Exact hand']]) {
    const o = document.createElement('option');
    o.value = val;
    o.textContent = label;
    modeSel.appendChild(o);
  }
  modeSel.value = state.mode;
  targetBar.appendChild(modeSel);

  const catSel = document.createElement('select');
  catSel.className = 'cat-select';
  for (const c of CATEGORIES) {
    const o = document.createElement('option');
    o.value = c.id;
    o.textContent = c.label;
    catSel.appendChild(o);
  }
  catSel.value = state.categoryId;
  targetBar.appendChild(catSel);

  // Exact-hand editor (five clickable dice)
  const exactWrap = document.createElement('div');
  exactWrap.className = 'exact-wrap';
  const exactLabel = document.createElement('span');
  exactLabel.className = 'exact-label';
  exactLabel.textContent = 'Target:';
  exactWrap.appendChild(exactLabel);
  const exactDice = document.createElement('div');
  exactDice.className = 'exact-dice';
  exactWrap.appendChild(exactDice);
  targetBar.appendChild(exactWrap);

  // ── Rows ──────────────────────────────────────────────────────────
  const rowsWrap = document.createElement('div');
  rowsWrap.className = 'rows';
  root.appendChild(rowsWrap);

  const rowEls = REROLLS.map(() => {
    const el = document.createElement('div');
    el.className = 'prob-row';
    rowsWrap.appendChild(el);
    return el;
  });

  // ── Helpers ───────────────────────────────────────────────────────
  function currentTarget() {
    if (state.mode === 'exact') return exactTarget(state.exactHand);
    return categoryById(state.categoryId).test;
  }

  function keptDice(row) {
    const out = [];
    for (let i = 0; i < 5; i++) if (row.keep[i]) out.push(row.dice[i]);
    return out.sort((a, b) => a - b);
  }

  function clearBelow(index) {
    for (let i = index + 1; i < 3; i++) {
      state.rows[i].dice = null;
      state.rows[i].keep = [true, true, true, true, true];
    }
  }

  function rollInto(index) {
    const src = state.rows[index];
    if (!src.dice) return;
    const next = src.dice.map((v, i) => (src.keep[i] ? v : randomDie()));
    state.rows[index + 1] = { dice: next, keep: [true, true, true, true, true] };
    for (let i = index + 2; i < 3; i++) {
      state.rows[i].dice = null;
      state.rows[i].keep = [true, true, true, true, true];
    }
  }

  // ── Render ────────────────────────────────────────────────────────
  function render() {
    const solver = makeSolver(currentTarget());

    catSel.style.display = state.mode === 'category' ? '' : 'none';
    exactWrap.style.display = state.mode === 'exact' ? '' : 'none';

    // Exact target dice
    exactDice.innerHTML = '';
    if (state.mode === 'exact') {
      state.exactHand.forEach((v, i) => {
        const d = createDieSVG(v, { size: 34, clickable: true });
        d.addEventListener('click', () => {
          state.exactHand[i] = (state.exactHand[i] % 6) + 1;
          render();
        });
        exactDice.appendChild(d);
      });
    }

    REROLLS.forEach((rLeft, index) => {
      const row = state.rows[index];
      const el = rowEls[index];
      el.innerHTML = '';
      el.classList.toggle('empty', !row.dice);

      const label = document.createElement('div');
      label.className = 'row-label';
      label.textContent = rLeft > 0 ? `${rLeft} reroll${rLeft > 1 ? 's' : ''} left` : 'final';
      el.appendChild(label);

      const diceEl = document.createElement('div');
      diceEl.className = 'row-dice';
      el.appendChild(diceEl);

      if (!row.dice) {
        const ph = document.createElement('div');
        ph.className = 'row-placeholder';
        ph.textContent = 'roll the row above';
        diceEl.appendChild(ph);
        return;
      }

      const bestK = rLeft > 0 ? solver.bestKeep(row.dice, rLeft) : [];
      const bestKeyCounts = {};
      for (const v of bestK) bestKeyCounts[v] = (bestKeyCounts[v] || 0) + 1;

      row.dice.forEach((v, i) => {
        const state2 = row.keep[i] ? 'kept' : 'will-reroll';
        const d = createDieSVG(v, { size: 44, state: state2, clickable: true });
        // Left click cycles value; the hold toggles via a small button below.
        d.addEventListener('click', () => {
          row.dice[i] = (row.dice[i] % 6) + 1;
          clearBelow(index);
          render();
        });
        const cell = document.createElement('div');
        cell.className = 'die-cell';
        cell.appendChild(d);

        const holdBtn = document.createElement('button');
        holdBtn.className = 'hold-btn' + (row.keep[i] ? ' held' : '');
        holdBtn.textContent = row.keep[i] ? 'keep' : 'reroll';
        holdBtn.addEventListener('click', () => {
          row.keep[i] = !row.keep[i];
          clearBelow(index);
          render();
        });
        cell.appendChild(holdBtn);
        diceEl.appendChild(cell);
      });

      // Probabilities / result
      const stats = document.createElement('div');
      stats.className = 'row-stats';
      if (rLeft > 0) {
        const you = solver.pYou(row.dice, rLeft, keptDice(row));
        const best = solver.pOpt(row.dice, rLeft);
        stats.innerHTML =
          `<span class="stat you">you <b>${pct(you)}</b></span>` +
          `<span class="stat best">best <b>${pct(best)}</b></span>`;
        const hint = document.createElement('div');
        hint.className = 'best-hint';
        hint.textContent = bestK.length ? `optimal: keep ${bestK.join(' ') || '(reroll all)'}` : 'optimal: reroll all';
        el.appendChild(diceEl.parentNode === el ? document.createComment('') : document.createComment(''));
        el.appendChild(stats);
        el.appendChild(hint);

        const roll = document.createElement('button');
        roll.className = 'roll-btn';
        roll.textContent = 'roll un-held';
        roll.addEventListener('click', () => {
          rollInto(index);
          render();
        });
        el.appendChild(roll);
      } else {
        const met = currentTarget()(row.dice);
        stats.innerHTML = met
          ? `<span class="result hit">target reached</span>`
          : `<span class="result miss">target missed</span>`;
        el.appendChild(stats);
      }
    });
  }

  // ── Events ────────────────────────────────────────────────────────
  modeSel.addEventListener('change', () => {
    state.mode = modeSel.value;
    render();
  });
  catSel.addEventListener('change', () => {
    state.categoryId = catSel.value;
    render();
  });

  render();
}
```

Note: the two `document.createComment('')` lines above are inert; replace the
whole `el.appendChild(diceEl.parentNode ...)` line with nothing if preferred.
For clarity, delete that line during implementation — it is a no-op guard left
out of the final code:

Final: remove the line
`el.appendChild(diceEl.parentNode === el ? document.createComment('') : document.createComment(''));`
entirely. It does nothing and should not ship.

- [ ] **Step 2: Wire `main.js`** — replace `prob/js/main.js` contents

```javascript
import { initProbTool } from './render.js';

const root = document.getElementById('prob-root');
if (root) initProbTool(root);
```

- [ ] **Step 3: Fill in `prob/css/prob.css`**

```css
.prob-tool { margin-top: 1rem; }

.target-bar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  padding: 0.75rem 0 1.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.25rem;
}

.mode-select, .cat-select {
  font-family: var(--font-serif);
  font-size: 1rem;
  padding: 0.3rem 0.5rem;
  background: var(--bg-alt);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 5px;
}

.exact-wrap { display: flex; align-items: center; gap: 0.5rem; }
.exact-label { color: var(--text-muted); }
.exact-dice { display: flex; gap: 4px; }

.rows { display: flex; flex-direction: column; gap: 0.75rem; }

.prob-row {
  display: grid;
  grid-template-columns: 110px 1fr auto;
  align-items: center;
  gap: 1rem;
  padding: 0.75rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg-alt);
}

.prob-row.empty { opacity: 0.55; }

.row-label { color: var(--text-muted); font-size: 0.95rem; }

.row-dice { display: flex; gap: 0.5rem; align-items: flex-start; }
.die-cell { display: flex; flex-direction: column; align-items: center; gap: 4px; }

.hold-btn {
  font-family: var(--font-serif);
  font-size: 0.7rem;
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text-muted);
  cursor: pointer;
}
.hold-btn.held { border-color: var(--accent); color: var(--accent); }

.row-placeholder { color: var(--text-muted); font-style: italic; }

.row-stats { display: flex; flex-direction: column; gap: 2px; text-align: right; }
.row-stats .stat { font-size: 0.95rem; color: var(--text-muted); }
.row-stats .stat b { color: var(--text); }
.row-stats .stat.best b { color: var(--accent); }

.best-hint { grid-column: 2 / 3; font-size: 0.8rem; color: var(--text-muted); }
.roll-btn {
  font-family: var(--font-serif);
  font-size: 0.85rem;
  padding: 0.35rem 0.7rem;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--bg);
  color: var(--text);
  cursor: pointer;
}
.roll-btn:hover { border-color: var(--accent); color: var(--accent); }

.result.hit { color: var(--color-success); font-weight: 600; }
.result.miss { color: var(--color-danger); font-weight: 600; }

@media (max-width: 620px) {
  .prob-row { grid-template-columns: 1fr; text-align: left; }
  .row-stats { text-align: left; flex-direction: row; gap: 1rem; }
}
```

- [ ] **Step 4: Verify in the browser**

Open `http://localhost:5173/yatzy/prob/`. Confirm:
- Three rows render; Row 0 shows [1,2,3,4,6], the 6 marked reroll.
- "you" and "best" percentages appear on rows 0 and 1.
- Clicking "roll un-held" on Row 0 populates Row 1; on Row 1 populates Row 2.
- Row 2 shows "target reached" / "target missed".
- Switching mode to "Exact hand" reveals the target dice editor and hides the category dropdown; probabilities update.
- Toggling a hold or clicking a die on an upper row clears the rows below.

- [ ] **Step 5: Re-run engine + target tests (guard against regressions)**

Run: `cd prob && node --test js/`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add prob/js/render.js prob/js/main.js prob/css/prob.css
git commit -m "feat(prob): three-row live simulation trace UI"
```

---

### Task 5: Production serving (nginx + deploy script + docs)

**Files:**
- Modify: `deploy/nginx.conf` (add `location /prob/` + redirect)
- Modify: `deploy/deploy.sh` (stage `prob/` into `apps/prob/`)
- Modify: `.claude/commands/deploy.md` (packaging + verification lines)

**Interfaces:**
- Consumes: the `prob/` directory from Tasks 1-4.
- Produces: `/yatzy/prob/` served in production, baked into the frontend image.

- [ ] **Step 1: Add nginx location** — edit `deploy/nginx.conf`

After the `location = /profile { ... }` redirect line, add:

```
    location = /prob { return 301 /yatzy/prob/; }
```

After the `location /profile/ { ... }` block, add:

```
    # Probabilities tool (baked into image)
    location /prob/ {
        alias /usr/share/nginx/apps/prob/;
        try_files $uri $uri/ /prob/index.html;
    }
```

- [ ] **Step 2: Add deploy packaging** — edit `deploy/deploy.sh`

After the profiler copy block (the `cp -r "$PROJECT_ROOT/profiler/data/"* ...` line), add:

```bash
# Probabilities tool at /prob/
mkdir -p "$FRONTEND_CTX/apps/prob/css" "$FRONTEND_CTX/apps/prob/js"
cp "$PROJECT_ROOT/prob/index.html" "$FRONTEND_CTX/apps/prob/index.html"
cp -r "$PROJECT_ROOT/prob/css/"* "$FRONTEND_CTX/apps/prob/css/"
cp -r "$PROJECT_ROOT/prob/js/"* "$FRONTEND_CTX/apps/prob/js/"
```

In the Step 9 `HEALTHCHECK` heredoc, after the Profiler check, add:

```bash
echo -n "Probabilities:  "
docker exec yatzy-frontend wget -q --spider http://localhost:8090/prob/ && echo "OK" || echo "FAIL"
```

In the final echo block, after the Profiler line, add:

```bash
echo "  Prob:      https://langkilde.se/yatzy/prob/"
```

- [ ] **Step 3: Update the deploy command doc** — edit `.claude/commands/deploy.md`

In the "Package" step 9 bullet list, append: "prob assets to apps/prob/".
In the post-deploy verification list (step 16), add a line for the `/prob/` spider check.

- [ ] **Step 4: Note — test JS files are excluded from the image**

The deploy copies `prob/js/*` including `*.test.js`. These are harmless static
files (never served/loaded by `index.html`) but to keep the image clean, remove
them after copy. Add after the `cp -r "$PROJECT_ROOT/prob/js/"* ...` line:

```bash
rm -f "$FRONTEND_CTX/apps/prob/js/"*.test.js
```

- [ ] **Step 5: Commit**

```bash
git add deploy/nginx.conf deploy/deploy.sh .claude/commands/deploy.md
git commit -m "feat(prob): production serving via nginx + deploy packaging"
```

---

## Deploy & verify (final)

- [ ] Ensure Docker Desktop is running.
- [ ] Run the full deploy: `just deploy` (cross-compiles solver, builds both images, redeploys, syncs treatise + shared/, purges Cloudflare cache).
- [ ] Verify external: `curl -sf https://langkilde.se/yatzy/prob/ | grep -o '<title>[^<]*</title>'` → `<title>Yatzy Probabilities</title>`.
- [ ] Verify the nav on the other UIs now shows the "Probabilities" tab (shared/nav.js is synced with the treatise).
- [ ] Manually load `https://langkilde.se/yatzy/prob/` and confirm the trace works end to end.

## Self-Review Notes

- **Spec coverage:** placement (Task 3), probability model you/best (Tasks 1, 4), target modes category+exact (Tasks 2, 4), live trace flow (Task 4), testing (Tasks 1, 2), deploy wiring (Task 5). All spec sections mapped.
- **Category refinement:** spec listed ones..sixes/chance; plan narrows Category mode to the 8 binary-achievement patterns (documented in Global Constraints and Task 2) because binary achievement is undefined for score-amount categories. Exact-hand mode covers any specific hand.
- **Type consistency:** `makeSolver` returns `{ pOpt, pYou, bestKeep }` used identically in Task 4. `CATEGORIES`/`categoryById`/`exactTarget` signatures match between Tasks 2 and 4.
