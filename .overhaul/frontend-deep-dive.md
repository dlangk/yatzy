# Frontend Deep Dive

## Current Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | React | 19.2.0 |
| Language | TypeScript | 5.9.3 |
| Bundler | Vite | 7.3.1 |
| Testing | Vitest | latest |
| Linting | ESLint | latest |
| Package manager | npm | — |

## Build Configuration

**vite.config.ts:** Minimal — React plugin, dev server port 5173.

**tsconfig.app.json:** Target ES2022, strict mode, JSX react-jsx, bundler module resolution.

**index.html:** Loads `/config.js` (dynamic API URL), then `/src/main.tsx` as module.

## File Inventory (2,097 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `reducer.test.ts` | 384 | Unit tests for game reducer |
| `reducer.ts` | 355 | Game state machine (14 actions) |
| `components/TrajectoryChart.tsx` | 272 | Canvas-based score evolution chart |
| `App.tsx` | 151 | Root component, orchestration |
| `components/ScorecardRow.tsx` | 112 | Category row with EV spark bar |
| `components/ActionBar.tsx` | 96 | Roll / Reroll / Reset buttons |
| `components/Scorecard.tsx` | 94 | Fixed-layout category table |
| `mask.test.ts` | 83 | Mask translation tests |
| `components/EvalPanel.tsx` | 81 | Eval metrics grid |
| `types.ts` | 80 | All TypeScript interfaces |
| `components/Die.tsx` | 71 | Single die with ±controls |
| `mask.ts` | 67 | Dice sort/mask translation |
| `components/DiceLegend.tsx` | 51 | Color legend (keep vs reroll) |
| `components/DiceBar.tsx` | 45 | 5-die row with optimal hints |
| `components/DebugPanel.tsx` | 44 | JSON state dump |
| `api.ts` | 64 | 4 API calls to solver |
| `constants.ts` | 27 | Category names, colors |
| `config.ts` | 10 | API_BASE_URL resolution |
| `main.tsx` | 9 | React root initialization |

## Component Hierarchy

```
App (useReducer + useEffect orchestration)
│
├── ActionBar
│   ├── Roll button (turnPhase === 'idle')
│   ├── Reroll button (turnPhase === 'rolled', rerollsRemaining > 0)
│   └── Reset button (turnPhase === 'game_over')
│
├── DiceBar
│   └── Die × 5
│       ├── Die face display (value or "?" placeholder)
│       ├── +/- increment buttons (debug)
│       ├── Hold toggle (click to toggle)
│       └── Optimal mask hint (green border=keep, red border=reroll)
│
├── DiceLegend
│   └── Color legend: green=keep, red=reroll, gray=idle
│
├── EvalPanel
│   └── 4-cell grid:
│       ├── State EV (expected final score from current state)
│       ├── Your mask EV (current held dice evaluation)
│       ├── Best mask EV (optimal reroll mask)
│       └── Delta (gap between your mask and best)
│
├── TrajectoryChart
│   └── HTML5 Canvas (DPR-scaled)
│       ├── Accumulated score (green shaded area)
│       ├── Expected final (blue line, main trajectory)
│       ├── Percentile bands (p10-p90, p25-p75 polygons)
│       └── Score events as larger circles
│
├── Scorecard
│   └── <table> (table-layout: fixed, colgroup with explicit widths)
│       ├── Upper section header
│       ├── ScorecardRow × 6 (Ones through Sixes)
│       ├── Upper subtotal + bonus row
│       ├── Lower section header
│       ├── ScorecardRow × 9 (One Pair through Yatzy)
│       └── Total row
│
└── DebugPanel (conditional on showDebug)
    └── JSON.stringify(state) + JSON.stringify(lastEvalResponse)
```

## State Management

### GameState Interface

```typescript
interface GameState {
  dice: DieState[];                        // 5 dice: { value, held }
  upperScore: number;                      // Sum of Ones-Sixes (capped 63)
  scoredCategories: number;                // 15-bit bitmask
  rerollsRemaining: number;                // 0-2
  categories: CategoryState[];             // 15 entries: { score, ev, ... }
  totalScore: number;                      // Running total + bonus
  bonus: number;                           // 50 if upper ≥ 63
  lastEvalResponse: EvaluateResponse | null;
  sortMap: number[] | null;                // Dice sort translation
  turnPhase: 'idle' | 'rolled' | 'game_over';
  trajectory: TrajectoryPoint[];           // Score history
  showDebug: boolean;
}
```

### 14 Action Types

| Action | Trigger | Effect |
|--------|---------|--------|
| `ROLL` | Roll button | Generate 5 random dice, set turnPhase='rolled' |
| `TOGGLE_DIE` | Click die | Toggle held state |
| `REROLL` | Reroll button | Re-randomize unheld dice, decrement rerolls |
| `SCORE_CATEGORY` | Score button | Lock category, update totals, check game_over |
| `SET_EVAL_RESPONSE` | API response | Store eval + compute sort map |
| `SET_DIE_VALUE` | ± button | Manual die value edit (debug) |
| `SET_REROLLS` | Debug | Override reroll count |
| `SET_CATEGORY_SCORE` | Debug | Manual category score |
| `UNSET_CATEGORY` | Debug | Undo scored category |
| `RESET_GAME` | Reset button | Fresh game state |
| `SET_INITIAL_EV` | On mount | Seed trajectory with initial EV |
| `SET_DENSITY_RESULT` | API response | Add percentiles to trajectory |
| `TOGGLE_DEBUG` | Keyboard | Show/hide debug panel |

### Turn Flow

```
idle → ROLL → rolled → [TOGGLE_DIE]* → REROLL → rolled →
  [TOGGLE_DIE]* → REROLL → rolled → SCORE_CATEGORY → idle
  (after 15 categories → game_over)
```

### Persistence

`saveState()` → localStorage. Excludes transient fields: `lastEvalResponse`, `sortMap`.

## API Integration Layer

### 4 Endpoints (api.ts)

```typescript
// POST /evaluate — called on every roll/reroll
evaluateRoll(dice: number[], upperScore: number, scoredCategories: number, rerolls: number)
  → EvaluateResponse { mask_evs, optimal_mask, categories[], state_ev, ... }

// GET /health — connectivity check
checkHealth() → 200 | error

// GET /state_value — initial EV seed
getStateValue(upperScore: number, scoredCategories: number)
  → { expected_final_score: number }

// POST /density — percentile bands for trajectory chart
getDensity(upperScore: number, scoredCategories: number, accumulatedScore: number)
  → { mean, std_dev, percentiles: { p10, p25, p50, p75, p90 } }
```

### API_BASE_URL Resolution Order

1. `window.__API_BASE_URL__` (set by `/config.js` in index.html)
2. `import.meta.env.VITE_API_BASE_URL` (Vite env variable)
3. Default: `http://localhost:9000`

## Key User Flows

### 1. Game Start
```
Mount → GET /state_value(0, 0) → SET_INITIAL_EV → trajectory seeded
```

### 2. Roll
```
Click Roll → ROLL (random dice) → POST /evaluate → SET_EVAL_RESPONSE
  → UI shows optimal hints (green/red borders on dice, EV bars on categories)
  → POST /density → SET_DENSITY_RESULT → trajectory updated with percentiles
```

### 3. Reroll
```
Click dice to toggle held → Click Reroll → REROLL (re-randomize unheld)
  → POST /evaluate → SET_EVAL_RESPONSE → UI updates
```

### 4. Score
```
Click Score button on category → SCORE_CATEGORY → state updated
  → If 15 categories scored → game_over
  → Otherwise → idle (ready for next roll)
```

## Visualizations

### TrajectoryChart (Canvas-based, 272 LOC)

- **Responsive:** Scales to container width × device pixel ratio
- **Y-axis:** Auto-ranged from data with 10% padding
- **Grid:** Nice-step calculation (2/5/10 multiples)
- **Three series:**
  1. Accumulated score — green shaded area + thin line
  2. Expected final score — bold blue line (main trajectory)
  3. Percentile bands — p10-p90 and p25-p75 as filled polygons
- **Points:** Gray (start), blue (roll), orange (reroll), green (score, larger radius)
- **X-axis:** Turn numbers at score events

### ScorecardRow Spark Bars

Each row shows a normalized background gradient:
- Score bar: width proportional to actual score / max possible
- EV bar: width proportional to ev_if_scored / max_ev

## Dependency List (Migration Baseline)

### Production
```json
"react": "^19.2.0",
"react-dom": "^19.2.0"
```

### Development
```json
"@types/react": "^19.0.0",
"@types/react-dom": "^19.0.0",
"@vitejs/plugin-react": "^4.3.4",
"eslint": "^9.9.1",
"eslint-plugin-react-hooks": "^5.1.0-rc.0",
"eslint-plugin-react-refresh": "^0.4.12",
"globals": "^15.11.0",
"typescript": "~5.9.3",
"typescript-eslint": "^8.15.0",
"vite": "^7.3.1",
"vitest": "^3.2.4"
```

### No External UI Libraries
- No component library (MUI, Chakra, etc.)
- No state management library (Redux, Zustand)
- No chart library (canvas is hand-written)
- No CSS framework (raw CSS)

### Existing Tests

- `reducer.test.ts` (384 LOC) — tests all reducer actions
- `mask.test.ts` (83 LOC) — tests dice sort/mask translation
- Run: `npm test` or `npx vitest`

## Blog Game (Parallel Implementation)

The blog at `blog/js/game/` contains a complete reimplementation of the frontend in vanilla JavaScript:

| Frontend (React) | Blog (Vanilla JS) | Shared Logic |
|-------------------|--------------------|-------------|
| `reducer.ts` (355 LOC) | `reducer.js` (290 LOC) | ~90% overlap |
| `api.ts` (64 LOC) | `api.js` (~50 LOC) | Identical endpoints |
| `mask.ts` (67 LOC) | `mask.js` (~60 LOC) | Identical logic |
| `constants.ts` (27 LOC) | `constants.js` (~25 LOC) | Identical values |
| 9 React components | 8 vanilla JS components | Parallel UI |

No shared component library exists between frontend and blog.

## Blog Profiling Quiz System

Separate from the game, the blog has a profiling quiz at `blog/js/profile/` (~2,500 LOC):

### Architecture
- **Flux store** (`store.js`, 163 LOC) — phases: loading → intro → answering → complete
- **MLE estimator** (`estimator.js`, 347 LOC) — multi-start Nelder-Mead for (θ, β, γ, d)
- **Renderer** (`render.js`) — builds DOM, subscribes to store, runs estimation
- **7 UI components:**
  - `scenario-card.js` (361 LOC) — interactive quiz question
  - `parameter-chart.js` (308 LOC) — real-time D3 convergence chart
  - `profile-scorecard.js` (231 LOC) — game state display
  - `player-card.js` (160 LOC) — simulation-backed performance analytics
  - `result-panel.js` (78 LOC) — natural language profile summary
  - `progress-bar.js` — quiz progress indicator
  - `question-list.js` — sticky sidebar with question status

### Data Flow
```
blog/data/scenarios.json (30 scenarios + Q-grids, pre-computed)
  → store.SCENARIOS_LOADED
  → User answers scenarios (ANSWER action)
  → estimator.estimate(answers, scenarios)
    → Nelder-Mead MLE: maximize P(answers | θ, β, γ, d)
    → Uses Q-grid lookup (no API calls)
  → store.UPDATE_PROFILE
  → result-panel + player-card render
```

### Blog Article Charts (11 D3 visualizations)

Built with `blog/js/yatzy-viz.js` (299 LOC) shared utility:

| Chart | File | LOC | Data Source |
|-------|------|-----|------------|
| Score Distribution | `score-distribution.js` | 166 | kde_curves.json |
| Mixture Decomposition | `mixture-decomposition.js` | 199 | mixture.json |
| Mean-Variance | `mean-variance.js` | 127 | sweep_summary.json |
| Risk-Reward | `risk-reward.js` | 238 | sweep_summary.json |
| Decision Explorer | `decision-explorer.js` | 117 | game_eval.json |
| Heuristic Gap | `heuristic-gap.js` | 146 | heuristic_gap.json |
| Surrogate Pareto | `surrogate-pareto.js` | 170 | (embedded) |
| Win Rate | `win-rate.js` | 139 | winrate.json |
| State Heatmap | `state-heatmap.js` | 163 | state_heatmap.json |
| Widget Explorer | `widget-explorer.js` | 267 | (API call) |
| Greedy vs Optimal | `greedy-vs-optimal.js` | 116 | greedy_vs_optimal.json |

All lazy-loaded via IntersectionObserver with 200px rootMargin.

## Blog CSS (2,041 LOC)

| File | LOC | Scope |
|------|-----|-------|
| `style.css` | 433 | Global theme, dark mode, layout |
| `profile.css` | 832 | Quiz layout, sticky sidebar, scenario card |
| `game.css` | 352 | Game UI (dice, scorecard) |
| `charts.css` | 424 | D3 chart styling, tooltips |

Dark mode via `html.dark` class, CSS custom properties: `--text`, `--bg`, `--accent`, `--border`.
