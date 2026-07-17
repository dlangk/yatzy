# CLAUDE.md — Frontend

Vanilla TypeScript + D3.js + Vite game-playing UI with real-time optimal-action hints.

## Commands

```bash
cd frontend
npm install       # Install dependencies
npm run dev       # Dev server (port 5173)
npm run build     # Production build
npm test          # Run vitest (58 tests)
npm run lint      # ESLint
```

## Architecture

- **Framework**: Vanilla TypeScript strict mode + Vite (no React)
- **State**: Flux-like store in `store.ts` with `dispatch()`/`subscribe()`/`getState()`
- **Reducer**: Pure function in `reducer.ts` with actions defined in `types.ts`
- **API**: All backend calls through `api.ts` (solver on port 9000)
- **Chart**: D3.js SVG trajectory visualization
- **Styles**: CSS custom properties in `style.css`
- **Tests**: vitest, node environment, no DOM. `mask.test.ts` (23), `reducer.test.ts` (25), `store.test.ts` (6), `api.test.ts` (4)

## Source Layout

| File | Purpose |
|------|---------|
| `main.ts` | Entry point, init store, wire side effects |
| `App.ts` | `initApp()` — creates DOM skeleton, inits components |
| `store.ts` | Flux store: `dispatch()`, `subscribe()`, `getState()`; persists on each dispatch |
| `reducer.ts` | Pure game state reducer (transitions only; persistence lives in `persistence.ts`) |
| `persistence.ts` | Versioned localStorage: `yatzy-game-state` (durable game data, cleared by New Game) + `yatzy-prefs` (preferences, survive New Game). Excludes derived/heavy fields (lastEvalResponse, undo/redo) |
| `types.ts` | TypeScript interfaces for all data contracts. `Prefs` (showHints, showDebug, guideOpen, autoplayDelay) is separate from game data and lives at `state.prefs` |
| `api.ts` | Backend API client (fetch wrappers) |
| `autoplay.ts` | Auto-play side-effect loop; delay is `state.prefs.autoplayDelay` |
| `hoverBus.ts` | Lightweight event bus for cross-component hover coordination |
| `tooltips.ts` | Centralized copy for the guidance tooltips |
| `mask.ts` | Pure dice-mask helpers and the DiceBar view model. All dice-bar display logic lives here so it is testable without a DOM |
| `constants.ts` | Colors, category names, game constants |
| `config.ts` | API base URL configuration |
| `style.css` | All styles via CSS custom properties |
| `components/ActionBar.ts` | Roll/reroll, reroll counter, Hints/Guide toggles, undo/redo, reset |
| `components/HowTo.ts` | Dismissible "how to play" intro box (toggled by the Guide button via `state.prefs.guideOpen`) |
| `components/Tooltip.ts` | Reusable accessible tooltip util (`attachTooltip` / `attachHoverTooltip`) |
| `components/DebugPanel.ts` | Collapsible raw state inspector |
| `components/DecisionLog.ts` | Per-turn reroll/score decision table with delta bars |
| `components/DiceBar.ts` | Renders `diceBarViewModel()`: five dice, the solver advice row, mismatch ticks, cost line |
| `components/Die.ts` | Single die: value arrows plus one button wrapping the die SVG and its HOLD/REROLL label |
| `components/EvalPanel.ts` | State value and optimal action summary |
| `components/Scorecard.ts` | 15-category scoring table with EV column |
| `components/ScorecardRow.ts` | Single scorecard row factory (score input + action button) |
| `components/TrajectoryChart.ts` | D3 SVG line chart with percentile bands |

## Component Pattern

```typescript
export function initComponent(container: HTMLElement): void {
  // 1. Create DOM elements once (fixed sizes)
  // 2. Subscribe to store for reactive updates
  // 3. Update content/styles on state change
}
```

## API Integration

The frontend calls these solver endpoints via `api.ts`:

| Frontend Call | Endpoint | Method | Latency |
|--------------|----------|--------|---------|
| Evaluate roll | `/evaluate` | POST | ~9μs |
| Get state value | `/state_value` | GET | <1ms |
| Health check | `/health` | GET | <1ms |
| Score density | `/density` | POST | <10ms-50ms |

## Colour Contract

Each colour means exactly one thing across the Play UI. Do not overload them.

| Colour | Meaning | Example |
|--------|---------|---------|
| ink (`--text`) | Hold: settled | Die label, advice row |
| orange (`--color-orange-text`) | Reroll: in motion | Die label, advice row |
| green (`--color-success`) | Good, ahead of par | Scorecard bonus |
| red (`--color-danger`) | Giving up points | Mismatch ticks, cost line, scorecard bonus |

`--color-orange` (`#F37021`) is the brand orange, for fills and accents only. Text
and focus rings use `--color-orange-text` (`#B54A0C` in light, `#F37021` in dark):
the brand orange measures 2.69:1 on the light background, under the WCAG AA 4.5:1
text floor and the SC 1.4.11 3:1 indicator floor.

Green and red previously encoded hold/reroll on the dice while also meaning
quality in the scorecard. That collision is the bug fixed in
`docs/superpowers/specs/2026-07-17-dice-hold-reroll-clarity-design.md`. Do not
reintroduce it.

## No Layout Shifts (Hard Requirement)

See `.claude/rules/frontend.md` for the full constraint list.
