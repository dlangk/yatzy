# CLAUDE.md — Frontend

Vanilla TypeScript + D3.js + Vite game-playing UI with real-time optimal-action hints.

## Commands

```bash
cd frontend
npm install       # Install dependencies
npm run dev       # Dev server (port 5173)
npm run build     # Production build
npm test          # Run vitest (46 tests)
npm run lint      # ESLint
```

## Architecture

- **Framework**: Vanilla TypeScript strict mode + Vite (no React)
- **State**: Flux-like store in `store.ts` with `dispatch()`/`subscribe()`/`getState()`
- **Reducer**: Pure function in `reducer.ts` with actions defined in `types.ts`
- **API**: All backend calls through `api.ts` (solver on port 9000)
- **Chart**: D3.js SVG trajectory visualization
- **Styles**: CSS custom properties in `style.css`
- **Tests**: vitest — `mask.test.ts` (11), `reducer.test.ts` (23), `store.test.ts` (6), `api.test.ts` (6)

## Source Layout

| File | Purpose |
|------|---------|
| `main.ts` | Entry point, init store, wire side effects |
| `App.ts` | `initApp()` — creates DOM skeleton, inits components |
| `store.ts` | Flux store: `dispatch()`, `subscribe()`, `getState()` |
| `reducer.ts` | Pure game state reducer + localStorage persistence |
| `types.ts` | TypeScript interfaces for all data contracts |
| `api.ts` | Backend API client (fetch wrappers) |
| `mask.ts` | Dice sort-mapping utilities |
| `constants.ts` | Colors, category names, game constants |
| `config.ts` | API base URL configuration |
| `style.css` | All styles via CSS custom properties |
| `components/*.ts` | Vanilla TS component modules (init + subscribe pattern) |

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
| Score density | `/density` | POST | ~100ms |

## No Layout Shifts (Hard Requirement)

See `.claude/rules/frontend.md` for the full rule set. Key points:
- Always render containers at fixed size. Use placeholders (`?`, `—`).
- Fixed row heights, fixed column widths
- Same element, different content — change text/style, never swap element types
