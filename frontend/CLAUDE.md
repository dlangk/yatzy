# CLAUDE.md — Frontend

React + TypeScript + Vite game-playing UI with real-time optimal-action hints.

## Commands

```bash
cd frontend
npm install       # Install dependencies
npm run dev       # Dev server (port 5173)
npm run build     # Production build
npm test          # Run vitest (34 tests)
npm run lint      # ESLint
```

## Architecture

- **Framework**: React 19 + TypeScript strict mode + Vite
- **State**: `useReducer` pattern in `reducer.ts` with actions defined in `types.ts`
- **API**: All backend calls through `api.ts` (solver on port 9000)
- **Tests**: vitest — `mask.test.ts` (11 tests), `reducer.test.ts` (23 tests)

## Source Layout

| File | Purpose |
|------|---------|
| `App.tsx` | Root component, game orchestration |
| `reducer.ts` | Game state reducer (useReducer) |
| `types.ts` | TypeScript interfaces for all data contracts |
| `api.ts` | Backend API client (fetch wrappers) |
| `mask.test.ts` | Bitmask operation tests |
| `reducer.test.ts` | Game state reducer tests |
| `components/TrajectoryChart.tsx` | Score trajectory visualization |

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
- Never return null — always render fixed-size containers
- Fixed row heights, fixed column widths
- Same element, different content (never swap element types)
