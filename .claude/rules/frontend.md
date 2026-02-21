---
paths:
  - "frontend/src/**/*.{ts,tsx}"
---

# Frontend Rules

- The frontend is React + TypeScript + Vite. State management via useReducer.
- Game logic lives in the Rust backend. The frontend does NOT reimplement it.
- All backend calls go through `api.ts` — no scattered fetch() calls.
- TypeScript strict mode. All data contracts defined in `types.ts`.

## No Layout Shifts (Hard Requirement)

The UI must be pixel-stable. No element may change size, move, or cause reflow when state changes.

1. Never return `null` — always render containers at fixed size. Use placeholders (`?`, `—`).
2. Fixed row/cell heights — use explicit `height` or `minHeight`.
3. Same element, different content — change text/style, never swap element types.
4. Fixed column widths — `table-layout: fixed` with explicit `<colgroup>`.
