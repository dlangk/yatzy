---
paths:
  - "frontend/src/**/*.ts"
---

# Frontend Rules

## No Layout Shifts (Hard Requirement)

The UI must be pixel-stable. No element may change size, move, or cause reflow when state changes.

1. Never return `null` — always render containers at fixed size. Use placeholders (`?`, `—`).
2. Fixed row/cell heights — use explicit `height` or `minHeight`.
3. Same element, different content — change text/style, never swap element types.
4. Fixed column widths — `table-layout: fixed` with explicit `<colgroup>`.
