# CLAUDE.md — Treatise

Markdown-driven static site: 8 sections covering DP theory, optimal strategy, risk sensitivity, and cognitive profiling.

## Commands

```bash
cd treatise && npm run build   # Markdown → HTML (node build.mjs)
just build-treatise            # Same, from repo root
just dev-frontend              # Vite serves at /yatzy/
```

## Build System

`build.mjs` converts `sections/*.md` → `sections/*.html` via markdown-it, then assembles `index.html` from `header.html` + all section HTML fragments. No bundler — output is a single HTML file with `<script type="module">` imports.

## Source Layout

| Path | Purpose |
|------|---------|
| `sections/*.md` | Markdown source (numbered 01-08) |
| `sections/*.html` | Generated HTML fragments (gitignored) |
| `index.html` | Assembled output (gitignored) |
| `css/style.css` | Layout, typography, theme |
| `css/charts.css` | D3 chart containers and controls |
| `css/treatise.css` | Section-specific styles |
| `js/yatzy-viz.js` | Main chart orchestrator (reads data, binds sliders) |
| `js/charts/*.js` | Individual chart modules |
| `js/charts/dice-symmetry.js` | Interactive 252-multiset explorer with reroll sidecar |
| `js/utils/dice-interactive.js` | Selectable dice renderer (shared by charts) |
| `scripts/gen-dice-symmetry.mjs` | Generator for `data/dice_symmetry.json` |
| `js/data-loader.js` | Fetch + cache JSON data files |
| `js/concept-drawer.js` | Concept glossary drawer |
| `js/depth-toggle.js` | Technical depth expand/collapse |
| `js/theme.js` | Dark/light theme toggle |
| `data/` | Pre-computed JSON for charts |

## D3 Chart Pattern

Charts are initialized by `yatzy-viz.js` which queries the DOM for `[data-chart]` containers. Each chart module exports an `init(container, data)` function. Slider controls use `<input type="range">` with a `.slider-value` span updated on input.

## Sections

1. Introduction — game overview
2. State Space — combinatorial structure
3. Solver — backward induction DP
4. Optimal Strategy — category analysis
5. Risk Parameter — θ sweep, score distributions
6. Compression — surrogate policy models
7. Profiling — cognitive parameter estimation
8. Conclusion
