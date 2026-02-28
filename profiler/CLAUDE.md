# CLAUDE.md — Profiler

Static site with 30-scenario cognitive profiling quiz. No build step, no framework.

## Commands

```bash
# Served via Vite dev server (frontend/vite.config.ts)
just dev-frontend
# open http://localhost:5173/yatzy/profile/
```

## Pages

| Page | File | Purpose |
|------|------|---------|
| Profile Quiz | `index.html` | 30-scenario cognitive profiling |

## Architecture

Flux-like state management with component-based rendering (no framework):

| Module | Purpose |
|--------|---------|
| `js/profile/store.js` | State management (ANSWER, CLEAR_ANSWER, ADVANCE, GO_TO) |
| `js/profile/estimator.js` | Multi-start Nelder-Mead MLE for (θ, β, γ, d) |
| `js/profile/render.js` | Quiz orchestrator: DOM building + estimation |
| `js/profile/api.js` | Scenario data loading from `data/scenarios.json` |
| `js/profile/player-card-data.js` | Grid loader, nearest-neighbor, counterfactuals |
| `js/profile/components/scenario-card.js` | Interactive quiz question (dice, scorecard, actions) |
| `js/profile/components/question-list.js` | Clickable sidebar with question status |
| `js/profile/components/result-panel.js` | Parameter estimates (visible after all answered) |
| `js/profile/components/parameter-chart.js` | Real-time convergence chart |
| `js/profile/components/player-card.js` | Simulation-backed performance analytics |
| `js/profile/components/progress-bar.js` | Quiz progress indicator |
| `js/profile/components/profile-scorecard.js` | Scenario scorecard display |

## Data Files

| File | Source | Content |
|------|--------|---------|
| `data/scenarios.json` | `just profile-deploy` | 30 scenarios + Q-grids (108 combos each) |
| `data/player_card_grid.json` | `just player-card-grid` | 648 combos × 10K games |

## Critical Notes

- Q-grid keys use Rust float formatting: `"0"` not `"0.0"`. This breaks if reformatted.
- Quiz layout: 675px container with sticky sidecar, hidden below 1020px viewport.
- 4-parameter model: θ (risk), β (precision), γ (myopia), d (depth).
- Game UI lives in `frontend/` (canonical Vite+TS app), not here.
