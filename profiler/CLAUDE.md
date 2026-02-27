# CLAUDE.md — Profiler

Static site with articles and interactive profiling quiz.

## Commands

```bash
python3 -m http.server # Serve locally for development
```

## Pages

| Page | File | Purpose |
|------|------|---------|
| Article | `index.html` | Data-driven Yatzy analysis with D3 charts |
| Profile Quiz | `profile/index.html` | 30-scenario cognitive profiling |

## Profile Quiz Architecture

Flux-like state management with component-based rendering (no framework):

| Module | Purpose |
|--------|---------|
| `js/profile/store.js` | State management (ANSWER, CLEAR_ANSWER, ADVANCE, GO_TO) |
| `js/profile/estimator.js` | Multi-start Nelder-Mead MLE for (θ, β, γ, d) |
| `js/profile/render.js` | Quiz orchestrator: DOM building + estimation |
| `js/profile/api.js` | Scenario data loading from `data/scenarios.json` |
| `js/profile/player-card-data.js` | Grid loader, nearest-neighbor, counterfactuals |
| `components/scenario-card.js` | Interactive quiz question (dice, scorecard, actions) |
| `components/question-list.js` | Clickable sidebar with question status |
| `components/result-panel.js` | Parameter estimates (visible after all answered) |
| `components/parameter-chart.js` | Real-time convergence chart |
| `components/player-card.js` | Simulation-backed performance analytics |

## Data Files

| File | Source | Content |
|------|--------|---------|
| `data/scenarios.json` | `just profile-deploy` | 30 scenarios + Q-grids (108 combos each) |
| `data/player_card_grid.json` | `just player-card-grid` | 648 combos × 10K games |
| `data/kde_curves.json` | Custom export | KDE data for article charts |
| `data/sweep_summary.json` | Custom export | Summary stats for article |

## Critical Notes

- Q-grid keys use Rust float formatting: `"0"` not `"0.0"`. This breaks if reformatted.
- Quiz layout: 675px container with sticky sidecar, hidden below 1020px viewport.
- 4-parameter model: θ (risk), β (precision), γ (myopia), d (depth).
- Game UI lives in `frontend/` (canonical Vite+TS app), not here.
