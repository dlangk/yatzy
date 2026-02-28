# Profiler — Cognitive Profiling Quiz

Static site with a 30-scenario cognitive profiling quiz. No build step, no framework.

## Quick Start

```bash
# Served by Vite in dev (via frontend/vite.config.ts)
just dev-frontend
# open http://localhost:5173/yatzy/profile/
```

## File Structure

```
profiler/
  index.html                  # Quiz entry point
  profile/index.html          # Legacy copy (used by deploy script)
  css/
    style.css                 # Layout, typography, dark mode
    game.css                  # Dice and scorecard components
    profile.css               # Quiz-specific styles
  js/
    theme.js                  # Dark mode toggle
    profile/
      store.js                # Flux-like state management
      estimator.js            # Multi-start Nelder-Mead MLE for (θ, β, γ, d)
      render.js               # Quiz orchestrator: DOM building + estimation
      api.js                  # Scenario data loading
      player-card-data.js     # Grid loader, nearest-neighbor, counterfactuals
      components/             # UI components (scenario-card, result-panel, etc.)
  data/
    scenarios.json            # 30 scenarios + Q-grids (108 combos each)
    player_card_grid.json     # 648 combos × 10K games
```

## Data Files

| File | Source | Content |
|------|--------|---------|
| `data/scenarios.json` | `just profile-deploy` | 30 scenarios + Q-grids |
| `data/player_card_grid.json` | `just player-card-grid` | 648 combos × 10K games |

## Critical Notes

- Q-grid keys use Rust float formatting: `"0"` not `"0.0"`. This breaks if reformatted.
- 4-parameter model: θ (risk), β (precision), γ (myopia), d (depth).
- Quiz layout: 675px container with sticky sidecar, hidden below 1020px viewport.
- The game UI lives in `frontend/` (Vite+TS app), not here.
