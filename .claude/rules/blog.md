---
paths:
  - "blog/**/*.{js,html,css}"
---

# Blog Rules

- Static site with vanilla JS (ES6 modules). No build step, no framework.
- Three pages: `index.html` (article), `profile.html` (quiz), `play.html` (game).
- Profile quiz uses Flux-like store (`js/profile/store.js`) and component-based rendering.
- Data files in `blog/data/` are pre-computed — the blog does NOT call the solver API at runtime (except `play.html` which embeds the game frontend).
- Quiz estimator: multi-start Nelder-Mead MLE for (θ, β, γ, d) parameters.
- Q-grid keys use Rust float formatting (e.g. "0" not "0.0") — this is critical for lookup.
- Layout: quiz container is 675px with absolutely-positioned sidecar, hidden below 1020px.
- Tests: `vitest run` (25 tests in mask.test.js + reducer.test.js).
