---
paths:
  - "profiler/**/*.{js,html,css}"
---

# Profiler Rules

- Static site with vanilla JS (ES6 modules). No build step, no framework.
- Two pages: `index.html` (article), `profile/index.html` (quiz). Game UI lives in `frontend/`.
- Profile quiz uses Flux-like store (`js/profile/store.js`) and component-based rendering.
- Data files in `profiler/data/` are pre-computed — the profiler does NOT call the solver API at runtime.
- Quiz estimator: multi-start Nelder-Mead MLE for (θ, β, γ, d) parameters.
- Q-grid keys use Rust float formatting (e.g. "0" not "0.0") — this is critical for lookup.
- Layout: quiz container is 675px with absolutely-positioned sidecar, hidden below 1020px.
