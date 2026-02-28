---
paths:
  - "profiler/**/*.{js,html,css}"
---

# Profiler Rules

- Static site with vanilla JS (ES6 modules). No build step, no framework.
- Data files in `profiler/data/` are pre-computed — the profiler does NOT call the solver API.
- Q-grid keys use Rust float formatting (e.g. `"0"` not `"0.0"`) — this is critical for lookup.
- Quiz layout: 675px container with sticky sidecar, hidden below 1020px viewport.
