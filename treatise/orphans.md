# Orphaned Content

Content removed during the 10→8 section restructuring (2026-02-26).

## Scroll Animation (State Space Counter)
- **chart-state-space-counter**: Interactive scrollytelling animation that counted through state space dimensions
- Removed by design: the information is conveyed in prose instead
- Files deleted: `js/charts/state-space-counter.js`, `js/utils/scroll-driver.js`, `js/utils/animated-counter.js`, `data/state_counter_steps.json`
- CSS removed: `.scroll-sticky`, `.scroll-steps .step`, `.counter-big`, `.counter-label`, `.counter-bar-*`

## Part Labels
- "Part I" through "Part VIII" and "Conclusion" labels (`:::part-title` blocks) removed from all sections
- Replaced by numbered section headings (02–08) without explicit part framing

## No Prose Orphaned
All narrative content has a home in the new 8-section structure:
- Geometry → State Space (section 2)
- Engineering → Solver (section 3)
- Decision Anatomy moved from Geometry → Optimal Strategy (section 4)
- Luck → Optimal Strategy (section 4)
- Thermodynamics → Risk Parameter (section 5)
- Multiplayer → Risk Parameter, subsection (section 5)
- Compression → Compression (section 6)
- Rosetta → Compression, subsection (section 6)
- Interpretability → Profiling (section 7)
- Conclusion → Conclusions (section 8)
