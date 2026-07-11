# Probabilities UI

Standalone client-side tool at `/yatzy/probabilities/`. Three always-visible dice rows
(first roll, second roll, target). Set each die with the up/down arrows, click a
die to keep it, and read the transition probabilities:

- P(first roll → second roll)
- P(second roll → target)
- Overall = the product of the two.

A die kept in one row that shows a different value in the next row makes that
transition impossible (0%).

- `js/render.js` — the three-row UI and interaction.
- `js/main.js` — entry point.
- Math lives in `/yatzy/shared/path-prob.js` (shared with the treatise
  path-probability chart so the two cannot drift). Unit-tested.

No build step. Run the math tests: `node --test js/path-prob.test.js`.
