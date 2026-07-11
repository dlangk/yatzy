# Probabilities UI

Standalone client-side tool at `/yatzy/probabilities/`. Three always-visible dice rows
(first roll, second roll, target). Set each die with the up/down arrows, click a
die to keep it, and read the transition probabilities:

- P(first roll → second roll)
- P(second roll → target)
- Overall = the product of the two.

A die kept in one row that shows a different value in the next row makes that
transition impossible (0%).

The tab has two modules:

1. **Roll transitions** (`js/render.js`) — three-row UI (first roll, second
   roll, target) with per-die arrows and keep toggles; shows the transition
   probabilities and their product. Math: `/yatzy/shared/path-prob.js`.
2. **Score probability** (`js/score.js`) — "how likely was my score?" A target
   score input and a θ (risk) slider drive a distribution chart (vendored d3)
   with the target marked, and P(score ≥ target). Data:
   `/yatzy/data/kde_curves.json` (same file the treatise risk-theta chart uses).
   Math: `/yatzy/shared/score-prob.js`.

`js/main.js` initializes both. All math lives in shared modules (shared with the
treatise charts so the two cannot drift) and is unit-tested.

No build step. Run the math tests: `node --test js/*.test.js`.
