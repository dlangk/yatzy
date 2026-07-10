# Probabilities UI

Standalone client-side tool at `/yatzy/prob/`. Given an opening hand and a
target, shows the probability of reaching the target across two rerolls under
your hold vs. optimal (target-maximizing) play.

- `js/engine.js` — pure probability engine (exact enumeration). Unit-tested.
- `js/targets.js` — category + exact-hand predicates. Unit-tested.
- `js/render.js` — three-row live simulation trace.
- `js/main.js` — entry point.

No build step. Run tests: `node --test js/*.test.js`.
