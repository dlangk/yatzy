# Probabilities Tab — Design

**Date:** 2026-07-10
**Status:** Approved (design), pending implementation plan

## Summary

A fourth standalone UI, "Probabilities", at `/yatzy/prob/`, that answers questions
like "if I start with [1,2,3,4,6], how likely am I to get a small straight during
the remaining two rerolls?"

It is a **live simulation trace**: three rows of dice (opening hand, after reroll 1,
after reroll 2), each with a hold toggle and a per-row randomizer, showing the
probability of reaching a chosen target from each stage forward. Two probabilities
are shown side by side at every stage: the probability under **your** current hold,
and the **best achievable** probability under optimal play.

The entire tool runs **client-side**. There is no solver endpoint, no `data/`
dependency, and no build step. It mirrors the profiler's static-UI pattern.

## Goals

- Let a user set an opening hand and a target, then explore how the probability of
  hitting the target evolves across two rerolls.
- Show, at each of the three stages, the probability under the user's current hold
  vs. the best possible hold, so the cost of a hold decision is visible.
- Support both "category" targets (small straight, full house, yatzy, …) and an
  "exact hand" target (a specific unordered multiset like {1,2,3,4,6}).
- Instant, exact computation in the browser. No network calls.

## Non-Goals

- Not tied to the score-maximizing game solver. "Optimal" here means
  target-probability-maximizing (see Probability Model).
- No persistence, no accounts, no sharing.
- No multiplayer, no risk parameter (θ), no expected-score analysis.

## Placement & Architecture

New standalone UI at `/yatzy/prob/`, mirroring `profiler/` exactly:

```
prob/
  index.html          # loads shared/nav.js, shared/dice.css, own css + js
  css/
    style.css         # page chrome (reuse profiler tokens: Newsreader, theme vars)
    prob.css          # tool-specific layout
  js/
    engine.js         # pure probability engine (no DOM) — unit-tested
    dice.js           # thin re-export/usage of /yatzy/shared/dice.js helpers
    targets.js        # target predicates (categories) + exact-hand matcher
    render.js         # DOM: three rows, toggles, randomizers, live updates
    main.js           # entry: wire render + initial state + theme
  README.md
```

- **Language:** plain ES-module JavaScript with JSDoc types. **No build step.**
  Matches the profiler; keeps deploy to a file copy. (Decision: chosen over TS to
  stay consistent with the sibling static UI and avoid a dist/build step.)
- **Shared assets:** `shared/nav.js` (add a 4th nav entry), `shared/dice.js` +
  `shared/dice.css` for dice rendering. Same Newsreader font + theme toggle as the
  other UIs.
- **Dev serving:** add one `serveDir('/yatzy/prob/', path.join(root, 'prob'))`
  plugin entry in `frontend/vite.config.ts`.
- **Production serving:** add a `location /prob/` block to `deploy/nginx.conf`
  (alias to `/usr/share/nginx/apps/prob/`, SPA fallback to `/prob/index.html`), a
  `/prob` → `/prob/` redirect, plus a copy step in `deploy/deploy.sh` that stages
  `prob/` into `apps/prob/`. Update the deploy command doc accordingly.

## Probability Model

Let a `hand` be a sorted multiset of five dice, each in 1..6. A target is a
predicate `target(hand) -> bool`. Rerolls remaining at each row: Row 1 = 2,
Row 2 = 1, Row 3 = 0.

Two quantities, both by exact enumeration:

**Best achievable (optimal, target-maximizing):**
```
P_opt(hand, 0) = target(hand) ? 1 : 0
P_opt(hand, r) = max over keep-sets K ⊆ hand of
                   Σ_outcomes Pr(outcome) · P_opt(K ∪ outcome, r−1)
```
where `K` ranges over which dice to keep (by value-multiset, 0..5 kept) and
`outcome` ranges over the reroll results of the `5 − |K|` freed dice, weighted by
multinomial probability.

**Under your hold (your current keep now, optimal after):**
```
P_you(hand, r, yourHolds) = Σ_outcomes Pr(outcome) · P_opt(yourHolds ∪ outcome, r−1)
```

Displayed per row: `P_you` ("you") and `P_opt` ("best"). The gap `P_opt − P_you`
is the cost of the current hold; the optimal keep-set is surfaced as a hint.

**Why "optimal" ≠ the game solver.** The main solver maximizes expected score
(risk-adjusted by θ). This tool maximizes the probability of reaching one specific
target, a different objective that yields a different policy. The UI states this
explicitly so the two are not conflated.

**Tractability.** There are only C(10,5) = 252 distinct 5-dice multisets. Reroll
outcomes for `k` freed dice (k = 0..5) are enumerated once as weighted multisets.
`P_opt` is memoized over `(hand, r)`. Full recompute on any target/dice change is
effectively instant.

## Target Modes

A toggle selects one of two modes:

- **Category mode:** ones, twos, …, sixes, one pair, two pairs, three of a kind,
  four of a kind, small straight (1-2-3-4-5), large straight (2-3-4-5-6),
  full house, chance, yatzy. Each is a `hand -> bool` predicate. (Scandinavian
  Yatzy: 15 categories; straights are the fixed 1-5 / 2-6 form.)
- **Exact hand mode:** the user picks a specific unordered multiset (e.g.
  {1,2,3,4,6}); `target(hand)` is true iff the sorted hand equals the target.

## Layout & Interaction (Live Trace)

```
Target:  ( Category ▾  Small straight )    |    ( Exact hand:  1 2 3 4 6 )

Row 1  (2 rerolls left)   [1][2][3][4][6]     you 61%   best 61%   [🎲 roll un-held]
                           held: 1 2 3 4                 hint: keep 1 2 3 4
Row 2  (1 reroll left)    [1][2][3][4][?]     you 33%   best 33%   [🎲]
Row 3  (final)            [1][2][3][4][5]     ✓ small straight
```

- **Dice:** click a die to cycle its value 1..6. A hold affordance toggles keep vs.
  reroll, reusing shared dice visual states (`kept` vs `will-reroll`).
- **Per-row randomizer:** rolls only the un-held dice of that row, populating the
  next row (held dice carry down unchanged).
- **Invalidation:** editing a die value or hold on an upper row clears the rows
  below it (they must be re-rolled).
- **Live probabilities:** `you %` and `best %` recompute on every toggle/edit/roll.
  Row 3 shows a pass/fail check instead of a forward probability (0 rerolls left).
- **Optimal-hold hint:** the argmax keep-set for each row is shown subtly so the
  user can compare their hold to the best one.

## Testing

- `engine.js` is pure (no DOM) and unit-tested against known Yatzy odds. Anchor
  cases:
  - P(yatzy | fresh hand, 2 rerolls, optimal) ≈ known value.
  - Small-straight cases from a near-complete hand (e.g. [1,2,3,4,6], keep 4,
    reroll 1) match hand-computed probabilities.
  - `P_you == P_opt` when the user's hold equals the optimal hold.
  - Degenerate: `r = 0` returns 1/0 exactly per `target`.
  - Exact-hand target of an already-matching hand at `r = 0` returns 1.
- Target predicates unit-tested per category (positive and negative hands).
- Runner: match the profiler's tooling (node-based test, no new heavy deps).

## Files Touched (outside `prob/`)

- `shared/nav.js` — add `{ id: 'prob', label: 'Probabilities', href: '/yatzy/prob/', … }`
  to `PAGES` and a branch in `detectActive()`.
- `frontend/vite.config.ts` — one `serveDir('/yatzy/prob/', …)` plugin entry.
- `deploy/nginx.conf` — `location /prob/` block + `/prob` → `/prob/` redirect.
- `deploy/deploy.sh` — stage `prob/` into `apps/prob/`.
- `.claude/commands/deploy.md` — document the new app in packaging + verification.

## Open Questions

None outstanding. All design decisions resolved during brainstorming:
placement (standalone UI), probability model (both you/best side by side),
target modes (category + exact), flow (live trace), language (plain JS).
