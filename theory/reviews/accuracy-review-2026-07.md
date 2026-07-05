# Accuracy Review 2026-07

Full adversarial accuracy review of theory/ and treatise/ (~11,400 lines of
prose across ~80 documents), conducted with a multi-agent pipeline: 17
claim-verification units and 9 adversarial derivation reviewers produced 654
findings plus 97 flagged mathematical flaws; every contested factual finding
then received an independent refutation verdict (219/219 verified: 209
confirmed, 10 dismissed as reviewer misreadings). The derivation flaws did
not receive an independent defense pass (cost cap); they are labeled
single-reviewer below. All verification ran against a fresh empirical ground
truth: the exact forward-DP density (mean 248.4400677, sigma 38.4964) and a
1M-game Monte Carlo run with full recording.

## 1. Headline resolutions

**Bonus rate (was: 83.6% vs 87% vs ~90%).** Settled. The theta=0
optimal-policy end-of-game bonus rate is **89.77% +- 0.04%** (1M games; the
exact mixture decomposition gives 89.8%). "~90%" was always correct. **87%
was wrong** everywhere it appeared (strategy-guide, external-review,
human-cognition-and-compression, treatise section 7); all fixed to ~90%.
**83.6% is a different, correct statistic**: the fraction of games that have
already secured the bonus when entering the final turn (exact forward-DP:
0.8378); its wording has been clarified at the source. The old
contradictions.md resolution ("~90% could round from 87%") was wrong in both
directions and has been rewritten.

**Reachable-state count.** The solver reports **1,430,528** reachable states
including the 64 terminal states (1,430,464 without). The "~1.43M" claims
are fine; external-review.md's "exactly 1,429,984" does not match any
solver-reported quantity. The for-fact-checking per-layer table (previously
approximate and self-flagged) now carries the exact per-level counts.

**Solver runtime table conflict.** theta-sweep-architecture.md's
1.3s/1.3s/7.4s were pre-NEON values presented as current; corrected to the
post-NEON canon (1.1s EV / 0.49s utility / 2.7s LSE on M1 Max) with the
7.4s noted as pre-NEON. contradictions.md's chronology and its "different
hardware runs" story for the 3.7h-vs-4.2h sweep (actually stage vs total of
the same run) were also corrected.

**Multiplayer win-rate framing.** The oracle ceiling is 50.14% against a
measured EV-vs-EV baseline of 49.58% (draws make the baseline sub-50), i.e.
a gain of ~+0.56pp; the previously stated "+0.146pp" only holds against an
unstated symmetric 50.00% and was corrected in treatise section 6. The
53-55% figures in the speculative 3D-tablebase documents were never
computed anywhere in the repo and are now labeled as such.

## 2. Fixes applied (confirmed errors)

Roughly 100 corrections across 30+ files. The larger ones:

- treatise/for-fact-checking.md: exact reachability table; "handful of games
  above 320" corrected to 330 (P(>320) = 1.76% is ~35 games per lifetime);
  reverse-LUT size 32 to 30 KB; residual-range 154 to 152; density file
  naming.
- treatise/sections/07-compression.md: "trees dominate at every budget"
  qualified (game-level ranking flips mid-range: mlp_64 at 11K params scores
  221 vs dt_d10 at 6K params scoring 216); MLP param counts (5,000 to
  11,023); optimal bonus rate 87 to ~90%; feature table (3 dice aggregates,
  5 game-state features); export files are .bin not .csv; the
  DecisionTreeClassifier snippet now matches surrogate.py; phase-2 range.
- theory/strategy/risk-sensitive-strategy.md: CARA convention stated
  sign-safely; theta=+0.2 row corrected against on-disk data (mean 225.3,
  p50 228); R^2 = 0.21 over all 37 points (0.37 was the 8-theta subset);
  near-origin mean fit linear coefficient -9.9 (was -0.9, transcription
  error); Gaussian p95 overshoot ~3 pts; turn-1 fill ranking; the 83.6%
  clarification.
- theory/strategy/applications-and-appendices.md: Appendix C rows for
  theta = -0.20 (mean 223.6, not 227.5), +0.200 and +0.300 corrected
  against scores.bin.
- treatise/sections/05-risk-parameter.md: regime table gains the
  0.2-1.5 "extreme" band (policy still shifts; frozen only past ~1.5);
  max-policy tail probability is ~1e-19, not "billions of games";
  degenerate-regime mean ~186, not 210; disagreement arithmetic.
- treatise/sections/06-multiplayer.md: win-rate framing (above); the
  adaptive-policy code snippet and CLI flags, which matched nothing in the
  repo, replaced with the real adaptive.rs / multiplayer.rs interfaces.
- treatise/concepts/: CARA/exponential-utility/log-sum-exp/bellman-equation/
  free-energy sign conventions aligned with the solver (u = e^(theta x),
  LSE at chance nodes with probability weights, hard max/min at decision
  nodes, Z = E[e^(theta score)], dV/dtheta|0 = sigma^2/2, convexity belongs
  to K(theta)); state-space/decision-tree/supervised-distillation table
  sizes (16 MB on disk, 8 MB live; 16x compression; 11K-param MLP);
  keep-multiset savings (47% keep dedup, ~1.9x phase / 1.4x whole;
  the 8.9x figure belongs to keep-EV dedup); widget-solver buffer
  description matched to the real in-place table + 250 KB scratch;
  45,000 is cycles per widget, not "intermediate values";
  backward-induction processes the 1.43M reachable states, in ~1s (M1 Max).
- treatise/sections/08-profiling.md: gamma/beta grids matched to the code;
  18 (not 24) Nelder-Mead starts client-side.
- treatise/sections/04-optimal-strategy.md: /evaluate returns 32 keep-mask
  EVs (252 is the internal multiset count).
- theory/foundations/pseudocode.md: the reachability recurrence gains the
  m = 63 cap semantics; the exact-total form as previously written would
  prune 1,024 reachable states (2,792 vs 2,794 (mask, m) pairs) and break
  backward induction. The implementation was always correct; the doc was not.
- theory/foundations/algorithm-and-dp.md: state_index stride 64 to 128,
  8 MB to 16 MB on-disk; runtime line labeled pre-NEON with current values.
- Citations: Pape (2025) trained on RTX 3090 + Tesla T4, not "NVIDIA A40"
  (fixed in three files); Häfner's Yahtzotron reached ~236-241 (rules
  dependent), not "~240-242"; "matching Larsson & Sjöberg's 248.63" softened
  (ours is exactly 248.44; the 0.19 gap is unexplained, and the earlier
  "50-point bonus correction" explanation was unsupported); RL bonus-deficit
  framing in rl-and-ml-approaches.md (the deficit exceeds the whole gap).
- theory/reviews/facts.md: header revision note, ~40 stale line references
  refreshed programmatically, value rows corrected.

## 3. Dismissed findings (docs were right, reviewers misread)

10 findings were dismissed on independent re-derivation, most notably: the
83.6% bonus statistic (correct under its section's start-of-turn convention,
verified two independent ways); treatise 03's "thread count pinned to 8"
(accurate when written; superseded by available_parallelism this branch);
the Four-of-a-Kind zero-rate conditional claim; the 3d-tablebase "16 MB"
tier-2 memory figure.

## 4. Derivation review (Phase B; flaws are single-reviewer)

Nine documents received line-by-line adversarial math review. Sound cores:
the DP recurrences, multinomial transitions, widget decomposition and
iteration order; the LSE recurrence and cumulant expansion in
risk-parameter-theta; the variance decomposition, mixture weights, bonus
+72 decomposition and tail-fit arithmetic in
optimal-play-and-distributions; the moment-polytope skeleton of
mean-variance-frontier.

Flagged and NOT independently confirmed (review notes added to the three
most affected docs): the three formal proofs in game-theory-and-mdp Part V
(Pexider domain misuse; counterexample fails its own constraints; vacuous
epsilon bound); the sign/labeling cluster in risk-parameter-theta (risk
premium, free-energy beta factor, overflow rationale, method-3 validity);
mean-variance-frontier's three logical errors and empty math blocks;
optimal-play-and-distributions section 3.4's KDE/bimodality account (the
exact PMF is unimodal at every bandwidth per the reviewer); the
human-cognition 16x-compression parameter accounting (~3x inflated) and
proxy-vs-actual EV-loss conflation in the 82.6-pt decomposition narrative;
dp-solution-walkthrough's American-Yahtzee state count and random-play
baseline. Each is a candidate for a focused follow-up.

## 5. Not re-verifiable from disk (recorded, unchanged)

37 claims need data that is not on disk: the full 37-theta sweep rows
beyond the 8 stored thetas, best-theta-per-quantile targets (theta = -0.03
p5 = 183; +0.07 p95 = 313 - the latter in mild tension with p95 = 312 at
both stored neighbors), the max-policy simulation (374/118.7), the
theta = 0.07 disagreement analysis, theta = 3 category rates, the
profiler's gitignored data files, and the 10^197 game-count (no derivation
exists in the repo; plausible range depends entirely on counting
convention). Regenerate via `just sweep` and `just simulate-max` to close.

## 6. Flagged for a human decision

1. `research/exact-solver-survey.md` and `research/yahtzee-research-landscape.md`
   are byte-identical duplicates; delete one (README now flags this).
2. The speculative research cluster (3d-threshold-tablebase,
   3d-god-algorithm-prompt, percentile-optimization,
   asymmetric-utility-functions, mean-variance-frontier) presents unbuilt
   systems with measured-sounding numbers. Banners added; consider moving
   to a `research/speculative/` subdirectory or rewriting.
3. `research/treatise-outline.md` is a single 25,000-character line with
   `[CLAUDE: INSERT]` placeholders; archive or delete.
4. The game-theory-and-mdp Part V proofs need either repair or demotion to
   conjecture status (attack details in this review's source data).
5. Probability conservation: the exact density run now measures
   1 + 3.1e-7 total probability, while theta-sweep-architecture claims
   1 - 2.5e-12; likely a code-version difference worth a fresh look.

## See Also

- `facts.md`: the re-verified claim registry
- `contradictions.md`: rewritten resolutions
- `../lab-reports/hardware-and-hot-path.md`: performance round-2 record
