# Theory

Theoretical foundations, experimental results, and research for the Delta Yatzy optimal solver.

## Directory Structure

| Directory | Files | Purpose |
|-----------|------:|---------|
| [foundations/](foundations/) | 6 | Core math: DP algorithm, game theory, risk model |
| [lab-reports/](lab-reports/) | 12 | Performance experiments with measured results |
| [research/](research/) | 11 | Exploratory ideas, surveys, ML approaches |
| [strategy/](strategy/) | 6 | Game strategy, sweep pipelines, player tools |
| [reviews/](reviews/) | 2 | Literature surveys and external assessments |

## Key Entry Points

- **How the solver works**: `foundations/algorithm-and-dp.md` → `foundations/pseudocode.md`
- **Why the solver is fast**: `lab-reports/hardware-and-hot-path.md` → `lab-reports/neon-intrinsics.md`
- **Risk-sensitive play**: `foundations/risk-parameter-theta.md` → `strategy/risk-sensitive-strategy.md`
- **Score distributions**: `foundations/optimal-play-and-distributions.md`
- **Human vs optimal**: `research/human-cognition-and-compression.md`
- **Full θ sweep data**: `strategy/applications-and-appendices.md` (Appendix C)
- **Literature context**: `reviews/literature-survey.md` → `reviews/external-review.md`
