# Delta Yatzy

Optimal-play Scandinavian Yatzy: backward-induction DP solver, risk-sensitive strategies, Monte Carlo simulation, exact density evolution, cognitive profiling, and a web-based game UI.

## Components

| Component | Location | Purpose | Key Constraint |
|-----------|----------|---------|----------------|
| Solver | `solver/` | HPC Rust engine: DP, simulation, REST API | **Performance is sacred** — see hot path rules |
| Frontend | `frontend/` | Vanilla TypeScript + D3.js game UI | No layout shifts |
| Analytics | `analytics/` | Python analysis, visualization, pipelines | Custom colormap (`#F37021` center) |
| Blog | `blog/` | Static site: articles, profiling quiz, game | Pre-computed data, no runtime API |

See component CLAUDE.md files for detailed guidance:
- `solver/CLAUDE.md` — architecture, hot paths, API reference, perf testing
- `frontend/CLAUDE.md` — Vanilla TS patterns, store, layout rules
- `analytics/CLAUDE.md` — CLI commands, colormap, data flow, plotting
- `blog/CLAUDE.md` — quiz system, data files, component architecture

## Critical Rules

- NEVER trade solver performance for code aesthetics — run `just bench-check` after Rust changes
- Intentionally duplicated hot-path code is marked `// PERF: intentional` — do not refactor it
- All state values use f32 throughout (storage + computation)
- STATE_STRIDE=128, state_index(up, scored) = scored * 128 + up
- Delete `data/strategy_tables/all_states_theta_*.bin` after changing solver code
- When modifying an API endpoint, update `solver/CLAUDE.md` AND `frontend/src/api.ts`
- When a conversation produces new insights, update the appropriate file in `theory/` (see `theory/README.md`)

## Commands

```bash
# Build + test
just setup              # Build solver + install analytics
just build              # Production build (solver + frontend)
just test               # Solver tests (184 tests)
just test-all           # All tests (solver + frontend + analytics)
just check              # Full quality gate (lint + typecheck + test + bench)
just bench-check        # Performance regression test (PASS/FAIL)

# Code quality
just fmt                # Format all components
just lint-all           # Lint all components
just typecheck          # Type-check solver + frontend

# Precompute + simulate
just precompute         # θ=0 strategy table (~504ms)
just sweep              # All 37 θ values (resumable)
just simulate           # 1M games lockstep

# Analyze + visualize
just pipeline           # compute → plot → categories → efficiency
just density            # Exact forward-DP PMFs

# Serve + dev
just serve              # API server on port 9000
just dev-backend        # Start backend server
just dev-frontend       # Start frontend dev server
```

Full recipe list: `just --list`

## Architecture

```
Frontend (Vanilla TS, :5173) ──POST /evaluate──→ Solver (axum, :9000)
Blog (static)           ──reads──→ blog/data/*.json (pre-computed)
Analytics (Python)      ──reads──→ data/simulations/theta/*/scores.bin
```

All game intelligence lives in the solver. Frontend and blog are thin clients. Analytics reads binary simulation files from disk.

### Computation Pipeline

1. **Precompute** — backward induction DP over 1.43M reachable states → `data/strategy_tables/`
2. **Simulate** — Monte Carlo with optimal policy → `data/simulations/`
3. **Analyze** — KDE, percentiles, CVaR, MER → `outputs/aggregates/`
4. **Serve** — mmap-load strategy table, stateless REST lookups
5. **Visualize** — 50+ PNG plots → `outputs/plots/`

## Data Layout

```
data/                              # Expensive (gitignored)
  strategy_tables/all_states*.bin  # 16 MB × 37 θ values
  simulations/theta/theta_*/       # scores.bin (~2 MB per θ)
  strategy_tables/oracle.bin       # 3.17 GB (θ=0 only, optional)

outputs/                           # Cheap to regenerate (gitignored)
  aggregates/parquet/              # summary, kde, mer, sdva
  aggregates/csv/                  # sweep_summary, category_stats
  plots/                           # ~50 PNGs at 200 DPI
  scenarios/                       # Pivotal/difficult scenario JSON
  profiling/                       # Quiz scenarios + Q-grids

blog/data/                         # Pre-computed for static site
  scenarios.json                   # Profiling quiz (copied from outputs/)
  player_card_grid.json            # 648 simulation grid entries
```

## Game Rules (Scandinavian Yatzy)

- 15 categories (not 13 as in American Yahtzee)
- 50-point upper bonus (not 35)
- No Yahtzee bonus flag
- State: upper_score (0-63) × scored_categories (15-bit bitmask) = 2,097,152 slots

## Theory

See `theory/README.md` for the full directory index. Key entry points:

- `theory/foundations/algorithm-and-dp.md` — DP algorithm and SOLVE_WIDGET
- `theory/foundations/pseudocode.md` — optimal algorithm pseudocode
- `theory/foundations/risk-parameter-theta.md` — risk-sensitive solver math
- `theory/strategy/risk-sensitive-strategy.md` — θ sweep results and decision analysis
- `theory/lab-reports/hardware-and-hot-path.md` — optimization history and hardware reference
- `theory/research/human-cognition-and-compression.md` — human vs optimal, surrogate compression
