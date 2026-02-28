# Delta Yatzy

Optimal-play Scandinavian Yatzy: backward-induction DP solver, risk-sensitive strategies, Monte Carlo simulation, exact density evolution, cognitive profiling, and a web-based game UI.

## Components

| Component | Location | Purpose | Key Constraint |
|-----------|----------|---------|----------------|
| Solver | `solver/` | HPC Rust engine: DP, simulation, REST API | **Performance is sacred** — see hot path rules |
| Frontend | `frontend/` | Vanilla TypeScript + D3.js game UI + Vite dev server | No layout shifts |
| Treatise | `treatise/` | Markdown-driven static site: theory, D3 charts | Build with `just build-treatise` |
| Profiler | `profiler/` | Cognitive profiling quiz (30 scenarios) | Pre-computed data, no runtime API |
| Analytics | `analytics/` | Python analysis, visualization, pipelines | Custom colormap (`#F37021` center) |

See component CLAUDE.md files for detailed guidance:
- `solver/CLAUDE.md` — architecture, hot paths, API reference, perf baselines, θ parameter
- `frontend/CLAUDE.md` — Vanilla TS patterns, store, layout rules
- `treatise/CLAUDE.md` — build system, section structure, D3 chart pattern
- `analytics/CLAUDE.md` — CLI commands, colormap, data flow, plotting
- `profiler/CLAUDE.md` — quiz architecture, data files, critical notes

## Critical Rules

- NEVER trade solver performance for code aesthetics — run `just bench-check` after Rust changes
- Intentionally duplicated hot-path code is marked `// PERF: intentional` — do not refactor it
- All state values use f32 throughout — see `solver/CLAUDE.md` for state layout and θ parameter details
- Delete `data/strategy_tables/all_states_theta_*.bin` after changing solver code
- When modifying an API endpoint, update `solver/CLAUDE.md` AND `frontend/src/api.ts` AND `frontend/CLAUDE.md`
- When a conversation produces new insights, update the appropriate file in `theory/` (see `theory/README.md`)

## Commands

```bash
# Build + test
just setup              # Build solver + install analytics
just build              # Production build (solver + frontend)
just test               # Solver tests (182 tests)
just test-all           # All tests (solver + frontend + analytics)
just check              # Full quality gate (lint + typecheck + test + bench)
just bench-check        # Performance regression test (PASS/FAIL)

# Code quality
just fmt                # Format all components
just lint-all           # Lint all components
just typecheck          # Type-check solver + frontend

# Precompute + simulate
just precompute         # θ=0 strategy table (~1.1s)
just sweep              # All 37 θ values (resumable)
just simulate           # 1M games lockstep

# Analyze + visualize
just pipeline           # compute → plot → categories → efficiency
just density            # Exact forward-DP PMFs

# Serve + dev (two servers needed)
just dev-backend        # Backend API on port 9000
just dev-frontend       # Vite on port 5173 (serves all UIs + proxies API)
```

## Local Development

Two servers, one URL. The Vite dev server serves all three UIs and proxies the API:

```bash
just dev-backend        # Terminal 1: Rust API on port 9000
just dev-frontend       # Terminal 2: Vite on port 5173
```

| URL | UI |
|-----|-----|
| `http://localhost:5173/yatzy/` | Treatise |
| `http://localhost:5173/yatzy/play/` | Game UI |
| `http://localhost:5173/yatzy/profile/` | Profiler |
| `http://localhost:5173/yatzy/api/health` | API (proxied to backend) |

This mirrors the production URL structure at `langkilde.se/yatzy/*`.

Full recipe list: `just --list`

## Architecture

```
Vite (:5173) ─── /yatzy/play/     → Game UI (frontend/)
             ├── /yatzy/           → Treatise (treatise/)
             ├── /yatzy/profile/   → Profiler (profiler/)
             └── /yatzy/api/       → proxy → Solver (axum, :9000)

Analytics (Python) ──reads──→ data/simulations/theta/*/scores.bin
```

All game intelligence lives in the solver. The three UIs are thin clients. Analytics reads binary simulation files from disk.

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

profiler/data/                     # Pre-computed for static site
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
