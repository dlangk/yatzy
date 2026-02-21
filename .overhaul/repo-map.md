# Repo Map

## Directory Tree (depth 4)

```
yatzy/
├── solver/                          # Rust — DP solver, API server, simulation
│   ├── Cargo.toml                   # Single crate, 28 binary targets
│   ├── Cargo.lock
│   ├── Dockerfile
│   └── src/
│       ├── lib.rs                   # Library root (re-exports all modules)
│       ├── bin/                     # 28 binaries (server, precompute, simulate, …)
│       │   ├── server.rs            # axum HTTP server (port 9000)
│       │   ├── precompute.rs        # Backward induction DP
│       │   ├── simulate.rs          # Monte Carlo simulation
│       │   ├── sweep.rs             # θ sweep infrastructure
│       │   ├── density.rs           # Exact forward-DP PMF
│       │   ├── scenarios.rs         # Scenario pipeline
│       │   ├── generate_profile_scenarios.rs
│       │   ├── player_card_grid.rs
│       │   ├── export_training_data.rs
│       │   ├── decision_sensitivity.rs
│       │   ├── decision_gaps.rs
│       │   ├── regret_export.rs
│       │   ├── eval_policy.rs
│       │   └── … (14 more analysis binaries)
│       ├── constants.rs             # STATE_STRIDE=128, NUM_STATES, enums
│       ├── types.rs                 # YatzyContext, KeepTable, StateValues
│       ├── game_mechanics.rs        # Yatzy scoring rules
│       ├── dice_mechanics.rs        # Dice operations, probabilities
│       ├── phase0_tables.rs         # Precompute lookup tables
│       ├── widget_solver.rs         # SOLVE_WIDGET: Groups 1→6 DP
│       ├── batched_solver.rs        # Batched SpMM (64 upper scores)
│       ├── state_computation.rs     # Phase 2: backward induction + rayon
│       ├── simd.rs                  # 14 NEON intrinsic kernels + fast exp
│       ├── api_computations.rs      # HTTP handler computation logic
│       ├── server.rs                # axum router, CORS, handlers
│       ├── storage.rs               # Binary I/O, zero-copy mmap
│       ├── simulation/              # MC simulation engine
│       │   ├── engine.rs            # Sequential simulate_game
│       │   ├── lockstep.rs          # Horizontal SIMD lockstep
│       │   ├── strategy.rs          # Adaptive multi-θ policies
│       │   ├── adaptive.rs          # Noisy/softmax policies
│       │   ├── statistics.rs        # Aggregate stats
│       │   ├── raw_storage.rs       # Binary simulation I/O
│       │   ├── sweep.rs             # θ sweep inventory
│       │   ├── multiplayer.rs       # 2-player simulation
│       │   ├── heuristic.rs         # Heuristic baselines
│       │   ├── radix_sort.rs        # O(N) counting sort
│       │   └── fast_prng.rs         # SplitMix64
│       ├── density/                 # Exact density evolution
│       │   ├── forward.rs           # Forward DP (dense Vec<f64>)
│       │   └── transitions.rs       # Per-state transition computation
│       ├── profiling/               # Cognitive profiling quiz
│       │   ├── scenarios.rs         # Build pool, assemble quiz, Q-grid
│       │   ├── qvalues.rs           # Q-value with (θ,γ,d) perturbation
│       │   └── player_card.rs       # Noisy simulation for player card
│       ├── scenarios/               # Scenario analysis pipeline
│       │   ├── select.rs            # Difficulty-based selection
│       │   ├── classify.rs          # Decision type/phase/tension
│       │   ├── collect.rs           # Candidate pool from simulations
│       │   ├── enrich.rs            # θ-sensitivity enrichment
│       │   ├── actions.rs           # Semantic action representation
│       │   ├── types.rs             # DecisionType, GamePhase, etc.
│       │   └── io.rs                # JSON I/O
│       └── rosetta/                 # Policy distillation
│           ├── policy.rs            # Oracle → human-readable rules
│           └── dsl.rs               # Rule DSL
│
├── analytics/                       # Python — analysis, plotting, pipelines
│   ├── pyproject.toml               # yatzy-analysis package (click CLI)
│   └── src/yatzy_analysis/
│       ├── cli.py                   # 40+ Click commands
│       ├── config.py                # Paths, theta grids, constants
│       ├── compute.py               # KDE, CVaR, MER, SDVA stats
│       ├── io.py                    # Binary format readers
│       ├── store.py                 # Parquet save/load
│       ├── density.py               # Forward-DP density (Python)
│       ├── surrogate.py             # DT/MLP training
│       ├── surrogate_eval.py        # Surrogate game simulation
│       ├── skill_ladder.py          # Human-readable rule induction
│       ├── theta_estimator.py       # Multi-start Nelder-Mead MLE
│       ├── theta_validation.py      # Parameter recovery validation
│       ├── tail_analysis.py         # P(374) extreme tail
│       ├── feature_engineering.py   # Surrogate features
│       ├── scorecard.py             # Scorecard rendering
│       ├── adaptive.py              # Adaptive policy analysis
│       ├── plots/                   # 25 plotting modules
│       │   ├── style.py             # Colormap, theme, fonts
│       │   ├── cdf.py, density.py, combined.py, mean_std.py
│       │   ├── quantile.py, percentiles.py, categories.py
│       │   ├── efficiency.py, frontier.py, sensitivity.py
│       │   ├── state_flow.py, difficulty_cards.py
│       │   ├── scenario_cards.py, percentile_sweep.py
│       │   ├── modality.py, compression.py, surrogate.py
│       │   ├── winrate.py, multiplayer.py, spec.py
│       │   └── yatzy_hypothesis.py, percentile_frontiers.py
│       │       difficult_sensitivity_cards.py
│       └── profiling/               # Validation subpackage
│           ├── synthetic.py, validation.py, estimator.py
│
├── frontend/                        # React + TypeScript + Vite
│   ├── package.json                 # React 19, Vite 7, Vitest
│   ├── vite.config.ts
│   ├── tsconfig.app.json
│   ├── index.html                   # SPA entry point
│   └── src/
│       ├── main.tsx                 # React root
│       ├── App.tsx                  # Root component (orchestrator)
│       ├── reducer.ts               # Flux state machine (14 actions)
│       ├── types.ts                 # All TypeScript interfaces
│       ├── api.ts                   # 4 API calls to solver
│       ├── config.ts                # API_BASE_URL resolution
│       ├── constants.ts             # Category names, colors
│       ├── mask.ts                  # Dice sort/mask translation
│       ├── components/
│       │   ├── ActionBar.tsx, DiceBar.tsx, Die.tsx
│       │   ├── DiceLegend.tsx, Scorecard.tsx, ScorecardRow.tsx
│       │   ├── EvalPanel.tsx, TrajectoryChart.tsx
│       │   └── DebugPanel.tsx
│       └── *.test.ts                # Vitest tests
│
├── blog/                            # Static site + interactive quiz
│   ├── index.html                   # Article with 11 D3 charts
│   ├── profile.html                 # 30-scenario profiling quiz
│   ├── play.html                    # Embedded game
│   ├── css/
│   │   ├── style.css, profile.css, game.css, charts.css
│   ├── js/
│   │   ├── yatzy-viz.js            # D3 shared utilities
│   │   ├── data-loader.js          # JSON data fetcher
│   │   ├── theme.js                # Dark mode toggle
│   │   ├── concept-drawer.js       # Concept panel animations
│   │   ├── charts/                  # 11 D3 chart modules
│   │   ├── profile/                 # Quiz system
│   │   │   ├── store.js, estimator.js, render.js, api.js
│   │   │   ├── player-card-data.js
│   │   │   └── components/          # 7 quiz UI components
│   │   └── game/                    # Vanilla JS game (parallel to frontend)
│   │       ├── store.js, reducer.js, render.js, api.js
│   │       ├── mask.js, constants.js
│   │       └── components/          # 8 game UI components
│   ├── data/                        # Pre-computed JSON for static site
│   │   ├── scenarios.json, player_card_grid.json
│   │   ├── kde_curves.json, sweep_summary.json
│   │   └── … (chart data files)
│   └── concepts/                    # Markdown concept definitions
│
├── theory/                          # Strategy documents & lab reports
│   ├── analysis_and_insights.md
│   ├── pseudocode.md
│   ├── performance_optimizations.md
│   ├── risk_parameter_theta.md
│   ├── performance_lab_report.md
│   ├── oracle_lab_report.md
│   ├── density_condensation_lab_report.md
│   └── yatzy_grandmaster_playbook.md
│
├── scripts/                         # One-off experiment scripts
│   ├── cfg_sweep.py                 # Config sweep (θ × games)
│   ├── scaling_sweep.py             # Scaling law experiments
│   └── generate_strategy_guide.py   # LLM-based playbook generator
│
├── configs/
│   └── theta_grid.toml              # Canonical θ grids (37/19/17 values)
│
├── data/                            # Expensive artifacts (.gitignored)
│   ├── strategy_tables/             # ~300 MB: all_states*.bin
│   └── simulations/                 # ~11 GB: scores.bin, simulation_raw.bin
│
├── outputs/                         # Regenerable (.gitignored)
│   ├── aggregates/parquet/          # summary, kde, mer, sdva
│   ├── aggregates/csv/              # sweep_summary, category_stats
│   ├── plots/                       # ~50+ PNG/SVG visualizations
│   ├── scenarios/                   # Decision scenario JSON
│   ├── profiling/                   # Quiz scenarios + Q-grids
│   ├── density/                     # Exact PMFs per θ
│   ├── surrogate/                   # Trained models + eval results
│   └── rosetta/                     # Distilled rules
│
├── justfile                         # 40+ task runner recipes
├── docker-compose.yml               # Frontend + backend containers
├── CLAUDE.md                        # Project instructions
├── OVERHAUL_PLAN.md                 # 10-phase overhaul roadmap
└── readme.md                        # Quick start
```

## Language / Framework Breakdown

| Component | Language | Framework | LOC |
|-----------|----------|-----------|-----|
| Solver | Rust | axum, rayon, memmap2, serde | ~37,700 |
| Analytics | Python 3.11+ | click, numpy, scipy, pandas, matplotlib, plotly | ~14,500 |
| Frontend | TypeScript | React 19, Vite 7, Vitest | ~2,100 |
| Blog JS | JavaScript (ES6) | Vanilla + D3.js v7 | ~5,600 |
| Blog CSS | CSS | Custom properties, flexbox, grid | ~2,000 |
| Blog HTML | HTML | Static pages | ~3 files |
| Scripts | Python | Standalone | ~17,600 |
| Theory | Markdown | — | ~8 files |
| **Total** | | | **~62,000+** |

## Build Systems and Toolchains

| Component | Build | Package Manager | Lock File |
|-----------|-------|-----------------|-----------|
| Solver | `cargo build --release` | Cargo | `solver/Cargo.lock` |
| Analytics | `uv pip install -e .` | uv | — (no lock) |
| Frontend | `vite build` | npm | `frontend/package-lock.json` |
| Blog | Static (no build) | npm (vitest only) | `blog/package-lock.json` |
| Orchestration | `just` | — | `justfile` |
| Containers | `docker-compose` | — | `docker-compose.yml` |

## Entry Points

| Binary / Command | Entry File | Purpose |
|------------------|-----------|---------|
| `yatzy` | `solver/src/bin/server.rs` | HTTP API server (port 9000) |
| `yatzy-precompute` | `solver/src/bin/precompute.rs` | Backward induction DP |
| `yatzy-simulate` | `solver/src/bin/simulate.rs` | Monte Carlo simulation |
| `yatzy-sweep` | `solver/src/bin/sweep.rs` | θ sweep (resumable) |
| `yatzy-density` | `solver/src/bin/density.rs` | Exact forward-DP PMF |
| `yatzy-analyze` | `analytics/src/yatzy_analysis/cli.py` | Python CLI (40+ commands) |
| Frontend dev | `frontend/src/main.tsx` | React SPA |
| Blog article | `blog/index.html` | Static article |
| Blog quiz | `blog/profile.html` | Profiling quiz |
| Blog game | `blog/play.html` | Embedded game |

## Inter-Component Communication

```
┌─────────────┐     HTTP (port 9000)     ┌──────────────┐
│  Frontend    │ ──────────────────────→  │  Solver API  │
│  (React)     │  POST /evaluate         │  (axum)      │
│  port 5173   │  GET  /state_value      │              │
│              │  POST /density           │              │
│              │  GET  /health            │              │
└─────────────┘                          └──────────────┘
                                               │
┌─────────────┐     HTTP (port 9000)           │
│  Blog Game  │ ──────────────────────→        │
│  (Vanilla)  │  Same 4 endpoints              │
└─────────────┘                                │
                                               │
┌─────────────┐     Pre-computed JSON          │
│  Blog Quiz  │ ←── blog/data/scenarios.json   │ (no runtime API)
│  (Vanilla)  │     blog/data/player_card.json │
└─────────────┘                                │
                                               │
┌─────────────┐     Binary files (disk)        │
│  Analytics  │ ←── data/simulations/*.bin  ←───┘ (solver writes)
│  (Python)   │     data/strategy_tables/*.bin
│             │ ──→ outputs/**/*  (plots, parquet, CSV)
└─────────────┘
```

## Dependency Manifests

| File | Contents |
|------|----------|
| `solver/Cargo.toml` | axum, tokio, rayon, serde, memmap2, rand, toml, tower-http |
| `analytics/pyproject.toml` | numpy, scipy, pandas, pyarrow, matplotlib, seaborn, plotly, click |
| `frontend/package.json` | react, react-dom, typescript, vite, vitest, eslint |
| `blog/package.json` | vitest (testing only; D3 loaded via CDN) |
