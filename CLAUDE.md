# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Delta Yatzy is a web-based implementation of the classic dice game with optimal action suggestions and multiplayer support. The project consists of:

- **Backend**: `backend/` — Rust, axum-based API server with rayon parallelism and memmap2 for zero-copy I/O
- **Analytics**: `analytics/` — Python package for simulation data analysis, plotting, and risk-sweep pipelines
- **Frontend**: Vanilla JavaScript single-page application with dynamic UI and Chart.js visualizations
- **Theory**: `theory/` — Analytical insights about Yatzy strategy and score distributions
- **Docker**: Containerized deployment for frontend + backend

Results (simulation binaries, aggregated parquet/csv/json, plots) live inside `analytics/results/`.

A legacy C implementation exists in `backend-legacy-c/` for reference only.

## Commands

### Convention: Run Everything from Repo Root

All commands below assume you are at the repository root (`yatzy/`).

### Backend Development

```bash
# Build the backend
cd backend && cargo build --release

# Run unit tests (from backend/)
cargo test

# Run precomputed integration tests (requires data/all_states.bin)
cargo test --test test_precomputed

# Precompute state values (required once, from repo root)
YATZY_BASE_PATH=backend backend/target/release/yatzy-precompute

# Run the backend server (port 9000, from repo root)
YATZY_BASE_PATH=backend backend/target/release/yatzy

# Simulate games and generate statistics (from repo root)
YATZY_BASE_PATH=backend backend/target/release/yatzy-simulate --games 1000000 --output analytics/results/bin_files
```

### Analytics (Python)

```bash
# Setup (once)
cd analytics && uv venv && uv pip install -e .

# Run analytics pipeline (from repo root)
analytics/.venv/bin/yatzy-analyze run

# Print summary table
analytics/.venv/bin/yatzy-analyze summary

# Run efficiency metrics
analytics/.venv/bin/yatzy-analyze efficiency
```

### Frontend Development

```bash
# Run frontend server (from frontend directory)
python3 frontend/serve.py  # Serves on port 8090

# Or using Docker
docker-compose up
```

### Docker Development

```bash
# Build and run both services
docker-compose up --build

# Access services
# Frontend: http://localhost:8090
# Backend: http://localhost:9000
```

## Architecture

### Backend Structure (`backend/`)

The backend is a multithreaded Rust application with precomputed game states:

- **Entry Points**: `src/bin/server.rs` (API server), `src/bin/precompute.rs` (offline precomputation), `src/bin/simulate.rs` (simulation + statistics)
- **Phase 0 — Precompute lookup tables**:
  - `phase0_tables.rs` - Builds all static lookup tables (scores, keep-multiset table, probabilities)
  - `game_mechanics.rs` - Yatzy scoring rules: s(S, r, c)
  - `dice_mechanics.rs` - Dice operations and probability calculations
- **Phase 2 — Backward induction**:
  - `state_computation.rs` - DP orchestrator with Rayon parallelism
  - `widget_solver.rs` - SOLVE_WIDGET — computes E(S) for a single state
- **API layer**:
  - `api_computations.rs` - Wrappers that compose widget solver primitives for HTTP handlers
  - `server.rs` - axum router, 9 HTTP handlers, CORS via tower-http
- **Simulation** (`simulation/`):
  - `engine.rs` - Simulate games with optimal strategy, with/without recording
  - `statistics.rs` - Aggregate statistics from recorded games
  - `raw_storage.rs` - Binary I/O for raw simulation data (mmap)
- **Infrastructure**:
  - `types.rs` - YatzyContext, KeepTable, StateValues (owned/mmap)
  - `constants.rs` - Category enum, state_index, NUM_STATES
  - `storage.rs` - Binary file I/O (16-byte header + float[2M], zero-copy mmap via memmap2)

### Analytics Structure (`analytics/`)

Python package (`yatzy-analysis`) with CLI entry point `yatzy-analyze`:

- **Source**: `analytics/src/yatzy_analysis/`
  - `cli.py` - CLI commands: extract, compute, plot, efficiency, run, summary, tail
  - `compute.py` - Summary stats, KDE, MER, SDVA, CVaR
  - `config.py` - Path resolution, binary format constants, theta grids
  - `io.py` - Binary file reading (simulation_raw.bin)
  - `store.py` - Parquet save/load
  - `plots/` - Plotting modules (cdf, density, efficiency, combined, etc.)

### Results Layout (`analytics/results/`)

```
analytics/results/
  bin_files/                    # Raw simulation binaries
    theta/theta_*/              # Per-theta simulation_raw.bin + game_statistics.json
    max_policy/scores.bin       # Max-policy simulation
  aggregates/                   # Processed data by format
    parquet/                    # kde, summary, scores, mer, sdva
    csv/                        # density_kde, scores_cdf, summary_stats
    json/                       # game_statistics
  plots/                        # Generated PNG visualizations (flat directory)
```

### Frontend Structure

The frontend uses ES6 modules for clean separation of concerns:

- **Entry Point**: `js/app.js` - Initializes game state and UI
- **Module Organization**:
  - `modules/game/gameState.js` - Player state management and persistence
  - `modules/utils/`:
    - `eventHandlers.js` - All user interaction handling
    - `uiBuilders.js` - Dynamic UI construction
    - `refreshers.js` - Periodic UI updates and API polling
    - `endpoints.js` - Backend API communication
    - `chartConfig.js` - Chart.js histogram configuration

### Key Design Patterns

1. **Precomputation**: Backend precomputes all 1.4M reachable game states offline. Runtime lookups are O(1).

2. **Keep-Multiset Dedup**: Multiple reroll masks can produce the same kept-dice multiset. The KeepTable collapses these into 462 unique keeps (avg 16.3 per dice set vs 31 masks), eliminating ~47% of redundant dot products in the DP hot path.

3. **Parallel Processing**: Rayon `par_iter` for state computation with unsafe direct writes (each state index is unique, so no data races).

4. **Binary Storage**: 16-byte header + float[2,097,152] in STATE_INDEX order (~8 MB). Zero-copy mmap loading in <1ms.

## Frontend UI Rules

### No Layout Shifts (Hard Requirement)
The UI must be pixel-stable: no element may change size, move, or cause reflow when state changes. This is a hard requirement on every component.

**Rules:**
1. **Never return `null`** — every component always renders its container at a fixed size. Use placeholders (`?`, `—`), `visibility: hidden`, `opacity`, or `disabled` states instead.
2. **Fixed row/cell heights** — table rows, grid cells, and flex items must use explicit `height` or `minHeight` so content changes (text ↔ button ↔ icon) cannot alter dimensions.
3. **Same element, different content** — when a cell switches between states (e.g. "Score" button → "✓"), keep the same DOM element (e.g. always a `<button>`) and change only its text/style. Never swap between different element types that have different intrinsic sizes.
4. **Fixed column widths** — tables must use `table-layout: fixed` with a `<colgroup>` defining explicit percentage widths for every column. Content changes (numbers appearing/disappearing, text length changes) must never cause columns to resize.

**Current examples:**
- Dice bar: always renders 5 dice; shows `?` at `opacity: 0.3` when idle
- Eval panel: always renders full grid; shows `—` dashes when no data
- Action bar: `minHeight: 40` container across all turn phases
- Scorecard rows: `height: 32` cells, button always rendered (shows "Score" or "✓"), `visibility: hidden` when not actionable

## Theory & Insights

The `theory/` directory contains four documents:

- `theory/pseudocode.md` — Optimal Scandinavian Yatzy algorithm pseudocode
- `theory/performance_optimizations.md` — Optimization history from ~20 min to ~2.3s
- `theory/risk_parameter_theta.md` — Mathematical framework for the risk-sensitive solver
- `theory/analysis_and_insights.md` — Living document of empirical findings and simulation results

**When a conversation produces new statistical or theoretical insights, update `theory/analysis_and_insights.md`.** Review the entire document before editing — restructure sections if needed to maintain a coherent structure rather than just appending.

## Important Considerations

- Frontend assumes backend is running on port 9000 (configurable via environment or `js/config.js`)
- Game state values are stored in `data/all_states.bin` (~8 MB) inside the backend directory
- Scandinavian Yatzy: 15 categories, 50-point upper bonus (not 35 as in American Yahtzee)
- State representation: `upper_score` (0-63) and `scored_categories` (15-bit bitmask), total 2,097,152 states
- Rayon thread count configurable via `RAYON_NUM_THREADS` env var (default: 8)
