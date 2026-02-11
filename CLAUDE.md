# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Delta Yatzy is a web-based implementation of the classic dice game with optimal action suggestions and multiplayer support. The project consists of:

- **Backend**: `backend/` — Rust, axum-based API server with rayon parallelism and memmap2 for zero-copy I/O
- **Frontend**: Vanilla JavaScript single-page application with dynamic UI and Chart.js visualizations
- **Docker**: Containerized deployment for frontend + backend

A legacy C implementation exists in `backend-legacy-c/` for reference only.

## Commands

### Backend Development

```bash
# Build the backend
cd backend && cargo build --release

# Run unit tests
cargo test

# Run precomputed integration tests (requires data/all_states.bin)
cargo test --test test_precomputed

# Precompute state values (required once)
YATZY_BASE_PATH=. target/release/yatzy-precompute

# Run the backend server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy
```

### Frontend Development

```bash
# Run frontend server (from frontend directory)
python3 serve.py  # Serves on port 8090

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

- **Entry Points**: `src/bin/server.rs` (API server), `src/bin/precompute.rs` (offline precomputation)
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
- **Infrastructure**:
  - `types.rs` - YatzyContext, KeepTable, StateValues (owned/mmap)
  - `constants.rs` - Category enum, state_index, NUM_STATES
  - `storage.rs` - Binary file I/O (16-byte header + float[2M], zero-copy mmap via memmap2)

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

## Important Considerations

- Frontend assumes backend is running on port 9000 (configurable via environment or `js/config.js`)
- Game state values are stored in `data/all_states.bin` (~8 MB) inside the backend directory
- Scandinavian Yatzy: 15 categories, 50-point upper bonus (not 35 as in American Yahtzee)
- State representation: `upper_score` (0-63) and `scored_categories` (15-bit bitmask), total 2,097,152 states
- Rayon thread count configurable via `RAYON_NUM_THREADS` env var (default: 8)
