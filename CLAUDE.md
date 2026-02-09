# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Delta Yatzy is a web-based implementation of the classic dice game with optimal action suggestions and multiplayer support. The project consists of:

- **Backend (Rust)**: `backend-rust/` — the primary backend. Axum-based API server with rayon parallelism and memmap2 for zero-copy I/O
- **Backend (C, legacy)**: `backend/` — original C implementation, kept for reference only. Not actively maintained. Produces bit-identical output to the Rust version
- **Frontend**: Vanilla JavaScript single-page application with dynamic UI and Chart.js visualizations
- **Docker**: Containerized deployment for frontend + Rust backend

## Commands

### Backend Development (Rust)

```bash
# Build the backend
cd backend-rust && cargo build --release

# Run unit tests
cargo test

# Run precomputed integration tests (requires data/all_states.bin)
YATZY_BASE_PATH=../backend cargo test --test test_precomputed

# Precompute state values (required once)
YATZY_BASE_PATH=. target/release/yatzy-precompute

# Run the backend server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy
```

### Backend Development (C, legacy — for reference only)

```bash
# Build the backend
cmake -S backend -B backend/build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_BUILD_TYPE=Release
make -C backend/build -j8

# Run tests
for t in test_context test_dice_mechanics test_game_mechanics test_phase0 test_storage test_widget; do
  backend/build/$t
done
YATZY_BASE_PATH=backend backend/build/test_precomputed
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

### Backend Structure (Rust — `backend-rust/`)

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
  - `storage.rs` - Binary file I/O (Storage v3 format, zero-copy mmap via memmap2)

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

3. **Parallel Processing**: Rayon `par_iter` for state computation (replaces OpenMP from C version)

4. **Binary Storage**: Storage v3 format — 16-byte header + float[2,097,152] in STATE_INDEX order (~8 MB). Zero-copy mmap loading in <1ms. Binary compatible between C and Rust backends.

## Important Considerations

- Frontend assumes backend is running on port 9000 (configurable via environment or `js/config.js`)
- Game state values are stored in `backend/data/all_states.bin` (~8 MB), symlinked from `backend-rust/data`. Binary format is shared between C and Rust backends
- Scandinavian Yatzy: 15 categories, 50-point upper bonus (not 35 as in American Yahtzee)
- State representation: `upper_score` (0-63) and `scored_categories` (15-bit bitmask), total 2,097,152 states
- Rayon thread count configurable via `RAYON_NUM_THREADS` or `OMP_NUM_THREADS` env vars (default: 8)
