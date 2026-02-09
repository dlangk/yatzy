# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Delta Yatzy is a web-based implementation of the classic dice game with optimal action suggestions and multiplayer support. The project consists of:

- **Backend**: C-based API server using libmicrohttpd for game logic, state precomputation, and RESTful endpoints
- **Frontend**: Vanilla JavaScript single-page application with dynamic UI and Chart.js visualizations
- **Docker**: Containerized deployment for both frontend and backend services

## Commands

### Backend Development

```bash
# Build the backend
cmake -S backend -B backend/build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_BUILD_TYPE=Release
make -C backend/build -j8

# Precompute state values (required once, ~81s)
YATZY_BASE_PATH=backend OMP_NUM_THREADS=8 backend/build/yatzy_precompute

# Run the backend server (port 9000)
YATZY_BASE_PATH=backend backend/build/yatzy

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

### Backend Structure

The backend is a multithreaded C application with precomputed game states for optimal performance:

- **Entry Points**: `src/yatzy.c` (API server), `src/precompute.c` (offline precomputation)
- **Phase 0 — Precompute lookup tables** (see `theory/optimal_yahtzee_pseudocode.md`):
  - `phase0_tables.c` - Builds all static lookup tables (scores, keep-multiset table, probabilities)
  - `game_mechanics.c` - Yatzy scoring rules: s(S, r, c)
  - `dice_mechanics.c` - Dice operations and probability calculations
- **Phase 2 — Backward induction**:
  - `state_computation.c` - DP orchestrator: processes states level by level (|C|=15 → 0)
  - `widget_solver.c` - Phase 2a: SOLVE_WIDGET — computes E(S) for a single state
- **API layer**:
  - `api_computations.c` - Wrappers that compose widget solver primitives for HTTP handlers
  - `webserver.c` - HTTP API server (libmicrohttpd + json-c)
- **Infrastructure**:
  - `context.c` - YatzyContext lifecycle and Phase 0 orchestration
  - `storage.c` - Binary file I/O (Storage v3 format, zero-copy mmap)
  - `utilities.c` - Logging and environment helpers

- **API Design**: RESTful endpoints that accept game state parameters and return JSON responses with optimal actions and expected values

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

- **State Management**: Local storage for game persistence, with support for multiple players

### Key Design Patterns

1. **Precomputation**: Backend precomputes all 1.4M reachable game states offline (~81s). Runtime lookups are O(1). Progress is tracked in real-time with ETA calculations.

2. **Keep-Multiset Dedup**: Multiple reroll masks can produce the same kept-dice multiset. The KeepTable collapses these into 462 unique keeps (avg 16.3 per dice set vs 31 masks), eliminating ~47% of redundant dot products in the DP hot path.

3. **Modular Frontend**: ES6 modules with clear separation between game logic, UI, and API communication

4. **Parallel Processing**: Backend uses OpenMP for parallel state computation (8 threads optimal on Apple Silicon)

5. **Binary Storage**: Storage v3 format — 16-byte header + float[2,097,152] in STATE_INDEX order (~8 MB). Zero-copy mmap loading in <1ms.

## Important Considerations

- Backend requires gcc (not clang) for OpenMP support — use `gcc-15` from Homebrew on macOS
- Backend requires libmicrohttpd and json-c libraries (installed via Homebrew on macOS)
- Frontend assumes backend is running on port 9000 (configurable via environment or `js/config.js`)
- Game state values are stored in `backend/data/all_states.bin` (~8 MB)
- Scandinavian Yatzy: 15 categories, 50-point upper bonus (not 35 as in American Yahtzee)
- State representation: `upper_score` (0-63) and `scored_categories` (15-bit bitmask), total 2,097,152 states
- IDE diagnostics may show `-fopenmp` errors — these are false positives from clang; the real build uses gcc
- OMP_NUM_THREADS=8 is optimal; 10 threads hits efficiency cores and hurts performance