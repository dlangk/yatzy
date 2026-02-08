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
# Build the backend (from backend directory)
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc-15  # Use gcc from Homebrew on macOS
make

# Run the backend server (from build directory)
./yatzy  # Runs on port 9000
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
  - `phase0_tables.c` - Builds all static lookup tables (scores, probabilities, transitions)
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
  - `storage.c` - All binary file I/O (per-level and consolidated, including mmap)
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

1. **Precomputation**: Backend precomputes all possible game states on startup for O(1) lookups during gameplay. Progress is tracked in real-time with ETA calculations.

2. **Modular Frontend**: ES6 modules with clear separation between game logic, UI, and API communication

3. **Parallel Processing**: Backend uses OpenMP for parallel state computation

4. **Binary Storage**: Precomputed states stored in binary files for fast loading. Supports both per-level files (`states_*.bin`) and consolidated file (`all_states.bin`) with memory-mapped I/O.

## Important Considerations

- Backend requires libmicrohttpd and json-c libraries (installed via Homebrew on macOS)
- Frontend assumes backend is running on port 9000 (configurable via environment or `js/config.js`)
- Game state values are stored in `backend/data/` directory
- The state computation system uses dynamic programming with backward induction, computing values from game end (level 15) to start (level 0)
- State representation: Each state is identified by `upper_score` (0-63) and `scored_categories` (bitmask)
- Error handling includes user-friendly notifications in frontend
- Modular backend architecture with separate HTTP utilities and API handlers