# Delta Yatzy

A web-based implementation of Scandinavian Yatzy with optimal action suggestions and multiplayer support. Uses dynamic programming to precompute expected values for all 2M+ game states, then serves real-time strategy advice via a REST API.

## Features

- **Optimal Play Advisor**: Shows the mathematically best action for any game situation
- **Score Evaluation**: Compares player choices against optimal strategy
- **Outcome Distributions**: Chart.js histograms showing probability distributions after rerolls
- **Multiplayer Support**: Tracks individual player states and scores
- **Fast**: Precomputed states load in <1ms via memory-mapped I/O; API responses are instant

## Quick Start

### Prerequisites

```bash
# macOS
brew install cmake libmicrohttpd json-c gcc

# Ubuntu
sudo apt install cmake libmicrohttpd-dev libjson-c-dev gcc build-essential -y
```

### Build & Run

```bash
# Build backend
cmake -S backend -B backend/build -DCMAKE_C_COMPILER=gcc-15 -DCMAKE_BUILD_TYPE=Release
make -C backend/build -j8

# Precompute state values (required once, ~81s)
YATZY_BASE_PATH=backend OMP_NUM_THREADS=8 backend/build/yatzy_precompute

# Start backend API server (port 9000)
YATZY_BASE_PATH=backend backend/build/yatzy

# Start frontend server (port 8090)
cd frontend && python3 serve.py
```

Or using Docker:

```bash
docker-compose up --build
# Frontend: http://localhost:8090
# Backend:  http://localhost:9000
```

## Project Structure

### Backend (`backend/`)

C-based API server and DP solver. See [`backend/README.md`](backend/README.md) for full details.

- **Computation Pipeline**: Phase 0 (lookup tables + keep-multiset transition table) -> Phase 2 (backward induction over 1.4M reachable states)
- **Key Optimization**: Keep-multiset deduplication collapses equivalent reroll masks (avg 16.3 unique keeps vs 31 masks per dice set)
- **Storage**: Storage v3 format, ~8 MB binary file with zero-copy mmap loading

Source modules:
- `phase0_tables.c` — Lookup tables (scores, keep-multiset table, probabilities, reachability)
- `state_computation.c` — DP backward induction loop with OpenMP parallelism
- `widget_solver.c` — SOLVE_WIDGET: computes E(S) for a single state
- `api_computations.c` — API wrappers for HTTP handlers
- `webserver.c` — REST API server (libmicrohttpd + json-c)
- `game_mechanics.c` / `dice_mechanics.c` — Yatzy rules and dice operations

### Frontend (`frontend/`)

Vanilla JavaScript SPA with ES6 modules:

- `js/app.js` — Entry point, initializes game state and UI
- `modules/game/gameState.js` — Player state management and local storage persistence
- `modules/utils/eventHandlers.js` — User interaction handling
- `modules/utils/uiBuilders.js` — Dynamic UI construction
- `modules/utils/refreshers.js` — Periodic UI updates and API polling
- `modules/utils/endpoints.js` — Backend API communication
- `modules/utils/chartConfig.js` — Chart.js histogram configuration

## API Example

```bash
# Get expected value for a game state
curl "http://localhost:9000/state_value?upper_score=50&scored_categories=3"

# Get optimal reroll strategy
curl "http://localhost:9000/best_reroll?upper_score=10&scored_categories=5&dice=1,3,3,4,6&rerolls=2"
```

## Scandinavian Yatzy Rules

- 15 scoring categories (Ones through Yatzy)
- 50-point upper section bonus (not 35 as in American Yahtzee)
- 3 rolls per turn (initial roll + 2 rerolls)
- Player chooses which dice to keep before each reroll
