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

- [Rust toolchain](https://rustup.rs/) (stable)
- Python 3 (for frontend dev server)

### Build & Run

```bash
# Build backend
cd backend-rust && cargo build --release

# Precompute state values (required once, ~90s)
YATZY_BASE_PATH=. RAYON_NUM_THREADS=8 target/release/yatzy-precompute

# Start backend API server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy

# Start frontend server (port 8090)
cd ../frontend && python3 serve.py
```

Or using Docker:

```bash
docker-compose up --build
# Frontend: http://localhost:8090
# Backend:  http://localhost:9000
```

### Run Tests

```bash
cd backend-rust
cargo test                    # 29 unit + 20 integration = 49 tests
cargo fmt --check             # formatting
cargo clippy                  # lints
```

## Project Structure

### Backend — Rust (`backend-rust/`)

Axum-based API server with rayon parallelism and memmap2 for zero-copy I/O. This is the primary backend implementation.

- **Computation Pipeline**: Phase 0 (lookup tables + keep-multiset transition table) -> Phase 2 (backward induction over 1.4M reachable states)
- **Key Optimization**: Keep-multiset deduplication collapses equivalent reroll masks (avg 16.3 unique keeps vs 31 masks per dice set)
- **Storage**: Storage v3 format, ~8 MB binary file with zero-copy mmap loading

Source modules:

| Module | Role |
|--------|------|
| `phase0_tables.rs` | Phase 0: all lookup tables (scores, keep-multiset table, probabilities, reachability) |
| `state_computation.rs` | Phase 2: DP backward induction loop with rayon parallelism |
| `widget_solver.rs` | SOLVE_WIDGET: computes E(S) for a single state via ping-pong buffers |
| `api_computations.rs` | API wrappers composing widget solver primitives for HTTP handlers |
| `server.rs` | Axum router, 9 HTTP handlers, CORS via tower-http |
| `game_mechanics.rs` | Yatzy scoring rules: s(S, r, c) |
| `dice_mechanics.rs` | Dice operations and probability calculations |
| `types.rs` | YatzyContext, KeepTable, StateValues (owned/mmap) |
| `storage.rs` | Binary file I/O (Storage v3 format, zero-copy mmap via memmap2) |

See `backend/theory/optimal_yahtzee_pseudocode.md` for the algorithm specification that the code implements.

### Frontend (`frontend/`)

Vanilla JavaScript SPA with ES6 modules:

- `js/app.js` — Entry point, initializes game state and UI
- `modules/game/gameState.js` — Player state management and local storage persistence
- `modules/utils/eventHandlers.js` — User interaction handling
- `modules/utils/uiBuilders.js` — Dynamic UI construction
- `modules/utils/refreshers.js` — Periodic UI updates and API polling
- `modules/utils/endpoints.js` — Backend API communication
- `modules/utils/chartConfig.js` — Chart.js histogram configuration

### Legacy C Backend (`backend/`)

The original C implementation is kept for reference. See [`backend/README.md`](backend/README.md). The Rust version produces bit-identical output and is the actively maintained backend.

## API Example

```bash
# Get expected value for a game state
curl "http://localhost:9000/state_value?upper_score=50&scored_categories=3"

# Suggest optimal action
curl -X POST http://localhost:9000/suggest_optimal_action \
  -H "Content-Type: application/json" \
  -d '{"dice":[1,3,3,4,6],"upper_score":10,"scored_categories":5,"rerolls_remaining":2}'
```

## Scandinavian Yatzy Rules

- 15 scoring categories (Ones through Yatzy)
- 50-point upper section bonus (not 35 as in American Yahtzee)
- 3 rolls per turn (initial roll + 2 rerolls)
- Player chooses which dice to keep before each reroll
