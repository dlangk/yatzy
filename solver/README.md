# Yatzy Solver

Rust HPC engine: backward-induction DP, Monte Carlo simulation, REST API.

See [`CLAUDE.md`](CLAUDE.md) for the complete reference: architecture, API endpoints, data structures, hot paths, and performance baselines.

## Quick Start

```bash
cargo build --release

# Precompute state values (required once, ~1.1s)
YATZY_BASE_PATH=. RAYON_NUM_THREADS=8 target/release/yatzy-precompute

# Start API server (port 9000)
YATZY_BASE_PATH=. target/release/yatzy
```

## Tests

```bash
cargo test             # 182 tests
cargo fmt --check      # Formatting
cargo clippy           # Lints
```
