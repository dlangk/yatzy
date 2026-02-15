# Yatzy project orchestration
# Run `just --list` to see all recipes.

default:
    @just --list

# ── Setup ─────────────────────────────────────────────────────────────────

# Build solver and install analytics package
setup:
    cd solver && cargo build --release
    cd analytics && uv venv && uv pip install -e .

# ── Stage 1: Compute (expensive) ─────────────────────────────────────────

# Precompute state values for θ=0 (~2.3s)
precompute:
    YATZY_BASE_PATH=. solver/target/release/yatzy-precompute

# Precompute state values for a specific θ (~7s)
precompute-theta theta:
    YATZY_BASE_PATH=. solver/target/release/yatzy-precompute --theta {{theta}}

# Simulate games (default: 1M)
simulate games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-simulate --games {{games}} --output data/simulations

# Simulate games for a specific θ
simulate-theta theta games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-simulate --theta {{theta}} --games {{games}} --output data/simulations/theta/theta_{{theta}}

# Per-category statistics across all θ values
category-sweep games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-category-sweep --games {{games}} --output outputs/aggregates/csv/category_stats.csv

# Conditional Yatzy hit-rate analysis
yatzy-conditional games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-conditional --games {{games}} --output outputs/aggregates/csv/yatzy_conditional.csv

# Generate pivotal scenarios for θ questionnaire
pivotal-scenarios games="100000":
    YATZY_BASE_PATH=. solver/target/release/pivotal-scenarios --games {{games}} --output outputs/scenarios/pivotal_scenarios.json

# Decision sensitivity analysis across θ ∈ [0, 0.2]
decision-sensitivity games="100000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-decision-sensitivity --games {{games}} --output outputs/scenarios

# Multiplayer simulation (default: 2 ev players, 100K games)
multiplayer *args:
    YATZY_BASE_PATH=. solver/target/release/yatzy-multiplayer {{args}}

# Multiplayer simulation with per-game recording
multiplayer-record *args:
    YATZY_BASE_PATH=. solver/target/release/yatzy-multiplayer --record {{args}}

# 1v1 ev vs ev baseline (1M games with recording)
multiplayer-ev-baseline games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-multiplayer \
        --record --strategy ev --strategy ev \
        --games {{games}} --output data/simulations/multiplayer/ev_vs_ev

# Underdog parameter sweep
underdog-sweep games="100000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-underdog-sweep --games {{games}}

# Quick underdog vs EV matchup
underdog-vs-ev games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-multiplayer \
        --strategy ev --strategy mp:underdog --games {{games}}

# Test state-dependent θ vs constant-θ Pareto frontier
frontier-test games="1000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-frontier-test --games {{games}} --output outputs/frontier

# Head-to-head win rate: constant-θ vs θ=0
winrate games="10000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-winrate --games {{games}} --output outputs/winrate

# Adaptive percentile sweep: find θ* for each percentile
percentile-sweep coarse="1000000" fine="10000000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-percentile-sweep --games-coarse {{coarse}} --games-fine {{fine}} --output outputs/aggregates/csv

# Human baseline: heuristic vs EV-optimal comparison
human-baseline games="100000":
    YATZY_BASE_PATH=. solver/target/release/yatzy-human-baseline --games {{games}} --output outputs/human_baseline

# Collect difficult scenarios for human skill estimation
difficult-scenarios games="1000000" top="200":
    YATZY_BASE_PATH=. solver/target/release/yatzy-difficult-scenarios --games {{games}} --top {{top}} --output outputs/scenarios

# Re-evaluate difficult scenarios across all θ strategy tables
scenario-sensitivity:
    YATZY_BASE_PATH=. solver/target/release/yatzy-scenario-sensitivity --output outputs/scenarios

# Analyze multiplayer results → plots
multiplayer-analyze *args:
    analytics/.venv/bin/yatzy-analyze multiplayer {{args}}

# ── Stage 2–3: Aggregate + Visualize (cheap) ─────────────────────────────

# Compute summary stats + KDE (reads scores.bin directly)
compute:
    analytics/.venv/bin/yatzy-analyze compute

# Generate plots
plot:
    analytics/.venv/bin/yatzy-analyze plot

# Plot per-category statistics
categories:
    analytics/.venv/bin/yatzy-analyze categories

# Compute efficiency metrics (MER, SDVA, CVaR)
efficiency:
    analytics/.venv/bin/yatzy-analyze efficiency

# Full analytics pipeline: compute → plot → categories → efficiency
pipeline: compute plot categories efficiency

# ── Dev ───────────────────────────────────────────────────────────────────

# Run solver tests
test:
    cd solver && cargo test

# Check formatting and lints
lint:
    cd solver && cargo fmt --check && cargo clippy

# Start API server on port 9000
serve:
    YATZY_BASE_PATH=. solver/target/release/yatzy

# Print summary table
summary:
    analytics/.venv/bin/yatzy-analyze summary

# Start frontend dev server
frontend:
    python3 frontend/serve.py
