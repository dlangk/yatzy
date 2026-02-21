# Output Map

## Simulation / Sweep Output Files

### Strategy Tables (`data/strategy_tables/`)

| File | Size | Producer | Format |
|------|------|----------|--------|
| `all_states.bin` | 16 MB | `yatzy-precompute` | Header(16B) + f32[4,194,304] |
| `all_states_theta_*.bin` | 16 MB each | `yatzy-precompute --theta T` | Same format, θ in header |
| `all_states_max_policy.bin` | 16 MB | `yatzy-precompute --max-policy` | Same format |
| `oracle.bin` | 3.17 GB | `yatzy-precompute --oracle` | Header(16B) + u8[1.05B] × 3 |

**Naming convention:** `all_states_theta_{theta_value:.3}.bin` (e.g., `all_states_theta_-0.030.bin`)

**Theta values:** 37 from configs/theta_grid.toml: [-3.0, ..., -0.005, 0.0, 0.005, ..., 3.0]

### Simulation Scores (`data/simulations/`)

| Path | Size | Producer | Format |
|------|------|----------|--------|
| `theta/theta_{T}/scores.bin` | ~2 MB/1M games | `yatzy-sweep` or `yatzy-simulate` | Header(32B) + i16[N] |
| `theta/theta_0/simulation_raw.bin` | ~276 MB/1M | `yatzy-simulate --full-recording` | Header(32B) + GameRecord(289B)×N |
| `max_policy/scores.bin` | ~2 MB | `yatzy-simulate --max-policy` | Same compact format |
| `multiplayer/ev_vs_ev/multiplayer_raw.bin` | ~61 MB/1M | `yatzy-multiplayer` | Header(32B) + MultiplayerRecord(64B)×N |
| `adaptive/bonus-adaptive/scores.bin` | ~2 MB | `yatzy-simulate --adaptive` | Compact format |
| `adaptive/phase-based/scores.bin` | ~2 MB | `yatzy-simulate --adaptive` | Compact format |

### Surrogate Training Data (`data/surrogate/`)

| File | Producer | Content |
|------|----------|---------|
| `category_decisions.bin` | `yatzy-export-training-data` | (features, label) pairs for category decisions |
| `reroll1_decisions.bin` | `yatzy-export-training-data` | (features, label) pairs for 1-reroll keep decisions |
| `reroll2_decisions.bin` | `yatzy-export-training-data` | (features, label) pairs for 2-reroll keep decisions |

## Analytics Outputs (`outputs/`)

### Aggregated Data (`outputs/aggregates/`)

#### Parquet (`outputs/aggregates/parquet/`)

| File | Columns | Rows | Producer |
|------|---------|------|----------|
| `summary.parquet` | theta + 37 stat columns | ~37 (one per θ) | `yatzy-analyze compute` |
| `kde.parquet` | theta, score, density, cdf, survival | ~37,000 | `yatzy-analyze compute` |
| `mer.parquet` | Marginal exchange rates | ~37 | `yatzy-analyze efficiency` |
| `sdva.parquet` | Stochastic dominance violation areas | ~37 | `yatzy-analyze efficiency` |
| `scores.parquet` | theta, score (legacy) | varies | `yatzy-analyze extract` |

#### CSV (`outputs/aggregates/csv/`)

| File | Columns | Producer |
|------|---------|----------|
| `sweep_summary.csv` | 34 summary columns | `yatzy-analyze compute --csv` |
| `category_stats.csv` | Per-category means/stds | `yatzy-category-sweep` |
| `yatzy_conditional.csv` | Conditional Yatzy hit rates | `yatzy-conditional` |
| `percentile_sweep.csv` | Adaptive θ* per percentile | `yatzy-percentile-sweep` |
| `percentile_peaks.csv` | Optimal θ per percentile | `yatzy-percentile-sweep` |

### Plots (`outputs/plots/`)

Flat directory, ~50+ files. All PNG at 200 DPI unless noted.

| File Pattern | Producer | Content |
|-------------|----------|---------|
| `cdf_full.png` | `yatzy-analyze plot` | Full CDF curves |
| `tails_zoomed.png` | `yatzy-analyze plot` | Tail CDF zoom |
| `density.png` | `yatzy-analyze plot` | 2D density heatmap |
| `density_3d.png` | `yatzy-analyze plot` | 3D density surface |
| `density_3d_interactive.html` | `yatzy-analyze plot` | Plotly interactive 3D |
| `percentiles_vs_theta.png` | `yatzy-analyze plot` | Quantile curves |
| `mean_vs_std.png` | `yatzy-analyze plot` | Efficient frontier |
| `efficiency.png` | `yatzy-analyze efficiency` | MER/SDVA/CVaR panel |
| `category_*.png` | `yatzy-analyze categories` | 15 category subplots |
| `scenarios/*.png` | `yatzy-analyze difficult-cards` | Scenario card renders |
| `sensitivity_*.png` | `yatzy-analyze sensitivity` | Decision sensitivity |
| `state_flow_*.png` | `yatzy-analyze state-flow` | Alluvial/DAG/stream |
| `modality_*.png` | `yatzy-analyze modality` | Mixture decomposition |
| `compression_*.png` | `yatzy-analyze compression` | Policy compression gaps |
| `winrate_*.png` | `yatzy-analyze winrate` | Win rate matrices |
| `multiplayer_*.png` | `yatzy-analyze multiplayer` | 2-player analysis |
| `percentile_sweep_*.png` | `yatzy-analyze percentile-sweep-plots` | Adaptive θ curves |
| `surrogate_*.png` | `yatzy-analyze surrogate-plot` | Pareto frontier |
| `frontier_*.png` | `yatzy-analyze frontier` | State-dependent vs fixed |

### Scenario Data (`outputs/scenarios/`)

| File | Producer | Content |
|------|----------|---------|
| `pivotal_scenarios.json` | `pivotal-scenarios` | 30 critical decision points |
| `answers.json` | `yatzy-analyze theta-questionnaire` | User questionnaire responses |
| `difficult_scenarios.json` | `difficult-scenarios` | High EV-gap decisions |
| `difficult_scenarios_sensitivity.json` | `scenario-sensitivity` | Multi-θ scenario eval |
| `state_frequency.csv` | `yatzy-analyze state-frequency` | Board state concentration |

### Profiling Data (`outputs/profiling/`)

| File | Producer | Content |
|------|----------|---------|
| `scenarios.json` | `yatzy-profile-scenarios` | 30 quiz scenarios + Q-grids (108 combos each) |

### Density Data (`outputs/density/`)

| File | Producer | Content |
|------|----------|---------|
| `density_theta_{T}.json` | `yatzy-density` | Exact PMF: (score, probability) pairs |

~90 files (37 θ × {oracle, non-oracle} variants).

### Surrogate Models (`outputs/surrogate/`)

| Path | Producer | Content |
|------|----------|---------|
| `models/dt_d5_*.pkl` | `yatzy-analyze surrogate-train` | Decision tree (depth 5) |
| `models/dt_d10_*.pkl` | `yatzy-analyze surrogate-train` | Decision tree (depth 10) |
| `models/dt_d20_*.pkl` | `yatzy-analyze surrogate-train` | Decision tree (depth 20) |
| `models/mlp_32_*.pkl` | `yatzy-analyze surrogate-train` | MLP (32 hidden units) |
| `models/mlp_64_*.pkl` | `yatzy-analyze surrogate-train` | MLP (64 hidden units) |
| `models/mlp_128_*.pkl` | `yatzy-analyze surrogate-train` | MLP (128 hidden units) |
| `game_eval_results.csv` | `yatzy-analyze surrogate-eval` | Full-game simulation results |

### Policy Analysis (`outputs/policy_compression/`)

| File | Producer | Content |
|------|----------|---------|
| `*.json` | `yatzy-decision-gaps` | Per-decision gap analysis |

### Rosetta Stone (`outputs/rosetta/`)

| File | Producer | Content |
|------|----------|---------|
| `rules.json` | `yatzy-rosetta` | Distilled human-readable rules |
| `rules.md` | `yatzy-rosetta` | Markdown rule summary |

### Other Outputs

| Path | Producer | Content |
|------|----------|---------|
| `outputs/frontier/` | `yatzy-frontier-test` | State-dependent vs fixed-θ data |
| `outputs/winrate/` | `yatzy-winrate` | Win rate matrices |
| `outputs/heuristic_gap/` | `yatzy-heuristic-gap` | Per-decision heuristic gaps |

## Blog Pre-Computed Data (`blog/data/`)

These files are copied from `outputs/` for the static blog:

| File | Source | Producer |
|------|--------|----------|
| `scenarios.json` | `outputs/profiling/scenarios.json` | `just profile-deploy` |
| `player_card_grid.json` | `yatzy-player-card-grid` direct | `just player-card-grid` |
| `kde_curves.json` | Extracted from kde.parquet | Custom export |
| `sweep_summary.json` | Extracted from summary.parquet | Custom export |
| `category_stats_theta0.json` | Extracted from category_stats.csv | Custom export |
| `game_eval.json` | `yatzy-eval-policy` | Direct output |
| `greedy_vs_optimal.json` | `export-greedy-game` | Direct output |
| `heuristic_gap.json` | `yatzy-heuristic-gap` | Direct output |
| `mixture.json` | `yatzy-analyze modality` | Custom export |
| `state_heatmap.json` | `export-state-heatmap` | Direct output |
| `winrate.json` | `yatzy-winrate` | Direct output |
| `percentile_peaks.json` | Extracted from percentile_peaks.csv | Custom export |

## Data Flow Summary

```
                    Precomputation
                    ─────────────
yatzy-precompute → data/strategy_tables/all_states*.bin (16 MB × 37)
                 → data/strategy_tables/oracle.bin (3.17 GB)

                    Simulation
                    ──────────
yatzy-sweep      → data/simulations/theta/theta_*/scores.bin (2 MB × 37)
yatzy-simulate   → data/simulations/theta/theta_0/simulation_raw.bin (276 MB)
yatzy-multiplayer→ data/simulations/multiplayer/*/multiplayer_raw.bin (61 MB)

                    Analysis Binaries
                    ─────────────────
yatzy-scenarios  → outputs/scenarios/*.json
yatzy-density    → outputs/density/density_*.json
yatzy-profile-*  → outputs/profiling/scenarios.json
yatzy-*-export   → data/surrogate/*.bin

                    Analytics Pipeline
                    ──────────────────
yatzy-analyze    → outputs/aggregates/parquet/*.parquet
                 → outputs/aggregates/csv/*.csv
                 → outputs/plots/*.png
                 → outputs/surrogate/models/*.pkl

                    Blog Deployment
                    ───────────────
just profile-deploy → blog/data/scenarios.json
just player-card-grid → blog/data/player_card_grid.json
Custom exports      → blog/data/*.json
```

## Storage Budget

| Category | Size | Persistence |
|----------|------|-------------|
| Strategy tables | ~600 MB (37θ × 16 MB) | Required (slow to recompute) |
| Oracle | 3.17 GB | Optional (θ=0 only) |
| Simulation scores | ~74 MB (37θ × 2 MB) | Cheap to regenerate |
| Full recordings | ~276 MB (θ=0 only) | Expensive per θ |
| Multiplayer | ~61 MB | Cheap to regenerate |
| Density JSON | ~50 MB | ~3s/θ with oracle |
| Analytics parquet | <10 MB | Seconds to regenerate |
| Plots | ~100 MB | Seconds to regenerate |
| Blog data | <5 MB | Copied from outputs |
| Surrogate models | <50 MB | Minutes to retrain |
| **Total** | **~4.3 GB typical** | |
