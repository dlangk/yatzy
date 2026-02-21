# Analytics Deep Dive

## Setup

- **Package:** `yatzy-analysis` (Python, `pyproject.toml`)
- **Python:** ≥3.11
- **Package manager:** `uv`
- **CLI entry point:** `yatzy-analyze` → `yatzy_analysis.cli:cli` (Click)
- **Install:** `cd analytics && uv venv && uv pip install -e .`
- **Run:** `analytics/.venv/bin/yatzy-analyze <command>`

## Dependencies

### Production
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26 | Numeric arrays, vectorized I/O |
| scipy | ≥1.11 | KDE, entropy, skewness, kurtosis |
| pandas | ≥2.0 | DataFrame manipulation |
| pyarrow | ≥14.0 | Parquet I/O |
| matplotlib | ≥3.8 | 2D plotting |
| seaborn | ≥0.13 | Statistical visualization |
| plotly | ≥6.0 | Interactive 3D density plots |
| click | ≥8.1 | CLI framework |

### Optional (surrogate models)
| Package | Purpose |
|---------|---------|
| scikit-learn ≥1.3 | Decision trees, MLPs |
| torch ≥2.0 | Neural network surrogates |
| joblib ≥1.3 | Model serialization |

## File Inventory

### Core Modules (14,473 LOC total)

| File | LOC | Purpose |
|------|-----|---------|
| `cli.py` | 1,501 | 40+ Click commands |
| `surrogate.py` | 927 | DT/MLP training, label noise analysis |
| `surrogate_eval.py` | 808 | Full-game simulation with surrogates |
| `skill_ladder.py` | 822 | Human-readable decision rules from regret |
| `theta_estimator.py` | 697 | Multi-start Nelder-Mead MLE for θ |
| `theta_validation.py` | 326 | Parameter recovery (Monte Carlo) |
| `density.py` | 305 | Forward-DP density (Python reimplementation) |
| `compute.py` | 305 | KDE, summary stats, CVaR, MER, SDVA |
| `tail_analysis.py` | 275 | P(374) extreme tail analysis |
| `io.py` | 222 | Binary file readers (3 formats) |
| `scorecard.py` | 218 | Scorecard rendering |
| `feature_engineering.py` | 214 | Feature extraction for surrogates |
| `config.py` | 126 | Paths, grids, constants |
| `adaptive.py` | 91 | Adaptive policy analysis |
| `store.py` | 56 | Parquet save/load |

### Plotting Modules (25 files, ~6,000 LOC)

| File | LOC | Purpose |
|------|-----|---------|
| `plots/categories.py` | 604 | Per-category statistics across θ |
| `plots/percentile_sweep.py` | 605 | Adaptive θ* per percentile |
| `plots/surrogate.py` | 552 | Surrogate Pareto frontier, error analysis |
| `plots/state_flow.py` | 523 | Alluvial, DAG, streamgraph plots |
| `plots/scenario_cards.py` | 510 | Profiling quiz scenario visualization |
| `plots/density.py` | 435 | 2D/3D density heatmaps, interactive HTML |
| `plots/percentile_frontiers.py` | 411 | Multipoint Pareto frontiers |
| `plots/modality.py` | 361 | Distribution decomposition by category |
| `plots/efficiency.py` | 341 | MER, SDVA, CVaR metrics |
| `plots/difficulty_cards.py` | 335 | Scenario cards (dice, board, decisions) |
| `plots/yatzy_hypothesis.py` | 327 | Conditional Yatzy hit-rate tests |
| `plots/sensitivity.py` | 308 | Decision sensitivity analysis |
| `plots/winrate.py` | 265 | Head-to-head win rates |
| `plots/compression.py` | 253 | Policy compression gap distributions |
| `plots/multiplayer.py` | 220 | 2-player game analysis |
| `plots/style.py` | 145 | Custom coolwarm colormap, theme |
| `plots/cdf.py` | 122 | CDF plots (full + tail zoom) |
| `plots/difficult_sensitivity_cards.py` | 113 | Sensitivity across θ |
| `plots/percentiles.py` | 99 | Percentile heatmaps |
| `plots/spec.py` | 97 | PlotSpec dataclass |
| `plots/combined.py` | 66 | Multi-panel dashboard |
| `plots/quantile.py` | 52 | Quantile curves vs θ |
| `plots/mean_std.py` | 49 | Efficient frontier |
| `plots/frontier.py` | 249 | State-dependent vs fixed-θ |

### Profiling Subpackage (3 files)

| File | Purpose |
|------|---------|
| `profiling/synthetic.py` | Synthetic human generators (θ, β, γ, d archetypes) |
| `profiling/validation.py` | Parameter recovery Monte Carlo |
| `profiling/estimator.py` | Nelder-Mead convergence analysis |

## Input Data Sources and Formats

### Binary Simulation Files (read by `io.py`)

| Format | Magic | Record Size | Content |
|--------|-------|-------------|---------|
| `scores.bin` (compact) | `0x59545353` | 2 bytes (i16) | Final scores only |
| `simulation_raw.bin` (full) | `0x59545352` | 289 bytes | 15 turns × 19B + metadata |
| `multiplayer_raw.bin` | `0x594C504D` | 64 bytes | 2-player scores + turn totals |

**Read functions (numpy vectorized):**
```python
read_scores(path) → NDArray[int32]           # Auto-detect format, return sorted
read_full_recording(path) → dict             # category_scores[N,15], bonus, upper
read_multiplayer_recording(path) → dict      # 2-player data
```

### Other Inputs

| Source | Format | Used By |
|--------|--------|---------|
| `configs/theta_grid.toml` | TOML | config.py (theta grids) |
| `outputs/scenarios/*.json` | JSON | scenario_cards, difficulty_cards |
| `outputs/surrogate/models/` | pickle/joblib | surrogate_eval |
| `data/surrogate/*.bin` | Binary | surrogate training |

## Every Analysis / Transformation

### Core Statistical Pipeline (`compute.py`)

**37-column summary per theta:**
- Percentiles: p1, p5, p10, p25, p50, p75, p90, p95, p99, p995, p999, p9999
- Moments: mean, std, min, max
- Extremes: bot5_avg, top5_avg, top5_pct_avg
- Risk: cvar_1, cvar_5, cvar_10 (Conditional Value at Risk / Expected Shortfall)
- Shape: skewness, kurtosis, iqr
- Tail: es_gain_1, es_gain_5, tail_ratio_95_5, tail_ratio_99_1
- Robust: trimmed_mean_5, winsorized_mean_5
- Information: entropy

**KDE (Kernel Density Estimation):**
- Method: `scipy.stats.gaussian_kde` with Silverman bandwidth
- Points: 1,000 evaluation points over [50, 374]
- Subsample: 100K games (speed vs accuracy)
- Output: density + CDF + survival per theta

**Derived Metrics:**
- MER (Marginal Exchange Rate): cost in mean per unit quantile gain
- SDVA (Stochastic Dominance Violation Area): CDF difference integral
- CVaR: mean of worst α% scores

### Surrogate Model Pipeline (`surrogate.py`, `surrogate_eval.py`)

1. **Label noise analysis** — conflicting vectors (same state, different optimal actions)
2. **Data scaling** — DT accuracy vs training set size
3. **Feature ablation** — drop-one-feature importance
4. **Forward selection** — greedy feature addition
5. **Model training** — DT (d5, d10, d20) + MLP (32, 64, 128)
6. **Full-game simulation** — play 10K games with each model, measure EV loss

### Cognitive Profiling (`theta_estimator.py`)

- **Model:** P(action | scenario, θ, β, γ, d) = softmax(β · Q(scenario, action | θ, γ, d))
- **Estimation:** Multi-start Nelder-Mead MLE (tol=1e-6, max 500 iters)
- **Validation:** Monte Carlo parameter recovery (100+ trials)

### Skill Ladder (`skill_ladder.py`)

- **Input:** Regret export from solver (yatzy-regret-export)
- **Process:** Parse CFG → cluster by decision type → sort by regret → consolidate
- **Output:** JSON + Markdown ranked decision rules

## CLI Command Inventory

### Stage 1: Core Pipeline
| Command | Input | Output |
|---------|-------|--------|
| `compute` | scores.bin or density JSON | summary.parquet, kde.parquet |
| `compute --csv` | scores.bin | + sweep_summary.csv (34 columns) |
| `plot` | parquet aggregates | CDF, density, percentiles, mean-std PNGs |
| `efficiency` | summary, kde | efficiency.png + MER/SDVA/CVaR |
| `categories` | category_stats.csv | 15 category plots |
| `run` | any | Full pipeline: compute → plot → efficiency |

### Stage 2: Specialized Analysis
| Command | Purpose |
|---------|---------|
| `adaptive` | Adaptive vs fixed-θ policies |
| `multiplayer` | 2-player matchup analysis |
| `winrate` | Head-to-head strategy comparison |
| `sensitivity` | Decision sensitivity across θ |
| `state-flow` | State frequency alluvial/DAG/stream |
| `modality` | Score distribution decomposition |
| `compression` | Policy compression gap analysis |
| `tail` | P(374) extreme tail analysis |
| `summary` | Print summary table to console |

### Stage 3: Profiling
| Command | Purpose |
|---------|---------|
| `theta-questionnaire` | Interactive CLI risk estimation |
| `theta-replay` | Replay saved answers |
| `theta-validate` | Validate estimator (synthetic humans) |
| `profile-validate` | 4-parameter recovery Monte Carlo |

### Stage 4: Surrogates
| Command | Purpose |
|---------|---------|
| `surrogate-diagnose` | Label noise + error distribution |
| `surrogate-scaling` | Data scaling experiment |
| `surrogate-features` | Feature ablation/forward selection |
| `surrogate-train` | Train DT/MLP models |
| `surrogate-plot` | Pareto frontier plots |
| `surrogate-eval` | Full-game surrogate simulation |

### Stage 5: Scenarios
| Command | Purpose |
|---------|---------|
| `difficult-cards` | Scenario cards for hard decisions |
| `difficult-sensitivity-cards` | Cross-θ sensitivity |
| `scenario-cards` | Quiz scenario rendering |
| `skill-ladder` | Human-readable rules |

## Output Formats and Destinations

### Parquet (intermediate)
| File | Columns | Rows |
|------|---------|------|
| `summary.parquet` | 37 stats columns + theta | 1 per θ (~37 rows) |
| `kde.parquet` | theta, score, density, cdf, survival | 37,000 (37θ × 1000 pts) |
| `mer.parquet` | Marginal exchange rates | ~37 rows |
| `sdva.parquet` | Dominance violation areas | ~37 rows |

### CSV (export)
| File | Purpose |
|------|---------|
| `sweep_summary.csv` | 34-column summary (human-readable) |
| `category_stats.csv` | Per-category statistics |
| `yatzy_conditional.csv` | Conditional Yatzy hit rates |
| `percentile_sweep.csv` | Adaptive θ* per percentile |
| `percentile_peaks.csv` | Optimal θ for each percentile |

### Plots (~50+ files in `outputs/plots/`)
All PNG at 200 DPI. Some HTML (interactive 3D density).

## Reusable Tools vs One-Off Scripts

### Reusable (in `analytics/src/yatzy_analysis/`)
- `io.py` — universal binary reader for all solver output formats
- `compute.py` — statistical kernel (KDE, CVaR, MER, SDVA)
- `config.py` — single source of truth for paths and constants
- `store.py` — parquet save/load
- `plots/style.py` — consistent visual theme
- `cli.py` — composable pipeline commands

### One-Off Scripts (in `scripts/`)
| Script | LOC | Purpose |
|--------|-----|---------|
| `cfg_sweep.py` | 2,471 | Configuration sweep (θ × games) |
| `scaling_sweep.py` | 3,778 | Scaling law experiments |
| `generate_strategy_guide.py` | 11,395 | LLM-based playbook generation |

These scripts are standalone (not part of the analytics package).

## Dependency on Backend API

The analytics package does **not** call the solver API at runtime. All data flows through binary files on disk:

```
Solver binaries → data/simulations/*.bin → analytics reads → outputs/
```

The only indirect coupling is through binary file format specifications (magic numbers, header sizes, record layouts) defined in both Rust (`storage.rs`, `raw_storage.rs`) and Python (`io.py`, `config.py`).
