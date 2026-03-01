# CLAUDE.md — Analytics

Python analysis package for Yatzy simulation data.

## Setup

```bash
cd analytics && uv venv && uv pip install -e ".[test]"
analytics/.venv/bin/yatzy-analyze --help
```

## Stack

Python 3.11+ with polars, numpy, scipy, matplotlib, seaborn, pyarrow, click.
Polars is the primary DataFrame library. Use `.to_pandas()` only at the seaborn/matplotlib plot boundary.
Optional: scikit-learn, joblib (surrogate models).

## CLI Commands

Entry point: `analytics/.venv/bin/yatzy-analyze <command>`

| Command | Purpose |
|---------|---------|
| `extract` | [Legacy] Read raw binaries to scores.parquet |
| `compute [--csv] [--source density]` | Summary stats (34 columns), KDE, from scores.bin or density JSON |
| `plot [--subset]` | Generate all standard plots (CDF, density, quantile, mean-std) |
| `efficiency` | MER, SDVA, CVaR metrics |
| `categories` | Per-category statistics across θ |
| `yatzy-hypothesis` | Yatzy conditional hit-rate hypothesis tests |
| `run` | Full pipeline: compute, plot, efficiency |
| `tail` | Tail distribution analysis for max-policy simulations |
| `adaptive` | Analyze adaptive θ policy simulations vs fixed-θ frontier |
| `theta-questionnaire` | Interactive questionnaire to estimate risk preference (θ) |
| `theta-replay` | Replay saved answers and print θ estimation results |
| `theta-validate` | Validate θ estimator convergence with synthetic humans |
| `sensitivity` | Plot decision sensitivity analysis (flip rates, gaps) |
| `state-flow` | State-flow visualizations (alluvial, DAG, streamgraph) |
| `state-frequency` | Board-state frequency distribution analysis |
| `multiplayer` | Analyze multiplayer simulation results |
| `winrate` | Win rate analysis plots (θ vs win rate, PMF overlay) |
| `summary` | Print sweep summary table to console |
| `percentile-sweep-plots` | Percentile sweep curves, heatmap, cost-benefit |
| `difficult-sensitivity-cards` | Sensitivity cards for difficult scenarios across θ |
| `modality` | Score distribution mixture decomposition |
| `compression` | Plot policy compression gaps |
| `surrogate-diagnose` | Label noise + error distribution |
| `surrogate-scaling` | Data scaling experiments |
| `surrogate-features` | Feature ablation / forward selection |
| `surrogate-train` | Train DT + MLP surrogate models |
| `surrogate-plot` | Plot surrogate Pareto frontier |
| `surrogate-eval [--games N]` | Full-game simulation with surrogates |
| `surrogate-eval-plot` | Plot game-level surrogate results (params vs mean score) |
| `difficult-cards` | Generate scenario cards for difficult scenarios |
| `profile-validate [--trials N]` | Monte Carlo parameter recovery validation |
| `skill-ladder` | Induce human-readable rules from regret data |

## Custom Colormap

All theta-indexed visualizations use the custom diverging colormap from `plots/style.py`:

```python
from yatzy_analysis.plots.style import CMAP, theta_color, make_norm, save_fig

# CMAP: ["#3b4cc0", "#8db0fe", "#F37021", "#f4987a", "#b40426"]
# Center = #F37021 (orange). Blue = risk-averse (θ<0), Red = risk-seeking (θ>0).
```

## Source Layout

| File | Purpose |
|------|---------|
| `cli.py` | 32 Click commands, main entry point |
| `compute.py` | KDE, summary stats, CVaR, MER, SDVA |
| `config.py` | Path resolution, binary format constants, theta grids |
| `io.py` | Binary file readers (scores.bin, simulation_raw.bin) |
| `store.py` | Parquet save/load |
| `adaptive.py` | Adaptive policy discovery, score extraction, summary computation |
| `density.py` | Python forward-DP density reimplementation |
| `tail_analysis.py` | Tail distribution analysis, log-linear extrapolation, P(374) |
| `surrogate.py` | DT/MLP training, label noise analysis |
| `surrogate_eval.py` | Full-game simulation with surrogates |
| `skill_ladder.py` | Human-readable decision rules from regret |
| `theta_estimator.py` | Multi-start Nelder-Mead MLE for θ |
| `theta_validation.py` | Parameter recovery (Monte Carlo) |
| `feature_engineering.py` | Feature extraction for surrogates |
| `scorecard.py` | Scorecard rendering |
| `profiling/estimator.py` | Reference MLE estimator for profiling (Python/scipy mirror of JS) |
| `profiling/synthetic.py` | Synthetic player simulation for profiling validation |
| `profiling/validation.py` | Parameter recovery validation for cognitive profiling |
| `plots/style.py` | CMAP, save_fig, theme, typography constants |
| `plots/cdf.py` | CDF and survival curves |
| `plots/density.py` | 2D/3D density heatmaps |
| `plots/quantile.py` | Percentile vs θ curves |
| `plots/mean_std.py` | Efficient frontier (mean vs std) |
| `plots/combined.py` | Multi-panel combined plots |
| `plots/categories.py` | Per-category statistics |
| `plots/efficiency.py` | MER/SDVA/CVaR panels |
| `plots/scenario_cards.py` | Scenario card renders |
| `plots/difficult_cards.py` | Difficult scenario renders |
| `plots/difficult_sensitivity_cards.py` | Difficult scenario sensitivity cards across θ |
| `plots/modality.py` | Score distribution modality analysis |
| `plots/multiplayer.py` | Multiplayer simulation plots (1v1) |
| `plots/percentiles.py` | Percentile curves vs θ |
| `plots/percentile_sweep.py` | Percentile sweep curves, heatmap, frontier |
| `plots/percentile_frontiers.py` | Percentile frontiers and risk tradeoff plots |
| `plots/sensitivity.py` | Decision sensitivity plots (flip rates, θ distributions, gaps) |
| `plots/state_flow.py` | State-flow visualizations (alluvial, DAG, streamgraph) |
| `plots/surrogate.py` | Surrogate model Pareto frontier and accuracy plots |
| `plots/compression.py` | Policy compression gap CDFs, heatmaps, visit coverage |
| `plots/frontier.py` | Adaptive θ(s) vs constant-θ Pareto frontier |
| `plots/winrate.py` | Win rate analysis (θ vs win rate, conditional breakdown) |
| `plots/yatzy_hypothesis.py` | Yatzy conditional hit-rate hypothesis tests |
| `plots/spec.py` | Plot specifications: purpose, design, theta-legend modes |

## Data Flow

```
Input:  data/simulations/theta/theta_*/scores.bin (binary, i16[N])
        data/simulations/theta/theta_0/simulation_raw.bin (optional)
        outputs/density/density_*.json (exact PMFs)

Process: yatzy-analyze compute → KDE, summary stats
         yatzy-analyze plot → PNG visualizations

Output: outputs/aggregates/parquet/{summary,kde,mer,sdva}.parquet
        outputs/aggregates/csv/sweep_summary.csv
        outputs/plots/*.png (200 DPI)
```

## Adding a New Plot

1. Create `analytics/src/yatzy_analysis/plots/<name>.py`
2. Import `CMAP`, `save_fig`, `setup_theme` from `plots.style`
3. Define a function that takes data + `out_dir: Path` and calls `save_fig()`
4. Register as a Click command in `cli.py`
5. Add to the justfile if it's a standard pipeline step

## Testing

22 tests via pytest:

```bash
analytics/.venv/bin/pytest analytics/tests/ -q
```

| File | Tests | Purpose |
|------|-------|---------|
| `test_config.py` | 8 | Path resolution, theta grids, binary constants |
| `test_io.py` | 5 | Binary file readers (scores.bin, simulation_raw) |
| `test_compute.py` | 6 | KDE, summary stats, CVaR, MER |
| `test_style.py` | 3 | Colormap, theta_color, save_fig |
