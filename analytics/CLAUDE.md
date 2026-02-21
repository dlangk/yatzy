# CLAUDE.md — Analytics

Python analysis package for Yatzy simulation data.

## Setup

```bash
cd analytics && uv venv && uv pip install -e .
analytics/.venv/bin/yatzy-analyze --help
```

## Stack

Python 3.11+ with pandas, numpy, scipy, matplotlib, seaborn, pyarrow, click.
Optional: scikit-learn, torch, joblib (surrogate models).

## CLI Commands

Entry point: `analytics/.venv/bin/yatzy-analyze <command>`

| Command | Purpose |
|---------|---------|
| `compute [--csv] [--source density]` | Summary stats (34 columns), KDE, from scores.bin or density JSON |
| `plot` | Generate all standard plots (CDF, density, quantile, mean-std) |
| `categories` | Per-category statistics across θ |
| `efficiency` | MER, SDVA, CVaR metrics |
| `modality` | Score distribution mixture decomposition |
| `summary` | Print sweep summary table |
| `multiplayer` | Analyze multiplayer simulation results |
| `surrogate-train` | Train DT + MLP surrogate models |
| `surrogate-eval [--games N]` | Full-game simulation with surrogates |
| `surrogate-plot` | Plot surrogate Pareto frontier |
| `surrogate-diagnose` | Label noise + error distribution |
| `surrogate-scaling` | Data scaling experiments |
| `surrogate-features` | Feature ablation / forward selection |
| `skill-ladder` | Induce human-readable rules from regret data |
| `profile-validate [--trials N]` | Monte Carlo parameter recovery validation |
| `compression` | Plot policy compression gaps |

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
| `cli.py` | 40+ Click commands, main entry point |
| `compute.py` | KDE, summary stats, CVaR, MER, SDVA |
| `config.py` | Path resolution, binary format constants, theta grids |
| `io.py` | Binary file readers (scores.bin, simulation_raw.bin) |
| `store.py` | Parquet save/load |
| `density.py` | Python forward-DP density reimplementation |
| `surrogate.py` | DT/MLP training, label noise analysis |
| `surrogate_eval.py` | Full-game simulation with surrogates |
| `skill_ladder.py` | Human-readable decision rules from regret |
| `theta_estimator.py` | Multi-start Nelder-Mead MLE for θ |
| `theta_validation.py` | Parameter recovery (Monte Carlo) |
| `feature_engineering.py` | Feature extraction for surrogates |
| `scorecard.py` | Scorecard rendering |
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
