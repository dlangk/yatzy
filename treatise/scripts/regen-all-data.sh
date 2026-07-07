#!/bin/bash
# Regenerate every treatise/data/*.json chart file from scratch.
#   just regen-treatise-data
#
# Ordered by dependency. Density and the (now-optimized) skill-ladder dominate
# wall-clock. Verify completeness at the end with: just check-treatise-data
set -euo pipefail
cd "$(dirname "$0")/../.."
export YATZY_BASE_PATH=.
SOLVER=solver/target/release
PY=analytics/.venv/bin/python
step() { echo; echo "── $1"; }

step "1/13 precompute theta=0 strategy table"
$SOLVER/yatzy-precompute

step "2/13 simulate 1M games with full recording"
$SOLVER/yatzy-simulate --games 1000000 --full-recording --output data/simulations/theta/theta_0.000
ln -sf theta/theta_0.000/simulation_raw.bin data/simulations/simulation_raw.bin

step "3/13 category sweep -> category_stats.csv"
$SOLVER/yatzy-category-sweep --games 1000000 --output outputs/aggregates/csv/category_stats.csv

step "4/13 analytics treatise files (landscape, pmfs, mixture, covariance, correlations, fill-turn)"
$PY analytics/scripts/generate_treatise_data.py

step "5/13 score spray"
$PY research/rosetta/generate_score_spray.py

step "6/13 dice symmetry (combinatorial)"
node treatise/scripts/gen-dice-symmetry.mjs

step "7/13 reachability (combinatorial, self-checks vs solver)"
node treatise/scripts/gen-reachability.mjs

step "8/13 heuristic-gap + category-stats -> profiler/data -> treatise/data"
$SOLVER/yatzy-heuristic-gap --games 100000 --output outputs/heuristic_gap
$PY - <<'PY'
import sys, shutil; from pathlib import Path
sys.path.insert(0, "profiler"); import convert
convert.convert_heuristic_gap(); convert.convert_category_stats()
for n in ("heuristic_gap.json","category_stats_theta0.json"):
    shutil.copy2(Path("profiler/data")/n, Path("treatise/data")/n)
PY

step "9/13 exact density evolution (37 thetas, grid all) -- LONG"
$SOLVER/yatzy-density --grid all --output outputs/density

step "10/13 exact-density charts (kde_curves, sweep_summary, tail_exact) + density_exact"
$PY treatise/scripts/gen-exact-data.py
cp outputs/density/density_0.json treatise/data/density_exact.json

step "11/13 rosetta pipeline (regret-export + optimized skill-ladder) -> rosetta_rules, filter_grammar"
$SOLVER/yatzy-regret-export --games 200000
analytics/.venv/bin/yatzy-analyze skill-ladder
$PY treatise/scripts/gen-rosetta-charts.py

step "12/13 surrogate pipeline (needs scikit-learn) -> game_eval"
$SOLVER/yatzy-export-training-data --games 200000 --output data/surrogate
analytics/.venv/bin/yatzy-analyze surrogate-train
analytics/.venv/bin/yatzy-analyze surrogate-eval --games 100000
$PY - <<'PY'
import sys, shutil; from pathlib import Path
sys.path.insert(0, "profiler"); import convert
convert.convert_game_eval()
shutil.copy2(Path("profiler/data/game_eval.json"), Path("treatise/data/game_eval.json"))
PY

step "13/13 umap placeholder (real embeddings need the UMAP manifold pipeline)"
[ -f treatise/data/umap_embeddings.json ] || echo '{"placeholder": true}' > treatise/data/umap_embeddings.json

step "verify completeness"
node treatise/scripts/check-data.mjs
echo; echo "✓ regen-treatise-data complete"
