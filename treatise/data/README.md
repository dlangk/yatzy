# Treatise chart data

These JSON files are the data behind the treatise's D3 charts. They are the
static site's source and are **committed to git** (unlike the expensive
top-level `data/`), so a fresh clone renders every chart with no pipeline run.

> **Why committed?** On a new machine, generated-but-uncommitted files are
> simply absent, and a `deploy-treatise` (`rsync --delete`) would then wipe them
> from the server. Versioning them here removes that failure mode. Keep them in
> git; regenerate deliberately with the command below.

## Regenerate everything

```bash
just regen-treatise-data      # full pipeline (minutes; density dominates)
just check-treatise-data      # verify completeness (also runs pre-deploy)
```

`check-treatise-data` requires a file **only if a chart that loads it has a
container in a section markdown**. Charts whose container was never placed in
the prose ("phantom" charts) do not create a requirement.

## What produces each file

| File | Generator | Upstream input |
|------|-----------|----------------|
| `category_landscape.json` | `analytics/scripts/generate_treatise_data.py` | θ=0 `simulation_raw.bin` + `category_stats.csv` |
| `category_pmfs.json` | ″ | θ=0 `simulation_raw.bin` |
| `mixture.json` | ″ | θ=0 `simulation_raw.bin` |
| `bonus_covariance.json` | ″ | θ=0 `simulation_raw.bin` |
| `category_correlations.json` | ″ | θ=0 `simulation_raw.bin` |
| `fill_turn_heatmap.json` | ″ | θ=0 `simulation_raw.bin` |
| `score_spray.json`, `score_spray_meta.json` | `research/rosetta/generate_score_spray.py` | θ=0 `simulation_raw.bin` |
| `category_stats_theta0.json` | `profiler/convert.py` → copy | `just category-sweep` → `category_stats.csv` |
| `heuristic_gap.json` | `profiler/convert.py` → copy | `just heuristic-gap` |
| `kde_curves.json`, `tail_exact.json`, `sweep_summary.json` | `treatise/scripts/gen-exact-data.py` | `just density --grid all` → `outputs/density/*.json` |
| `density_exact.json` | copy of `outputs/density/density_0.json` | `just density` (θ=0) |
| `rosetta_rules.json`, `filter_grammar.json` | `treatise/scripts/gen-rosetta-charts.py` | `just rosetta` → `outputs/rosetta/skill_ladder.json` |
| `game_eval.json` | `profiler/convert.py` → copy | surrogate pipeline (`scikit-learn`): `export-training-data` → `surrogate-train` → `surrogate-eval` |
| `dice_symmetry.json` | `treatise/scripts/gen-dice-symmetry.mjs` | none (combinatorial) |
| `reachability.json` | `treatise/scripts/gen-reachability.mjs` | none (combinatorial; self-checks vs solver's 2794/4096) |
| `graph_race_to_63.json`, `graph_category_sankey.json`, `graph_ev_funnel.json` | `yatzy-forward-pass` | `just precompute` |
| `backward_wave.json`, `decision_anatomy.json` | hand-authored (no generator) | — |
| `umap_embeddings.json` | placeholder `{ "placeholder": true }` | real embeddings need the UMAP manifold pipeline (emits PNG, not JSON) |

## Notes

- **`reachability.json`** currently backs a phantom chart (no section container),
  but is generated and committed because it is cheap and self-validating.
- **`umap_embeddings.json`** is an intentional placeholder; the chart renders
  illustrative clusters with a "PLACEHOLDER" watermark until a real embeddings
  export exists.
- Files with **no generator** (`backward_wave`, `decision_anatomy`) are
  hand-authored: do not delete them expecting regeneration.
