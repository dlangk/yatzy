/**
 * Centralized JSON fetch + cache for treatise data files.
 */

const cache = new Map();

async function load(name) {
  if (cache.has(name)) return cache.get(name);
  const resp = await fetch(`data/${name}`);
  if (!resp.ok) throw new Error(`Failed to load data/${name}: ${resp.status}`);
  const data = await resp.json();
  cache.set(name, data);
  return data;
}

export const DataLoader = {
  // Copied from blog
  sweepSummary: () => load('sweep_summary.json'),
  kdeCurves: () => load('kde_curves.json'),
  winrate: () => load('winrate.json'),
  gameEval: () => load('game_eval.json'),
  heuristicGap: () => load('heuristic_gap.json'),
  mixture: () => load('mixture.json'),
  stateHeatmap: () => load('state_heatmap.json'),
  greedyVsOptimal: () => load('greedy_vs_optimal.json'),
  categoryStats: () => load('category_stats_theta0.json'),

  // New treatise data
  stateCounterSteps: () => load('state_counter_steps.json'),
  backwardWave: () => load('backward_wave.json'),
  widgetScenarios: () => load('widget_scenarios.json'),
  decisionAnatomy: () => load('decision_anatomy.json'),
  reachability: () => load('reachability.json'),
  optimizationTimeline: () => load('optimization_timeline.json'),
  maxPolicy: () => load('max_policy.json'),
  bonusCovariance: () => load('bonus_covariance.json'),
  thresholdPolicy: () => load('threshold_policy.json'),
  compressionTail: () => load('compression_tail.json'),
  rosettaRules: () => load('rosetta_rules.json'),
  filterGrammar: () => load('filter_grammar.json'),
  umapEmbeddings: () => load('umap_embeddings.json'),
};
