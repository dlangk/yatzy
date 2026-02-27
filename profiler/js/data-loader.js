/**
 * Centralized JSON fetch + cache for blog data files.
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
  sweepSummary: () => load('sweep_summary.json'),
  kdeCurves: () => load('kde_curves.json'),
  winrate: () => load('winrate.json'),
  gameEval: () => load('game_eval.json'),
  heuristicGap: () => load('heuristic_gap.json'),
  mixture: () => load('mixture.json'),
  percentilePeaks: () => load('percentile_peaks.json'),
  categoryStats: () => load('category_stats_theta0.json'),
  stateHeatmap: () => load('state_heatmap.json'),
  greedyVsOptimal: () => load('greedy_vs_optimal.json'),
};
