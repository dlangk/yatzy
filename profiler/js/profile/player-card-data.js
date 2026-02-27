/**
 * Player card grid data: load, lookup, and counterfactual computation.
 *
 * The grid JSON contains simulation results for a 4D parameter grid
 * (θ × β × γ × d). Index: ti*108 + bi*18 + gi*3 + di.
 */

const GRID_URL = './data/player_card_grid.json';

let gridData = null;
let loadPromise = null;

/** Load the player card grid JSON. Caches the result. */
export async function loadPlayerCardGrid() {
  if (gridData) return gridData;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const res = await fetch(GRID_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      gridData = await res.json();
      return gridData;
    } catch (err) {
      console.warn('Player card grid not available:', err.message);
      loadPromise = null;
      return null;
    }
  })();

  return loadPromise;
}

/** Check if grid data is loaded. */
export function isGridLoaded() {
  return gridData !== null;
}

/** Get the raw grid data (or null). */
export function getGridData() {
  return gridData;
}

/** Get the optimal baseline stats (or null). */
export function getOptimalStats() {
  return gridData ? gridData.optimal : null;
}

/**
 * Find the nearest index in an array for a given value.
 */
function nearestIndex(arr, value) {
  let best = 0;
  let bestDist = Math.abs(arr[0] - value);
  for (let i = 1; i < arr.length; i++) {
    const dist = Math.abs(arr[i] - value);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

/**
 * Look up grid stats for given parameters (nearest-neighbor).
 * Returns the stats object or null if grid not loaded.
 */
export function lookupGrid(theta, beta, gamma, d) {
  if (!gridData) return null;

  const ti = nearestIndex(gridData.theta_values, theta);
  const bi = nearestIndex(gridData.beta_values, beta);
  const gi = nearestIndex(gridData.gamma_values, gamma);
  const di = nearestIndex(gridData.d_values, d);

  const idx = ti * 108 + bi * 18 + gi * 3 + di;
  return gridData.grid[idx] || null;
}

/**
 * Compute counterfactuals: for each parameter, fix it to optimal and compare.
 *
 * Returns array of { param, label, yourMean, fixedMean, delta } sorted by delta desc.
 * delta = fixedMean - yourMean (positive = points gained by fixing that param).
 */
export function computeCounterfactuals(theta, beta, gamma, d) {
  if (!gridData) return [];

  const yours = lookupGrid(theta, beta, gamma, d);
  if (!yours) return [];

  const fixes = [
    { param: 'theta', label: 'Risk Attitude', optVal: 0, current: theta,
      lookup: () => lookupGrid(0, beta, gamma, d) },
    { param: 'beta', label: 'Decision Precision', optVal: 10, current: beta,
      lookup: () => lookupGrid(theta, 10, gamma, d) },
    { param: 'gamma', label: 'Planning Horizon', optVal: 1.0, current: gamma,
      lookup: () => lookupGrid(theta, beta, 1.0, d) },
    { param: 'd', label: 'Strategic Depth', optVal: 999, current: d,
      lookup: () => lookupGrid(theta, beta, gamma, 999) },
  ];

  const results = [];
  for (const fix of fixes) {
    const fixed = fix.lookup();
    if (fixed) {
      results.push({
        param: fix.param,
        label: fix.label,
        yourMean: yours.mean,
        fixedMean: fixed.mean,
        delta: fixed.mean - yours.mean,
      });
    }
  }

  // Sort by delta descending (largest improvement first)
  results.sort((a, b) => b.delta - a.delta);
  return results;
}
