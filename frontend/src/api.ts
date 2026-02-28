/**
 * Solver API client.
 *
 * All endpoints are stateless lookups against the precomputed strategy table.
 * See solver/CLAUDE.md § API Reference for the canonical endpoint spec.
 *
 * Request contracts (validated server-side in solver/src/server.rs):
 *   - dice:                5-element array, each value in [1, 6]
 *   - upper_score:         integer in [0, 63]
 *   - scored_categories:   15-bit bitmask (0 = no categories scored)
 *   - rerolls_remaining:   0, 1, or 2
 *   - accumulated_score:   non-negative integer
 */
import { API_BASE_URL } from './config.ts';
import type { EvaluateRequest, EvaluateResponse } from './types.ts';

/** Evaluate all keep-mask and category EVs for the current dice + game state. */
export async function evaluate(req: EvaluateRequest): Promise<EvaluateResponse> {
  const res = await fetch(`${API_BASE_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`/evaluate failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<EvaluateResponse>;
}

/** Check if the solver backend is reachable. Returns false on any error. */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/** Look up the expected final score for a given scorecard state (mmap table lookup). */
export async function getStateValue(
  upperScore: number,
  scoredCategories: number,
): Promise<{ expected_final_score: number }> {
  const params = new URLSearchParams({
    upper_score: String(upperScore),
    scored_categories: String(scoredCategories),
  });
  const res = await fetch(`${API_BASE_URL}/state_value?${params}`);
  if (!res.ok) {
    throw new Error(`/state_value failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export interface DensityResponse {
  mean: number;
  std_dev: number;
  percentiles: Record<string, number>;
}

/** Compute exact score distribution from a mid-game state via forward density evolution. */
export async function fetchDensity(
  upperScore: number,
  scoredCategories: number,
  accumulatedScore: number,
): Promise<DensityResponse> {
  const res = await fetch(`${API_BASE_URL}/density`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      upper_score: upperScore,
      scored_categories: scoredCategories,
      accumulated_score: accumulatedScore,
    }),
  });
  if (!res.ok) {
    throw new Error(`/density failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
