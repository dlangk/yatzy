import { API_BASE_URL } from './config.ts';
import type { EvaluateRequest, EvaluateResponse } from './types.ts';

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

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

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
