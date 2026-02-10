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
