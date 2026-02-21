import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock config module to provide a known base URL
vi.mock('./config.ts', () => ({
  API_BASE_URL: 'http://test:9000',
}));

import { evaluate, healthCheck, getStateValue, fetchDensity } from './api.ts';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  mockFetch.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('evaluate', () => {
  it('sends POST to /evaluate and returns parsed JSON', async () => {
    const body = { categories: [], optimal_category: 0, optimal_category_ev: 0, state_ev: 245 };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));

    const result = await evaluate({
      dice: [1, 2, 3, 4, 5],
      upper_score: 0,
      scored_categories: 0,
      rerolls_remaining: 2,
    });

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://test:9000/evaluate');
    expect(opts.method).toBe('POST');
    expect(result.state_ev).toBe(245);
  });

  it('throws on non-OK response', async () => {
    mockFetch.mockResolvedValueOnce(new Response('bad', { status: 400, statusText: 'Bad Request' }));

    await expect(
      evaluate({ dice: [0, 1, 2, 3, 4], upper_score: 0, scored_categories: 0, rerolls_remaining: 2 }),
    ).rejects.toThrow('/evaluate failed');
  });
});

describe('healthCheck', () => {
  it('returns true on 200', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: 'OK' }));
    const result = await healthCheck();
    expect(result).toBe(true);
  });

  it('returns false on network error', async () => {
    mockFetch.mockRejectedValueOnce(new Error('network error'));
    const result = await healthCheck();
    expect(result).toBe(false);
  });
});

describe('getStateValue', () => {
  it('sends GET with correct query params', async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ expected_final_score: 247.5 }));

    const result = await getStateValue(10, 3);
    expect(result.expected_final_score).toBe(247.5);

    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toContain('/state_value?');
    expect(url).toContain('upper_score=10');
    expect(url).toContain('scored_categories=3');
  });
});

describe('fetchDensity', () => {
  it('sends POST to /density with correct body', async () => {
    const body = { mean: 250.0, std_dev: 30.0, percentiles: { '50': 250 } };
    mockFetch.mockResolvedValueOnce(jsonResponse(body));

    const result = await fetchDensity(10, 3, 100);
    expect(result.mean).toBe(250.0);

    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://test:9000/density');
    expect(opts.method).toBe('POST');
    const parsed = JSON.parse(opts.body as string) as Record<string, number>;
    expect(parsed.upper_score).toBe(10);
    expect(parsed.scored_categories).toBe(3);
    expect(parsed.accumulated_score).toBe(100);
  });
});
