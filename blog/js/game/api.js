const API_BASE_URL = window.__API_BASE_URL__ || 'http://localhost:9000';

export async function evaluate(req) {
  const res = await fetch(`${API_BASE_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`/evaluate failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function density(req) {
  const res = await fetch(`${API_BASE_URL}/density`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error(`/density failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function stateValue(upperScore, scoredCategories) {
  const res = await fetch(
    `${API_BASE_URL}/state_value?upper_score=${upperScore}&scored_categories=${scoredCategories}`
  );
  if (!res.ok) {
    throw new Error(`/state_value failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
