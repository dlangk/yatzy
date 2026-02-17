/**
 * Load profiling scenarios from static JSON.
 */

const SCENARIOS_URL = './data/scenarios.json';

export async function loadScenarios() {
  const res = await fetch(SCENARIOS_URL);
  if (!res.ok) {
    throw new Error(`Failed to load scenarios: ${res.status}`);
  }
  const data = await res.json();
  return data.scenarios;
}
