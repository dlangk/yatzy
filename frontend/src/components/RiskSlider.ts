import { getState, dispatch, subscribe } from '../store.ts';

const LABELS = ['Very safe', 'Safe', 'Neutral', 'Aggressive', 'Very aggressive'];

/**
 * Five-detent risk slider. A native range input (min 0, max 4, step 1) so
 * keyboard arrows move detents for free. Dispatches SET_RISK_LEVEL, which
 * re-centers the whole screen client-side (no network). The current level is
 * named above; below-neutral vs above is not colour-coded (monochrome), to
 * avoid overloading the green/red/orange meanings.
 */
export function initRiskSlider(container: HTMLElement): void {
  container.className = 'risk-slider';

  const heading = document.createElement('div');
  heading.className = 'risk-slider-heading';
  const title = document.createElement('span');
  title.textContent = 'Risk appetite:';
  const current = document.createElement('span');
  current.className = 'risk-slider-current';
  heading.appendChild(title);
  heading.appendChild(current);
  container.appendChild(heading);

  const input = document.createElement('input');
  input.type = 'range';
  input.min = '0';
  input.max = '4';
  input.step = '1';
  input.className = 'risk-slider-input';
  input.setAttribute('aria-label', 'Risk appetite');
  input.addEventListener('input', () => {
    dispatch({ type: 'SET_RISK_LEVEL', level: Number(input.value) });
  });
  container.appendChild(input);

  const ticks = document.createElement('div');
  ticks.className = 'risk-slider-ticks';
  for (const label of LABELS) {
    const t = document.createElement('span');
    t.className = 'risk-slider-tick';
    t.textContent = label;
    ticks.appendChild(t);
  }
  container.appendChild(ticks);

  // A one-line note shown only off-neutral: the values are risk-adjusted.
  const note = document.createElement('div');
  note.className = 'risk-slider-note';
  container.appendChild(note);

  function render() {
    const level = getState().prefs.riskLevel;
    if (input.value !== String(level)) input.value = String(level);
    current.textContent = LABELS[level];
    input.setAttribute('aria-valuetext', LABELS[level]);
    const neutral = level === 2;
    note.textContent = neutral
      ? ''
      : 'Values shown are risk-adjusted (certainty-equivalent) points, not expected scores.';
    note.style.visibility = neutral ? 'hidden' : 'visible';
  }

  render();
  subscribe((state, prev) => {
    if (state.prefs.riskLevel === prev.prefs.riskLevel) return;
    render();
  });
}
