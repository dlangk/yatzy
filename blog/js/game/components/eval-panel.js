import { subscribe, getState } from '../store.js';
import { computeRerollMask, mapMaskToSorted } from '../mask.js';

export function initEvalPanel(container) {
  const panel = document.createElement('div');
  panel.className = 'game-eval-panel';
  container.appendChild(panel);

  const grid = document.createElement('div');
  grid.className = 'game-eval-grid';
  panel.appendChild(grid);

  const rows = [
    { label: 'Banked', key: 'banked' },
    { label: 'Remaining EV', key: 'stateEv' },
    { label: 'Expected final', key: 'expectedFinal' },
    { label: 'Your mask EV', key: 'maskEv' },
    { label: 'Best mask EV', key: 'bestMaskEv' },
    { label: 'Delta', key: 'delta' },
    { label: 'Best category', key: 'bestCat' },
    { label: 'Category EV', key: 'catEv' },
  ];

  const valueEls = {};
  for (const row of rows) {
    const labelSpan = document.createElement('span');
    labelSpan.textContent = row.label + ':';
    const valueSpan = document.createElement('span');
    valueSpan.className = 'eval-value';
    valueSpan.textContent = '\u2014';
    grid.append(labelSpan, valueSpan);
    valueEls[row.key] = valueSpan;
  }

  function getCurrentMaskEv(state) {
    if (!state.lastEvalResponse?.mask_evs || !state.sortMap) return null;
    const originalMask = computeRerollMask(state.dice.map(d => d.held));
    const sortedMask = mapMaskToSorted(originalMask, state.sortMap);
    return state.lastEvalResponse.mask_evs[sortedMask] ?? null;
  }

  function render(state) {
    const dash = '\u2014';
    const resp = state.lastEvalResponse;
    const hasData = state.turnPhase === 'rolled' && resp && resp.state_ev != null;

    valueEls.banked.textContent = state.totalScore;
    valueEls.stateEv.textContent = hasData ? resp.state_ev.toFixed(1) : dash;
    valueEls.expectedFinal.textContent = hasData
      ? (state.totalScore + resp.state_ev).toFixed(1)
      : dash;

    const maskEv = hasData ? getCurrentMaskEv(state) : null;
    const showMask = hasData && state.rerollsRemaining > 0;

    valueEls.maskEv.textContent = showMask && maskEv !== null ? maskEv.toFixed(2) : dash;

    const bestMaskEv = hasData ? (resp.optimal_mask_ev ?? null) : null;
    valueEls.bestMaskEv.textContent = showMask && bestMaskEv !== null ? bestMaskEv.toFixed(2) : dash;

    // Delta
    const deltaEl = valueEls.delta;
    if (showMask && maskEv !== null && bestMaskEv !== null) {
      const delta = maskEv - bestMaskEv;
      deltaEl.textContent = delta.toFixed(2);
      deltaEl.className = 'eval-value ' + (Math.abs(delta) < 0.01 ? 'eval-positive' : 'eval-negative');
    } else {
      deltaEl.textContent = dash;
      deltaEl.className = 'eval-value';
    }

    // Best category
    const optCat = hasData && resp.optimal_category != null && resp.optimal_category >= 0
      ? state.categories[resp.optimal_category]?.name ?? '?'
      : null;
    valueEls.bestCat.textContent = hasData && optCat ? optCat : dash;

    valueEls.catEv.textContent = hasData && resp.optimal_category_ev != null
      ? resp.optimal_category_ev.toFixed(2) : dash;
  }

  render(getState());
  subscribe((state) => render(state));
}
