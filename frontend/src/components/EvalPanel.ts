import { getState, subscribe } from '../store.ts';
import { computeRerollMask, mapMaskToSorted } from '../mask.ts';

/** Render the evaluation panel showing state EV, mask EV, delta, and best category. */
export function initEvalPanel(container: HTMLElement): void {
  container.className = 'eval-panel';

  const grid = document.createElement('div');
  grid.className = 'grid';
  container.appendChild(grid);

  const dash = '\u2014';
  const rows: [string, HTMLSpanElement][] = [];
  const labels = ['Expected final score', 'Your reroll value', 'Best reroll value', 'Difference', 'Best category', 'Best category value'];

  for (const label of labels) {
    const lbl = document.createElement('span');
    lbl.textContent = label + ':';
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = dash;
    grid.appendChild(lbl);
    grid.appendChild(val);
    rows.push([label, val]);
  }

  function render() {
    const s = getState();
    const hasData = s.turnPhase === 'rolled' && s.lastEvalResponse !== null;
    const rerolls = s.rerollsRemaining;
    const hints = s.showHints;

    let currentMaskEv: number | null = null;
    if (s.lastEvalResponse?.mask_evs && s.sortMap) {
      const originalMask = computeRerollMask(s.dice.map(d => d.held));
      const sortedMask = mapMaskToSorted(originalMask, s.sortMap);
      currentMaskEv = s.lastEvalResponse.mask_evs[sortedMask] ?? null;
    }

    const optMaskEv = s.lastEvalResponse?.optimal_mask_ev ?? null;
    const optCat = s.lastEvalResponse?.optimal_category ?? null;
    const optCatEv = s.lastEvalResponse?.optimal_category_ev ?? null;
    const stateEv = s.lastEvalResponse?.state_ev ?? null;

    const optCatName = optCat !== null && optCat >= 0
      ? s.categories[optCat]?.name ?? '?'
      : null;

    // State EV
    rows[0][1].textContent = hasData && stateEv !== null ? stateEv.toFixed(2) : dash;

    // Your mask EV
    rows[1][1].textContent = hasData && hints && rerolls > 0 && currentMaskEv !== null
      ? currentMaskEv.toFixed(2) : dash;

    // Best mask EV
    rows[2][1].textContent = hasData && hints && rerolls > 0 && optMaskEv !== null
      ? optMaskEv.toFixed(2) : dash;

    // Delta
    if (hasData && hints && rerolls > 0 && currentMaskEv !== null && optMaskEv !== null) {
      const delta = currentMaskEv - optMaskEv;
      rows[3][1].textContent = delta.toFixed(2);
      rows[3][1].style.color = Math.abs(delta) < 0.01 ? 'var(--color-success)' : 'var(--color-danger)';
    } else {
      rows[3][1].textContent = dash;
      rows[3][1].style.color = 'inherit';
    }

    // Best category
    rows[4][1].textContent = hasData && hints && optCatName ? optCatName : dash;

    // Category EV
    rows[5][1].textContent = hasData && hints && optCatEv !== null ? optCatEv.toFixed(2) : dash;
  }

  render();
  subscribe(render);
}
