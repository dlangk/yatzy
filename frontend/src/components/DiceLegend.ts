import { getState, subscribe } from '../store.ts';

/** Render the dice color legend. Always shows kept/reroll; adds hint colors when hints are on. */
export function initDiceLegend(container: HTMLElement): void {
  container.className = 'dice-legend';

  const keptSpan = makeSwatch('var(--bg)', '1px solid var(--border)', 'Kept');
  const rerollSpan = makeSwatch('var(--bg-alt)', '1px solid var(--border)', 'Reroll');
  const optKeepSpan = makeSwatch('var(--bg)', '2px solid var(--color-success)', 'Optimal');
  const optRerollSpan = makeSwatch('var(--bg-alt)', '2px solid var(--color-danger)', 'Suboptimal');

  container.appendChild(keptSpan);
  container.appendChild(rerollSpan);
  container.appendChild(optKeepSpan);
  container.appendChild(optRerollSpan);

  function render() {
    const s = getState();
    const active = s.turnPhase === 'rolled' && s.rerollsRemaining > 0;
    const showHints = s.showHints && active;

    // Always show kept/reroll when dice are active
    keptSpan.style.display = active ? '' : 'none';
    rerollSpan.style.display = active ? '' : 'none';
    optKeepSpan.style.display = showHints ? '' : 'none';
    optRerollSpan.style.display = showHints ? '' : 'none';
  }

  render();
  subscribe((state, prev) => {
    if (state.showHints === prev.showHints &&
        state.turnPhase === prev.turnPhase &&
        state.rerollsRemaining === prev.rerollsRemaining) return;
    render();
  });
}

function makeSwatch(bg: string, border: string, label: string): HTMLSpanElement {
  const span = document.createElement('span');
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = bg;
  swatch.style.border = border;
  span.appendChild(swatch);
  span.appendChild(document.createTextNode(label));
  return span;
}
