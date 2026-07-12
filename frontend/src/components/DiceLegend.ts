import { getState, subscribe } from '../store.ts';
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';

/** Render the dice color legend. Always shows kept/reroll; adds hint colors when hints are on. */
export function initDiceLegend(container: HTMLElement): void {
  container.className = 'dice-legend';

  const keptSpan = makeSwatch('var(--bg)', '1px solid var(--border)', 'Kept', TIP.legendKept);
  const rerollSpan = makeSwatch('var(--bg-alt)', '1px solid var(--border)', 'Reroll', TIP.legendReroll);
  const optKeepSpan = makeSwatch('var(--bg)', '2px solid var(--color-success)', 'Optimal', TIP.legendOptimal);
  const optRerollSpan = makeSwatch('var(--bg-alt)', '2px solid var(--color-danger)', 'Suboptimal', TIP.legendSuboptimal);

  container.appendChild(keptSpan);
  container.appendChild(rerollSpan);
  container.appendChild(optKeepSpan);
  container.appendChild(optRerollSpan);

  function render() {
    const s = getState();
    const active = s.turnPhase === 'rolled' && s.rerollsRemaining > 0;
    const showHints = s.prefs.showHints && active;

    // Always show kept/reroll when dice are active
    keptSpan.style.display = active ? '' : 'none';
    rerollSpan.style.display = active ? '' : 'none';
    optKeepSpan.style.display = showHints ? '' : 'none';
    optRerollSpan.style.display = showHints ? '' : 'none';
  }

  render();
  subscribe((state, prev) => {
    if (state.prefs.showHints === prev.prefs.showHints &&
        state.turnPhase === prev.turnPhase &&
        state.rerollsRemaining === prev.rerollsRemaining) return;
    render();
  });
}

function makeSwatch(bg: string, border: string, label: string, tip: string): HTMLSpanElement {
  const span = document.createElement('span');
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = bg;
  swatch.style.border = border;
  span.appendChild(swatch);
  const labelSpan = document.createElement('span');
  labelSpan.textContent = label;
  attachTooltip(labelSpan, tip);
  span.appendChild(labelSpan);
  return span;
}
