import { getState, subscribe } from '../store.ts';

/** Render the dice color legend (held/reroll/optimal keep/optimal reroll). */
export function initDiceLegend(container: HTMLElement): void {
  container.className = 'dice-legend';

  const items: [string, string, string][] = [
    ['var(--bg)', '1px solid var(--text)', 'Held'],
    ['var(--bg-alt)', '1px solid var(--text)', 'Reroll'],
    ['var(--bg)', '2px solid var(--color-success)', 'Optimal keep'],
    ['var(--bg-alt)', '2px solid var(--color-danger)', 'Optimal reroll'],
  ];

  for (const [bg, border, label] of items) {
    const span = document.createElement('span');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = bg;
    swatch.style.border = border;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(label));
    container.appendChild(span);
  }

  function render() {
    const s = getState();
    const active = s.showHints && s.turnPhase === 'rolled' && s.rerollsRemaining > 0;
    container.style.opacity = active ? '1' : '0';
  }

  render();
  subscribe(render);
}
