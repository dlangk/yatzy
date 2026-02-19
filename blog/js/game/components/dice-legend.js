import { subscribe, getState } from '../store.js';

export function initDiceLegend(container) {
  const legend = document.createElement('div');
  legend.className = 'game-dice-legend';
  legend.style.opacity = '0';
  container.appendChild(legend);

  const items = [
    { bg: 'var(--bg)', border: '1px solid var(--border)', label: 'Held' },
    { bg: 'var(--bg-alt)', border: '1px solid var(--border)', label: 'Reroll' },
    { bg: 'var(--bg)', border: '2px solid var(--color-success)', label: 'Optimal keep' },
    { bg: 'var(--bg-alt)', border: '2px solid var(--color-danger)', label: 'Optimal reroll' },
  ];

  for (const item of items) {
    const span = document.createElement('span');
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = item.bg;
    swatch.style.border = item.border;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(item.label));
    legend.appendChild(span);
  }

  function render(state) {
    const active = state.turnPhase === 'rolled' && state.rerollsRemaining > 0;
    legend.style.opacity = active ? '1' : '0';
  }

  render(getState());
  subscribe((state) => render(state));
}
