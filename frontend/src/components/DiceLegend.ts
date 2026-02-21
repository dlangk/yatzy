import { getState, subscribe } from '../store.ts';
import { COLORS } from '../constants.ts';

export function initDiceLegend(container: HTMLElement): void {
  container.className = 'dice-legend';

  const items: [string, string, string][] = [
    [COLORS.bg, `1px solid ${COLORS.text}`, 'Held'],
    [COLORS.bgAlt, `1px solid ${COLORS.text}`, 'Reroll'],
    [COLORS.bg, `2px solid ${COLORS.success}`, 'Optimal keep'],
    [COLORS.bgAlt, `2px solid ${COLORS.danger}`, 'Optimal reroll'],
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
    const active = s.turnPhase === 'rolled' && s.rerollsRemaining > 0;
    container.style.opacity = active ? '1' : '0';
  }

  render();
  subscribe(render);
}
