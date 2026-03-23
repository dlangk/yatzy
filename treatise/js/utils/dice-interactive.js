/**
 * Selectable dice renderer — extends shared die with click-to-toggle.
 * Re-exports PIPS from shared module for backward compatibility.
 */

export { PIPS, createDieSVG } from '/yatzy/shared/dice.js';
import { createDieSVG } from '/yatzy/shared/dice.js';

export function renderDiceSelectable(container, values, options = {}) {
  const { selected = new Set(), onToggle = null, size = 48, locked = false } = options;

  function render() {
    container.innerHTML = '';
    values.forEach((v, idx) => {
      const isSelected = selected.has(idx);
      const state = isSelected ? 'kept' : 'normal';
      const svg = createDieSVG(v, { size, state, clickable: !locked });

      if (!locked && onToggle) {
        svg.addEventListener('click', () => onToggle(idx));
      }
      container.appendChild(svg);
    });
  }

  render();

  return {
    update(newValues, newSelected) {
      values = newValues;
      selected.clear();
      newSelected.forEach((i) => selected.add(i));
      render();
    },
    destroy() {
      container.innerHTML = '';
    },
  };
}
