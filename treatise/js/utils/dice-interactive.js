/**
 * Selectable dice renderer — extends renderDice with click-to-toggle.
 */

const PIPS = [
  [],
  [{ cx: 24, cy: 24 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 24, cy: 24 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 24, cy: 24 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
  [{ cx: 14, cy: 12 }, { cx: 34, cy: 12 }, { cx: 14, cy: 24 }, { cx: 34, cy: 24 }, { cx: 14, cy: 36 }, { cx: 34, cy: 36 }],
];

export function renderDiceSelectable(container, values, options = {}) {
  const { selected = new Set(), onToggle = null, size = 48, locked = false } = options;

  function render() {
    container.innerHTML = '';
    values.forEach((v, idx) => {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 48 48');
      svg.setAttribute('width', String(size));
      svg.setAttribute('height', String(size));
      svg.classList.add('die-svg');
      if (selected.has(idx)) svg.classList.add('die-selected');
      if (!locked) svg.style.cursor = 'pointer';

      const isSelected = selected.has(idx);
      const strokeColor = isSelected ? 'var(--accent)' : 'var(--border)';
      const strokeWidth = isSelected ? 3 : 2;
      svg.innerHTML = `<rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg-alt)" stroke="${strokeColor}" stroke-width="${strokeWidth}"/>`;
      (PIPS[v] || []).forEach((p) => {
        svg.innerHTML += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)"/>`;
      });

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
