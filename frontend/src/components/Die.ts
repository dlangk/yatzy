// @ts-ignore - shared module served at runtime
import { createDieSVG } from '/yatzy/shared/dice.js';

/** DOM handles and update function returned by {@link createDie}. */
export interface DieElements {
  container: HTMLDivElement;
  update: (opts: {
    value: number;
    kept: boolean;
    isOptimalReroll: boolean;
    isOptimalKeep: boolean;
    disabled: boolean;
    faded: boolean;
  }) => void;
}

/**
 * Creates a single die SVG with up/down increment arrows and keep toggle.
 *
 * Callbacks: onToggle (keep/reroll), onIncrement (+1), onDecrement (-1).
 * Returned `update()` sets value, kept state, optimal-keep/reroll borders,
 * and disabled/faded appearance.
 */
export function createDie(
  onToggle: () => void,
  onIncrement: () => void,
  onDecrement: () => void,
): DieElements {
  const container = document.createElement('div');
  container.className = 'die-container';

  const upBtn = document.createElement('button');
  upBtn.className = 'die-arrow';
  upBtn.innerHTML = '&#9650;';
  upBtn.addEventListener('click', onIncrement);

  // Placeholder SVG die (will be replaced on first update)
  const dieWrap = document.createElement('div');
  dieWrap.className = 'die-wrap';
  dieWrap.style.width = '56px';
  dieWrap.style.height = '56px';
  dieWrap.addEventListener('click', onToggle);

  const downBtn = document.createElement('button');
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.addEventListener('click', onDecrement);

  container.appendChild(upBtn);
  container.appendChild(dieWrap);
  container.appendChild(downBtn);

  function update(opts: {
    value: number;
    kept: boolean;
    isOptimalReroll: boolean;
    isOptimalKeep: boolean;
    disabled: boolean;
    faded: boolean;
  }) {
    // Background: white if kept, gray if will-reroll
    // Border: green if correct (matches optimal), red if wrong, neutral otherwise
    // Pips: full if kept, dimmed if will-reroll
    let fill: string, stroke: string, sw: number, pipOp: number;

    if (opts.faded) {
      fill = 'var(--bg-alt)';
      stroke = 'var(--border)';
      sw = 2;
      pipOp = 0.5;
    } else if (opts.kept) {
      fill = 'var(--bg)';
      pipOp = 1;
      if (opts.isOptimalKeep) { stroke = 'var(--color-success)'; sw = 3; }
      else if (opts.isOptimalReroll) { stroke = 'var(--color-danger)'; sw = 3; }
      else { stroke = 'var(--border)'; sw = 2; }
    } else {
      fill = 'var(--bg-alt)';
      pipOp = 0.45;
      if (opts.isOptimalReroll) { stroke = 'var(--color-success)'; sw = 3; }
      else if (opts.isOptimalKeep) { stroke = 'var(--color-danger)'; sw = 3; }
      else { stroke = 'var(--border)'; sw = 2; }
    }

    const svg = createDieSVG(opts.value || 0, {
      size: 56,
      state: 'normal',
      clickable: !opts.disabled,
      fill, stroke, sw, pipOp,
    });

    dieWrap.innerHTML = '';
    dieWrap.appendChild(svg);
    dieWrap.style.cursor = opts.disabled ? 'default' : 'pointer';
  }

  return { container, update };
}
