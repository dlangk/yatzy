/** DOM handles and update function returned by {@link createDie}. */
export interface DieElements {
  container: HTMLDivElement;
  update: (opts: {
    value: number;
    held: boolean;
    isOptimalReroll: boolean;
    isOptimalKeep: boolean;
    disabled: boolean;
    faded: boolean;
  }) => void;
}

/**
 * Creates a single die button with up/down increment arrows and hold toggle.
 *
 * Callbacks: onToggle (hold/reroll), onIncrement (+1), onDecrement (-1).
 * Returned `update()` sets value, held state, optimal-keep/reroll borders,
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

  const btn = document.createElement('button');
  btn.className = 'die-btn';
  btn.addEventListener('click', onToggle);

  const downBtn = document.createElement('button');
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.addEventListener('click', onDecrement);

  container.appendChild(upBtn);
  container.appendChild(btn);
  container.appendChild(downBtn);

  function update(opts: {
    value: number;
    held: boolean;
    isOptimalReroll: boolean;
    isOptimalKeep: boolean;
    disabled: boolean;
    faded: boolean;
  }) {
    btn.textContent = opts.value === 0 ? '?' : String(opts.value);
    btn.disabled = opts.disabled;
    btn.style.background = opts.held ? 'var(--bg)' : 'var(--bg-alt)';
    btn.style.cursor = opts.disabled ? 'default' : 'pointer';
    btn.style.opacity = opts.faded ? '0.5' : '1';
    btn.title = opts.held ? 'Held (click to reroll)' : 'Will reroll (click to hold)';

    if (opts.isOptimalReroll) {
      btn.style.border = '3px solid var(--color-danger)';
    } else if (opts.isOptimalKeep) {
      btn.style.border = '3px solid var(--color-success)';
    } else {
      btn.style.border = '2px solid var(--text)';
    }
  }

  return { container, update };
}
