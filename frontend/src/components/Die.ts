// @ts-expect-error - shared module served at runtime
import { createDieSVG } from '/yatzy/shared/dice.js';
import type { DieAdvice } from '../mask.ts';

/** DOM handles and update function returned by {@link createDie}. */
export interface DieElements {
  container: HTMLDivElement;
  update: (opts: {
    value: number;
    held: boolean;
    label: DieAdvice | '';
    disabled: boolean;
    faded: boolean;
  }) => void;
}

/**
 * Creates a single die: an up arrow, one hold button containing the die SVG and
 * its HOLD/REROLL label, and a down arrow.
 *
 * The die and its label are deliberately ONE button, not a die with a button
 * beneath it. That gives one focus stop, one 56x74 touch target, and one control
 * serving mouse, keyboard, and touch. A separate 56x18 label button would fail
 * WCAG 2.2 SC 2.5.8 (24x24 minimum).
 *
 * This component carries no hint logic. The solver's advice lives in DiceBar's
 * advice row, so that it stays readable without toggling dice.
 *
 * Callbacks: onToggle (hold/reroll), onIncrement (+1), onDecrement (-1).
 */
export function createDie(
  onToggle: () => void,
  onIncrement: () => void,
  onDecrement: () => void,
): DieElements {
  const container = document.createElement('div');
  container.className = 'die-container';

  const upBtn = document.createElement('button');
  upBtn.type = 'button';
  upBtn.className = 'die-arrow';
  upBtn.innerHTML = '&#9650;';
  upBtn.setAttribute('aria-label', 'Increase die value');
  upBtn.addEventListener('click', onIncrement);

  const holdBtn = document.createElement('button');
  holdBtn.type = 'button';
  holdBtn.className = 'die-hold';
  holdBtn.addEventListener('click', onToggle);

  const dieWrap = document.createElement('span');
  dieWrap.className = 'die-wrap';
  holdBtn.appendChild(dieWrap);

  const labelEl = document.createElement('span');
  labelEl.className = 'die-label';
  holdBtn.appendChild(labelEl);

  const downBtn = document.createElement('button');
  downBtn.type = 'button';
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.setAttribute('aria-label', 'Decrease die value');
  downBtn.addEventListener('click', onDecrement);

  container.appendChild(upBtn);
  container.appendChild(holdBtn);
  container.appendChild(downBtn);

  function update(opts: {
    value: number;
    held: boolean;
    label: DieAdvice | '';
    disabled: boolean;
    faded: boolean;
  }) {
    // Border is always neutral. Green/red outlines used to encode agreement with
    // the solver here, which contradicted the legend and moved as the player
    // toggled. Fill and pips echo held state as a redundant cue only.
    let fill: string, pipOp: number;
    if (opts.faded) {
      fill = 'var(--bg-alt)';
      pipOp = 0.5;
    } else if (opts.held) {
      fill = 'var(--bg)';
      pipOp = 1;
    } else {
      fill = 'var(--bg-alt)';
      pipOp = 0.45;
    }

    const svg = createDieSVG(opts.value || 0, {
      size: 56,
      state: 'normal',
      clickable: !opts.disabled,
      fill,
      stroke: 'var(--border)',
      sw: 2,
      pipOp,
    });

    dieWrap.innerHTML = '';
    dieWrap.appendChild(svg);

    labelEl.textContent = opts.label;
    holdBtn.disabled = opts.disabled;
    holdBtn.setAttribute('aria-pressed', String(opts.held));
    holdBtn.classList.toggle('die-hold--reroll', !opts.held && !opts.faded);
    holdBtn.setAttribute(
      'aria-label',
      opts.value === 0
        ? 'Die, not yet rolled'
        : `Die showing ${opts.value}, ${opts.held ? 'held' : 'will be rerolled'}`,
    );
  }

  return { container, update };
}
