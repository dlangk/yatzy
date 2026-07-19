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
    advice: DieAdvice | null;
    showTick: boolean;
    disabled: boolean;
    faded: boolean;
  }) => void;
}

/**
 * One die column, top to bottom:
 *   up arrow, your HOLD/REROLL choice, the die, the optimal HOLD/REROLL,
 *   down arrow, a mismatch line.
 *
 * The die is the anchor: your choice sits directly above it and the solver's
 * advice directly below, so a disagreement reads on the die itself. Your choice
 * and the die are ONE button (one focus stop, one target); the advice and the
 * line are display-only. The advice slot and the line reserve their height, so
 * turning Hints on or off never moves the arrows.
 *
 * Colour: your choice is bold (you control it), advice is regular (secondary).
 * Both use ink for HOLD and orange for REROLL. The red line, not the advice
 * text, carries disagreement, so the advice row reads as calm guidance.
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

  // Your choice and the die are one button. The choice label is appended first,
  // so it renders above the die.
  const holdBtn = document.createElement('button');
  holdBtn.type = 'button';
  holdBtn.className = 'die-hold';
  holdBtn.addEventListener('click', onToggle);

  const labelEl = document.createElement('span');
  labelEl.className = 'die-label';
  holdBtn.appendChild(labelEl);

  const dieWrap = document.createElement('span');
  dieWrap.className = 'die-wrap';
  holdBtn.appendChild(dieWrap);

  // The solver's advice: display-only, directly below the die.
  const adviceEl = document.createElement('span');
  adviceEl.className = 'die-advice';

  const downBtn = document.createElement('button');
  downBtn.type = 'button';
  downBtn.className = 'die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.setAttribute('aria-label', 'Decrease die value');
  downBtn.addEventListener('click', onDecrement);

  // Mismatch line at the base of the column.
  const tickEl = document.createElement('span');
  tickEl.className = 'die-tick';

  container.appendChild(upBtn);
  container.appendChild(holdBtn);
  container.appendChild(adviceEl);
  container.appendChild(downBtn);
  container.appendChild(tickEl);

  function update(opts: {
    value: number;
    held: boolean;
    label: DieAdvice | '';
    advice: DieAdvice | null;
    showTick: boolean;
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

    // Advice slot: reserved even when empty, so the arrows never move. The full
    // plan is always shown, but agreeing (or costless) advice is muted so only a
    // meaningful disagreement takes colour. showTick marks that disagreement.
    adviceEl.textContent = opts.advice ?? '';
    adviceEl.classList.toggle('die-advice--reroll', opts.advice === 'REROLL');
    adviceEl.classList.toggle('die-advice--mismatch', opts.showTick);
    if (opts.advice) {
      adviceEl.setAttribute('aria-label', `Optimal play, ${opts.advice.toLowerCase()}`);
      adviceEl.removeAttribute('aria-hidden');
    } else {
      adviceEl.removeAttribute('aria-label');
      adviceEl.setAttribute('aria-hidden', 'true');
    }

    tickEl.classList.toggle('die-tick--on', opts.showTick);
  }

  return { container, update };
}
