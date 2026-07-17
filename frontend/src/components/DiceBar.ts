import { getState, dispatch, subscribe } from '../store.ts';
import { createDie, type DieElements } from './Die.ts';
import { diceBarViewModel, type DiceBarViewModel } from '../mask.ts';
import { TOTAL_DICE } from '../constants.ts';
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';

/**
 * The five dice, plus the solver's advice row beneath them.
 *
 * The player's choice and the solver's advice are two separate rows sharing the
 * same columns. They used to be one object: the die's fill carried the choice and
 * its border carried the solver's verdict, which meant the advice was only
 * readable by toggling dice until the border turned green.
 *
 * All display logic lives in diceBarViewModel(). This function renders it.
 */
export function initDiceBar(container: HTMLElement): void {
  container.className = 'dice-bar-group';

  const diceRow = document.createElement('div');
  diceRow.className = 'dice-bar';
  container.appendChild(diceRow);

  const dice: DieElements[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const idx = i;
    const die = createDie(
      () => dispatch({ type: 'TOGGLE_DIE', index: idx }),
      () => {
        const s = getState();
        const v = s.turnPhase === 'rolled' ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v >= 6 || v === 0 ? 1 : v + 1 });
      },
      () => {
        const s = getState();
        const v = s.turnPhase === 'rolled' ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v <= 1 || v === 0 ? 6 : v - 1 });
      },
    );
    dice.push(die);
    diceRow.appendChild(die.container);
  }

  // Mismatch ticks sit between the two rows so the eye lands on the disagreement.
  const tickRow = document.createElement('div');
  tickRow.className = 'dice-ticks';
  container.appendChild(tickRow);
  const tickCells: HTMLSpanElement[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const cell = document.createElement('span');
    cell.className = 'dice-tick';
    tickRow.appendChild(cell);
    tickCells.push(cell);
  }

  const adviceHeader = document.createElement('div');
  adviceHeader.className = 'dice-advice-header';
  adviceHeader.textContent = 'OPTIMAL PLAY';
  attachTooltip(adviceHeader, TIP.adviceRow);
  container.appendChild(adviceHeader);

  const adviceRow = document.createElement('div');
  adviceRow.className = 'dice-advice';
  container.appendChild(adviceRow);
  const adviceCells: HTMLSpanElement[] = [];
  for (let i = 0; i < TOTAL_DICE; i++) {
    const cell = document.createElement('span');
    cell.className = 'dice-advice-cell';
    adviceRow.appendChild(cell);
    adviceCells.push(cell);
  }

  const costLine = document.createElement('div');
  costLine.className = 'dice-cost';
  container.appendChild(costLine);
  attachTooltip(costLine, TIP.diceCost);

  function render(vm: DiceBarViewModel) {
    for (let i = 0; i < TOTAL_DICE; i++) {
      dice[i].update(vm.dice[i]);
    }

    // Heights stay reserved in every branch below: contents change, nothing moves.
    const showAdvice = vm.advice !== null;
    adviceHeader.style.visibility = showAdvice ? 'visible' : 'hidden';
    for (let i = 0; i < TOTAL_DICE; i++) {
      const advice = vm.advice?.[i] ?? '';
      adviceCells[i].textContent = advice;
      adviceCells[i].classList.toggle('dice-advice-cell--reroll', advice === 'REROLL');
      tickCells[i].classList.toggle('dice-tick--on', vm.ticks[i]);
    }

    costLine.textContent = vm.cost === null
      ? ''
      : `Costs you ${Math.abs(vm.cost).toFixed(1)} pts vs optimal`;
  }

  render(diceBarViewModel(getState()));
  subscribe((state, prev) => {
    if (state.dice === prev.dice &&
        state.lastEvalResponse === prev.lastEvalResponse &&
        state.prefs.showHints === prev.prefs.showHints &&
        state.rerollsRemaining === prev.rerollsRemaining &&
        state.turnPhase === prev.turnPhase) return;
    render(diceBarViewModel(state));
  });
}
