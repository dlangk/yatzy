import { getState, dispatch, subscribe } from '../store.ts';
import { createDie, type DieElements } from './Die.ts';
import { diceBarViewModel, type DiceBarViewModel } from '../mask.ts';
import { TOTAL_DICE } from '../constants.ts';
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';

/**
 * The five dice, plus the one-line cost beneath them.
 *
 * Each column carries its own choice, advice, and mismatch line (see Die.ts):
 * your HOLD/REROLL choice hugs the die from above, the solver's advice from
 * below, and a red line at the base flags any disagreement. This function maps
 * the view model onto the columns and renders the cost line.
 *
 * The choice and the advice used to fight over one die: the die's fill carried
 * the choice and its border carried the solver's verdict, so the advice was
 * only readable by toggling dice until the border turned green. All display
 * logic now lives in diceBarViewModel(); this function only renders it.
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

  // The one-line cost, set apart below the dice. Its height is reserved and it
  // is hidden (not merely emptied) when absent, so attachTooltip's tabindex does
  // not leave a focusable, help-cursored blank strip in the tab order.
  const costLine = document.createElement('div');
  costLine.className = 'dice-cost';
  attachTooltip(costLine, TIP.diceCost);
  container.appendChild(costLine);

  function render(vm: DiceBarViewModel) {
    for (let i = 0; i < TOTAL_DICE; i++) {
      dice[i].update({
        ...vm.dice[i],
        advice: vm.advice?.[i] ?? null,
        showTick: vm.ticks[i],
      });
    }

    // The status line, only while hints are showing (advice !== null):
    //   - no cost: you match optimal (or a costless tie), shown green.
    //   - a cost: red, two decimals to match DeltaBox and DecisionLog, which
    //     report the same number. Two decimals is also what COST_EPS assumes:
    //     at one decimal, a cost of -0.006 clears the gate and prints "0.0 pts".
    // Hidden (not merely emptied) when absent, so attachTooltip's tabindex does
    // not leave a focusable, help-cursored blank strip in the tab order.
    const showStatus = vm.advice !== null;
    const optimal = showStatus && vm.cost === null;
    costLine.textContent = !showStatus
      ? ''
      : optimal
        ? 'Optimal choice'
        : `Costs you ${Math.abs(vm.cost!).toFixed(2)} pts vs optimal`;
    costLine.style.visibility = showStatus ? 'visible' : 'hidden';
    costLine.classList.toggle('dice-cost--optimal', optimal);
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
