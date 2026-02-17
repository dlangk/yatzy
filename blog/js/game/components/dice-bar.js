import { subscribe, getState } from '../store.js';
import { createDie } from './die.js';
import { computeRerollMask, mapMaskToSorted, unmapMask } from '../mask.js';

export function initDiceBar(container) {
  const row = document.createElement('div');
  row.className = 'game-dice-row';
  container.appendChild(row);

  const dice = [];
  for (let i = 0; i < 5; i++) {
    const die = createDie(i);
    row.appendChild(die.el);
    dice.push(die);
  }

  function getOptimalMaskOriginal(state) {
    if (state.lastEvalResponse?.optimal_mask == null || !state.sortMap) return null;
    return unmapMask(state.lastEvalResponse.optimal_mask, state.sortMap);
  }

  function render(state) {
    const active = state.turnPhase === 'rolled';
    const hasEval = state.lastEvalResponse !== null;
    const canToggle = active && hasEval && state.rerollsRemaining > 0;
    const optimalMask = getOptimalMaskOriginal(state);
    const showMaskHints = active && hasEval && optimalMask !== null && state.rerollsRemaining > 0;

    for (let i = 0; i < 5; i++) {
      dice[i].update(state.dice[i], active, canToggle, showMaskHints, optimalMask);
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
