import { getState, dispatch, subscribe } from '../store.ts';
import { createDie, type DieElements } from './Die.ts';

/** Render 5 interactive dice with keep/toggle and optimal-keep hints. */
export function initDiceBar(container: HTMLElement): void {
  container.className = 'dice-bar';

  const dice: DieElements[] = [];
  for (let i = 0; i < 5; i++) {
    const idx = i;
    const die = createDie(
      () => dispatch({ type: 'TOGGLE_DIE', index: idx }),
      () => {
        const s = getState();
        const active = s.turnPhase === 'rolled';
        const v = active ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v >= 6 || v === 0 ? 1 : v + 1 });
      },
      () => {
        const s = getState();
        const active = s.turnPhase === 'rolled';
        const v = active ? s.dice[idx].value : 0;
        dispatch({ type: 'SET_DIE_VALUE', index: idx, value: v <= 1 || v === 0 ? 6 : v - 1 });
      },
    );
    dice.push(die);
    container.appendChild(die.container);
  }

  function render() {
    const s = getState();
    const active = s.turnPhase === 'rolled';
    const hasEval = s.lastEvalResponse !== null;
    const canToggle = active && hasEval && s.rerollsRemaining > 0;

    const optimalMask = s.lastEvalResponse?.optimal_mask ?? null;
    const showMaskHints = s.showHints && active && hasEval && optimalMask !== null && s.rerollsRemaining > 0;

    for (let i = 0; i < 5; i++) {
      const v = active ? s.dice[i].value : 0;
      dice[i].update({
        value: v,
        kept: active ? s.dice[i].kept : true,
        isOptimalReroll: showMaskHints && !!(optimalMask! & (1 << i)),
        isOptimalKeep: showMaskHints && !(optimalMask! & (1 << i)),
        disabled: !canToggle,
        faded: !active,
      });
    }
  }

  render();
  subscribe((state, prev) => {
    if (state.dice === prev.dice &&
        state.lastEvalResponse === prev.lastEvalResponse &&
        state.showHints === prev.showHints &&
        state.rerollsRemaining === prev.rerollsRemaining &&
        state.turnPhase === prev.turnPhase) return;
    render();
  });
}
