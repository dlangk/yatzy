import { getState, dispatch, subscribe } from '../store.ts';
import { unmapMask } from '../mask.ts';
import { createDie, type DieElements } from './Die.ts';

/** Render 5 interactive dice with hold/toggle and optimal-keep hints. */
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

    let optimalMask: number | null = null;
    if (s.lastEvalResponse?.optimal_mask != null && s.sortMap) {
      optimalMask = unmapMask(s.lastEvalResponse.optimal_mask, s.sortMap);
    }
    const showMaskHints = s.showHints && active && hasEval && optimalMask !== null && s.rerollsRemaining > 0;

    for (let i = 0; i < 5; i++) {
      const v = active ? s.dice[i].value : 0;
      dice[i].update({
        value: v,
        held: active ? s.dice[i].held : true,
        isOptimalReroll: showMaskHints && !!(optimalMask! & (1 << i)),
        isOptimalKeep: showMaskHints && !(optimalMask! & (1 << i)),
        disabled: !canToggle,
        faded: !active,
      });
    }
  }

  render();
  subscribe(render);
}
