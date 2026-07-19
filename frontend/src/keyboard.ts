import { getState, dispatch } from './store.ts';
import { TOTAL_DICE } from './constants.ts';

/**
 * Keyboard shortcuts for fast desktop play:
 *   1-5    toggle hold/reroll for that die (press again to flip it back)
 *   Space  the primary action: roll, reroll, or new game (mirrors the main button)
 *
 * Scoring stays mouse-only by design. Shortcuts are ignored while typing in a
 * field or with a modifier held, so they never fight text entry or browser
 * shortcuts. Space also blurs a focused button and preventDefaults, so it does
 * not scroll the page or double-fire a button's own activation.
 */
export function initKeyboard(): void {
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target as HTMLElement | null;
    const tag = target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) return;

    const s = getState();

    // 1-5: toggle a die, only while a reroll is still possible (mirrors the
    // dice buttons, which disable when no rerolls remain).
    const n = Number(e.key);
    if (e.key.length === 1 && n >= 1 && n <= TOTAL_DICE) {
      if (s.turnPhase === 'rolled' && s.rerollsRemaining > 0) {
        e.preventDefault();
        dispatch({ type: 'TOGGLE_DIE', index: n - 1 });
      }
      return;
    }

    // Space: the primary action, mirroring the main button's states.
    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      const active = document.activeElement;
      if (active instanceof HTMLElement && active.tagName === 'BUTTON') active.blur();

      if (s.turnPhase === 'game_over') dispatch({ type: 'RESET_GAME' });
      else if (s.turnPhase === 'idle') dispatch({ type: 'ROLL' });
      else if (s.rerollsRemaining > 0) dispatch({ type: 'REROLL' });
    }
  });
}
