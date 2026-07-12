/**
 * "How to play" intro box shown above the action buttons.
 *
 * Explains how to start a game and what the controls (Hints, Auto, New Game) do.
 * Open/closed lives in prefs (state.prefs.guideOpen), so it is remembered across
 * sessions and survives New Game, and stays in sync with the "Guide" button in
 * the action row. The box's own × closes it.
 */
import { getState, dispatch, subscribe } from '../store.ts';

export function initHowTo(container: HTMLElement): void {
  container.className = 'howto-wrap';

  const box = document.createElement('div');
  box.className = 'howto';
  const body = document.createElement('div');
  body.className = 'howto-body';
  body.innerHTML =
    '<strong>New here?</strong> Roll the dice, click dice to keep them, and reroll up to ' +
    'twice before scoring a category. Fill all 15 categories to finish. Turn on ' +
    '<strong>Hints</strong> to see the optimal dice to keep (green) and reroll (red) and ' +
    'the best category to score. <strong>Auto</strong> plays the optimal strategy for you, one ' +
    'step at a time, and <strong>New Game</strong> starts over. Hover any underlined term to see ' +
    'what it means.';
  box.appendChild(body);

  const dismiss = document.createElement('button');
  dismiss.className = 'howto-dismiss';
  dismiss.textContent = '×';
  dismiss.setAttribute('aria-label', 'Close the guide');
  dismiss.addEventListener('click', () => {
    if (getState().prefs.guideOpen) dispatch({ type: 'TOGGLE_GUIDE' });
  });
  box.appendChild(dismiss);

  container.appendChild(box);

  function render(): void {
    container.style.display = getState().prefs.guideOpen ? '' : 'none';
  }

  render();
  subscribe((state, prev) => {
    if (state.prefs.guideOpen === prev.prefs.guideOpen) return;
    render();
  });
}
