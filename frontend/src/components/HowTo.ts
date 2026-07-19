/**
 * "How to play" intro box shown above the action buttons.
 *
 * Explains how to start a game and what the controls (Optimal hints, Auto, Override mode) do.
 * Open/closed lives in prefs (state.prefs.guideOpen), so it is remembered across
 * sessions and survives New Game, and stays in sync with the "Intro guide" button in
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
    '<strong>Delta Yatzy is a game of skill, not luck.</strong> Instead of chasing a high ' +
    'score, you try to play every decision as close to optimal as possible. The box above the ' +
    'scorecard tracks your <strong>delta versus optimal play</strong>; the goal is to keep it at ' +
    'zero. Because it measures decisions rather than dice, luck cancels out.<br><br>' +
    '<strong>New here?</strong> Roll the dice, then click a die to hold or release it. Each die ' +
    'is labelled <strong>HOLD</strong> or <strong>REROLL</strong>, so you can always see what ' +
    'will happen. Reroll up to twice before scoring a category, and fill all 15 categories to ' +
    'finish. Turn on <strong>Optimal hints</strong> to show, below each die, what optimal play ' +
    'would do, plus the best category to score. <strong>Auto</strong> plays the optimal strategy ' +
    'for you, one step at a time. <strong>Override mode</strong> reveals manual controls for ' +
    'editing the dice, rerolls, and undo.<br><br>' +
    '<strong>Keyboard:</strong> press <strong>1</strong> to <strong>5</strong> to hold or ' +
    'release a die, and <strong>Space</strong> to roll and reroll. Scoring is done with the ' +
    'mouse. Hover any underlined term to see what it means.';
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
