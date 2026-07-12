/**
 * "How to play" intro box shown above the action buttons.
 *
 * Explains how to start a game and what the controls (Show Hints, Auto, New
 * Game) do. Dismissible; the choice is remembered in localStorage so returning
 * players are not nagged. When dismissed it collapses to a small "How to play"
 * button that brings it back.
 */
const KEY = 'yatzy_howto_dismissed';

export function initHowTo(container: HTMLElement): void {
  container.className = 'howto-wrap';

  const box = document.createElement('div');
  box.className = 'howto';
  const body = document.createElement('div');
  body.className = 'howto-body';
  body.innerHTML =
    '<strong>New here?</strong> Roll the dice, click dice to keep them, and reroll up to ' +
    'twice before scoring a category. Fill all 15 categories to finish. Turn on ' +
    '<strong>Show Hints</strong> to see the optimal dice to keep (green) and reroll (red) and ' +
    'the best category to score. <strong>Auto</strong> plays the optimal strategy for you, one ' +
    'step at a time, and <strong>New Game</strong> starts over. Hover any underlined term to see ' +
    'what it means.';
  box.appendChild(body);

  const dismiss = document.createElement('button');
  dismiss.className = 'howto-dismiss';
  dismiss.textContent = '×';
  dismiss.setAttribute('aria-label', 'Hide how-to');
  box.appendChild(dismiss);

  const restore = document.createElement('button');
  restore.className = 'howto-restore';
  restore.textContent = '? How to play';

  container.appendChild(box);
  container.appendChild(restore);

  function apply(dismissed: boolean): void {
    box.style.display = dismissed ? 'none' : '';
    restore.style.display = dismissed ? '' : 'none';
  }

  dismiss.addEventListener('click', () => {
    try {
      localStorage.setItem(KEY, '1');
    } catch { /* ignore storage errors */ }
    apply(true);
  });
  restore.addEventListener('click', () => {
    try {
      localStorage.removeItem(KEY);
    } catch { /* ignore storage errors */ }
    apply(false);
  });

  let dismissed = false;
  try {
    dismissed = localStorage.getItem(KEY) === '1';
  } catch { /* ignore */ }
  apply(dismissed);
}
