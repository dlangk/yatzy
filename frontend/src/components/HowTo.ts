/**
 * "How to play" intro box shown above the action buttons.
 *
 * Explains how to start a game and what the controls (Show Hints, Auto, New
 * Game) do. Its open/closed state is remembered in localStorage (independent of
 * the game state, so it survives New Game). The box is toggled by the "Guide"
 * button in the action row via the returned controller; the box's own × closes
 * it too.
 */
const KEY = 'yatzy_howto_dismissed';

export interface GuideController {
  toggle(): void;
  isOpen(): boolean;
  /** Register a listener; called immediately with the current state. */
  subscribe(cb: (open: boolean) => void): void;
}

export function initHowTo(container: HTMLElement): GuideController {
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
  box.appendChild(dismiss);

  container.appendChild(box);

  const listeners: ((open: boolean) => void)[] = [];

  let open = true;
  try {
    open = localStorage.getItem(KEY) !== '1';
  } catch { /* ignore */ }

  function apply(): void {
    container.style.display = open ? '' : 'none';
    for (const l of listeners) l(open);
  }

  function setOpen(v: boolean): void {
    open = v;
    try {
      if (open) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, '1');
    } catch { /* ignore */ }
    apply();
  }

  dismiss.addEventListener('click', () => setOpen(false));
  apply();

  return {
    toggle: () => setOpen(!open),
    isOpen: () => open,
    subscribe: (cb) => {
      listeners.push(cb);
      cb(open);
    },
  };
}
