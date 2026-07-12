/**
 * Auto-play module.
 *
 * Subscribes to the store and automatically executes optimal moves
 * (roll, reroll with optimal mask, score optimal category) with a
 * configurable delay. The delay is a user preference (state.prefs.autoplayDelay).
 */
import { getState, dispatch, subscribe } from './store.ts';

const MIN_DELAY = 200;
const MAX_DELAY = 3000;

let playing = false;
let pendingTimeout: number | null = null;

function clampDelay(v: number): number {
  return Math.max(MIN_DELAY, Math.min(MAX_DELAY, v || 500));
}

function cancelPending(): void {
  if (pendingTimeout !== null) {
    clearTimeout(pendingTimeout);
    pendingTimeout = null;
  }
}

function executeStep(): void {
  pendingTimeout = null;
  if (!playing) return;

  const s = getState();

  if (s.turnPhase === 'idle') {
    dispatch({ type: 'ROLL' });
    return;
  }

  if (s.turnPhase === 'game_over') {
    playing = false;
    updateButton();
    return;
  }

  // turnPhase === 'rolled'
  if (!s.lastEvalResponse) return; // wait for eval

  const ev = s.lastEvalResponse;
  const shouldReroll =
    s.rerollsRemaining > 0 &&
    ev.optimal_mask_ev !== undefined &&
    ev.optimal_mask_ev > ev.optimal_category_ev;

  if (shouldReroll && ev.optimal_mask !== undefined) {
    // Set keeps: bit=1 in rerollMask means reroll, so kept = !(bit set)
    for (let i = 0; i < s.dice.length; i++) {
      const shouldKeep = !(ev.optimal_mask & (1 << i));
      if (s.dice[i].kept !== shouldKeep) {
        dispatch({ type: 'TOGGLE_DIE', index: i });
      }
    }
    dispatch({ type: 'REROLL' });
  } else {
    dispatch({ type: 'SCORE_CATEGORY', categoryId: ev.optimal_category });
  }
}

function scheduleNextMove(): void {
  if (!playing || pendingTimeout !== null) return;

  const s = getState();

  // Nothing to do if waiting for eval response
  if (s.turnPhase === 'rolled' && !s.lastEvalResponse) return;

  // Game over — pause
  if (s.turnPhase === 'game_over') {
    playing = false;
    updateButton();
    return;
  }

  pendingTimeout = window.setTimeout(executeStep, getState().prefs.autoplayDelay);
}

let btnEl: HTMLButtonElement | null = null;

function updateButton(): void {
  if (btnEl) btnEl.textContent = playing ? 'Pause' : 'Auto';
}

/** Create auto-play controls and wire up the store subscription. */
export function initAutoPlay(container: HTMLElement): void {
  container.className = 'autoplay-bar';

  const btn = document.createElement('button');
  btn.className = 'game-btn-secondary';
  btn.style.width = '72px';
  btn.textContent = 'Auto';
  btnEl = btn;

  const input = document.createElement('input');
  input.type = 'number';
  input.className = 'autoplay-delay-input';
  input.min = String(MIN_DELAY);
  input.max = String(MAX_DELAY);
  input.step = '100';
  input.value = String(getState().prefs.autoplayDelay);

  const label = document.createElement('span');
  label.className = 'autoplay-delay-label';
  label.textContent = 'ms';

  function commitDelay(): void {
    const delay = clampDelay(parseInt(input.value, 10));
    input.value = String(delay);
    if (delay !== getState().prefs.autoplayDelay) {
      dispatch({ type: 'SET_AUTOPLAY_DELAY', delay });
    }
  }

  btn.addEventListener('click', () => {
    playing = !playing;
    updateButton();
    if (playing) {
      commitDelay();
      scheduleNextMove();
    } else {
      cancelPending();
    }
  });

  input.addEventListener('change', commitDelay);

  container.appendChild(btn);
  container.appendChild(input);
  container.appendChild(label);

  // React to every state change while playing
  subscribe(() => {
    if (playing) scheduleNextMove();
  });
}
