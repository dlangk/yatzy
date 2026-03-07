/**
 * Auto-play module.
 *
 * Subscribes to the store and automatically executes optimal moves
 * (roll, reroll with optimal mask, score optimal category) with a
 * configurable delay. Purely a side-effect loop — no reducer state.
 */
import { getState, dispatch, subscribe } from './store.ts';

let playing = false;
let delay = 500;
let pendingTimeout: number | null = null;

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

  pendingTimeout = window.setTimeout(executeStep, delay);
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
  input.min = '200';
  input.max = '3000';
  input.step = '100';
  input.value = String(delay);

  const label = document.createElement('span');
  label.className = 'autoplay-delay-label';
  label.textContent = 'ms';

  btn.addEventListener('click', () => {
    playing = !playing;
    updateButton();
    if (playing) {
      delay = Math.max(200, Math.min(3000, parseInt(input.value, 10) || 500));
      input.value = String(delay);
      scheduleNextMove();
    } else {
      cancelPending();
    }
  });

  input.addEventListener('change', () => {
    delay = Math.max(200, Math.min(3000, parseInt(input.value, 10) || 500));
    input.value = String(delay);
  });

  container.appendChild(btn);
  container.appendChild(input);
  container.appendChild(label);

  // React to every state change while playing
  subscribe(() => {
    if (playing) scheduleNextMove();
  });
}
