/**
 * Progress bar: N/total counter with fill animation.
 */
import { subscribe, getState } from '../store.js';

export function initProgressBar(container) {
  const el = document.createElement('div');
  el.className = 'profile-progress';
  container.appendChild(el);

  const track = document.createElement('div');
  track.className = 'profile-progress-track';
  const fill = document.createElement('div');
  fill.className = 'profile-progress-fill';
  track.appendChild(fill);

  const label = document.createElement('div');
  label.className = 'profile-progress-label';
  label.textContent = '0 / 0';

  el.append(track, label);

  function render(state) {
    const total = state.scenarios.length || 1;
    const done = state.answers.length;
    const pct = Math.min(100, (done / total) * 100);

    fill.style.width = `${pct}%`;
    label.textContent = `${done} / ${total}`;

    el.style.visibility = state.phase === 'answering' ? 'visible' : 'hidden';
  }

  render(getState());
  subscribe((state) => render(state));
}
