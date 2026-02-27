/**
 * Result panel: natural-language profile summary shown after quiz completion.
 */
import { subscribe, getState } from '../store.js';
import { describeProfile } from '../estimator.js';

export function initResultPanel(container) {
  const el = document.createElement('div');
  el.className = 'profile-result';
  container.appendChild(el);

  function render(state) {
    const allAnswered = state.scenarios.length > 0 && state.answers.length >= state.scenarios.length;
    if (!allAnswered || !state.profile) {
      el.style.visibility = 'hidden';
      el.innerHTML = '<div class="profile-result-placeholder"></div>';
      return;
    }

    el.style.visibility = 'visible';
    const p = state.profile;
    const desc = describeProfile(p);

    // Count correct answers
    const correct = state.answers.filter(ans => {
      const s = state.scenarios.find(sc => sc.id === ans.scenarioId);
      return s && ans.actionId === s.optimal_action_id;
    }).length;

    const pct = Math.round((correct / state.answers.length) * 100);

    el.innerHTML = `
      <h2 class="profile-result-title">Your Yatzy Profile</h2>

      <div class="profile-result-score">
        <span class="profile-result-correct">${correct}/${state.answers.length}</span>
        <span class="profile-result-pct">(${pct}% optimal)</span>
      </div>

      <div class="profile-result-params">
        <div class="profile-result-param">
          <span class="profile-result-param-label">Risk</span>
          <span class="profile-result-param-value">${p.theta.toFixed(3)}</span>
        </div>
        <div class="profile-result-param">
          <span class="profile-result-param-label">Precision</span>
          <span class="profile-result-param-value">${p.beta.toFixed(1)}</span>
        </div>
        <div class="profile-result-param">
          <span class="profile-result-param-label">Horizon</span>
          <span class="profile-result-param-value">${p.gamma.toFixed(2)}</span>
        </div>
        <div class="profile-result-param">
          <span class="profile-result-param-label">Depth</span>
          <span class="profile-result-param-value">${p.d === 999 ? 'Optimal' : p.d}</span>
        </div>
      </div>

      <div class="profile-result-description">${formatMarkdown(desc)}</div>

      <button class="game-btn-primary profile-retry-btn" id="profile-retry">Try Again</button>
    `;

    el.querySelector('#profile-retry')?.addEventListener('click', () => {
      import('../store.js').then(m => m.dispatch({ type: 'START_QUIZ' }));
    });
  }

  render(getState());
  subscribe((state) => render(state));
}

function formatMarkdown(text) {
  return text
    .split('\n\n')
    .map(p => `<p>${p.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</p>`)
    .join('');
}
