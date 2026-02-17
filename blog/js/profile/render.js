/**
 * Profile quiz orchestrator.
 * Loads scenarios, initializes components, runs estimation after each answer.
 */
import { subscribe, getState, dispatch } from './store.js';
import { loadScenarios } from './api.js';
import { estimateProfile } from './estimator.js';
import { initScenarioCard } from './components/scenario-card.js';
import { initProgressBar } from './components/progress-bar.js';
import { initParameterChart } from './components/parameter-chart.js';
import { initResultPanel } from './components/result-panel.js';

// Load scenarios on init
(async function init() {
  try {
    const scenarios = await loadScenarios();
    dispatch({ type: 'SCENARIOS_LOADED', scenarios });
  } catch (err) {
    console.error('Failed to load profiling scenarios:', err);
    dispatch({ type: 'LOAD_ERROR', error: err.message });
  }
})();

// Build UI
const root = document.getElementById('profile-root');
if (root) {
  // Intro panel
  const intro = document.createElement('div');
  intro.className = 'profile-intro';
  intro.id = 'profile-intro';
  root.appendChild(intro);

  // Quiz components (progress bar after scenario card = at bottom)
  initScenarioCard(root);
  initProgressBar(root);
  initParameterChart(root);
  initResultPanel(root);

  // Render intro/loading states
  function renderIntro(state) {
    if (state.phase === 'loading') {
      intro.style.display = 'block';
      intro.innerHTML = `
        <div class="profile-loading">Loading scenarios...</div>
      `;
    } else if (state.phase === 'intro') {
      intro.style.display = 'block';
      intro.innerHTML = `
        <div class="profile-intro-content">
          <h2>Profile Your Yatzy Style</h2>
          <p>We'll show you <strong>${state.scenarios.length} game situations</strong>.
             Pick what you'd do in each one. There's no time limit.</p>
          <p>After a few answers, we'll start estimating your strategic profile across
             four dimensions: risk attitude, decision precision, planning horizon, and
             strategic resolution.</p>
          <button class="game-btn-primary" id="start-quiz">Start Quiz</button>
        </div>
      `;
      intro.querySelector('#start-quiz')?.addEventListener('click', () => {
        dispatch({ type: 'START_QUIZ' });
      });
    } else {
      intro.style.display = 'none';
    }
  }

  renderIntro(getState());
  subscribe((state) => renderIntro(state));
}

// Run estimation after each answer (starting from answer #5)
subscribe((state, prev, action) => {
  if (action?.type !== 'ANSWER') return;
  if (state.answers.length < 1) return;

  // Run estimation asynchronously to avoid blocking UI
  requestAnimationFrame(() => {
    const profile = estimateProfile(state.scenarios, state.answers);
    if (profile) {
      dispatch({ type: 'UPDATE_PROFILE', profile });
    }
  });
});

// Re-run estimation on restore from localStorage (after scenarios load)
subscribe((state, prev, action) => {
  if (action?.type !== 'SCENARIOS_LOADED') return;
  if (state.answers.length >= 1 && !state.profile) {
    requestAnimationFrame(() => {
      const profile = estimateProfile(state.scenarios, state.answers);
      if (profile) {
        dispatch({ type: 'UPDATE_PROFILE', profile });
      }
    });
  }
});
