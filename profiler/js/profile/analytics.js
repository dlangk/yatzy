// Profiler analytics. Fires GA4 events via window.yatzyTrack (set up by
// shared/nav.js, which no-ops off production). Subscribed to the profile
// store so each meaningful transition maps to one event.
import { subscribe } from './store.js';

export function initProfileAnalytics() {
  subscribe((state, prev, action) => {
    const track = window.yatzyTrack;
    if (!track) return;

    if (action.type === 'START_QUIZ') {
      track('quiz_start');
      return;
    }

    if (action.type === 'ADVANCE') {
      // Advancing commits the current scenario's answer.
      if (prev.phase === 'answering') {
        track('scenario_answered', { index: prev.currentIndex });
      }
      // Final advance flips the quiz to complete.
      if (state.phase === 'complete' && prev.phase !== 'complete') {
        const params = { answers: state.answers.length };
        const p = state.profile;
        if (p) { params.theta = p.theta; params.beta = p.beta; params.gamma = p.gamma; params.d = p.d; }
        track('quiz_complete', params);
      }
    }
  });
}
