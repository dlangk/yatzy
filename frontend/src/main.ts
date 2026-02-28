import { initStore, getState, dispatch, subscribe } from './store.ts';
import { initApp } from './App.ts';
import { evaluate, getStateValue, fetchDensity } from './api.ts';
import { sortDiceWithMapping } from './mask.ts';

initStore();

const root = document.getElementById('root')!;
initApp(root);

// Side effect: auto-evaluate after roll/reroll
subscribe((state, prev) => {
  if (state.turnPhase === 'rolled' && !state.lastEvalResponse) {
    const allValid = state.dice.every(d => d.value >= 1 && d.value <= 6);
    if (allValid) {
      const { sorted, sortMap } = sortDiceWithMapping(state.dice.map(d => d.value));
      evaluate({
        dice: sorted,
        upper_score: state.upperScore,
        scored_categories: state.scoredCategories,
        rerolls_remaining: state.rerollsRemaining,
      })
        .then(response => dispatch({ type: 'SET_EVAL_RESPONSE', response, sortMap }))
        .catch(err => console.error('Evaluate failed:', err));
    }
  }

  // Side effect: density fetch after new score trajectory point
  const traj = state.trajectory;
  if (traj.length > prev.trajectory.length) {
    const latest = traj[traj.length - 1];
    if (latest.event === 'score' && !latest.percentiles) {
      if (latest.turn < 15) {
        const rawScoredSum = state.categories.reduce((sum, c) => c.isScored ? sum + c.score : sum, 0);
        fetchDensity(state.upperScore, state.scoredCategories, rawScoredSum)
          .then(res => dispatch({ type: 'SET_DENSITY_RESULT', index: latest.index, percentiles: res.percentiles }))
          .catch(() => { /* density endpoint not available */ });
      } else {
        const finalScore = state.totalScore;
        const pctiles: Record<string, number> = {};
        for (const k of ['p1','p5','p10','p25','p50','p75','p90','p95','p99']) {
          pctiles[k] = finalScore;
        }
        dispatch({ type: 'SET_DENSITY_RESULT', index: latest.index, percentiles: pctiles });
      }
    }
  }
});

// Side effect: fetch initial EV and density at turn 0
{
  const s = getState();
  if (s.trajectory.length === 0 && s.turnPhase === 'idle') {
    getStateValue(s.upperScore, s.scoredCategories)
      .then(res => {
        dispatch({ type: 'SET_INITIAL_EV', ev: res.expected_final_score });
      })
      .catch(() => { /* server not available */ });

    // Fetch density for the initial state (turn 0)
    fetchDensity(s.upperScore, s.scoredCategories, 0)
      .then(res => {
        // Dispatch as the first trajectory point (index 0) if it exists
        const state = getState();
        const traj = state.trajectory;
        if (traj.length > 0 && !traj[0].percentiles) {
          dispatch({ type: 'SET_DENSITY_RESULT', index: traj[0].index, percentiles: res.percentiles });
        }
      })
      .catch(() => { /* density endpoint not available */ });
  }
}
