import { initStore, getState, dispatch, subscribe } from './store.ts';
import { initApp } from './App.ts';
import { evaluate, getStateValue, fetchDensity } from './api.ts';
import { sortDiceWithMapping } from './mask.ts';

initStore();

const root = document.getElementById('root')!;
initApp(root);

// Side effect: auto-evaluate after roll/reroll
subscribe((state) => {
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
});

// Side effect: backfill missing percentiles on any trajectory point
const pendingDensity = new Set<number>();
const PCTL_KEYS = ['p1','p5','p10','p25','p50','p75','p90','p95','p99'];

subscribe((state) => {
  for (const p of state.trajectory) {
    if (p.event !== 'start' && p.event !== 'score') continue;
    if (p.percentiles) continue;
    if (pendingDensity.has(p.index)) continue;

    if (p.turn >= 15) {
      const pctiles: Record<string, number> = {};
      for (const k of PCTL_KEYS) pctiles[k] = p.accumulatedScore;
      dispatch({ type: 'SET_DENSITY_RESULT', index: p.index, percentiles: pctiles });
      continue;
    }

    const up = p.upperScore ?? 0;
    const scored = p.scoredCategories ?? 0;
    const rawAcc = p.accumulatedScore - (up >= 63 ? 50 : 0);
    pendingDensity.add(p.index);
    fetchDensity(up, scored, rawAcc)
      .then(res => {
        pendingDensity.delete(p.index);
        dispatch({ type: 'SET_DENSITY_RESULT', index: p.index, percentiles: res.percentiles });
      })
      .catch(() => { pendingDensity.delete(p.index); });
  }
});

// Side effect: fetch initial EV at startup
{
  const s = getState();
  if (s.trajectory.length === 0 && s.turnPhase === 'idle') {
    getStateValue(s.upperScore, s.scoredCategories)
      .then(res => {
        dispatch({ type: 'SET_INITIAL_EV', ev: res.expected_final_score });
      })
      .catch(() => { /* server not available */ });
  }
}
