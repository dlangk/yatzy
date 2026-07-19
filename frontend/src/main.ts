// Load shared nav bar (absolute URL to bypass Vite base path rewriting)
const s = document.createElement('script');
s.type = 'module';
s.src = '/yatzy/shared/nav.js';
document.head.appendChild(s);

import { TOTAL_DICE } from './constants.ts';
import { getState, dispatch, subscribe } from './store.ts';
import { initApp } from './App.ts';
import { evaluate, getStateValue, fetchDensity } from './api.ts';
import { initPlayAnalytics } from './analytics.ts';
import { initKeyboard } from './keyboard.ts';
import { rerollMaskFromDice, maskCostVsOptimal } from './mask.ts';
import type { GameState } from './types.ts';

const root = document.getElementById('root')!;
initApp(root);
initKeyboard();
initPlayAnalytics();

// Side effect: auto-evaluate after roll/reroll.
// Captures trajectory event info from the action that triggered the eval,
// then passes it along with the SET_EVAL_RESPONSE action.
let pendingTrajectoryEvent: 'roll' | 'reroll' | undefined;
let pendingRerollLabel: string | undefined;
let pendingRerollDeltas: number[] | undefined;

subscribe((state, prev, action) => {
  // Capture trajectory context when a roll/reroll occurs
  if (action.type === 'ROLL') {
    pendingTrajectoryEvent = 'roll';
    pendingRerollLabel = undefined;
    pendingRerollDeltas = undefined;
  } else if (action.type === 'REROLL') {
    pendingTrajectoryEvent = 'reroll';
    // Compute label from pre-reroll state
    const keptValues = prev.dice.filter(d => d.kept).map(d => d.value).sort((a, b) => a - b);
    pendingRerollLabel = keptValues.length === TOTAL_DICE
      ? 'Kept all'
      : keptValues.length === 0
        ? 'Rerolled all'
        : `Kept ${keptValues.join(',')}`;
    // Compute cost at each detent from the pre-reroll eval
    pendingRerollDeltas = undefined;
    const mask = rerollMaskFromDice(prev.dice);
    if (prev.detents && prev.detents.length === 5) {
      pendingRerollDeltas = prev.detents.map((d) => maskCostVsOptimal(d, mask) ?? 0);
    } else {
      const cost = maskCostVsOptimal(prev.lastEvalResponse, mask);
      if (cost !== null) pendingRerollDeltas = Array(5).fill(cost);
    }
  }

  evaluateIfNeeded(state);
});

/**
 * Fetch the eval for a rolled hand that has none yet.
 *
 * Called from the store subscription and once at startup. The startup call
 * matters because lastEvalResponse is deliberately not persisted, so a reload
 * mid-roll restores a 'rolled' state with no eval. Without hints the dice are
 * disabled (they need an eval to know the optimal mask), so nothing the player
 * does to a die can dispatch the action that would fetch one. Mobile browsers
 * discard and reload backgrounded tabs, which makes that state easy to reach.
 */
function evaluateIfNeeded(state: GameState): void {
  if (state.turnPhase !== 'rolled' || state.lastEvalResponse) return;
  if (!state.dice.every(d => d.value >= 1 && d.value <= 6)) return;

  const trajEvent = pendingTrajectoryEvent;
  const trajLabel = pendingRerollLabel;
  const trajDeltas = pendingRerollDeltas;
  pendingTrajectoryEvent = undefined;
  pendingRerollLabel = undefined;
  pendingRerollDeltas = undefined;

  evaluate({
    dice: state.dice.map(d => d.value),
    upper_score: state.upperScore,
    scored_categories: state.scoredCategories,
    rerolls_remaining: state.rerollsRemaining,
  })
    .then(response => dispatch({
      type: 'SET_EVAL_RESPONSE',
      response,
      trajectoryEvent: trajEvent,
      rerollLabel: trajLabel,
      rerollDeltas: trajDeltas,
    }))
    .catch(err => console.error('Evaluate failed:', err));
}

// Restored mid-roll state needs an eval that no dispatch will ask for.
evaluateIfNeeded(getState());

// Side effect: fetch initial EV when trajectory is empty (startup + reset)
subscribe((state) => {
  if (state.trajectory.length === 0 && state.turnPhase === 'idle') {
    getStateValue(state.upperScore, state.scoredCategories)
      .then(res => {
        dispatch({ type: 'SET_INITIAL_EV', ev: res.expected_final_score });
      })
      .catch(() => { /* server not available */ });
  }
});

// Side effect: backfill missing percentiles on any trajectory point.
// Density requests are serialized to avoid overloading the server.
const pendingDensity = new Set<number>();
const PCTL_KEYS = ['p1','p5','p10','p25','p50','p75','p90','p95','p99'];
let densityQueueRunning = false;
const densityQueue: Array<{ index: number; up: number; scored: number; rawAcc: number }> = [];

async function drainDensityQueue() {
  if (densityQueueRunning) return;
  densityQueueRunning = true;
  while (densityQueue.length > 0) {
    const { index, up, scored, rawAcc } = densityQueue.shift()!;
    try {
      const res = await fetchDensity(up, scored, rawAcc);
      pendingDensity.delete(index);
      dispatch({ type: 'SET_DENSITY_RESULT', index, percentiles: res.percentiles });
    } catch {
      pendingDensity.delete(index);
    }
  }
  densityQueueRunning = false;
}

function backfillPercentiles(state: { trajectory: readonly import('./types.ts').TrajectoryPoint[] }) {
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
    densityQueue.push({ index: p.index, up, scored, rawAcc });
  }
  drainDensityQueue();
}

subscribe((state) => backfillPercentiles(state));

// Kick backfill on startup for restored games (subscribers don't fire on registration)
backfillPercentiles(getState());
