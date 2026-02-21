import { subscribe, getState, dispatch } from './store.js';
import { evaluate, density, stateValue } from './api.js';
import { sortDiceWithMapping, computeRerollMask, mapMaskToSorted } from './mask.js';
import { clearLegacyKeys } from './reducer.js';
import { initActionBar } from './components/action-bar.js';
import { initDiceBar } from './components/dice-bar.js';
import { initDiceLegend } from './components/dice-legend.js';
import { initEvalPanel } from './components/eval-panel.js';
import { initScorecard } from './components/scorecard.js';
import { initDebugPanel } from './components/debug-panel.js';
import { initTrajectoryChart } from './components/trajectory-chart.js';
import { initDecisionLog } from './components/decision-log.js';

// Clean up old per-component localStorage keys
clearLegacyKeys();

// --- Helpers ---

function scoredCountFromMask(mask) {
  let n = 0, m = mask;
  while (m) { n += m & 1; m >>= 1; }
  return n;
}

function lastDensity(history) {
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].density && (history[i].type === 'start' || history[i].type === 'score'))
      return history[i].density;
  }
  return null;
}

// Monotonic game counter — stale async responses are discarded
let gameId = 0;
let densityAvailable = true;

async function fetchDensityFor(upperScore, scoredCategories, accumulatedScore, historyIndex, gid) {
  try {
    const resp = await density({
      upper_score: upperScore,
      scored_categories: scoredCategories,
      accumulated_score: accumulatedScore,
    });
    if (gid !== gameId) return;
    dispatch({ type: 'PATCH_HISTORY', index: historyIndex, patch: { density: resp.percentiles } });
    // Propagate to subsequent roll/reroll entries that share this state
    const state = getState();
    for (let i = historyIndex + 1; i < state.history.length; i++) {
      const e = state.history[i];
      if (e.type === 'roll' || e.type === 'reroll') {
        dispatch({ type: 'PATCH_HISTORY', index: i, patch: { density: resp.percentiles } });
      } else {
        break; // hit next score/start — different state
      }
    }
  } catch {
    densityAvailable = false;
  }
}

async function fetchStateEv(upperScore, scoredCategories) {
  try {
    const resp = await stateValue(upperScore, scoredCategories);
    return resp.expected_final_score;
  } catch {
    return null;
  }
}

// --- Auto-evaluate after roll/reroll/manual edits ---

async function callEvaluate(state) {
  const { sorted, sortMap } = sortDiceWithMapping(state.dice.map(d => d.value));
  try {
    const response = await evaluate({
      dice: sorted,
      upper_score: state.upperScore,
      scored_categories: state.scoredCategories,
      rerolls_remaining: state.rerollsRemaining,
    });
    dispatch({ type: 'SET_EVAL_RESPONSE', response, sortMap });
  } catch (err) {
    console.error('Evaluate failed:', err);
  }
}

subscribe((state) => {
  if (state.turnPhase === 'rolled' && !state.lastEvalResponse) {
    const allValid = state.dice.every(d => d.value >= 1 && d.value <= 6);
    if (allValid) {
      callEvaluate(state);
    }
  }
});

// --- Event-capture subscriber: builds history from actions ---

subscribe((state, prev, action) => {
  if (!action) return;

  switch (action.type) {
    case 'REROLL': {
      const resp = prev.lastEvalResponse;
      if (!resp || !resp.mask_evs || !prev.sortMap) return;

      const originalMask = computeRerollMask(prev.dice.map(d => d.held));
      const sortedMask = mapMaskToSorted(originalMask, prev.sortMap);
      const playerMaskEv = resp.mask_evs[sortedMask] ?? null;
      const optimalMaskEv = resp.optimal_mask_ev ?? null;
      if (playerMaskEv === null || optimalMaskEv === null) return;

      const delta = playerMaskEv - optimalMaskEv;
      const kept = prev.dice.filter(d => d.held).map(d => d.value).sort((a, b) => a - b);
      const label = kept.length > 0 ? 'Kept ' + kept.join(',') : 'Reroll all';
      const turn = scoredCountFromMask(state.scoredCategories);
      const inherited = lastDensity(state.history);

      dispatch({
        type: 'PUSH_HISTORY',
        entry: {
          type: 'reroll',
          turn,
          label,
          delta,
          stateEv: null,
          expectedFinal: null,
          accumulatedScore: state.totalScore,
          density: inherited,
        },
      });
      break;
    }

    case 'SET_EVAL_RESPONSE': {
      if (!state.lastEvalResponse) return;
      const ev = state.lastEvalResponse.state_ev;
      if (ev == null) return;

      const turn = scoredCountFromMask(state.scoredCategories);
      const expectedFinal = state.totalScore + ev;
      const inherited = lastDensity(state.history);

      // Check if last history entry is a reroll with null stateEv — backfill it
      const hist = state.history;
      if (hist.length > 0 && hist[hist.length - 1].type === 'reroll' && hist[hist.length - 1].stateEv === null) {
        dispatch({
          type: 'PATCH_HISTORY',
          index: hist.length - 1,
          patch: { stateEv: ev, expectedFinal },
        });
      } else {
        // Roll or other — push a new trajectory point
        const eventType = state.rerollsRemaining === 2 ? 'roll' : 'reroll';
        dispatch({
          type: 'PUSH_HISTORY',
          entry: {
            type: eventType,
            turn,
            label: null,
            delta: null,
            stateEv: ev,
            expectedFinal,
            accumulatedScore: state.totalScore,
            density: inherited,
          },
        });
      }
      break;
    }

    case 'SCORE_CATEGORY':
    case 'SET_CATEGORY_SCORE': {
      const resp = prev.lastEvalResponse;
      const turn = scoredCountFromMask(state.scoredCategories);
      const gid = gameId;

      // Decision log delta (only for SCORE_CATEGORY with eval data)
      let delta = null;
      let label = null;
      if (action.type === 'SCORE_CATEGORY' && resp) {
        const catId = action.categoryId;
        const playerCatEv = prev.categories[catId]?.evIfScored ?? null;
        const optimalCatEv = resp.optimal_category_ev ?? null;
        if (playerCatEv !== null && optimalCatEv !== null) {
          delta = playerCatEv - optimalCatEv;
        }
        label = prev.categories[catId]?.name ?? 'Category ' + catId;
      }

      if (state.turnPhase === 'game_over') {
        const finalPercentiles = {};
        for (const k of ['p1','p5','p10','p25','p50','p75','p90','p95','p99']) {
          finalPercentiles[k] = state.totalScore;
        }
        dispatch({
          type: 'PUSH_HISTORY',
          entry: {
            type: 'game_over',
            turn,
            label,
            delta,
            stateEv: 0,
            expectedFinal: state.totalScore,
            accumulatedScore: state.totalScore,
            density: finalPercentiles,
          },
        });
      } else {
        // Mid-game score: push entry, then async fetch stateEv + density
        const histIndex = state.history.length; // index of the entry we're about to push
        dispatch({
          type: 'PUSH_HISTORY',
          entry: {
            type: 'score',
            turn,
            label,
            delta,
            stateEv: null,
            expectedFinal: null,
            accumulatedScore: state.totalScore,
            density: null,
          },
        });

        (async () => {
          const ev = await fetchStateEv(state.upperScore, state.scoredCategories);
          if (ev == null || gid !== gameId) return;
          dispatch({
            type: 'PATCH_HISTORY',
            index: histIndex,
            patch: { stateEv: ev, expectedFinal: state.totalScore + ev },
          });
          if (densityAvailable) {
            fetchDensityFor(state.upperScore, state.scoredCategories, state.totalScore, histIndex, gid);
          }
        })();
      }
      break;
    }

    case 'RESET_GAME': {
      gameId++;
      densityAvailable = true;
      // Seed start entry for the fresh game
      seedStart(state);
      break;
    }
  }
});

async function seedStart(state) {
  const gid = gameId;
  const turn = scoredCountFromMask(state.scoredCategories);
  const ev = await fetchStateEv(state.upperScore, state.scoredCategories);
  if (ev == null || gid !== gameId) return;

  const histIndex = getState().history.length;
  dispatch({
    type: 'PUSH_HISTORY',
    entry: {
      type: 'start',
      turn,
      label: null,
      delta: null,
      stateEv: ev,
      expectedFinal: state.totalScore + ev,
      accumulatedScore: state.totalScore,
      density: null,
    },
  });

  if (densityAvailable) {
    fetchDensityFor(state.upperScore, state.scoredCategories, state.totalScore, histIndex, gid);
  }
}

// Seed initial start entry if history is empty and game is in progress (or fresh)
{
  const state = getState();
  if (state.history.length === 0) {
    seedStart(state);
  }
}

// --- Build the UI — 2×2 grid layout ---

const scorecard = document.getElementById('game-scorecard');
const controls = document.getElementById('game-controls');
const charts = document.getElementById('game-charts');
const debug = document.getElementById('game-debug');

if (controls) {
  initActionBar(controls);
  initDiceBar(controls);
  initDiceLegend(controls);
  initEvalPanel(controls);
}

if (scorecard) {
  initScorecard(scorecard);
}

if (debug) {
  initDebugPanel(debug);
}

if (charts) {
  initTrajectoryChart(charts);
}

const log = document.getElementById('game-log');
if (log) {
  initDecisionLog(log);
}
