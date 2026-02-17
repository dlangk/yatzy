import { subscribe, getState, dispatch } from './store.js';
import { evaluate } from './api.js';
import { sortDiceWithMapping } from './mask.js';
import { initActionBar } from './components/action-bar.js';
import { initDiceBar } from './components/dice-bar.js';
import { initDiceLegend } from './components/dice-legend.js';
import { initEvalPanel } from './components/eval-panel.js';
import { initScorecard } from './components/scorecard.js';
import { initDebugPanel } from './components/debug-panel.js';

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

// Auto-evaluate after roll/reroll/manual edits
subscribe((state) => {
  if (state.turnPhase === 'rolled' && !state.lastEvalResponse) {
    const allValid = state.dice.every(d => d.value >= 1 && d.value <= 6);
    if (allValid) {
      callEvaluate(state);
    }
  }
});

// Build the UI
const root = document.getElementById('game-root');
if (root) {
  initActionBar(root);
  initDiceBar(root);
  initDiceLegend(root);
  initEvalPanel(root);
  initScorecard(root);
  initDebugPanel(root);
}
