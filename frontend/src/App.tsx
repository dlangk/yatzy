import { useReducer, useEffect, useCallback } from 'react';
import { gameReducer, initialState } from './reducer.ts';
import { evaluate } from './api.ts';
import { sortDiceWithMapping, computeRerollMask, mapMaskToSorted, unmapMask } from './mask.ts';
import type { GameState } from './types.ts';
import { Scorecard } from './components/Scorecard.tsx';
import { DiceBar } from './components/DiceBar.tsx';
import { ActionBar } from './components/ActionBar.tsx';
import { EvalPanel } from './components/EvalPanel.tsx';
import { DebugPanel } from './components/DebugPanel.tsx';

function getCurrentMaskEv(state: GameState): number | null {
  if (!state.lastEvalResponse?.mask_evs || !state.sortMap) return null;
  const originalMask = computeRerollMask(state.dice.map((d) => d.held));
  const sortedMask = mapMaskToSorted(originalMask, state.sortMap);
  return state.lastEvalResponse.mask_evs[sortedMask] ?? null;
}

function getOptimalMaskOriginal(state: GameState): number | null {
  if (state.lastEvalResponse?.optimal_mask == null || !state.sortMap) return null;
  return unmapMask(state.lastEvalResponse.optimal_mask, state.sortMap);
}

export function App() {
  const [state, dispatch] = useReducer(gameReducer, undefined, initialState);

  const callEvaluate = useCallback(async (s: GameState) => {
    const { sorted, sortMap } = sortDiceWithMapping(s.dice.map((d) => d.value));
    try {
      const response = await evaluate({
        dice: sorted,
        upper_score: s.upperScore,
        scored_categories: s.scoredCategories,
        rerolls_remaining: s.rerollsRemaining,
      });
      dispatch({ type: 'SET_EVAL_RESPONSE', response, sortMap });
    } catch (err) {
      console.error('Evaluate failed:', err);
    }
  }, []);

  // Auto-evaluate after ROLL or REROLL (when lastEvalResponse is cleared)
  useEffect(() => {
    if (state.turnPhase === 'rolled' && !state.lastEvalResponse) {
      callEvaluate(state);
    }
  }, [state.turnPhase, state.lastEvalResponse, callEvaluate]); // eslint-disable-line react-hooks/exhaustive-deps

  const currentMaskEv = getCurrentMaskEv(state);
  const optimalMask = getOptimalMaskOriginal(state);

  return (
    <div style={{ maxWidth: 600, margin: '0 auto', padding: 16, fontFamily: 'monospace' }}>
      <h1 style={{ textAlign: 'center', marginBottom: 8 }}>Delta Yatzy</h1>

      <ActionBar
        turnPhase={state.turnPhase}
        rerollsRemaining={state.rerollsRemaining}
        onRoll={() => dispatch({ type: 'ROLL' })}
        onReroll={() => dispatch({ type: 'REROLL' })}
        onReset={() => dispatch({ type: 'RESET_GAME' })}
      />

      <DiceBar
        dice={state.dice}
        onToggle={(i) => dispatch({ type: 'TOGGLE_DIE', index: i })}
        optimalMask={optimalMask}
        rerollsRemaining={state.rerollsRemaining}
        turnPhase={state.turnPhase}
        hasEval={state.lastEvalResponse !== null}
      />

      <EvalPanel
        currentMaskEv={currentMaskEv}
        optimalMaskEv={state.lastEvalResponse?.optimal_mask_ev ?? null}
        optimalCategory={state.lastEvalResponse?.optimal_category ?? null}
        optimalCategoryEv={state.lastEvalResponse?.optimal_category_ev ?? null}
        stateEv={state.lastEvalResponse?.state_ev ?? null}
        rerollsRemaining={state.rerollsRemaining}
        categories={state.categories}
        turnPhase={state.turnPhase}
      />

      <Scorecard
        categories={state.categories}
        upperScore={state.upperScore}
        bonus={state.bonus}
        totalScore={state.totalScore}
        turnPhase={state.turnPhase}
        onScoreCategory={(id) => dispatch({ type: 'SCORE_CATEGORY', categoryId: id })}
      />

      <div style={{ textAlign: 'center', marginTop: 12 }}>
        <button onClick={() => dispatch({ type: 'TOGGLE_DEBUG' })} style={{ fontSize: 12, opacity: 0.6 }}>
          {state.showDebug ? 'Hide Debug' : 'Show Debug'}
        </button>
      </div>

      {state.showDebug && (
        <DebugPanel state={state} />
      )}
    </div>
  );
}
