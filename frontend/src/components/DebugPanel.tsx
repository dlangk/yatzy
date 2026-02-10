import type { GameState } from '../types.ts';

interface DebugPanelProps {
  state: GameState;
}

export function DebugPanel({ state }: DebugPanelProps) {
  return (
    <div style={{
      background: '#1e1e1e',
      color: '#d4d4d4',
      borderRadius: 6,
      padding: 12,
      marginTop: 12,
      fontSize: 11,
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-all',
      maxHeight: 400,
      overflow: 'auto',
    }}>
      <div style={{ marginBottom: 8, color: '#569cd6', fontWeight: 'bold' }}>Game State</div>
      <div>{JSON.stringify({
        dice: state.dice,
        upperScore: state.upperScore,
        scoredCategories: state.scoredCategories,
        rerollsRemaining: state.rerollsRemaining,
        totalScore: state.totalScore,
        bonus: state.bonus,
        turnPhase: state.turnPhase,
        sortMap: state.sortMap,
      }, null, 2)}</div>

      {state.lastEvalResponse && (
        <>
          <div style={{ marginTop: 12, marginBottom: 8, color: '#569cd6', fontWeight: 'bold' }}>
            Last Eval Response
          </div>
          <div>{JSON.stringify(state.lastEvalResponse, null, 2)}</div>
        </>
      )}
    </div>
  );
}
