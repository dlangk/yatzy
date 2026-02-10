import type { CategoryState, TurnPhase } from '../types.ts';

interface EvalPanelProps {
  currentMaskEv: number | null;
  optimalMaskEv: number | null;
  optimalCategory: number | null;
  optimalCategoryEv: number | null;
  stateEv: number | null;
  rerollsRemaining: number;
  categories: CategoryState[];
  turnPhase: TurnPhase;
}

export function EvalPanel({
  currentMaskEv,
  optimalMaskEv,
  optimalCategory,
  optimalCategoryEv,
  stateEv,
  rerollsRemaining,
  categories,
  turnPhase,
}: EvalPanelProps) {
  if (turnPhase !== 'rolled' || stateEv === null) return null;

  const optCatName = optimalCategory !== null && optimalCategory >= 0
    ? categories[optimalCategory]?.name ?? '?'
    : null;

  return (
    <div style={{
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: 6,
      padding: 10,
      margin: '12px 0',
      fontFamily: 'monospace',
      fontSize: 13,
    }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        <span>State EV:</span>
        <span style={{ textAlign: 'right' }}>{stateEv.toFixed(2)}</span>

        {rerollsRemaining > 0 && currentMaskEv !== null && (
          <>
            <span>Your mask EV:</span>
            <span style={{ textAlign: 'right' }}>{currentMaskEv.toFixed(2)}</span>
          </>
        )}

        {rerollsRemaining > 0 && optimalMaskEv !== null && (
          <>
            <span>Best mask EV:</span>
            <span style={{ textAlign: 'right' }}>{optimalMaskEv.toFixed(2)}</span>
          </>
        )}

        {rerollsRemaining > 0 && currentMaskEv !== null && optimalMaskEv !== null && (
          <>
            <span>Delta:</span>
            <span style={{
              textAlign: 'right',
              color: Math.abs(currentMaskEv - optimalMaskEv) < 0.01 ? '#28a745' : '#dc3545',
            }}>
              {(currentMaskEv - optimalMaskEv).toFixed(2)}
            </span>
          </>
        )}

        {rerollsRemaining === 0 && optCatName && (
          <>
            <span>Best category:</span>
            <span style={{ textAlign: 'right' }}>{optCatName}</span>
            <span>Category EV:</span>
            <span style={{ textAlign: 'right' }}>{optimalCategoryEv?.toFixed(2)}</span>
          </>
        )}
      </div>
    </div>
  );
}
