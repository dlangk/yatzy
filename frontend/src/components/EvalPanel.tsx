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
  const hasData = turnPhase === 'rolled' && stateEv !== null;

  const optCatName = optimalCategory !== null && optimalCategory >= 0
    ? categories[optimalCategory]?.name ?? '?'
    : null;

  const dash = '\u2014';

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
        <span style={{ textAlign: 'right' }}>{hasData ? stateEv!.toFixed(2) : dash}</span>

        <span>Your mask EV:</span>
        <span style={{ textAlign: 'right' }}>
          {hasData && rerollsRemaining > 0 && currentMaskEv !== null ? currentMaskEv.toFixed(2) : dash}
        </span>

        <span>Best mask EV:</span>
        <span style={{ textAlign: 'right' }}>
          {hasData && rerollsRemaining > 0 && optimalMaskEv !== null ? optimalMaskEv.toFixed(2) : dash}
        </span>

        <span>Delta:</span>
        <span style={{
          textAlign: 'right',
          color: hasData && rerollsRemaining > 0 && currentMaskEv !== null && optimalMaskEv !== null
            ? (Math.abs(currentMaskEv - optimalMaskEv) < 0.01 ? '#28a745' : '#dc3545')
            : 'inherit',
        }}>
          {hasData && rerollsRemaining > 0 && currentMaskEv !== null && optimalMaskEv !== null
            ? (currentMaskEv - optimalMaskEv).toFixed(2)
            : dash}
        </span>

        <span>Best category:</span>
        <span style={{ textAlign: 'right' }}>
          {hasData && optCatName ? optCatName : dash}
        </span>

        <span>Category EV:</span>
        <span style={{ textAlign: 'right' }}>
          {hasData && optimalCategoryEv !== null ? optimalCategoryEv.toFixed(2) : dash}
        </span>
      </div>
    </div>
  );
}
