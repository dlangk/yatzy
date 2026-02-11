import type { CategoryState } from '../types.ts';

interface ScorecardRowProps {
  category: CategoryState;
  isOptimal: boolean;
  canScore: boolean;
  onScore: () => void;
  onSetScore: (score: number) => void;
  onUnsetCategory: () => void;
}

const cellStyle: React.CSSProperties = {
  padding: '4px 8px',
  borderBottom: '1px solid #ddd',
  height: 32,
  boxSizing: 'border-box',
};

export function ScorecardRow({ category, isOptimal, canScore, onScore, onSetScore, onUnsetCategory }: ScorecardRowProps) {
  const bg = category.isScored ? '#f0f0f0' : isOptimal ? '#d4edda' : 'transparent';
  const showAction = canScore && !category.isScored && category.available;

  return (
    <tr style={{ background: bg }}>
      <td style={cellStyle}>
        {category.name}
      </td>
      <td style={{ ...cellStyle, textAlign: 'center' }}>
        <input
          type="number"
          value={category.isScored ? category.score : (category.suggestedScore ?? 0)}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            if (!isNaN(v) && v >= 0) onSetScore(v);
          }}
          style={{
            width: '100%',
            border: 'none',
            background: 'transparent',
            textAlign: 'center',
            fontFamily: 'monospace',
            fontSize: 14,
            padding: 0,
            height: 24,
            MozAppearance: 'textfield',
          }}
        />
      </td>
      <td style={{ ...cellStyle, textAlign: 'center', fontSize: 12 }}>
        {!category.isScored && category.available ? category.evIfScored.toFixed(1) : ''}
      </td>
      <td style={{ ...cellStyle, textAlign: 'center' }}>
        <button
          onClick={category.isScored ? onUnsetCategory : showAction ? onScore : undefined}
          disabled={!showAction && !category.isScored}
          style={{
            fontSize: 12,
            padding: '2px 8px',
            border: category.isScored ? 'none' : undefined,
            background: category.isScored ? 'transparent' : undefined,
            cursor: showAction || category.isScored ? 'pointer' : 'default',
            visibility: showAction || category.isScored ? 'visible' : 'hidden',
          }}
          title={category.isScored ? 'Click to un-score' : undefined}
        >
          {category.isScored ? '\u2713' : 'Score'}
        </button>
      </td>
    </tr>
  );
}
