import type { CategoryState } from '../types.ts';

interface ScorecardRowProps {
  category: CategoryState;
  isOptimal: boolean;
  canScore: boolean;
  onScore: () => void;
}

const cellStyle: React.CSSProperties = {
  padding: '4px 8px',
  borderBottom: '1px solid #ddd',
  height: 32,
  boxSizing: 'border-box',
};

export function ScorecardRow({ category, isOptimal, canScore, onScore }: ScorecardRowProps) {
  const bg = category.isScored ? '#f0f0f0' : isOptimal ? '#d4edda' : 'transparent';
  const showAction = canScore && !category.isScored && category.available;

  return (
    <tr style={{ background: bg }}>
      <td style={cellStyle}>
        {category.name}
      </td>
      <td style={{ ...cellStyle, textAlign: 'center' }}>
        {category.isScored ? category.score : category.suggestedScore ?? '\u2014'}
      </td>
      <td style={{ ...cellStyle, textAlign: 'center', fontSize: 12 }}>
        {!category.isScored && category.available ? category.evIfScored.toFixed(1) : ''}
      </td>
      <td style={{ ...cellStyle, textAlign: 'center' }}>
        <button
          onClick={showAction ? onScore : undefined}
          disabled={!showAction}
          style={{
            fontSize: 12,
            padding: '2px 8px',
            border: category.isScored ? 'none' : undefined,
            background: category.isScored ? 'transparent' : undefined,
            cursor: showAction ? 'pointer' : 'default',
            visibility: showAction || category.isScored ? 'visible' : 'hidden',
          }}
        >
          {category.isScored ? '\u2713' : 'Score'}
        </button>
      </td>
    </tr>
  );
}
