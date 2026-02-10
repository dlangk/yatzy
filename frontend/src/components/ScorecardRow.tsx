import type { CategoryState } from '../types.ts';

interface ScorecardRowProps {
  category: CategoryState;
  isOptimal: boolean;
  canScore: boolean;
  onScore: () => void;
}

export function ScorecardRow({ category, isOptimal, canScore, onScore }: ScorecardRowProps) {
  const bg = category.isScored ? '#f0f0f0' : isOptimal ? '#d4edda' : 'transparent';

  return (
    <tr style={{ background: bg }}>
      <td style={{ padding: '4px 8px', borderBottom: '1px solid #ddd' }}>
        {category.name}
      </td>
      <td style={{ padding: '4px 8px', borderBottom: '1px solid #ddd', textAlign: 'center' }}>
        {category.isScored ? category.score : category.suggestedScore || '—'}
      </td>
      <td style={{ padding: '4px 8px', borderBottom: '1px solid #ddd', textAlign: 'center', fontSize: 12 }}>
        {!category.isScored && category.available ? category.evIfScored.toFixed(1) : ''}
      </td>
      <td style={{ padding: '4px 8px', borderBottom: '1px solid #ddd', textAlign: 'center' }}>
        {canScore && !category.isScored && category.available ? (
          <button onClick={onScore} style={{ fontSize: 12, padding: '2px 8px' }}>
            Score
          </button>
        ) : category.isScored ? '✓' : ''}
      </td>
    </tr>
  );
}
