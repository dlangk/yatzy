import type { CategoryState, TurnPhase } from '../types.ts';
import { ScorecardRow } from './ScorecardRow.tsx';
import { UPPER_CATEGORIES, BONUS_THRESHOLD } from '../constants.ts';

interface ScorecardProps {
  categories: CategoryState[];
  upperScore: number;
  bonus: number;
  totalScore: number;
  turnPhase: TurnPhase;
  onScoreCategory: (id: number) => void;
}

export function Scorecard({ categories, upperScore, bonus, totalScore, turnPhase, onScoreCategory }: ScorecardProps) {
  const optimalId = findOptimalCategory(categories);
  const canScore = turnPhase === 'rolled';

  const upperCats = categories.slice(0, UPPER_CATEGORIES);
  const lowerCats = categories.slice(UPPER_CATEGORIES);

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12, fontFamily: 'monospace', fontSize: 14 }}>
      <thead>
        <tr style={{ borderBottom: '2px solid #333' }}>
          <th style={{ textAlign: 'left', padding: '4px 8px' }}>Category</th>
          <th style={{ textAlign: 'center', padding: '4px 8px' }}>Score</th>
          <th style={{ textAlign: 'center', padding: '4px 8px', fontSize: 12 }}>EV</th>
          <th style={{ textAlign: 'center', padding: '4px 8px' }}></th>
        </tr>
      </thead>
      <tbody>
        {upperCats.map((cat) => (
          <ScorecardRow
            key={cat.id}
            category={cat}
            isOptimal={cat.id === optimalId && canScore}
            canScore={canScore}
            onScore={() => onScoreCategory(cat.id)}
          />
        ))}
        <tr style={{ background: '#f8f8f8', fontWeight: 'bold' }}>
          <td style={{ padding: '4px 8px', borderBottom: '2px solid #333' }}>
            Upper ({upperScore}/{BONUS_THRESHOLD})
          </td>
          <td style={{ padding: '4px 8px', borderBottom: '2px solid #333', textAlign: 'center' }}>
            {bonus > 0 ? `+${bonus}` : '—'}
          </td>
          <td colSpan={2} style={{ borderBottom: '2px solid #333' }}></td>
        </tr>
        {lowerCats.map((cat) => (
          <ScorecardRow
            key={cat.id}
            category={cat}
            isOptimal={cat.id === optimalId && canScore}
            canScore={canScore}
            onScore={() => onScoreCategory(cat.id)}
          />
        ))}
        <tr style={{ fontWeight: 'bold', borderTop: '2px solid #333' }}>
          <td style={{ padding: '6px 8px' }}>Total</td>
          <td style={{ padding: '6px 8px', textAlign: 'center' }}>{totalScore}</td>
          <td colSpan={2}></td>
        </tr>
      </tbody>
    </table>
  );
}

function findOptimalCategory(categories: CategoryState[]): number | null {
  let bestId: number | null = null;
  let bestEv = -Infinity;
  for (const cat of categories) {
    if (!cat.isScored && cat.available && cat.evIfScored > bestEv) {
      bestEv = cat.evIfScored;
      bestId = cat.id;
    }
  }
  return bestId;
}
