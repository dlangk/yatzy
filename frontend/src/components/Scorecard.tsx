import type { CategoryState, TurnPhase } from '../types.ts';
import { ScorecardRow } from './ScorecardRow.tsx';
import { UPPER_CATEGORIES, BONUS_THRESHOLD, COLORS } from '../constants.ts';

interface ScorecardProps {
  categories: CategoryState[];
  upperScore: number;
  bonus: number;
  totalScore: number;
  turnPhase: TurnPhase;
  optimalCategoryId: number | null;
  onScoreCategory: (id: number) => void;
  onSetCategoryScore: (id: number, score: number) => void;
  onUnsetCategory: (id: number) => void;
}

function normalize(value: number, min: number, max: number): number {
  if (max <= min) return value > 0 ? 1 : 0;
  return (value - min) / (max - min);
}

export function Scorecard({ categories, upperScore, bonus, totalScore, turnPhase, optimalCategoryId, onScoreCategory, onSetCategoryScore, onUnsetCategory }: ScorecardProps) {
  const canScore = turnPhase === 'rolled';

  // Compute normalization ranges for spark bars
  const scoreValues = categories.map(c => c.isScored ? c.score : c.suggestedScore);
  const scoreMin = Math.min(...scoreValues);
  const scoreMax = Math.max(...scoreValues);

  const unscoredAvailable = categories.filter(c => !c.isScored && c.available);
  const evValues = unscoredAvailable.map(c => c.evIfScored);
  const evMin = evValues.length > 0 ? Math.min(...evValues) : 0;
  const evMax = evValues.length > 0 ? Math.max(...evValues) : 0;

  const upperCats = categories.slice(0, UPPER_CATEGORIES);
  const lowerCats = categories.slice(UPPER_CATEGORIES);

  function renderRow(cat: CategoryState) {
    const scoreVal = cat.isScored ? cat.score : cat.suggestedScore;
    const scoreFraction = normalize(scoreVal, scoreMin, scoreMax);
    const evFraction = (!cat.isScored && cat.available) ? normalize(cat.evIfScored, evMin, evMax) : null;

    return (
      <ScorecardRow
        key={cat.id}
        category={cat}
        isOptimal={cat.id === optimalCategoryId && canScore}
        canScore={canScore}
        onScore={() => onScoreCategory(cat.id)}
        onSetScore={(score) => onSetCategoryScore(cat.id, score)}
        onUnsetCategory={() => onUnsetCategory(cat.id)}
        scoreFraction={scoreFraction}
        evFraction={evFraction}
      />
    );
  }

  return (
    <table style={{ width: '100%', tableLayout: 'fixed', borderCollapse: 'collapse', marginTop: 12, fontFamily: 'monospace', fontSize: 14 }}>
      <colgroup>
        <col style={{ width: '45%' }} />
        <col style={{ width: '20%' }} />
        <col style={{ width: '15%' }} />
        <col style={{ width: '20%' }} />
      </colgroup>
      <thead>
        <tr style={{ borderBottom: `2px solid ${COLORS.text}` }}>
          <th style={{ textAlign: 'left', padding: '4px 8px' }}>Category</th>
          <th style={{ textAlign: 'center', padding: '4px 8px' }}>Score</th>
          <th style={{ textAlign: 'center', padding: '4px 8px', fontSize: 12 }}>EV</th>
          <th style={{ textAlign: 'center', padding: '4px 8px' }}></th>
        </tr>
      </thead>
      <tbody>
        {upperCats.map(renderRow)}
        <tr style={{ background: COLORS.bgAlt2, fontWeight: 'bold' }}>
          <td style={{ padding: '4px 8px', borderBottom: `2px solid ${COLORS.text}` }}>
            Upper ({upperScore}/{BONUS_THRESHOLD})
          </td>
          <td style={{ padding: '4px 8px', borderBottom: `2px solid ${COLORS.text}`, textAlign: 'center' }}>
            {bonus > 0 ? `+${bonus}` : '\u2014'}
          </td>
          <td colSpan={2} style={{ borderBottom: `2px solid ${COLORS.text}` }}></td>
        </tr>
        {lowerCats.map(renderRow)}
        <tr style={{ fontWeight: 'bold', borderTop: `2px solid ${COLORS.text}` }}>
          <td style={{ padding: '6px 8px' }}>Total</td>
          <td style={{ padding: '6px 8px', textAlign: 'center' }}>{totalScore}</td>
          <td colSpan={2}></td>
        </tr>
      </tbody>
    </table>
  );
}
