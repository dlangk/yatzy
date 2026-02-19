import type { CategoryState } from '../types.ts';
import { COLORS } from '../constants.ts';

interface ScorecardRowProps {
  category: CategoryState;
  isOptimal: boolean;
  canScore: boolean;
  onScore: () => void;
  onSetScore: (score: number) => void;
  onUnsetCategory: () => void;
  scoreFraction: number;
  evFraction: number | null;
}

const cellStyle: React.CSSProperties = {
  padding: '4px 8px',
  borderBottom: `1px solid ${COLORS.bgAlt}`,
  height: 32,
  boxSizing: 'border-box',
};

function sparkGradient(fraction: number, color: string): string {
  const pct = `${(fraction * 100).toFixed(1)}%`;
  return `linear-gradient(to right, ${color} ${pct}, transparent ${pct})`;
}

export function ScorecardRow({ category, isOptimal, canScore, onScore, onSetScore, onUnsetCategory, scoreFraction, evFraction }: ScorecardRowProps) {
  const displayedScore = category.isScored ? category.score : category.suggestedScore;
  const isZero = !category.isScored && displayedScore === 0;
  const dimmed = isZero && !isOptimal;

  const bg = category.isScored
    ? COLORS.bgAlt
    : isOptimal
      ? 'rgba(44, 160, 44, 0.12)'
      : 'transparent';

  const nameColor = dimmed ? COLORS.textMuted : 'inherit';

  const scoreBarColor = isOptimal
    ? 'rgba(44, 160, 44, 0.18)'
    : category.isScored
      ? 'rgba(0, 0, 0, 0.06)'
      : 'rgba(59, 76, 192, 0.15)';

  const evBarColor = isOptimal
    ? 'rgba(44, 160, 44, 0.18)'
    : 'rgba(59, 76, 192, 0.15)';

  const showAction = canScore && !category.isScored && category.available;
  const hasEv = !category.isScored && category.available;

  return (
    <tr style={{ background: bg }}>
      <td style={{ ...cellStyle, color: nameColor }}>
        {category.name}
      </td>
      <td style={{
        ...cellStyle,
        textAlign: 'center',
        background: scoreFraction > 0 ? sparkGradient(scoreFraction, scoreBarColor) : undefined,
      }}>
        <input
          type="number"
          value={displayedScore}
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
            color: dimmed ? COLORS.textMuted : 'inherit',
            MozAppearance: 'textfield',
          }}
        />
      </td>
      <td style={{
        ...cellStyle,
        textAlign: 'center',
        fontSize: 12,
        color: dimmed ? COLORS.textMuted : 'inherit',
        background: evFraction != null && evFraction > 0 ? sparkGradient(evFraction, evBarColor) : undefined,
      }}>
        {hasEv ? category.evIfScored.toFixed(1) : ''}
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
