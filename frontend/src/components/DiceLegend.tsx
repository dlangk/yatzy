import type { TurnPhase } from '../types.ts';
import { COLORS } from '../constants.ts';

interface DiceLegendProps {
  turnPhase: TurnPhase;
  rerollsRemaining: number;
}

const swatchBase: React.CSSProperties = {
  display: 'inline-block',
  width: 12,
  height: 12,
  borderRadius: 2,
  marginRight: 4,
  verticalAlign: 'middle',
};

export function DiceLegend({ turnPhase, rerollsRemaining }: DiceLegendProps) {
  const active = turnPhase === 'rolled' && rerollsRemaining > 0;

  return (
    <div style={{
      display: 'flex',
      gap: 16,
      justifyContent: 'center',
      fontSize: 11,
      color: COLORS.textMuted,
      margin: '4px 0 8px',
      opacity: active ? 1 : 0,
      transition: 'opacity 0.2s',
      minHeight: 16,
    }}>
      <span>
        <span style={{ ...swatchBase, background: COLORS.bg, border: `1px solid ${COLORS.text}` }} />
        Held
      </span>
      <span>
        <span style={{ ...swatchBase, background: COLORS.bgAlt, border: `1px solid ${COLORS.text}` }} />
        Reroll
      </span>
      <span>
        <span style={{ ...swatchBase, background: COLORS.bg, border: `2px solid ${COLORS.success}` }} />
        Optimal keep
      </span>
      <span>
        <span style={{ ...swatchBase, background: COLORS.bgAlt, border: `2px solid ${COLORS.danger}` }} />
        Optimal reroll
      </span>
    </div>
  );
}
