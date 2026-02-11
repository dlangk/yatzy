import type { TurnPhase } from '../types.ts';

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
      color: '#666',
      margin: '4px 0 8px',
      opacity: active ? 1 : 0,
      transition: 'opacity 0.2s',
      minHeight: 16,
    }}>
      <span>
        <span style={{ ...swatchBase, background: '#fff', border: '1px solid #333' }} />
        Held
      </span>
      <span>
        <span style={{ ...swatchBase, background: '#ddd', border: '1px solid #333' }} />
        Reroll
      </span>
      <span>
        <span style={{ ...swatchBase, background: '#fff', border: '2px solid #28a745' }} />
        Optimal keep
      </span>
      <span>
        <span style={{ ...swatchBase, background: '#ddd', border: '2px solid #e74c3c' }} />
        Optimal reroll
      </span>
    </div>
  );
}
