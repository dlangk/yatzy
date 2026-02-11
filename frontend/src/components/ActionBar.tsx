import type { TurnPhase } from '../types.ts';

interface ActionBarProps {
  turnPhase: TurnPhase;
  rerollsRemaining: number;
  onRoll: () => void;
  onReroll: () => void;
  onReset: () => void;
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  justifyContent: 'center',
  alignItems: 'center',
  margin: '12px 0',
  minHeight: 40,
};

export function ActionBar({ turnPhase, rerollsRemaining, onRoll, onReroll, onReset }: ActionBarProps) {
  if (turnPhase === 'game_over') {
    return (
      <div style={containerStyle}>
        <strong>Game Over!</strong>
        <button onClick={onReset} style={{ fontSize: 16, padding: '6px 16px' }}>
          New Game
        </button>
        <button style={{ fontSize: 12, padding: '4px 10px', visibility: 'hidden' }}>Reset</button>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      {turnPhase === 'idle' ? (
        <button onClick={onRoll} style={{ fontSize: 16, padding: '6px 20px' }}>
          Roll
        </button>
      ) : (
        <button
          onClick={onReroll}
          disabled={rerollsRemaining <= 0}
          style={{ fontSize: 16, padding: '6px 20px' }}
        >
          Reroll ({rerollsRemaining})
        </button>
      )}
      <button
        onClick={() => {
          if (window.confirm('Reset game? All progress will be lost.')) {
            onReset();
          }
        }}
        style={{
          fontSize: 12,
          padding: '4px 10px',
          opacity: 0.6,
        }}
      >
        Reset
      </button>
    </div>
  );
}
