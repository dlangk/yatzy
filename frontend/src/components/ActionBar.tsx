import type { TurnPhase } from '../types.ts';

interface ActionBarProps {
  turnPhase: TurnPhase;
  rerollsRemaining: number;
  onRoll: () => void;
  onReroll: () => void;
  onReset: () => void;
  onSetRerolls: (rerolls: number) => void;
}

const containerStyle: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  justifyContent: 'center',
  alignItems: 'center',
  margin: '12px 0',
  minHeight: 40,
};

const smallBtnStyle: React.CSSProperties = {
  fontSize: 14,
  width: 24,
  height: 24,
  padding: 0,
  border: '1px solid #ccc',
  background: '#f0f0f0',
  cursor: 'pointer',
  borderRadius: 3,
  lineHeight: '22px',
};

export function ActionBar({ turnPhase, rerollsRemaining, onRoll, onReroll, onReset, onSetRerolls }: ActionBarProps) {
  const showRerollControls = turnPhase === 'rolled';

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
      <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center', visibility: showRerollControls ? 'visible' : 'hidden' }}>
        <button
          onClick={() => onSetRerolls(rerollsRemaining - 1)}
          disabled={rerollsRemaining <= 0}
          style={smallBtnStyle}
        >
          &minus;
        </button>
        <button
          onClick={() => onSetRerolls(rerollsRemaining + 1)}
          disabled={rerollsRemaining >= 2}
          style={smallBtnStyle}
        >
          +
        </button>
      </span>
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
