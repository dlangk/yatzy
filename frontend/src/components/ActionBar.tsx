import type { TurnPhase } from '../types.ts';

interface ActionBarProps {
  turnPhase: TurnPhase;
  rerollsRemaining: number;
  onRoll: () => void;
  onReroll: () => void;
  onReset: () => void;
}

export function ActionBar({ turnPhase, rerollsRemaining, onRoll, onReroll, onReset }: ActionBarProps) {
  if (turnPhase === 'game_over') {
    return (
      <div style={{ textAlign: 'center', margin: '12px 0' }}>
        <strong>Game Over!</strong>{' '}
        <button onClick={onReset} style={{ fontSize: 16, padding: '6px 16px', marginLeft: 8 }}>
          New Game
        </button>
      </div>
    );
  }

  if (turnPhase === 'idle') {
    return (
      <div style={{ display: 'flex', gap: 12, justifyContent: 'center', alignItems: 'center', margin: '12px 0' }}>
        <button onClick={onRoll} style={{ fontSize: 16, padding: '6px 20px' }}>
          Roll
        </button>
      </div>
    );
  }

  // turnPhase === 'rolled'
  return (
    <div style={{ display: 'flex', gap: 12, justifyContent: 'center', alignItems: 'center', margin: '12px 0' }}>
      <button
        onClick={onReroll}
        disabled={rerollsRemaining <= 0}
        style={{ fontSize: 16, padding: '6px 20px' }}
      >
        Reroll ({rerollsRemaining})
      </button>
    </div>
  );
}
