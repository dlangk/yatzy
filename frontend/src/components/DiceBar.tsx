import type { DieState, TurnPhase } from '../types.ts';
import { Die } from './Die.tsx';

interface DiceBarProps {
  dice: DieState[];
  onToggle: (index: number) => void;
  optimalMask: number | null;
  rerollsRemaining: number;
  turnPhase: TurnPhase;
  hasEval: boolean;
}

export function DiceBar({ dice, onToggle, optimalMask, rerollsRemaining, turnPhase, hasEval }: DiceBarProps) {
  if (turnPhase !== 'rolled') return null;

  const canToggle = hasEval && rerollsRemaining > 0;

  return (
    <div style={{ display: 'flex', gap: 8, justifyContent: 'center', margin: '12px 0' }}>
      {dice.map((die, i) => {
        const isOptimalReroll = optimalMask !== null && rerollsRemaining > 0 && !!(optimalMask & (1 << i));
        return (
          <Die
            key={i}
            value={die.value}
            held={die.held}
            isOptimalReroll={isOptimalReroll}
            onClick={() => onToggle(i)}
            disabled={!canToggle}
          />
        );
      })}
    </div>
  );
}
