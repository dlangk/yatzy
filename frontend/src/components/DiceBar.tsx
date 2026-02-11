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
  const active = turnPhase === 'rolled';
  const canToggle = active && hasEval && rerollsRemaining > 0;
  const showMaskHints = active && hasEval && optimalMask !== null && rerollsRemaining > 0;

  return (
    <div style={{ display: 'flex', gap: 8, justifyContent: 'center', margin: '12px 0', opacity: active ? 1 : 0.3 }}>
      {dice.map((die, i) => {
        const inOptimalReroll = showMaskHints && !!(optimalMask! & (1 << i));
        const inOptimalKeep = showMaskHints && !(optimalMask! & (1 << i));
        return (
          <Die
            key={i}
            value={active ? die.value : 0}
            held={active ? die.held : true}
            isOptimalReroll={inOptimalReroll}
            isOptimalKeep={inOptimalKeep}
            onClick={() => onToggle(i)}
            disabled={!canToggle}
            faded={!active}
          />
        );
      })}
    </div>
  );
}
