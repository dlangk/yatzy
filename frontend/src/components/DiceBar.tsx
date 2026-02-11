import type { DieState, TurnPhase } from '../types.ts';
import { Die } from './Die.tsx';

interface DiceBarProps {
  dice: DieState[];
  onToggle: (index: number) => void;
  onSetDie: (index: number, value: number) => void;
  optimalMask: number | null;
  rerollsRemaining: number;
  turnPhase: TurnPhase;
  hasEval: boolean;
}

export function DiceBar({ dice, onToggle, onSetDie, optimalMask, rerollsRemaining, turnPhase, hasEval }: DiceBarProps) {
  const active = turnPhase === 'rolled';
  const canToggle = active && hasEval && rerollsRemaining > 0;
  const showMaskHints = active && hasEval && optimalMask !== null && rerollsRemaining > 0;

  return (
    <div style={{ display: 'flex', gap: 8, justifyContent: 'center', margin: '12px 0' }}>
      {dice.map((die, i) => {
        const inOptimalReroll = showMaskHints && !!(optimalMask! & (1 << i));
        const inOptimalKeep = showMaskHints && !(optimalMask! & (1 << i));
        const v = active ? die.value : 0;
        const upValue = v >= 6 || v === 0 ? 1 : v + 1;
        const downValue = v <= 1 || v === 0 ? 6 : v - 1;
        return (
          <Die
            key={i}
            value={v}
            held={active ? die.held : true}
            isOptimalReroll={inOptimalReroll}
            isOptimalKeep={inOptimalKeep}
            onClick={() => onToggle(i)}
            disabled={!canToggle}
            faded={!active}
            onIncrement={() => onSetDie(i, upValue)}
            onDecrement={() => onSetDie(i, downValue)}
            showManualControls={true}
          />
        );
      })}
    </div>
  );
}
