import { COLORS } from '../constants.ts';

interface DieProps {
  value: number;
  held: boolean;
  isOptimalReroll: boolean;
  isOptimalKeep: boolean;
  onClick: () => void;
  disabled: boolean;
  faded: boolean;
  onIncrement?: () => void;
  onDecrement?: () => void;
  showManualControls: boolean;
}

const arrowBtnStyle: React.CSSProperties = {
  width: 56,
  height: 16,
  fontSize: 10,
  padding: 0,
  border: `1px solid ${COLORS.border}`,
  background: COLORS.bgAlt,
  cursor: 'pointer',
  borderRadius: 3,
  lineHeight: '14px',
};

export function Die({ value, held, isOptimalReroll, isOptimalKeep, onClick, disabled, faded, onIncrement, onDecrement, showManualControls }: DieProps) {
  const bg = held ? COLORS.bg : COLORS.bgAlt;
  const border = isOptimalReroll
    ? `3px solid ${COLORS.danger}`
    : isOptimalKeep
      ? `3px solid ${COLORS.success}`
      : `2px solid ${COLORS.text}`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
      <button
        onClick={onIncrement}
        style={{ ...arrowBtnStyle, visibility: showManualControls ? 'visible' : 'hidden' }}
      >
        &#9650;
      </button>
      <button
        onClick={onClick}
        disabled={disabled}
        style={{
          width: 56,
          height: 56,
          fontSize: 28,
          fontWeight: 'bold',
          fontFamily: 'monospace',
          background: bg,
          border,
          borderRadius: 8,
          cursor: disabled ? 'default' : 'pointer',
          opacity: faded ? 0.5 : 1,
        }}
        title={held ? 'Held (click to reroll)' : 'Will reroll (click to hold)'}
      >
        {value === 0 ? '?' : value}
      </button>
      <button
        onClick={onDecrement}
        style={{ ...arrowBtnStyle, visibility: showManualControls ? 'visible' : 'hidden' }}
      >
        &#9660;
      </button>
    </div>
  );
}
