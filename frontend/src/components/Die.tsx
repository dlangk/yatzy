interface DieProps {
  value: number;
  held: boolean;
  isOptimalReroll: boolean;
  onClick: () => void;
  disabled: boolean;
}

export function Die({ value, held, isOptimalReroll, onClick, disabled }: DieProps) {
  const bg = held ? '#fff' : '#ddd';
  const border = isOptimalReroll ? '3px solid #e74c3c' : '2px solid #333';

  return (
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
        opacity: disabled ? 0.5 : 1,
      }}
      title={held ? 'Held (click to reroll)' : 'Will reroll (click to hold)'}
    >
      {value}
    </button>
  );
}
