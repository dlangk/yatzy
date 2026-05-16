// Static accent colors — same in light and dark mode.
// Only used by D3/SVG which can't use CSS var().
export const COLORS = {
  success: '#2ca02c',
  danger: '#b40426',
  orange: '#F37021',
  blue: '#3b4cc0',
} as const;

/** Read theme-sensitive colors from CSS custom properties at call time. */
export function getColors() {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    ...COLORS,
    text: v('--text') || '#050505',
    textMuted: v('--text-muted') || '#555',
    border: v('--border') || '#d4d3cd',
    bg: v('--bg') || '#f6f5ef',
    bgPanel: v('--bg-alt') || '#eae9e3',
    borderPanel: v('--border') || '#d4d3cd',
    success: v('--color-success') || COLORS.success,
    danger: v('--color-danger') || COLORS.danger,
  };
}

export const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
] as const;

export const TOTAL_DICE = 5;
export const UPPER_SCORE_CAP = 63;
export const UPPER_BONUS = 50;
export const CATEGORY_COUNT = 15;
export const UPPER_CATEGORIES = 6;
