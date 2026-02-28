// Unified color palette (synced with treatise design tokens)
// Only used by D3/SVG which can't use CSS var()
export const COLORS = {
  success: '#2ca02c',
  danger: '#b40426',
  orange: '#F37021',
  blue: '#3b4cc0',
  text: '#050505',
  textMuted: '#555',
  border: '#d4d3cd',
  bg: '#f6f5ef',
  bgAlt: '#eae9e3',
  bgPanel: '#eae9e3',
  borderPanel: '#d4d3cd',
} as const;

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
