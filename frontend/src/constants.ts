// Unified color palette (matches analytics coolwarm_mid)
export const COLORS = {
  success: '#2ca02c',
  danger: '#b40426',
  orange: '#F37021',
  blue: '#3b4cc0',
  text: '#333',
  textMuted: '#aaa',
  border: '#ccc',
  bg: '#fff',
  bgAlt: '#f0f0f0',
  bgAlt2: '#f8f8f8',
  bgPanel: '#f8f9fa',
  borderPanel: '#dee2e6',
} as const;

export const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
] as const;

export const TOTAL_DICE = 5;
export const BONUS_THRESHOLD = 63;
export const BONUS_SCORE = 50;
export const CATEGORY_COUNT = 15;
export const UPPER_CATEGORIES = 6;
