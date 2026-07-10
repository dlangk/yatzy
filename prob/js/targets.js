/**
 * Target predicates for the probabilities tool.
 * Categories are binary-achievement patterns (achieved or not), unlike the
 * upper-section scoring categories which are score amounts, not patterns.
 * @module targets
 */

/** Tally of value -> count for a hand. */
function counts(hand) {
  const c = {};
  for (const v of hand) c[v] = (c[v] || 0) + 1;
  return c;
}

/** Sorted array of the count values, e.g. full house -> [2,3]. */
function shape(hand) {
  return Object.values(counts(hand)).sort((a, b) => a - b);
}

function hasNOfAKind(hand, n) {
  return Object.values(counts(hand)).some((c) => c >= n);
}

function keyOf(hand) {
  return hand.slice().sort((a, b) => a - b).join('');
}

/** @type {Array<{id:string,label:string,test:(hand:number[])=>boolean}>} */
export const CATEGORIES = [
  { id: 'one_pair', label: 'One pair', test: (h) => hasNOfAKind(h, 2) },
  {
    id: 'two_pairs',
    label: 'Two pairs',
    test: (h) => Object.values(counts(h)).filter((c) => c >= 2).length >= 2,
  },
  { id: 'three_kind', label: 'Three of a kind', test: (h) => hasNOfAKind(h, 3) },
  { id: 'four_kind', label: 'Four of a kind', test: (h) => hasNOfAKind(h, 4) },
  {
    id: 'full_house',
    label: 'Full house',
    test: (h) => {
      const s = shape(h);
      return s.length === 2 && s[0] === 2 && s[1] === 3;
    },
  },
  { id: 'small_straight', label: 'Small straight', test: (h) => keyOf(h) === '12345' },
  { id: 'large_straight', label: 'Large straight', test: (h) => keyOf(h) === '23456' },
  { id: 'yatzy', label: 'Yatzy', test: (h) => new Set(h).size === 1 },
];

/** @param {string} id */
export function categoryById(id) {
  return CATEGORIES.find((c) => c.id === id);
}

/**
 * Predicate that matches a specific unordered five-die hand.
 * @param {number[]} targetHand
 * @returns {(hand:number[])=>boolean}
 */
export function exactTarget(targetHand) {
  const target = keyOf(targetHand);
  return (hand) => keyOf(hand) === target;
}
