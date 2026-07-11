/**
 * Shared path-probability math for Yatzy three-roll transitions.
 * Pure functions, no DOM, no imports. Used by the treatise path-probability
 * chart and the standalone Probabilities tab so their math cannot drift.
 * @module path-prob
 */

/** n! for small n. */
export function factorial(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

/** Count of each face value 1..6 in a dice array. */
export function frequencies(dice) {
  const freq = [0, 0, 0, 0, 0, 0];
  for (const d of dice) freq[d - 1]++;
  return freq;
}

/** Multinomial coefficient n! / (f1! f2! ...). */
export function multinomial(n, freqs) {
  let denom = 1;
  for (const f of freqs) denom *= factorial(f);
  return factorial(n) / denom;
}

/**
 * Probability that rerolling `dice.length` fair dice lands on exactly this
 * multiset of values (order-independent). Empty set (nothing rerolled) is 1.
 * @param {number[]} dice the free (rerolled) dice values
 * @returns {number}
 */
export function rollProbability(dice) {
  if (dice.length === 0) return 1;
  const freqs = frequencies(dice);
  return multinomial(dice.length, freqs) / Math.pow(6, dice.length);
}

/** Greatest common divisor. */
export function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

/**
 * Probability of transitioning from `row` to `next` given which positions were
 * kept. A kept position whose value changes makes the transition impossible (0).
 * Otherwise the probability is that of rerolling the free positions into exactly
 * `next`'s free-position values.
 * @param {number[]} row current five dice
 * @param {number[]} next target five dice
 * @param {Set<number>} keptSet positions (0..4) kept from `row`
 * @returns {number}
 */
export function transitionProb(row, next, keptSet) {
  const free = [];
  for (let i = 0; i < 5; i++) {
    if (keptSet.has(i)) {
      if (row[i] !== next[i]) return 0; // kept die cannot change value
    } else {
      free.push(next[i]);
    }
  }
  return rollProbability(free);
}

/**
 * Probability of a one-reroll transition from a kept multiset to an outcome
 * multiset (order-independent, matching the treatise transition matrix). You
 * keep `keptValues` (0..5 dice) and reroll the rest; this is the chance the
 * five-die `outcome` results. Impossible (0) if a kept value is not present in
 * the outcome.
 * @param {number[]} keptValues values of the dice you hold (length 0..5)
 * @param {number[]} outcome the five-die outcome multiset
 * @returns {number}
 */
export function outcomeProbability(keptValues, outcome) {
  const kf = frequencies(keptValues);
  const of = frequencies(outcome);
  const free = [];
  for (let v = 0; v < 6; v++) {
    const d = of[v] - kf[v];
    if (d < 0) return 0; // kept a die the outcome does not contain
    for (let i = 0; i < d; i++) free.push(v + 1);
  }
  return rollProbability(free);
}

/**
 * Reduced fraction {num, den} for the probability of rerolling into `dice`.
 * @param {number[]} dice
 * @returns {{num:string, den:string}}
 */
export function formatFraction(dice) {
  if (dice.length === 0) return { num: '1', den: '1' };
  const freqs = frequencies(dice);
  const num = multinomial(dice.length, freqs);
  const den = Math.pow(6, dice.length);
  const g = gcd(num, den);
  return { num: String(num / g), den: String(den / g) };
}

/**
 * Human "1 in N" phrasing for a probability.
 * @param {number} p
 * @returns {string}
 */
export function formatOneInN(p) {
  if (p >= 1) return '1 in 1';
  if (p <= 0) return 'impossible';
  const n = Math.round(1 / p);
  return `1 in ${n.toLocaleString()}`;
}
