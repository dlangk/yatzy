/**
 * Pure probability engine for target reachability across dice rerolls.
 * A `hand` is a sorted-ascending array of five dice values in 1..6.
 * @module engine
 */

/** @type {Map<number, Array<{dice:number[], p:number}>>} */
const _rerollCache = new Map();

/** k! for small k. */
function factorial(n) {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
}

/**
 * All sorted multiset outcomes of rolling `k` fair dice, with probabilities.
 * @param {number} k dice to roll (0..5)
 * @returns {Array<{dice:number[], p:number}>}
 */
export function rerollOutcomes(k) {
  if (_rerollCache.has(k)) return _rerollCache.get(k);
  /** @type {Array<{dice:number[], p:number}>} */
  const out = [];
  if (k === 0) {
    out.push({ dice: [], p: 1 });
  } else {
    const total = Math.pow(6, k);
    const rec = (start, current) => {
      if (current.length === k) {
        const counts = {};
        for (const v of current) counts[v] = (counts[v] || 0) + 1;
        let perms = factorial(k);
        for (const v in counts) perms /= factorial(counts[v]);
        out.push({ dice: current.slice(), p: perms / total });
        return;
      }
      for (let v = start; v <= 6; v++) {
        current.push(v);
        rec(v, current);
        current.pop();
      }
    };
    rec(1, []);
  }
  _rerollCache.set(k, out);
  return out;
}

/**
 * Distinct sorted sub-multisets of a 5-die hand (which dice to keep).
 * @param {number[]} hand sorted 5-die hand
 * @returns {number[][]}
 */
export function keepSets(hand) {
  const seen = new Set();
  /** @type {number[][]} */
  const res = [];
  for (let mask = 0; mask < 32; mask++) {
    const keep = [];
    for (let i = 0; i < 5; i++) if (mask & (1 << i)) keep.push(hand[i]);
    keep.sort((a, b) => a - b);
    const key = keep.join('');
    if (!seen.has(key)) {
      seen.add(key);
      res.push(keep);
    }
  }
  return res;
}

/**
 * Build a memoized solver for a fixed target predicate.
 * @param {(hand:number[])=>boolean} target
 */
export function makeSolver(target) {
  /** @type {Map<string, number>} */
  const memo = new Map();

  /** Expected pOpt(next, r-1) over rerolling the non-kept dice. */
  function applyKeep(keep, r) {
    const k = 5 - keep.length;
    let sum = 0;
    for (const { dice, p } of rerollOutcomes(k)) {
      const next = keep.concat(dice).sort((a, b) => a - b);
      sum += p * pOpt(next, r - 1);
    }
    return sum;
  }

  /** Best achievable P(target) from `hand` with `r` rerolls left. */
  function pOpt(hand, r) {
    if (r <= 0) return target(hand) ? 1 : 0;
    const key = r + ':' + hand.join('');
    const cached = memo.get(key);
    if (cached !== undefined) return cached;
    let best = 0;
    for (const keep of keepSets(hand)) {
      const v = applyKeep(keep, r);
      if (v > best) best = v;
    }
    memo.set(key, best);
    return best;
  }

  /** P(target) if you keep exactly `keep` now, then play optimally. */
  function pYou(hand, r, keep) {
    if (r <= 0) return target(hand) ? 1 : 0;
    return applyKeep(keep.slice().sort((a, b) => a - b), r);
  }

  /** The sorted keep-set achieving pOpt(hand, r). */
  function bestKeep(hand, r) {
    let best = -1;
    let bestK = hand.slice();
    for (const keep of keepSets(hand)) {
      const v = applyKeep(keep, r);
      if (v > best) {
        best = v;
        bestK = keep;
      }
    }
    return bestK;
  }

  return { pOpt, pYou, bestKeep };
}
