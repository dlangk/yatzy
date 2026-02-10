/**
 * Dice sort-mapping utilities.
 *
 * The backend sorts dice ascending before computing. Mask bit positions
 * correspond to sorted-order indices. The frontend must translate between
 * the player's original dice order and the backend's sorted order.
 */

/**
 * Sort dice values ascending, returning the sorted values and a mapping
 * from sorted index → original index.
 */
export function sortDiceWithMapping(dice: number[]): {
  sorted: number[];
  sortMap: number[];
} {
  const indexed = dice.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  return {
    sorted: indexed.map((x) => x.v),
    sortMap: indexed.map((x) => x.i),
  };
}

/**
 * Convert a mask in original-order to sorted-order.
 * sortMap[sortedIdx] = originalIdx, so bit sortedIdx of the output
 * takes its value from bit sortMap[sortedIdx] of the input.
 */
export function mapMaskToSorted(mask: number, sortMap: number[]): number {
  let out = 0;
  for (let si = 0; si < sortMap.length; si++) {
    if (mask & (1 << sortMap[si])) {
      out |= 1 << si;
    }
  }
  return out;
}

/**
 * Convert a mask in sorted-order to original-order.
 * Inverse of mapMaskToSorted.
 */
export function unmapMask(sortedMask: number, sortMap: number[]): number {
  let out = 0;
  for (let si = 0; si < sortMap.length; si++) {
    if (sortedMask & (1 << si)) {
      out |= 1 << sortMap[si];
    }
  }
  return out;
}

/**
 * Compute the reroll mask from held state.
 * Bit i = 1 means die i is NOT held (will be rerolled).
 * This is in original order.
 */
export function computeRerollMask(held: boolean[]): number {
  let mask = 0;
  for (let i = 0; i < held.length; i++) {
    if (!held[i]) {
      mask |= 1 << i;
    }
  }
  return mask;
}
