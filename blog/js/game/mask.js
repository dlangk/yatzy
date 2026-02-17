/**
 * Dice sort-mapping utilities.
 *
 * The backend sorts dice ascending before computing. Mask bit positions
 * correspond to sorted-order indices. The frontend must translate between
 * the player's original dice order and the backend's sorted order.
 */

export function sortDiceWithMapping(dice) {
  const indexed = dice.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  return {
    sorted: indexed.map((x) => x.v),
    sortMap: indexed.map((x) => x.i),
  };
}

export function mapMaskToSorted(mask, sortMap) {
  let out = 0;
  for (let si = 0; si < sortMap.length; si++) {
    if (mask & (1 << sortMap[si])) {
      out |= 1 << si;
    }
  }
  return out;
}

export function unmapMask(sortedMask, sortMap) {
  let out = 0;
  for (let si = 0; si < sortMap.length; si++) {
    if (sortedMask & (1 << si)) {
      out |= 1 << sortMap[si];
    }
  }
  return out;
}

export function computeRerollMask(held) {
  let mask = 0;
  for (let i = 0; i < held.length; i++) {
    if (!held[i]) {
      mask |= 1 << i;
    }
  }
  return mask;
}
