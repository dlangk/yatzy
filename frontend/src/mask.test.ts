import { describe, it, expect } from 'vitest';
import {
  sortDiceWithMapping,
  mapMaskToSorted,
  unmapMask,
  computeRerollMask,
} from './mask.ts';

describe('sortDiceWithMapping', () => {
  it('returns identity mapping for already-sorted input', () => {
    const { sorted, sortMap } = sortDiceWithMapping([1, 2, 3, 4, 5]);
    expect(sorted).toEqual([1, 2, 3, 4, 5]);
    expect(sortMap).toEqual([0, 1, 2, 3, 4]);
  });

  it('sorts descending input and provides correct mapping', () => {
    const { sorted, sortMap } = sortDiceWithMapping([5, 4, 3, 2, 1]);
    expect(sorted).toEqual([1, 2, 3, 4, 5]);
    // sortMap[sortedIdx] = originalIdx
    expect(sortMap).toEqual([4, 3, 2, 1, 0]);
  });

  it('handles duplicates', () => {
    const { sorted, sortMap } = sortDiceWithMapping([3, 1, 3, 1, 5]);
    expect(sorted).toEqual([1, 1, 3, 3, 5]);
    // Each sortMap entry points back to a valid original index
    for (const origIdx of sortMap) {
      expect(origIdx).toBeGreaterThanOrEqual(0);
      expect(origIdx).toBeLessThan(5);
    }
    // Verify the mapping reconstructs the sorted array
    for (let si = 0; si < 5; si++) {
      expect([3, 1, 3, 1, 5][sortMap[si]]).toBe(sorted[si]);
    }
  });
});

describe('mapMaskToSorted / unmapMask roundtrip', () => {
  it('roundtrips for identity sortMap', () => {
    const sortMap = [0, 1, 2, 3, 4];
    for (let mask = 0; mask < 32; mask++) {
      expect(unmapMask(mapMaskToSorted(mask, sortMap), sortMap)).toBe(mask);
    }
  });

  it('roundtrips for reversed sortMap', () => {
    const sortMap = [4, 3, 2, 1, 0];
    for (let mask = 0; mask < 32; mask++) {
      expect(unmapMask(mapMaskToSorted(mask, sortMap), sortMap)).toBe(mask);
    }
  });

  it('known example: mask=0b00001 with reversed sortMap', () => {
    // Original bit 0 set → sortMap[4]=0, so sorted bit 4 should be set
    const sortMap = [4, 3, 2, 1, 0];
    expect(mapMaskToSorted(0b00001, sortMap)).toBe(0b10000);
  });

  it('known example: mask=0b10101 with reversed sortMap', () => {
    // Original bits 0,2,4 → sorted bits 4,2,0
    const sortMap = [4, 3, 2, 1, 0];
    expect(mapMaskToSorted(0b10101, sortMap)).toBe(0b10101);
  });
});

describe('computeRerollMask', () => {
  it('all held → mask 0 (nothing to reroll)', () => {
    expect(computeRerollMask([true, true, true, true, true])).toBe(0);
  });

  it('none held → mask 31 (all bits set)', () => {
    expect(computeRerollMask([false, false, false, false, false])).toBe(31);
  });

  it('mixed: dice 1 and 3 unheld', () => {
    // bits 1 and 3 set → 0b01010 = 10
    expect(computeRerollMask([true, false, true, false, true])).toBe(0b01010);
  });

  it('single die unheld at position 0', () => {
    expect(computeRerollMask([false, true, true, true, true])).toBe(0b00001);
  });
});
