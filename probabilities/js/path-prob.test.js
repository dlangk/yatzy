import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  rollProbability,
  transitionProb,
  outcomeProbability,
  formatFraction,
  formatOneInN,
} from '../../shared/path-prob.js';

test('rollProbability: empty (all kept) is certain', () => {
  assert.equal(rollProbability([]), 1);
});

test('rollProbability: one specific die is 1/6', () => {
  assert.ok(Math.abs(rollProbability([5]) - 1 / 6) < 1e-12);
});

test('rollProbability: five specific equal dice is 1/7776', () => {
  assert.ok(Math.abs(rollProbability([1, 1, 1, 1, 1]) - 1 / 7776) < 1e-15);
});

test('rollProbability: two distinct dice via multinomial (2/36)', () => {
  // {1,2} as a multiset: 2 arrangements out of 36 ordered = 1/18.
  assert.ok(Math.abs(rollProbability([1, 2]) - 2 / 36) < 1e-12);
});

test('transitionProb: keep 1,2,3,4 reroll one, need a 5 -> 1/6', () => {
  const row = [1, 2, 3, 4, 6];
  const next = [1, 2, 3, 4, 5];
  const keep = new Set([0, 1, 2, 3]);
  assert.ok(Math.abs(transitionProb(row, next, keep) - 1 / 6) < 1e-12);
});

test('transitionProb: kept die that changes value is impossible (0)', () => {
  const row = [1, 2, 3, 4, 6];
  const next = [1, 2, 3, 4, 5];
  const keep = new Set([4]); // kept the 6, but next shows 5 there
  assert.equal(transitionProb(row, next, keep), 0);
});

test('transitionProb: nothing kept, target all ones -> 1/7776', () => {
  const row = [1, 2, 3, 4, 6];
  const next = [1, 1, 1, 1, 1];
  const keep = new Set();
  assert.ok(Math.abs(transitionProb(row, next, keep) - 1 / 7776) < 1e-15);
});

test('transitionProb: all kept and all match -> certain (1)', () => {
  const row = [2, 3, 4, 5, 6];
  const next = [2, 3, 4, 5, 6];
  const keep = new Set([0, 1, 2, 3, 4]);
  assert.equal(transitionProb(row, next, keep), 1);
});

test('transitionProb: free-position multiset probability ignores which free slot', () => {
  // keep pos 0 (a 1); reroll 4 dice to become {2,3,4,5} in some order.
  const row = [1, 6, 6, 6, 6];
  const next = [1, 2, 3, 4, 5];
  const keep = new Set([0]);
  // 4 distinct free dice: 4! = 24 arrangements / 6^4 = 1296 -> 24/1296.
  assert.ok(Math.abs(transitionProb(row, next, keep) - 24 / 1296) < 1e-12);
});

test('outcomeProbability: keep 1,2,3,4 reroll one into a 5 -> 1/6', () => {
  assert.ok(Math.abs(outcomeProbability([1, 2, 3, 4], [1, 2, 3, 4, 5]) - 1 / 6) < 1e-12);
});

test('outcomeProbability: order-independent (kept 6 lands anywhere in outcome)', () => {
  // Keep a single 6, reroll four dice into {1,2,3,4}: 4!/6^4 = 24/1296.
  assert.ok(Math.abs(outcomeProbability([6], [1, 2, 3, 4, 6]) - 24 / 1296) < 1e-12);
});

test('outcomeProbability: impossible when a kept value is absent from outcome', () => {
  assert.equal(outcomeProbability([6], [1, 2, 3, 4, 5]), 0);
});

test('outcomeProbability: keep all five (matching) is certain', () => {
  assert.equal(outcomeProbability([2, 3, 4, 5, 6], [2, 3, 4, 5, 6]), 1);
});

test('formatFraction reduces to lowest terms', () => {
  assert.deepEqual(formatFraction([5]), { num: '1', den: '6' });
  assert.deepEqual(formatFraction([]), { num: '1', den: '1' });
});

test('formatOneInN', () => {
  assert.equal(formatOneInN(1 / 6), '1 in 6');
  assert.equal(formatOneInN(0), 'impossible');
  assert.equal(formatOneInN(1), '1 in 1');
});
