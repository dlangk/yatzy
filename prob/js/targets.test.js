import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CATEGORIES, categoryById, exactTarget } from './targets.js';

const cat = (id) => categoryById(id).test;

test('one pair', () => {
  assert.equal(cat('one_pair')([1, 1, 3, 4, 6]), true);
  assert.equal(cat('one_pair')([1, 2, 3, 4, 6]), false);
});

test('two pairs (yatzy is not two pairs)', () => {
  assert.equal(cat('two_pairs')([1, 1, 4, 4, 6]), true);
  assert.equal(cat('two_pairs')([1, 1, 4, 6, 6]), true);
  assert.equal(cat('two_pairs')([1, 1, 1, 4, 4]), true);
  assert.equal(cat('two_pairs')([1, 1, 3, 4, 6]), false);
  assert.equal(cat('two_pairs')([5, 5, 5, 5, 5]), false);
});

test('three and four of a kind', () => {
  assert.equal(cat('three_kind')([2, 2, 2, 4, 6]), true);
  assert.equal(cat('three_kind')([2, 2, 4, 4, 6]), false);
  assert.equal(cat('four_kind')([2, 2, 2, 2, 6]), true);
  assert.equal(cat('four_kind')([2, 2, 2, 4, 6]), false);
});

test('full house is exactly 3+2 of distinct values', () => {
  assert.equal(cat('full_house')([2, 2, 2, 5, 5]), true);
  assert.equal(cat('full_house')([2, 2, 2, 2, 5]), false);
  assert.equal(cat('full_house')([5, 5, 5, 5, 5]), false);
  assert.equal(cat('full_house')([2, 2, 3, 5, 5]), false);
});

test('small and large straight are the fixed patterns', () => {
  assert.equal(cat('small_straight')([1, 2, 3, 4, 5]), true);
  assert.equal(cat('small_straight')([2, 3, 4, 5, 6]), false);
  assert.equal(cat('large_straight')([2, 3, 4, 5, 6]), true);
  assert.equal(cat('large_straight')([1, 2, 3, 4, 5]), false);
});

test('yatzy is five of a kind', () => {
  assert.equal(cat('yatzy')([4, 4, 4, 4, 4]), true);
  assert.equal(cat('yatzy')([4, 4, 4, 4, 6]), false);
});

test('CATEGORIES has 8 entries with unique ids and labels', () => {
  assert.equal(CATEGORIES.length, 8);
  assert.equal(new Set(CATEGORIES.map((c) => c.id)).size, 8);
  for (const c of CATEGORIES) assert.ok(c.label && !c.label.includes('--'));
});

test('exactTarget matches regardless of order', () => {
  const t = exactTarget([6, 4, 3, 2, 1]);
  assert.equal(t([1, 2, 3, 4, 6]), true);
  assert.equal(t([1, 2, 3, 4, 5]), false);
});
