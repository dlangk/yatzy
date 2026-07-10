import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rerollOutcomes, keepSets, makeSolver } from './engine.js';

const sorted = (a) => a.slice().sort((x, y) => x - y);

test('rerollOutcomes: k=0 is the empty certain outcome', () => {
  assert.deepEqual(rerollOutcomes(0), [{ dice: [], p: 1 }]);
});

test('rerollOutcomes: k=1 is six equally likely faces', () => {
  const out = rerollOutcomes(1);
  assert.equal(out.length, 6);
  for (const o of out) assert.ok(Math.abs(o.p - 1 / 6) < 1e-12);
});

test('rerollOutcomes: probabilities sum to 1 for k=0..5', () => {
  for (let k = 0; k <= 5; k++) {
    const s = rerollOutcomes(k).reduce((acc, o) => acc + o.p, 0);
    assert.ok(Math.abs(s - 1) < 1e-9, `k=${k} sum=${s}`);
  }
});

test('keepSets: distinct sub-multisets include empty and full', () => {
  const ks = keepSets([1, 1, 2, 3, 4]).map((k) => k.join(''));
  assert.ok(ks.includes(''));
  assert.ok(ks.includes('11234'));
  assert.equal(new Set(ks).size, ks.length);
});

const smallStraight = (h) => sorted(h).join('') === '12345';

test('pOpt: r=0 is 1 when target met, 0 otherwise', () => {
  const s = makeSolver(smallStraight);
  assert.equal(s.pOpt([1, 2, 3, 4, 5], 0), 1);
  assert.equal(s.pOpt([1, 2, 3, 4, 6], 0), 0);
});

test('pYou == 1/6: keep 1234, reroll one die, need a 5 (one reroll)', () => {
  const s = makeSolver(smallStraight);
  const p = s.pYou([1, 2, 3, 4, 6], 1, [1, 2, 3, 4]);
  assert.ok(Math.abs(p - 1 / 6) < 1e-12, `p=${p}`);
});

test('pOpt with one reroll from [1,2,3,4,6] equals best hold (1/6)', () => {
  const s = makeSolver(smallStraight);
  assert.ok(Math.abs(s.pOpt([1, 2, 3, 4, 6], 1) - 1 / 6) < 1e-12);
  assert.deepEqual(s.bestKeep([1, 2, 3, 4, 6], 1), [1, 2, 3, 4]);
});

test('pYou equals pOpt when your keep is the optimal keep', () => {
  const s = makeSolver(smallStraight);
  const hand = [1, 2, 3, 4, 6];
  const best = s.bestKeep(hand, 2);
  assert.ok(Math.abs(s.pYou(hand, 2, best) - s.pOpt(hand, 2)) < 1e-12);
});

test('more rerolls never hurt: pOpt(hand,2) >= pOpt(hand,1)', () => {
  const s = makeSolver(smallStraight);
  const hand = [1, 2, 3, 4, 6];
  assert.ok(s.pOpt(hand, 2) >= s.pOpt(hand, 1) - 1e-12);
});

test('yatzy over a fresh game ~ 4.603% (known odds)', () => {
  // The classic Yahtzee probability averages over the opening roll then plays
  // two optimal rerolls: Σ p(openingHand) · pOpt(openingHand, 2).
  const yatzy = (h) => new Set(h).size === 1;
  const s = makeSolver(yatzy);
  let p = 0;
  for (const { dice, p: pr } of rerollOutcomes(5)) p += pr * s.pOpt(dice, 2);
  assert.ok(Math.abs(p - 0.04603) < 0.0005, `p=${p}`);
});
