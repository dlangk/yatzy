import { test } from 'node:test';
import assert from 'node:assert/strict';
import { integrateTail, pAtLeast } from '../../shared/score-prob.js';

// Uniform density on [0,100] with height 0.01 integrates to 1.
const uniform = [];
for (let s = 0; s <= 100; s += 1) uniform.push({ x: s, y: 0.01 });

test('integrateTail: full mass above 0 is ~1', () => {
  assert.ok(Math.abs(integrateTail(uniform, 0, true) - 1) < 1e-9);
});

test('integrateTail: above 50 of a uniform is ~0.5', () => {
  assert.ok(Math.abs(integrateTail(uniform, 50, true) - 0.5) < 1e-9);
});

test('integrateTail: below 50 of a uniform is ~0.5', () => {
  assert.ok(Math.abs(integrateTail(uniform, 50, false) - 0.5) < 1e-9);
});

test('integrateTail: above a threshold past the support is ~0', () => {
  assert.ok(integrateTail(uniform, 100, true) < 1e-9);
});

test('integrateTail: handles a threshold between grid points (above 75.5 ~ 0.245)', () => {
  assert.ok(Math.abs(integrateTail(uniform, 75.5, true) - 0.245) < 1e-9);
});

test('pAtLeast is integrateTail above and is monotone decreasing', () => {
  const a = pAtLeast(uniform, 20);
  const b = pAtLeast(uniform, 60);
  assert.ok(Math.abs(a - 0.8) < 1e-9);
  assert.ok(Math.abs(b - 0.4) < 1e-9);
  assert.ok(a > b);
});
