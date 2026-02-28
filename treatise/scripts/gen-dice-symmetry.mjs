#!/usr/bin/env node
/**
 * Generate dice_symmetry.json — all 252 sorted 5-dice multisets with
 * pattern families, multinomial probabilities, and unique keeps.
 */

import { writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(__dirname, '..', 'data', 'dice_symmetry.json');

// ------- Families by sorted frequency signature -------
const FAMILIES = [
  { key: 'abcde', name: 'All Different',   sig: '1,1,1,1,1', color: '#4e79a7' },
  { key: 'aabcd', name: 'One Pair',        sig: '1,1,1,2',   color: '#f28e2b' },
  { key: 'aabbc', name: 'Two Pair',        sig: '1,2,2',     color: '#e15759' },
  { key: 'aaabc', name: 'Three of a Kind', sig: '1,1,3',     color: '#76b7b2' },
  { key: 'aaabb', name: 'Full House',      sig: '2,3',       color: '#59a14f' },
  { key: 'aaaab', name: 'Four of a Kind',  sig: '1,4',       color: '#edc948' },
  { key: 'aaaaa', name: 'Yatzy',           sig: '5',         color: '#b07aa1' },
];

function familyBySignature(sig) {
  return FAMILIES.findIndex(f => f.sig === sig);
}

/** Sorted frequency signature of a sorted multiset */
function signature(dice) {
  const counts = [];
  let i = 0;
  while (i < dice.length) {
    let j = i + 1;
    while (j < dice.length && dice[j] === dice[i]) j++;
    counts.push(j - i);
    i = j;
  }
  return counts.sort((a, b) => a - b).join(',');
}

/** Multinomial coefficient: n! / (k1! * k2! * ... ) */
function multinomial(dice) {
  const counts = new Map();
  for (const d of dice) counts.set(d, (counts.get(d) || 0) + 1);
  // 5! / product(count_i!)
  let num = 120; // 5!
  for (const c of counts.values()) {
    let f = 1;
    for (let i = 2; i <= c; i++) f *= i;
    num /= f;
  }
  return num;
}

/** Enumerate unique keep sub-multisets for a roll via all 32 bitmasks */
function uniqueKeeps(dice) {
  const seen = new Set();
  const keeps = [];
  for (let mask = 0; mask < 32; mask++) {
    const kept = [];
    for (let b = 0; b < 5; b++) {
      if (mask & (1 << b)) kept.push(dice[b]);
    }
    kept.sort((a, b) => a - b);
    const key = kept.join(',');
    if (!seen.has(key)) {
      seen.add(key);
      keeps.push({ dice: kept, masks: 1 });
    } else {
      // Count how many masks produce this keep
      const existing = keeps.find(k => k.dice.join(',') === key);
      existing.masks++;
    }
  }
  return keeps;
}

// ------- Enumerate all 252 sorted multisets -------
const multisets = [];
let id = 0;
for (let a = 1; a <= 6; a++)
  for (let b = a; b <= 6; b++)
    for (let c = b; c <= 6; c++)
      for (let d = c; d <= 6; d++)
        for (let e = d; e <= 6; e++) {
          const dice = [a, b, c, d, e];
          const sig = signature(dice);
          const fi = familyBySignature(sig);
          const perms = multinomial(dice);
          const keeps = uniqueKeeps(dice);
          multisets.push({
            id: id++,
            dice,
            family: FAMILIES[fi].key,
            familyIndex: fi,
            permutations: perms,
            probability: +(perms / 7776).toFixed(6),
            keeps: keeps.map(k => ({ dice: k.dice, masks: k.masks })),
          });
        }

// ------- Build family summaries -------
const familyCounts = FAMILIES.map(() => 0);
const familyPerms = FAMILIES.map(() => 0);
for (const ms of multisets) {
  familyCounts[ms.familyIndex]++;
  familyPerms[ms.familyIndex] += ms.permutations;
}

const families = FAMILIES.map((f, i) => ({
  ...f,
  count: familyCounts[i],
  totalPermutations: familyPerms[i],
  example: multisets.find(m => m.familyIndex === i).dice,
}));

// ------- Assertions -------
console.assert(multisets.length === 252, `Expected 252 multisets, got ${multisets.length}`);

const permSum = multisets.reduce((s, m) => s + m.permutations, 0);
console.assert(permSum === 7776, `Expected permutation sum 7776, got ${permSum}`);

const expectedCounts = [6, 60, 60, 60, 30, 30, 6];
for (let i = 0; i < 7; i++) {
  console.assert(
    familyCounts[i] === expectedCounts[i],
    `Family ${FAMILIES[i].key}: expected ${expectedCounts[i]}, got ${familyCounts[i]}`
  );
}

console.log(`✓ 252 multisets, 7776 permutations, family partition correct`);
console.log(`  Families: ${familyCounts.join(' + ')} = ${familyCounts.reduce((a, b) => a + b)}`);

// ------- Write output -------
const output = {
  families,
  multisets,
  totalMultisets: 252,
  totalOrdered: 7776,
};

writeFileSync(OUT, JSON.stringify(output, null, 2));
console.log(`Wrote ${OUT} (${(JSON.stringify(output).length / 1024).toFixed(1)} KB)`);
