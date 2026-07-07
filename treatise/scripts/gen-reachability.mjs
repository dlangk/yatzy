#!/usr/bin/env node
// Generate treatise/data/reachability.json — the state-space pruning chart data.
//
// State = (upper_score 0..63) × (15-bit scored-category mask). upper_score
// reachability depends ONLY on which of the 6 upper categories (bits 0..5) are
// scored. This mirrors the solver's exact DP in solver/src/phase0_tables.rs
// (precompute_reachability): each scored upper category i (face = i+1)
// contributes k·face for k in 0..5 (0 allowed), and upper_score 63 means "≥63".
//
// Output shape (consumed by treatise/js/charts/reachability-pruning.js):
//   { by_popcount: [ { popcount, total, reachable }, ... ] }  // popcount 0..15
//
//   node treatise/scripts/gen-reachability.mjs

import { writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

// --- Exact upper-section reachability DP (ports precompute_reachability) ---
// r[n][mask] for n in 0..105, mask in 0..63 (6-bit upper mask)
const NMAX = 105;
const r = Array.from({ length: NMAX + 1 }, () => new Uint8Array(64));
r[0][0] = 1;
for (let face = 1; face <= 6; face++) {
  const bit = 1 << (face - 1);
  for (let n = NMAX; n >= 0; n--) {
    for (let mask = 0; mask < 64; mask++) {
      if ((mask & bit) === 0 || r[n][mask]) continue;
      const prev = mask ^ bit;
      for (let k = 0; k <= 5; k++) {
        const contrib = k * face;
        if (contrib > n) break;
        if (r[n - contrib][prev]) {
          r[n][mask] = 1;
          break;
        }
      }
    }
  }
}

// reachableCount[um] = # of upper_scores 0..63 reachable for upper-mask um
// (with 63 meaning "≥63": OR of exact values 63..105)
const reachableCount = new Array(64).fill(0);
for (let mask = 0; mask < 64; mask++) {
  let cnt = 0;
  for (let n = 0; n < 63; n++) if (r[n][mask]) cnt++;
  for (let n = 63; n <= NMAX; n++) {
    if (r[n][mask]) {
      cnt++;
      break;
    }
  }
  reachableCount[mask] = cnt;
}

// Sanity check against the solver's documented "2794 / 4096 reachable pairs".
const totalReachablePairs = reachableCount.reduce((a, b) => a + b, 0);
if (totalReachablePairs !== 2794) {
  console.error(
    `✗ reachable-pair check failed: got ${totalReachablePairs}, expected 2794. ` +
      `Reachability DP does not match the solver — aborting.`
  );
  process.exit(1);
}

// --- Aggregate by popcount of the full 15-bit category mask ---
// A 15-bit mask with popcount p splits into a 6-bit upper part (um, u bits set)
// and a 9-bit lower part (p-u bits set). Lower categories don't constrain
// upper_score, so:
//   reachable(p) = Σ_um reachableCount(um) · C(9, p - popcount(um))
//   total(p)     = C(15, p) · 64
const popcount = (m) => {
  let c = 0;
  while (m) {
    c += m & 1;
    m >>= 1;
  }
  return c;
};
const choose = (() => {
  const C = Array.from({ length: 16 }, () => new Array(16).fill(0));
  for (let n = 0; n <= 15; n++) {
    C[n][0] = 1;
    for (let k = 1; k <= n; k++) C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
  }
  return (n, k) => (k < 0 || k > n ? 0 : C[n][k]);
})();

const byPopcount = [];
for (let p = 0; p <= 15; p++) {
  let reachable = 0;
  for (let um = 0; um < 64; um++) {
    reachable += reachableCount[um] * choose(9, p - popcount(um));
  }
  byPopcount.push({ popcount: p, total: choose(15, p) * 64, reachable });
}

const outPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "data",
  "reachability.json"
);
writeFileSync(outPath, JSON.stringify({ by_popcount: byPopcount }, null, 2));

const grandTotal = byPopcount.reduce((a, d) => a + d.total, 0);
const grandReach = byPopcount.reduce((a, d) => a + d.reachable, 0);
console.log(
  `✓ reachable-pair check: ${totalReachablePairs} / 4096 upper pairs\n` +
    `Wrote ${outPath}\n` +
    `  full state space: ${grandReach.toLocaleString()} reachable / ` +
    `${grandTotal.toLocaleString()} total (${((1 - grandReach / grandTotal) * 100).toFixed(1)}% pruned)`
);
