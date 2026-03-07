/**
 * Transition Matrix — sparse keep-to-outcome probability grid.
 *
 * Pure JS computation: enumerates all 462 keep-multisets and 252 outcome-multisets,
 * checks sub-multiset reachability, computes multinomial probabilities.
 * Renders a canvas pixel grid with k-group coloring and hover details.
 */

import { COLORS, resolveColor } from '../yatzy-viz.js';

// ── Multiset enumeration ─────────────────────────────────────────

/** Generate all sorted multisets of size k from {1..maxVal}. */
function enumerateMultisets(k, maxVal = 6) {
  const results = [];
  const current = [];
  function recurse(depth, minVal) {
    if (depth === k) { results.push([...current]); return; }
    for (let v = minVal; v <= maxVal; v++) {
      current.push(v);
      recurse(depth + 1, v);
      current.pop();
    }
  }
  recurse(0, 1);
  return results;
}

/** Count occurrences of each face value (1-6) in a multiset. */
function frequencies(ms) {
  const f = [0, 0, 0, 0, 0, 0];
  for (const v of ms) f[v - 1]++;
  return f;
}

/** Check if keep h is a sub-multiset of outcome d. */
function isSubMultiset(hFreq, dFreq) {
  for (let i = 0; i < 6; i++) {
    if (hFreq[i] > dFreq[i]) return false;
  }
  return true;
}

/** Compute P(keep -> outcome) = multinomial(free dice) / 6^(5-k). */
function transitionProb(hFreq, dFreq, k) {
  const free = 5 - k;
  if (free === 0) return 1; // keep all: deterministic

  // Free dice frequencies
  const freeFreq = [];
  for (let i = 0; i < 6; i++) freeFreq.push(dFreq[i] - hFreq[i]);

  // Multinomial coefficient: free! / product(freeFreq[i]!)
  let numer = 1;
  for (let i = 2; i <= free; i++) numer *= i;
  let denom = 1;
  for (const f of freeFreq) {
    for (let i = 2; i <= f; i++) denom *= i;
  }

  return (numer / denom) / Math.pow(6, free);
}

// ── Data builder ─────────────────────────────────────────────────

function buildTransitionData() {
  const outcomes = enumerateMultisets(5); // 252

  // All keeps grouped by k
  const kGroups = [];
  const keeps = [];
  for (let k = 0; k <= 5; k++) {
    const ms = enumerateMultisets(k);
    kGroups.push({ k, start: keeps.length, count: ms.length });
    for (const h of ms) keeps.push({ ms: h, k, freq: frequencies(h) });
  }

  const outcomeFreqs = outcomes.map(d => frequencies(d));

  // Build sparse rows
  let nnz = 0;
  const rows = keeps.map(h => {
    const entries = [];
    for (let j = 0; j < 252; j++) {
      if (isSubMultiset(h.freq, outcomeFreqs[j])) {
        const p = transitionProb(h.freq, outcomeFreqs[j], h.k);
        entries.push({ col: j, prob: p });
        nnz++;
      }
    }
    return entries;
  });

  return { keeps, outcomes, outcomeFreqs, rows, kGroups, nnz };
}

// ── K-group colors ───────────────────────────────────────────────

const K_COLORS = [
  COLORS.accent,      // k=0: orange
  COLORS.riskAverse,  // k=1: blue
  '#2ca02c',          // k=2: green
  '#7b3294',          // k=3: purple
  '#e7298a',          // k=4: pink
  COLORS.riskSeeking, // k=5: red
];

const K_LABELS = [
  'k=0 (keep nothing)',
  'k=1 (keep 1)',
  'k=2 (keep 2)',
  'k=3 (keep 3)',
  'k=4 (keep 4)',
  'k=5 (keep all)',
];

// ── Format helpers ───────────────────────────────────────────────

function formatMultiset(ms) {
  if (ms.length === 0) return '()';
  return '(' + ms.join(', ') + ')';
}

function formatProb(p) {
  // Express as fraction of 6^free if clean
  if (p === 1) return '1';
  return p.toFixed(4);
}

function formatFraction(p, k) {
  const free = 5 - k;
  if (free === 0) return '1/1';
  const denom = Math.pow(6, free);
  const numer = Math.round(p * denom);
  return `${numer}/${denom}`;
}

// ── Canvas rendering ─────────────────────────────────────────────

export async function initTransitionMatrix() {
  const container = document.getElementById('chart-transition-matrix');
  if (!container) return;

  const data = buildTransitionData();
  const caption = container.querySelector('.chart-caption');
  container.innerHTML = '';

  // Wrapper
  const wrapper = document.createElement('div');

  // Canvas
  const canvas = document.createElement('canvas');
  canvas.style.display = 'block';
  canvas.style.margin = '0 auto';
  canvas.style.cursor = 'crosshair';
  wrapper.appendChild(canvas);

  // Hover panel
  const hoverPanel = document.createElement('div');
  hoverPanel.className = 'transition-hover-panel';
  hoverPanel.innerHTML = '<span style="opacity:0.5">Hover over the matrix to inspect entries</span>';
  wrapper.appendChild(hoverPanel);

  // Legend
  const legend = document.createElement('div');
  legend.className = 'chart-legend';
  legend.style.marginTop = '0.5rem';
  for (let k = 0; k <= 5; k++) {
    const item = document.createElement('div');
    item.className = 'chart-legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = K_COLORS[k];
    const label = document.createElement('span');
    label.textContent = `k=${k} (${data.kGroups[k].count})`;
    item.append(swatch, label);
    legend.appendChild(item);
  }
  wrapper.appendChild(legend);

  // Stats bar
  const statsPanel = document.createElement('div');
  statsPanel.className = 'chart-stats-panel';
  statsPanel.style.gridTemplateColumns = 'repeat(4, 1fr)';
  const stats = [
    { value: '462', label: 'unique keeps' },
    { value: '252', label: 'outcomes' },
    { value: data.nnz.toLocaleString(), label: 'non-zero' },
    { value: ((1 - data.nnz / (462 * 252)) * 100).toFixed(1) + '%', label: 'sparse' },
  ];
  for (const s of stats) {
    const card = document.createElement('div');
    card.className = 'chart-stat';
    card.innerHTML = `<div class="chart-stat-value">${s.value}</div><div class="chart-stat-label">${s.label}</div>`;
    statsPanel.appendChild(card);
  }
  wrapper.appendChild(statsPanel);

  container.appendChild(wrapper);
  if (caption) container.appendChild(caption);

  // Build a fast lookup for probabilities: key = keepIdx * 252 + outcomeIdx
  const probMap = new Map();
  for (let i = 0; i < data.rows.length; i++) {
    for (const e of data.rows[i]) {
      probMap.set(i * 252 + e.col, e.prob);
    }
  }

  // ── Draw (horizontal: x=keeps 462, y=outcomes 252) ─────────

  let cellSize = 1;
  let canvasW = 0;
  let canvasH = 0;

  function draw() {
    cellSize = 2;
    canvasW = cellSize * 462;  // 924px — bleeds past column, centered
    canvasH = cellSize * 252;  // 504px

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.width = canvasW + 'px';
    canvas.style.height = canvasH + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = resolveColor('--bg-alt');
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Draw k-group vertical separators
    ctx.strokeStyle = resolveColor('--border');
    ctx.lineWidth = 0.5;
    for (const g of data.kGroups) {
      if (g.start > 0) {
        const x = g.start * cellSize;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasH);
        ctx.stroke();
      }
    }

    // Draw non-zero cells: x=keep index, y=outcome index
    for (let i = 0; i < data.keeps.length; i++) {
      const color = K_COLORS[data.keeps[i].k];
      ctx.fillStyle = color;
      for (const e of data.rows[i]) {
        ctx.fillRect(i * cellSize, e.col * cellSize, cellSize, cellSize);
      }
    }
  }

  draw();

  // ── Hover ────────────────────────────────────────────────────

  let lastKeepIdx = -1;
  let lastOutIdx = -1;

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const keepIdx = Math.floor(x / cellSize);
    const outIdx = Math.floor(y / cellSize);

    if (keepIdx === lastKeepIdx && outIdx === lastOutIdx) return;
    lastKeepIdx = keepIdx;
    lastOutIdx = outIdx;

    if (keepIdx < 0 || keepIdx >= 462 || outIdx < 0 || outIdx >= 252) {
      hoverPanel.innerHTML = '<span style="opacity:0.5">Hover over the matrix to inspect entries</span>';
      return;
    }

    const keep = data.keeps[keepIdx];
    const outcome = data.outcomes[outIdx];
    const prob = probMap.get(keepIdx * 252 + outIdx);

    if (prob !== undefined) {
      const free = 5 - keep.k;
      hoverPanel.innerHTML =
        `<span class="keep-label">Keep: ${formatMultiset(keep.ms)}</span>` +
        ` &rarr; <span class="outcome-label">Outcome: ${formatMultiset(outcome)}</span>` +
        ` &middot; <span class="prob-value">P = ${formatFraction(prob, keep.k)} = ${prob.toFixed(4)}</span>` +
        ` &middot; ${free} free ${free === 1 ? 'die' : 'dice'}`;
    } else {
      hoverPanel.innerHTML =
        `<span class="keep-label">Keep: ${formatMultiset(keep.ms)}</span>` +
        ` &rarr; <span class="outcome-label">Outcome: ${formatMultiset(outcome)}</span>` +
        ` &middot; <span style="opacity:0.5">not reachable</span>`;
    }
  });

  canvas.addEventListener('mouseleave', () => {
    lastKeepIdx = -1;
    lastOutIdx = -1;
    hoverPanel.innerHTML = '<span style="opacity:0.5">Hover over the matrix to inspect entries</span>';
  });

  // ── Theme observer ───────────────────────────────────────────

  const themeObserver = new MutationObserver(() => draw());
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  });

  // ── Resize observer ──────────────────────────────────────────

  const resizeObserver = new ResizeObserver(() => draw());
  resizeObserver.observe(container);
}
