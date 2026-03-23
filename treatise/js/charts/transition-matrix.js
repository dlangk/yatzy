/**
 * Transition Matrix: sparse keep-to-outcome probability grid.
 *
 * Pure JS computation: enumerates all 462 keep-multisets and 252 outcome-multisets,
 * checks sub-multiset reachability, computes multinomial probabilities.
 * Renders a canvas pixel grid with interactive dice on left (keep) and right (outcome).
 * Dice drive crosshairs + pulse on the matrix and an info panel below.
 */

import { COLORS, resolveColor } from '../yatzy-viz.js';

// ── Multiset enumeration ─────────────────────────────────────────

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

function frequencies(ms) {
  const f = [0, 0, 0, 0, 0, 0];
  for (const v of ms) f[v - 1]++;
  return f;
}

function isSubMultiset(hFreq, dFreq) {
  for (let i = 0; i < 6; i++) {
    if (hFreq[i] > dFreq[i]) return false;
  }
  return true;
}

function transitionProb(hFreq, dFreq, k) {
  const free = 5 - k;
  if (free === 0) return 1;
  const freeFreq = [];
  for (let i = 0; i < 6; i++) freeFreq.push(dFreq[i] - hFreq[i]);
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
  const outcomes = enumerateMultisets(5);
  const kGroups = [];
  const keeps = [];
  for (let k = 0; k <= 5; k++) {
    const ms = enumerateMultisets(k);
    kGroups.push({ k, start: keeps.length, count: ms.length });
    for (const h of ms) keeps.push({ ms: h, k, freq: frequencies(h) });
  }
  const outcomeFreqs = outcomes.map(d => frequencies(d));
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

// ── Format helpers ───────────────────────────────────────────────

function formatMultiset(ms) {
  if (ms.length === 0) return '(none)';
  return '(' + ms.join(', ') + ')';
}

function formatFraction(p, k) {
  const free = 5 - k;
  if (free === 0) return '1/1';
  const denom = Math.pow(6, free);
  const numer = Math.round(p * denom);
  return `${numer}/${denom}`;
}

// ── Lookup builders ──────────────────────────────────────────────

function buildKeepLookup(keeps) {
  const map = new Map();
  for (let i = 0; i < keeps.length; i++) {
    map.set(keeps[i].ms.join(','), i);
  }
  return map;
}

function buildOutcomeLookup(outcomes) {
  const map = new Map();
  for (let i = 0; i < outcomes.length; i++) {
    map.set(outcomes[i].join(','), i);
  }
  return map;
}

import { createDieSVG } from '/yatzy/shared/dice.js';

// ── Die component (vertical, matches path-probability pattern) ──

function makeDie(value, { kept = false, onToggle, onInc, onDec }) {
  const container = document.createElement('div');
  container.className = 'tm-die-container';

  const upBtn = document.createElement('button');
  upBtn.className = 'tm-die-arrow';
  upBtn.innerHTML = '&#9650;';
  upBtn.addEventListener('click', onInc);

  const state = kept ? 'kept' : 'normal';
  const die = createDieSVG(value, { size: 36, state, clickable: true });
  die.addEventListener('click', onToggle);

  const downBtn = document.createElement('button');
  downBtn.className = 'tm-die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.addEventListener('click', onDec);

  container.appendChild(upBtn);
  container.appendChild(die);
  container.appendChild(downBtn);
  return container;
}

function makeOutcomeDie(value, { onInc, onDec }) {
  const container = document.createElement('div');
  container.className = 'tm-die-container';

  const upBtn = document.createElement('button');
  upBtn.className = 'tm-die-arrow';
  upBtn.innerHTML = '&#9650;';
  upBtn.addEventListener('click', onInc);

  const die = createDieSVG(value, { size: 36, state: 'normal' });

  const downBtn = document.createElement('button');
  downBtn.className = 'tm-die-arrow';
  downBtn.innerHTML = '&#9660;';
  downBtn.addEventListener('click', onDec);

  container.appendChild(upBtn);
  container.appendChild(die);
  container.appendChild(downBtn);
  return container;
}

// ── Canvas rendering ─────────────────────────────────────────────

export async function initTransitionMatrix() {
  const container = document.getElementById('chart-transition-matrix');
  if (!container) return;

  const data = buildTransitionData();
  const caption = container.querySelector('.chart-caption');
  container.innerHTML = '';

  const keepLookup = buildKeepLookup(data.keeps);
  const outcomeLookup = buildOutcomeLookup(data.outcomes);

  const probMap = new Map();
  for (let i = 0; i < data.rows.length; i++) {
    for (const e of data.rows[i]) {
      probMap.set(i * 252 + e.col, e.prob);
    }
  }

  // ── State ──
  const state = {
    dice: [3, 3, 4, 5, 6],     // 5 dice values
    kept: new Set([0, 1]),      // indices of kept dice
    outcome: [1, 3, 3, 4, 5],  // 5 outcome dice values
  };

  // ── Layout: [left dice] [matrix] [right dice] ──
  const row = document.createElement('div');
  row.className = 'tm-layout';

  // Left: keep dice column
  const leftCol = document.createElement('div');
  leftCol.className = 'tm-dice-col';
  const leftLabel = document.createElement('div');
  leftLabel.className = 'tm-dice-label';
  leftLabel.textContent = 'Keep';
  leftCol.appendChild(leftLabel);
  const leftDice = document.createElement('div');
  leftDice.className = 'tm-dice-stack';
  leftCol.appendChild(leftDice);

  // Center: canvas + overlay
  const centerCol = document.createElement('div');
  centerCol.className = 'tm-center';
  centerCol.style.position = 'relative';

  const canvas = document.createElement('canvas');
  canvas.style.display = 'block';

  const overlay = document.createElement('canvas');
  overlay.style.position = 'absolute';
  overlay.style.top = '0';
  overlay.style.left = '0';
  overlay.style.pointerEvents = 'none';

  centerCol.appendChild(canvas);
  centerCol.appendChild(overlay);

  // Right: outcome dice column
  const rightCol = document.createElement('div');
  rightCol.className = 'tm-dice-col';
  const rightLabel = document.createElement('div');
  rightLabel.className = 'tm-dice-label';
  rightLabel.textContent = 'Outcome';
  rightCol.appendChild(rightLabel);
  const rightDice = document.createElement('div');
  rightDice.className = 'tm-dice-stack';
  rightCol.appendChild(rightDice);

  row.appendChild(leftCol);
  row.appendChild(centerCol);
  row.appendChild(rightCol);

  // Floating label (positioned over canvas)
  const floatLabel = document.createElement('div');
  floatLabel.className = 'tm-float-label';
  centerCol.appendChild(floatLabel);

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

  container.appendChild(row);
  container.appendChild(statsPanel);
  if (caption) container.appendChild(caption);

  // ── Draw matrix ────────────────────────────────────────────

  const cellSize = 2;
  const canvasW = cellSize * 252;
  const canvasH = cellSize * 462;

  function drawMatrix() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.width = canvasW + 'px';
    canvas.style.height = canvasH + 'px';

    overlay.width = canvasW * dpr;
    overlay.height = canvasH * dpr;
    overlay.style.width = canvasW + 'px';
    overlay.style.height = canvasH + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    ctx.fillStyle = resolveColor('--bg-alt');
    ctx.fillRect(0, 0, canvasW, canvasH);

    ctx.strokeStyle = resolveColor('--border');
    ctx.lineWidth = 0.5;
    for (const g of data.kGroups) {
      if (g.start > 0) {
        const y = g.start * cellSize;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasW, y);
        ctx.stroke();
      }
    }

    for (let i = 0; i < data.keeps.length; i++) {
      const color = K_COLORS[data.keeps[i].k];
      ctx.fillStyle = color;
      for (const e of data.rows[i]) {
        ctx.fillRect(e.col * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
  }

  // ── Crosshairs + pulse ─────────────────────────────────────

  let pulseFrame = null;

  function clearOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const ctx = overlay.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.scale(dpr, dpr);
    return ctx;
  }

  function drawCrosshairs(keepIdx, outIdx) {
    if (pulseFrame) cancelAnimationFrame(pulseFrame);

    if (keepIdx < 0 || outIdx < 0) {
      clearOverlay();
      pulseFrame = null;
      return;
    }

    const cx = outIdx * cellSize + cellSize / 2;
    const cy = keepIdx * cellSize + cellSize / 2;
    const prob = probMap.get(keepIdx * 252 + outIdx);
    const color = prob !== undefined ? K_COLORS[data.keeps[keepIdx].k] : resolveColor('--text-muted');

    function pulse() {
      const t = (Date.now() % 1000) / 1000;
      const r = cellSize + 2 + Math.sin(t * Math.PI * 2) * 2;
      const alpha = 0.6 + Math.sin(t * Math.PI * 2) * 0.3;

      const ctx = clearOverlay();

      // Crosshair lines
      ctx.strokeStyle = resolveColor('--text');
      ctx.globalAlpha = 0.15;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(canvasW, cy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, canvasH);
      ctx.stroke();

      // Pulse circle
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      pulseFrame = requestAnimationFrame(pulse);
    }
    pulse();
  }

  // ── Resolve dice state to matrix indices ───────────────────

  function getKeepMultiset() {
    const kept = [];
    for (let i = 0; i < 5; i++) {
      if (state.kept.has(i)) kept.push(state.dice[i]);
    }
    return kept.sort((a, b) => a - b);
  }

  function getOutcomeMultiset() {
    return [...state.outcome].sort((a, b) => a - b);
  }

  function lookupIndices() {
    const keepMs = getKeepMultiset();
    const outMs = getOutcomeMultiset();
    const keepIdx = keepLookup.get(keepMs.join(','));
    const outIdx = outcomeLookup.get(outMs.join(','));
    return {
      keepIdx: keepIdx !== undefined ? keepIdx : -1,
      outIdx: outIdx !== undefined ? outIdx : -1,
    };
  }

  // ── Update everything ──────────────────────────────────────

  function renderAll() {
    renderLeftDice();
    renderRightDice();
    const { keepIdx, outIdx } = lookupIndices();
    drawCrosshairs(keepIdx, outIdx);
    renderInfo(keepIdx, outIdx);
  }

  function renderLeftDice() {
    leftDice.innerHTML = '';
    for (let i = 0; i < 5; i++) {
      const die = makeDie(state.dice[i], {
        kept: state.kept.has(i),
        onToggle: () => {
          if (state.kept.has(i)) state.kept.delete(i);
          else state.kept.add(i);
          renderAll();
        },
        onInc: () => {
          state.dice[i] = (state.dice[i] % 6) + 1;
          renderAll();
        },
        onDec: () => {
          state.dice[i] = ((state.dice[i] - 2 + 6) % 6) + 1;
          renderAll();
        },
      });
      leftDice.appendChild(die);
    }
  }

  function renderRightDice() {
    rightDice.innerHTML = '';
    for (let i = 0; i < 5; i++) {
      const die = makeOutcomeDie(state.outcome[i], {
        onInc: () => {
          state.outcome[i] = (state.outcome[i] % 6) + 1;
          renderAll();
        },
        onDec: () => {
          state.outcome[i] = ((state.outcome[i] - 2 + 6) % 6) + 1;
          renderAll();
        },
      });
      rightDice.appendChild(die);
    }
  }

  function renderInfo(keepIdx, outIdx) {
    const keepMs = getKeepMultiset();
    const outMs = getOutcomeMultiset();
    const keptCount = state.kept.size;

    if (keepIdx < 0 || outIdx < 0) {
      floatLabel.style.opacity = '0';
      return;
    }

    // Position near the crosshair point
    const cx = outIdx * cellSize + cellSize / 2;
    const cy = keepIdx * cellSize + cellSize / 2;

    // Flip sides to keep label on-canvas
    const offsetX = cx < canvasW / 2 ? 10 : -10;
    const anchor = cx < canvasW / 2 ? 'left' : 'right';

    floatLabel.style.top = Math.max(0, Math.min(cy - 12, canvasH - 40)) + 'px';
    floatLabel.style[anchor] = (anchor === 'left' ? cx + offsetX : canvasW - cx + Math.abs(offsetX)) + 'px';
    floatLabel.style[anchor === 'left' ? 'right' : 'left'] = 'auto';

    const prob = probMap.get(keepIdx * 252 + outIdx);
    if (prob !== undefined) {
      floatLabel.innerHTML =
        `${formatMultiset(keepMs)} &rarr; ${formatMultiset(outMs)}<br>` +
        `<span class="tm-float-prob">P = ${formatFraction(prob, keptCount)}</span>`;
    } else {
      floatLabel.innerHTML =
        `${formatMultiset(keepMs)} &rarr; ${formatMultiset(outMs)}<br>` +
        `<span style="opacity:0.5">not reachable</span>`;
    }
    floatLabel.style.opacity = '1';
  }

  // ── Initial draw ───────────────────────────────────────────

  drawMatrix();
  renderAll();

  // ── Theme observer ─────────────────────────────────────────

  const themeObserver = new MutationObserver(() => { drawMatrix(); renderAll(); });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  });

  // ── Scoped styles ──────────────────────────────────────────

  if (!document.getElementById('tm-styles')) {
    const style = document.createElement('style');
    style.id = 'tm-styles';
    style.textContent = `
      .tm-layout {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 0.75rem;
      }
      .tm-dice-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
        padding-top: 0.25rem;
      }
      .tm-dice-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
      }
      .tm-dice-stack {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .tm-die-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
      }
      .tm-die-arrow {
        width: 36px;
        height: 14px;
        font-size: 9px;
        padding: 0;
        border: 1px solid var(--border);
        background: var(--bg-alt);
        cursor: pointer;
        border-radius: 3px;
        line-height: 12px;
        color: var(--text);
      }
      .tm-die-arrow:hover {
        background: var(--accent-light);
      }
      .tm-die-btn {
        width: 36px;
        height: 36px;
        font-size: 18px;
        font-weight: bold;
        font-family: var(--font-mono);
        border-radius: 6px;
        color: var(--text);
      }
      .tm-center {
        flex-shrink: 0;
      }
      .tm-float-label {
        position: absolute;
        pointer-events: none;
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.3rem 0.5rem;
        font-size: 0.75rem;
        line-height: 1.4;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        opacity: 0;
        transition: opacity 0.15s;
        z-index: 10;
      }
      .tm-float-prob {
        font-family: var(--font-mono);
        color: var(--accent);
      }
      @media (max-width: 620px) {
        .tm-layout { flex-direction: column; align-items: center; }
        .tm-dice-col { flex-direction: row; }
        .tm-dice-stack { flex-direction: row; gap: 4px; }
      }
    `;
    document.head.appendChild(style);
  }
}
