/**
 * Keep Funnel Chart — visualizes how 32 raw keep subsets collapse
 * into N unique keep-multisets for a given dice roll.
 *
 * Three zones: dice tray (left), dedup grid (center), unique keeps (right).
 */

import { DataLoader } from '../data-loader.js';
import { renderDiceSelectable } from '../utils/dice-interactive.js';
import { createDieSVG } from '/yatzy/shared/dice.js';

function makeMinDie(face, kept) {
  const state = kept ? 'kept' : 'reroll';
  const wrap = document.createElement('div');
  wrap.className = kept ? 'keep-die keep-die-kept' : 'keep-die keep-die-reroll';
  wrap.appendChild(createDieSVG(face, { size: 20, state }));
  return wrap;
}

// Categorical palette — enough distinct hues for up to 32 keeps
function keepPalette(n) {
  const colors = [];
  for (let i = 0; i < n; i++) {
    const hue = (i * 360 / n + 20) % 360;
    colors.push(`hsl(${hue}, 62%, 55%)`);
  }
  return colors;
}

export async function initKeepFunnel() {
  const container = document.getElementById('chart-keep-funnel');
  if (!container) return;

  const data = await DataLoader.diceSymmetry();
  const { families, multisets } = data;

  // ------- State -------
  let currentMs = null;
  let displayDice = [];
  let hoveredKeepIdx = -1;

  // ------- Build DOM -------
  container.innerHTML = '';

  // Top row: dice + roll button
  const topRow = document.createElement('div');
  topRow.className = 'dice-tray';
  const trayTop = document.createElement('div');
  trayTop.className = 'dice-tray-top';
  const diceRow = document.createElement('div');
  diceRow.className = 'dice-tray-row';
  const rollBtn = document.createElement('button');
  rollBtn.className = 'chart-btn dice-roll-btn';
  rollBtn.textContent = 'Roll';
  const patternLabel = document.createElement('span');
  patternLabel.className = 'keep-funnel-pattern';
  trayTop.appendChild(diceRow);
  trayTop.appendChild(rollBtn);
  trayTop.appendChild(patternLabel);
  topRow.appendChild(trayTop);
  container.appendChild(topRow);

  // Main content: grid (left) + keeps (right)
  const funnel = document.createElement('div');
  funnel.className = 'keep-funnel';

  // Center: dedup grid
  const centerZone = document.createElement('div');
  centerZone.className = 'keep-funnel-center';
  const centerLabel = document.createElement('div');
  centerLabel.className = 'keep-funnel-center-label';
  centerLabel.textContent = '32 Possible Keeps';
  const grid = document.createElement('div');
  grid.className = 'keep-funnel-grid';
  centerZone.appendChild(centerLabel);
  centerZone.appendChild(grid);
  funnel.appendChild(centerZone);

  // Right: unique keeps
  const rightZone = document.createElement('div');
  rightZone.className = 'keep-funnel-right';
  const rightLabel = document.createElement('div');
  rightLabel.className = 'keep-funnel-right-label';
  rightLabel.textContent = 'Unique keeps';
  const keepsList = document.createElement('div');
  keepsList.className = 'keep-funnel-keeps';
  rightZone.appendChild(rightLabel);
  rightZone.appendChild(keepsList);
  funnel.appendChild(rightZone);

  container.appendChild(funnel);

  // Summary
  const summary = document.createElement('div');
  summary.className = 'keep-funnel-summary';
  container.appendChild(summary);

  // ------- Roll -------
  function rollDice() {
    const display = new Array(5);
    for (let i = 0; i < 5; i++) {
      display[i] = Math.floor(Math.random() * 6) + 1;
    }
    const sorted = [...display].sort((a, b) => a - b);
    const key = sorted.join(',');
    const ms = multisets.find(m => m.dice.join(',') === key);
    currentMs = ms;
    displayDice = display;
    hoveredKeepIdx = -1;
    render();
  }

  // ------- Render -------
  function render() {
    if (!currentMs) return;

    // Dice tray (read-only)
    diceRow.innerHTML = '';
    renderDiceSelectable(diceRow, displayDice, { size: 40, locked: true });

    // Pattern label
    const fam = families[currentMs.familyIndex];
    patternLabel.textContent = fam.name;

    const keeps = currentMs.keeps;
    const palette = keepPalette(keeps.length);

    // Build grid cells: expand each keep's masks into individual cells
    grid.innerHTML = '';
    const cellKeepMap = []; // cellIndex → keepIndex
    for (let ki = 0; ki < keeps.length; ki++) {
      for (let m = 0; m < keeps[ki].masks; m++) {
        cellKeepMap.push(ki);
      }
    }

    for (let ci = 0; ci < 32; ci++) {
      const ki = cellKeepMap[ci];
      const cell = document.createElement('div');
      cell.className = 'keep-funnel-cell';
      cell.style.background = palette[ki];
      cell.dataset.keepIdx = ki;
      cell.addEventListener('mouseenter', () => setHover(ki));
      cell.addEventListener('mouseleave', () => setHover(-1));
      grid.appendChild(cell);
    }

    // Build keeps list
    keepsList.innerHTML = '';
    for (let ki = 0; ki < keeps.length; ki++) {
      const keep = keeps[ki];
      const row = document.createElement('div');
      row.className = 'keep-funnel-keep-row';
      row.style.borderLeftColor = palette[ki];
      row.dataset.keepIdx = ki;

      // Mini dice: show which are kept
      const diceWrap = document.createElement('div');
      diceWrap.className = 'keep-row-dice';
      const keptSet = new Set();
      const usedRoll = new Array(5).fill(false);
      for (const v of keep.dice) {
        for (let ri = 0; ri < 5; ri++) {
          if (!usedRoll[ri] && displayDice[ri] === v) {
            keptSet.add(ri);
            usedRoll[ri] = true;
            break;
          }
        }
      }
      for (let i = 0; i < 5; i++) {
        diceWrap.appendChild(makeMinDie(displayDice[i], keptSet.has(i)));
      }
      row.appendChild(diceWrap);

      // Masks badge
      if (keep.masks > 1) {
        const badge = document.createElement('span');
        badge.className = 'keep-masks';
        badge.textContent = `×${keep.masks}`;
        row.appendChild(badge);
      }

      row.addEventListener('mouseenter', () => setHover(ki));
      row.addEventListener('mouseleave', () => setHover(-1));
      keepsList.appendChild(row);
    }

    // Summary line
    summary.innerHTML =
      `<strong>32</strong> subsets → <strong>${keeps.length}</strong> unique keeps` +
      `&nbsp;&nbsp;·&nbsp;&nbsp;Across all 252 rolls: <strong>462</strong> total unique keeps`;

    updateHighlight();
  }

  function setHover(ki) {
    hoveredKeepIdx = ki;
    updateHighlight();
  }

  function updateHighlight() {
    // Grid cells
    const cells = grid.querySelectorAll('.keep-funnel-cell');
    cells.forEach(cell => {
      const ki = parseInt(cell.dataset.keepIdx);
      if (hoveredKeepIdx < 0) {
        cell.classList.remove('highlight', 'dimmed');
      } else {
        cell.classList.toggle('highlight', ki === hoveredKeepIdx);
        cell.classList.toggle('dimmed', ki !== hoveredKeepIdx);
      }
    });

    // Keep rows
    const rows = keepsList.querySelectorAll('.keep-funnel-keep-row');
    rows.forEach(row => {
      const ki = parseInt(row.dataset.keepIdx);
      row.classList.toggle('highlight', ki === hoveredKeepIdx);
    });
  }

  rollBtn.addEventListener('click', rollDice);

  // Initial roll: match the 3,3,3,5,5 example in the prose
  const defaultMs = multisets.find(m => m.dice.join(',') === '3,3,3,5,5');
  if (defaultMs) {
    currentMs = defaultMs;
    displayDice = [3, 3, 3, 5, 5];
    render();
  } else {
    rollDice(); // fallback
  }
}
