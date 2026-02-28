/**
 * Dice Symmetry Explorer — interactive visualization of the 252 multisets.
 *
 * Three zones: dice tray (roll/hold), pattern table (7 families),
 * 252-grid (hover/click quintuplets).
 */

import { DataLoader } from '../data-loader.js';
import { tooltip, getTextColor, getMutedColor, getGridColor, COLORS } from '../yatzy-viz.js';
import { renderDiceSelectable } from '../utils/dice-interactive.js';

export async function initDiceSymmetry() {
  const container = document.getElementById('chart-dice-symmetry');
  if (!container) return;

  const data = await DataLoader.diceSymmetry();
  const { families, multisets } = data;

  // ------- State -------
  const state = {
    currentId: -1,        // active multiset id
    heldIndices: new Set(),
    hoveredId: -1,
  };

  // Die face → color ramp (1=light → 6=dark)
  const FACE_COLORS = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'];

  // ------- Build DOM -------
  container.innerHTML = '';
  container.classList.add('dice-symmetry-explorer');

  // Zone 1: Dice Tray
  const tray = document.createElement('div');
  tray.className = 'dice-tray';

  const diceRow = document.createElement('div');
  diceRow.className = 'dice-tray-row';

  const rollBtn = document.createElement('button');
  rollBtn.className = 'chart-btn dice-roll-btn';
  rollBtn.textContent = 'Roll';

  const trayTop = document.createElement('div');
  trayTop.className = 'dice-tray-top';
  trayTop.appendChild(diceRow);
  trayTop.appendChild(rollBtn);
  tray.appendChild(trayTop);
  container.appendChild(tray);

  // Zone 1b: Keeps Panel (absolute-positioned sidecar outside content column)
  const keepsPanel = document.createElement('div');
  keepsPanel.className = 'keeps-panel';
  container.appendChild(keepsPanel);

  // Zone 2: Pattern Table
  const tableWrap = document.createElement('div');
  tableWrap.className = 'pattern-table-wrap';
  tableWrap.innerHTML = buildPatternTable(families);
  container.appendChild(tableWrap);

  // Zone 3: 252 Grid
  const gridWrap = document.createElement('div');
  gridWrap.className = 'grid-252-wrap';
  gridWrap.innerHTML = buildGrid(families, multisets);
  container.appendChild(gridWrap);

  const tt = tooltip(container);

  // ------- Pip layout for mini dice (scaled from 48px base) -------
  const PIPS = [
    [],
    [{ cx: 24, cy: 24 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 24, cy: 24 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 24, cy: 24 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 12 }, { cx: 34, cy: 12 }, { cx: 14, cy: 24 }, { cx: 34, cy: 24 }, { cx: 14, cy: 36 }, { cx: 34, cy: 36 }],
  ];

  function makeMinDie(face, held) {
    // Build SVG content as single string to avoid innerHTML+= re-parse issues
    const pips = (PIPS[face] || []).map(p =>
      held
        ? `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)"/>`
        : `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)" opacity="0.2"/>`
    ).join('');

    const rect = held
      ? `<rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg)" stroke="var(--accent)" stroke-width="3"/>`
      : `<rect x="1" y="1" width="46" height="46" rx="8" fill="none" stroke="var(--border)" stroke-width="2" stroke-dasharray="4 3"/>`;

    // Wrap in a div for reliable CSS sizing (classList on SVG can be flaky)
    const wrap = document.createElement('div');
    wrap.className = held ? 'keep-die keep-die-held' : 'keep-die keep-die-reroll';
    wrap.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">${rect}${pips}</svg>`;
    return wrap;
  }

  // ------- Dice Renderer -------
  let diceRenderer = null;

  function renderDice() {
    if (state.currentId < 0) return;
    const ms = multisets[state.currentId];
    diceRow.innerHTML = '';

    diceRenderer = renderDiceSelectable(diceRow, ms.dice, {
      selected: state.heldIndices,
      size: 56,
      onToggle(idx) {
        if (state.heldIndices.has(idx)) state.heldIndices.delete(idx);
        else state.heldIndices.add(idx);
        renderDice();
        renderKeepsPanel();
      },
    });
    renderKeepsPanel();
  }

  // ------- Keeps Panel -------
  function getActiveKeepKey() {
    if (state.currentId < 0 || state.heldIndices.size === 0) return '';
    const ms = multisets[state.currentId];
    const kept = [];
    for (const i of state.heldIndices) kept.push(ms.dice[i]);
    kept.sort((a, b) => a - b);
    return kept.join(',');
  }

  function indicesForKeep(roll, keepDice) {
    const indices = new Set();
    const used = new Array(5).fill(false);
    for (const v of keepDice) {
      for (let j = 0; j < 5; j++) {
        if (!used[j] && roll[j] === v) {
          indices.add(j);
          used[j] = true;
          break;
        }
      }
    }
    return indices;
  }

  function renderKeepsPanel() {
    if (state.currentId < 0) { keepsPanel.innerHTML = ''; return; }
    const ms = multisets[state.currentId];
    const activeKey = getActiveKeepKey();

    keepsPanel.innerHTML = '';
    for (const keep of ms.keeps) {
      const key = keep.dice.join(',');
      const row = document.createElement('div');
      row.className = 'keep-row';
      if (key === activeKey) row.classList.add('active');

      // 5 mini dice
      const diceWrap = document.createElement('div');
      diceWrap.className = 'keep-row-dice';
      const keptSet = new Set();
      // Map keep values to roll positions (greedy left-to-right)
      const usedRoll = new Array(5).fill(false);
      for (let ki = 0; ki < keep.dice.length; ki++) {
        for (let ri = 0; ri < 5; ri++) {
          if (!usedRoll[ri] && ms.dice[ri] === keep.dice[ki]) {
            keptSet.add(ri);
            usedRoll[ri] = true;
            break;
          }
        }
      }
      for (let i = 0; i < 5; i++) {
        diceWrap.appendChild(makeMinDie(ms.dice[i], keptSet.has(i)));
      }
      row.appendChild(diceWrap);

      // Mask count badge
      if (keep.masks > 1) {
        const badge = document.createElement('span');
        badge.className = 'keep-masks';
        badge.textContent = `×${keep.masks}`;
        row.appendChild(badge);
      }

      // Click → update held indices
      row.addEventListener('click', () => {
        state.heldIndices = indicesForKeep(ms.dice, keep.dice);
        renderDice();
      });

      keepsPanel.appendChild(row);
    }
  }

  // ------- Roll -------
  function rollDice() {
    // Keep held dice values, re-roll the rest
    const prev = state.currentId >= 0 ? multisets[state.currentId] : null;
    const heldValues = [];
    if (prev) {
      for (const i of state.heldIndices) heldValues.push(prev.dice[i]);
    }

    const newDice = [...heldValues];
    for (let i = 0; i < 5 - heldValues.length; i++) {
      newDice.push(Math.floor(Math.random() * 6) + 1);
    }
    newDice.sort((a, b) => a - b);

    const key = newDice.join(',');
    const ms = multisets.find(m => m.dice.join(',') === key);
    state.currentId = ms.id;

    // Map held values back to positions in the sorted result
    const newHeld = new Set();
    const used = new Array(5).fill(false);
    for (const v of heldValues) {
      for (let j = 0; j < 5; j++) {
        if (!used[j] && newDice[j] === v) {
          newHeld.add(j);
          used[j] = true;
          break;
        }
      }
    }
    state.heldIndices = newHeld;

    renderDice();
    updateTable();
    updateGrid();
  }

  rollBtn.addEventListener('click', rollDice);

  // ------- Pattern Table -------
  function buildPatternTable(fams) {
    let html = '<table class="pattern-table">';
    html += '<colgroup><col class="col-swatch"><col class="col-pattern"><col class="col-num"><col class="col-num"><col class="col-num"></colgroup>';
    html += '<thead><tr>';
    html += '<th></th><th>Pattern</th><th class="num">Combinations</th><th class="num">Ordered</th><th class="num">Keeps</th>';
    html += '</tr></thead><tbody>';
    let totalKeeps = 0;
    for (let i = 0; i < fams.length; i++) {
      const f = fams[i];
      const exMs = multisets.find(m => m.familyIndex === i);
      const keepCount = exMs ? exMs.keeps.length : 0;
      totalKeeps += keepCount * f.count;
      html += `<tr data-family="${i}" class="pattern-row">`;
      html += `<td><span class="pattern-swatch" style="background:${f.color}"></span></td>`;
      html += `<td class="pattern-glyph">${f.key}</td>`;
      html += `<td class="num">${f.count}</td>`;
      html += `<td class="num">${f.totalPermutations.toLocaleString()}</td>`;
      html += `<td class="num">${keepCount}</td>`;
      html += `</tr>`;
    }
    html += '<tr class="pattern-total"><td></td><td><strong>Total</strong></td>';
    html += `<td class="num"><strong>252</strong> unique</td>`;
    html += `<td class="num"><strong>7,776</strong> ordered</td>`;
    html += `<td></td></tr>`;
    html += '</tbody></table>';
    return html;
  }

  function updateTable() {
    const rows = tableWrap.querySelectorAll('.pattern-row');
    rows.forEach(row => row.classList.remove('active'));
    if (state.currentId >= 0) {
      const fi = multisets[state.currentId].familyIndex;
      const activeRow = tableWrap.querySelector(`[data-family="${fi}"]`);
      if (activeRow) activeRow.classList.add('active');
    }
  }

  // ------- 252 Grid -------
  function buildGrid(fams, msets) {
    let html = '<div class="grid-252">';
    for (let fi = 0; fi < fams.length; fi++) {
      const f = fams[fi];
      html += `<div class="grid-family-group">`;
      html += `<div class="grid-family-label" style="border-left: 3px solid ${f.color}"><span class="pattern-glyph">${f.key}</span> <span class="grid-family-count">(${f.count})</span></div>`;
      html += `<div class="grid-family-dice">`;
      const famMsets = msets.filter(m => m.familyIndex === fi);
      for (const ms of famMsets) {
        html += `<div class="grid-quintuplet" data-msid="${ms.id}">`;
        for (const d of ms.dice) {
          const opacity = 0.3 + (d - 1) * 0.14;
          html += `<div class="grid-die" style="background:${f.color};opacity:${opacity}"></div>`;
        }
        html += `</div>`;
      }
      html += `</div></div>`;
    }
    html += '</div>';
    return html;
  }

  function updateGrid() {
    const quints = gridWrap.querySelectorAll('.grid-quintuplet');
    quints.forEach(q => {
      const msid = parseInt(q.dataset.msid);
      q.classList.toggle('active', msid === state.currentId);
    });
  }

  // Grid hover/click events
  gridWrap.addEventListener('mouseover', (e) => {
    const q = e.target.closest('.grid-quintuplet');
    if (!q) return;
    const msid = parseInt(q.dataset.msid);
    state.hoveredId = msid;
    const ms = multisets[msid];
    const fam = families[ms.familyIndex];
    tt.show(
      `<strong>[${ms.dice.join(', ')}]</strong><br>` +
      `${fam.name} — ${ms.permutations} ordered / ${(ms.probability * 100).toFixed(2)}%<br>` +
      `${ms.keeps.length} unique keeps`,
      e
    );
  });

  gridWrap.addEventListener('mousemove', (e) => tt.move(e));

  gridWrap.addEventListener('mouseout', (e) => {
    if (!e.target.closest('.grid-quintuplet')) return;
    state.hoveredId = -1;
    tt.hide();
  });

  gridWrap.addEventListener('click', (e) => {
    const q = e.target.closest('.grid-quintuplet');
    if (!q) return;
    const msid = parseInt(q.dataset.msid);
    state.currentId = msid;
    state.heldIndices.clear();
    renderDice();
    updateTable();
    updateGrid();
    tt.hide();
  });

  // ------- Initial roll -------
  rollDice();
}
