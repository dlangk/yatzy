/**
 * Dice Symmetry Explorer — interactive visualization of the 252 multisets.
 *
 * Three zones: dice tray (roll/keep), pattern table (7 families),
 * 252-grid (hover/click quintuplets).
 */

import { DataLoader } from '../data-loader.js';
import { tooltip, getTextColor, getMutedColor, getGridColor, COLORS } from '../yatzy-viz.js';
import { renderDiceSelectable } from '../utils/dice-interactive.js';
import { createDieSVG } from '/yatzy/shared/dice.js';

export async function initDiceSymmetry() {
  const container = document.getElementById('chart-dice-symmetry');
  if (!container) return;

  const data = await DataLoader.diceSymmetry();
  const { families, multisets } = data;

  // ------- State -------
  const state = {
    currentId: -1,        // active multiset id
    displayDice: [],      // unsorted dice for display
    keptIndices: new Set(),
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

  const diceLegend = document.createElement('div');
  diceLegend.className = 'dice-symmetry-legend';
  diceLegend.innerHTML =
    '<span><span class="legend-swatch legend-swatch-keep"></span>Keep</span>' +
    '<span><span class="legend-swatch legend-swatch-reroll"></span>Reroll</span>';
  trayTop.appendChild(diceLegend);

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

  function makeMinDie(face, kept) {
    const state = kept ? 'kept' : 'reroll';
    const wrap = document.createElement('div');
    wrap.className = kept ? 'keep-die keep-die-kept' : 'keep-die keep-die-reroll';
    wrap.appendChild(createDieSVG(face, { size: 20, state }));
    return wrap;
  }

  // ------- Dice Renderer -------
  let diceRenderer = null;

  function renderDice() {
    if (state.currentId < 0) return;
    diceRow.innerHTML = '';

    diceRenderer = renderDiceSelectable(diceRow, state.displayDice, {
      selected: state.keptIndices,
      size: 56,
      onToggle(idx) {
        if (state.keptIndices.has(idx)) state.keptIndices.delete(idx);
        else state.keptIndices.add(idx);
        renderDice();
        renderKeepsPanel();
      },
    });
    renderKeepsPanel();
  }

  // ------- Keeps Panel -------
  function getActiveKeepKey() {
    if (state.currentId < 0 || state.keptIndices.size === 0) return '';
    const kept = [];
    for (const i of state.keptIndices) kept.push(state.displayDice[i]);
    kept.sort((a, b) => a - b);
    return kept.join(',');
  }

  function indicesForKeep(keepDice) {
    const indices = new Set();
    const used = new Array(5).fill(false);
    for (const v of keepDice) {
      for (let j = 0; j < 5; j++) {
        if (!used[j] && state.displayDice[j] === v) {
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

    keepsPanel.innerHTML = '<div class="keeps-panel-header">Possible Rerolls</div>';
    for (const keep of ms.keeps) {
      const key = keep.dice.join(',');
      const row = document.createElement('div');
      row.className = 'keep-row';
      if (key === activeKey) row.classList.add('active');

      // 5 mini dice
      const diceWrap = document.createElement('div');
      diceWrap.className = 'keep-row-dice';
      const keptSet = new Set();
      // Map keep values to display positions (greedy left-to-right)
      const usedRoll = new Array(5).fill(false);
      for (let ki = 0; ki < keep.dice.length; ki++) {
        for (let ri = 0; ri < 5; ri++) {
          if (!usedRoll[ri] && state.displayDice[ri] === keep.dice[ki]) {
            keptSet.add(ri);
            usedRoll[ri] = true;
            break;
          }
        }
      }
      for (let i = 0; i < 5; i++) {
        diceWrap.appendChild(makeMinDie(state.displayDice[i], keptSet.has(i)));
      }
      row.appendChild(diceWrap);

      // Mask count badge
      if (keep.masks > 1) {
        const badge = document.createElement('span');
        badge.className = 'keep-masks';
        badge.textContent = `×${keep.masks}`;
        row.appendChild(badge);
      }

      // Click → update kept indices
      row.addEventListener('click', () => {
        state.keptIndices = indicesForKeep(keep.dice);
        renderDice();
      });

      keepsPanel.appendChild(row);
    }
  }

  // ------- Roll -------
  function rollDice() {
    // Keep kept dice at their positions, re-roll the rest
    const display = new Array(5);
    const newKept = new Set();

    if (state.currentId >= 0) {
      // Preserve kept dice in place
      for (const i of state.keptIndices) {
        display[i] = state.displayDice[i];
        newKept.add(i);
      }
    }
    // Fill empty slots with new rolls
    for (let i = 0; i < 5; i++) {
      if (display[i] === undefined) {
        display[i] = Math.floor(Math.random() * 6) + 1;
      }
    }

    // Look up multiset from sorted values
    const sorted = [...display].sort((a, b) => a - b);
    const key = sorted.join(',');
    const ms = multisets.find(m => m.dice.join(',') === key);
    state.currentId = ms.id;
    state.displayDice = display;
    state.keptIndices = newKept;

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
    html += '<th></th><th>Pattern</th><th class="num">Combinations</th><th class="num">Ordered</th><th class="num">Rerolls</th>';
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
      html += `<div class="grid-family-label" style="--family-color: ${f.color}"><span class="pattern-glyph">${f.key}</span> <span class="grid-family-count">(${f.count})</span></div>`;
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
    state.displayDice = [...multisets[msid].dice];
    state.keptIndices.clear();
    renderDice();
    updateTable();
    updateGrid();
    tt.hide();
  });

  // ------- Initial roll -------
  rollDice();
}
