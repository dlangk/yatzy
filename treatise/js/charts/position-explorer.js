/**
 * Position Explorer: interactive solver API lookup.
 *
 * The reader sets dice and scorecard state, then sees the solver's
 * evaluation: optimal keep mask, category EVs, and action ranking.
 */

import { createDieSVG } from '/yatzy/shared/dice.js';
import { getMutedColor, COLORS } from '../yatzy-viz.js';

const API_BASE = '/yatzy/api';

const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
];

const DIE_SIZE = 44;

// ── API ─────────────────────────────────────────────────────────

async function apiHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch { return false; }
}

async function callEvaluate(dice, upperScore, scoredCategories, rerolls) {
  const r = await fetch(`${API_BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dice,
      upper_score: upperScore,
      scored_categories: scoredCategories,
      rerolls_remaining: rerolls,
    }),
  });
  if (!r.ok) throw new Error(`evaluate ${r.status}`);
  return r.json();
}

// ── Dice rendering ──────────────────────────────────────────────

function createDie(value, options = {}) {
  const { clickable = false } = options;
  return createDieSVG(value, { size: DIE_SIZE, state: 'normal', clickable });
}

// ── Main ────────────────────────────────────────────────────────

export async function initPositionExplorer() {
  const container = document.getElementById('chart-position-explorer');
  if (!container) return;

  let apiAvailable = await apiHealth();
  let loading = false;
  let evalResp = null;

  // State
  let dice = [3, 3, 4, 5, 6];
  let upperScore = 0;
  let scoredCategories = 0;
  let rerolls = 2;

  function mk(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  // ── Build DOM ─────────────────────────────────────────────────

  const root = mk('div', 'position-explorer');
  const caption = container.querySelector('.chart-caption');
  container.innerHTML = '';

  // Fallback
  const fallbackEl = mk('div', 'widget-api-fallback');

  // Controls row
  const controlsEl = mk('div', 'explorer-controls');

  // Dice row
  const diceLabel = mk('div', 'explorer-section-label');
  diceLabel.textContent = 'Dice (click to cycle)';
  const diceRow = mk('div', 'explorer-dice-row');

  // Rerolls selector
  const rerollLabel = mk('div', 'explorer-section-label');
  rerollLabel.textContent = 'Rerolls remaining';
  const rerollRow = mk('div', 'explorer-reroll-row');
  [2, 1, 0].forEach(v => {
    const btn = mk('button', 'explorer-reroll-btn');
    btn.textContent = String(v);
    btn.dataset.val = String(v);
    btn.addEventListener('click', () => { rerolls = v; evaluate(); });
    rerollRow.appendChild(btn);
  });

  // Scorecard (compact checkboxes)
  const scorecardLabel = mk('div', 'explorer-section-label');
  scorecardLabel.textContent = 'Already scored (click to toggle)';
  const scorecardEl = mk('div', 'explorer-scorecard');

  CATEGORY_NAMES.forEach((name, i) => {
    const item = mk('label', 'explorer-cat-toggle');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.dataset.cat = String(i);
    cb.addEventListener('change', () => {
      if (cb.checked) scoredCategories |= (1 << i);
      else scoredCategories &= ~(1 << i);
      evaluate();
    });
    item.appendChild(cb);
    const span = mk('span');
    span.textContent = name;
    item.appendChild(span);
    scorecardEl.appendChild(item);
  });

  // Upper score
  const upperLabel = mk('div', 'explorer-section-label');
  upperLabel.textContent = 'Upper section score';
  const upperRow = mk('div', 'explorer-upper-row');
  const upperInput = document.createElement('input');
  upperInput.type = 'range';
  upperInput.min = '0';
  upperInput.max = '63';
  upperInput.value = '0';
  const upperVal = mk('span', 'slider-value');
  upperVal.textContent = '0';
  upperInput.addEventListener('input', () => {
    upperScore = parseInt(upperInput.value);
    upperVal.textContent = String(upperScore);
    evaluate();
  });
  upperRow.append(upperInput, upperVal);

  // Random button
  const randomBtn = mk('button', 'explorer-random-btn');
  randomBtn.textContent = 'Random Dice';
  randomBtn.addEventListener('click', () => {
    dice = Array.from({ length: 5 }, () => Math.floor(Math.random() * 6) + 1);
    evaluate();
  });

  // Results
  const resultsEl = mk('div', 'explorer-results');

  controlsEl.append(
    diceLabel, diceRow,
    rerollLabel, rerollRow,
    upperLabel, upperRow,
    scorecardLabel, scorecardEl,
    randomBtn
  );
  root.append(fallbackEl, controlsEl, resultsEl);
  if (caption) root.appendChild(caption);
  container.appendChild(root);

  // ── Evaluate ──────────────────────────────────────────────────

  let debounceTimer = null;

  async function evaluate() {
    render();
    if (!apiAvailable) return;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      loading = true;
      render();
      try {
        evalResp = await callEvaluate(dice, upperScore, scoredCategories, rerolls);
      } catch (err) {
        console.error('Evaluate failed:', err);
        evalResp = null;
      }
      loading = false;
      render();
    }, 150);
  }

  // ── Render ────────────────────────────────────────────────────

  function render() {
    // Fallback
    if (!apiAvailable) {
      fallbackEl.style.display = '';
      fallbackEl.innerHTML = 'Start the backend with <code>just dev-backend</code> for the position explorer.';
      controlsEl.style.display = 'none';
      resultsEl.style.display = 'none';
      return;
    }
    fallbackEl.style.display = 'none';
    controlsEl.style.display = '';
    resultsEl.style.display = '';

    // Dice
    diceRow.innerHTML = '';
    dice.forEach((v, i) => {
      const d = createDie(v, { clickable: true });
      d.addEventListener('click', () => {
        dice[i] = (dice[i] % 6) + 1;
        evaluate();
      });
      diceRow.appendChild(d);
    });

    // Reroll buttons
    rerollRow.querySelectorAll('.explorer-reroll-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.val) === rerolls);
    });

    // Results
    resultsEl.innerHTML = '';

    if (loading) {
      resultsEl.innerHTML = '<div class="widget-loading"><div class="spinner"></div>Evaluating...</div>';
      return;
    }

    if (!evalResp) return;

    // Optimal action summary
    const summary = mk('div', 'explorer-summary');

    if (rerolls > 0) {
      // Keep mask result
      const optMask = evalResp.optimal_mask;
      const keptDice = [];
      const rerolledDice = [];
      for (let i = 0; i < 5; i++) {
        if (optMask & (1 << i)) rerolledDice.push(dice[i]);
        else keptDice.push(dice[i]);
      }
      const keepStr = keptDice.length > 0 ? keptDice.sort().join(', ') : 'none';
      const rerollStr = rerolledDice.length > 0 ? rerolledDice.sort().join(', ') : 'none';

      summary.innerHTML =
        `<div class="explorer-action"><strong>Optimal:</strong> Keep [${keepStr}], reroll [${rerollStr}]</div>` +
        `<div class="explorer-ev">Expected final score: <span class="ev-value">${evalResp.optimal_mask_ev.toFixed(1)}</span></div>`;
    } else {
      // Category scoring result
      const optCat = evalResp.optimal_category;
      const optName = CATEGORY_NAMES[optCat];
      const optEv = evalResp.categories[optCat].ev_if_scored;

      summary.innerHTML =
        `<div class="explorer-action"><strong>Optimal:</strong> Score ${optName}</div>` +
        `<div class="explorer-ev">Expected final score: <span class="ev-value">${optEv.toFixed(1)}</span></div>`;
    }
    resultsEl.appendChild(summary);

    // Category table (when rerolls = 0, show all available categories)
    if (rerolls === 0 && evalResp.categories) {
      const table = mk('div', 'explorer-cat-table');
      const header = mk('div', 'explorer-cat-header');
      header.innerHTML = '<span>Category</span><span>Score</span><span>Expected Total</span>';
      table.appendChild(header);

      const available = evalResp.categories
        .map((c, i) => ({ ...c, idx: i }))
        .filter(c => c.available)
        .sort((a, b) => b.ev_if_scored - a.ev_if_scored);

      available.forEach(c => {
        const row = mk('div', 'explorer-cat-row');
        if (c.idx === evalResp.optimal_category) row.classList.add('optimal');
        row.innerHTML =
          `<span class="explorer-cat-name">${CATEGORY_NAMES[c.idx]}</span>` +
          `<span class="explorer-cat-score">${c.score}</span>` +
          `<span class="explorer-cat-ev">${c.ev_if_scored.toFixed(1)}</span>`;
        table.appendChild(row);
      });
      resultsEl.appendChild(table);
    }

    // Top keep masks (when rerolls > 0)
    if (rerolls > 0 && evalResp.mask_evs) {
      const maskTable = mk('div', 'explorer-mask-table');
      const mHeader = mk('div', 'explorer-mask-header');
      mHeader.innerHTML = '<span>Keep</span><span>Expected Total</span>';
      maskTable.appendChild(mHeader);

      // Get top 8 keep masks
      const masks = evalResp.mask_evs
        .map((ev, mask) => ({ mask, ev }))
        .sort((a, b) => b.ev - a.ev)
        .slice(0, 8);

      masks.forEach(({ mask, ev }) => {
        const kept = [];
        for (let i = 0; i < 5; i++) {
          if (!(mask & (1 << i))) kept.push(dice[i]);
        }
        const keepStr = kept.length > 0 ? kept.sort().join(', ') : 'reroll all';

        const row = mk('div', 'explorer-mask-row');
        if (mask === evalResp.optimal_mask) row.classList.add('optimal');
        row.innerHTML =
          `<span class="explorer-mask-keep">[${keepStr}]</span>` +
          `<span class="explorer-mask-ev">${ev.toFixed(1)}</span>`;
        maskTable.appendChild(row);
      });
      resultsEl.appendChild(maskTable);
    }
  }

  render();
  if (apiAvailable) evaluate();
}
