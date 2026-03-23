/**
 * Widget Interactive: live solver visualization.
 *
 * Two modes:
 *   Explain: pre-computed optimal turn, rendered bottom-to-top (backward)
 *   Play:    user makes choices, rows accumulate top-to-bottom
 */

import { createDieSVG } from '/yatzy/shared/dice.js';

const SCENARIO = {
  upper_score: 18,
  scored_categories: 0b000000000000111,
};

const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
];

const API_BASE = '/yatzy/api';
const DIE_SIZE = 32;

// Explain texts indexed by step: 0=score, 1=keep2, 2=keep1, 3=state
const EXPLAIN_TEXTS = [
  'Starting from the end: given these final dice, the solver tries every unscored category and picks the one with the best immediate score plus future value.',
  'One step back: for each possible keep, compute the weighted average over reroll outcomes using the expected scores just computed.',
  'Same logic again: two identical passes, reading from one buffer, writing to the other.',
  'Before any dice are thrown, the expected score is the average over all 252 initial rolls. One number: the state value.',
];

// ── Dice rendering ─────────────────────────────────────────────

function createDie(value, options = {}) {
  const { selected = false, rerolling = false, clickable = false } = options;
  const state = rerolling ? 'faded' : (selected ? 'kept' : 'normal');
  return createDieSVG(value, { size: DIE_SIZE, state, clickable });
}

// ── API helpers ────────────────────────────────────────────────

async function apiHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch { return false; }
}

async function callStateValue() {
  const r = await fetch(`${API_BASE}/state_value?upper_score=${SCENARIO.upper_score}&scored_categories=${SCENARIO.scored_categories}`);
  if (!r.ok) throw new Error(`state_value ${r.status}`);
  return r.json();
}

async function callEvaluate(dice, rerolls) {
  const r = await fetch(`${API_BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dice,
      upper_score: SCENARIO.upper_score,
      scored_categories: SCENARIO.scored_categories,
      rerolls_remaining: rerolls,
    }),
  });
  if (!r.ok) throw new Error(`evaluate ${r.status}`);
  return r.json();
}

// ── Dice utilities ─────────────────────────────────────────────

function randomDice() {
  return Array.from({ length: 5 }, () => Math.floor(Math.random() * 6) + 1);
}

function applyMask(mask, dice) {
  const r = [...dice];
  for (let i = 0; i < 5; i++) {
    if (mask & (1 << i)) r[i] = Math.floor(Math.random() * 6) + 1;
  }
  return r;
}

function keptSetToMask(kept) {
  let m = 0;
  for (let i = 0; i < 5; i++) if (!kept.has(i)) m |= (1 << i);
  return m;
}

function maskToKeptSet(mask) {
  const s = new Set();
  for (let i = 0; i < 5; i++) if (!(mask & (1 << i))) s.add(i);
  return s;
}

// ── Main ───────────────────────────────────────────────────────

export async function initWidgetInteractive() {
  const container = document.getElementById('chart-widget-interactive');
  if (!container) return;

  let mode = 'explain';
  let apiAvailable = false;
  let loading = false;

  // Explain
  let explainStep = 0; // 0=score, 1=keep2, 2=keep1, 3=state
  let seq = null;

  // Play: accumulate history
  let playPhase = 'idle';
  let dice = [1, 1, 1, 1, 1];
  let keptIndices = new Set();
  let evalResp = null;    // current step's eval
  let playHistory = [];   // frozen rows: { label, type, dice, kept, ev, evalData }
  let playCategory = null;
  let playScore = null;

  // ── DOM ──────────────────────────────────────────────────────

  function mk(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  const root = mk('div', 'widget-solver');
  const headerEl = mk('div', 'widget-solver-header');
  const stateInfoEl = mk('div', 'widget-solver-state');
  const toggleEl = mk('div', 'widget-mode-toggle');
  const btnExplain = mk('button'); btnExplain.textContent = 'Explain';
  const btnPlay = mk('button'); btnPlay.textContent = 'Play';
  toggleEl.append(btnExplain, btnPlay);
  headerEl.append(stateInfoEl, toggleEl);

  const flowEl = mk('div', 'widget-turn-flow');
  const explainPanelEl = mk('div', 'widget-explain-panel');
  const actionBarEl = mk('div', 'widget-action-bar');
  const fallbackEl = mk('div', 'widget-api-fallback');

  const caption = container.querySelector('.chart-caption');
  container.innerHTML = '';
  root.append(headerEl, flowEl, explainPanelEl, actionBarEl, fallbackEl);
  if (caption) root.appendChild(caption);
  container.appendChild(root);

  btnExplain.addEventListener('click', () => switchMode('explain'));
  btnPlay.addEventListener('click', () => switchMode('play'));

  // ── Mode switching ───────────────────────────────────────────

  function switchMode(m) {
    mode = m;
    if (m === 'explain') startExplain();
    else { resetPlay(); render(); }
  }

  // ── Explain ──────────────────────────────────────────────────

  async function startExplain() {
    if (!apiAvailable) { render(); return; }
    loading = true; render();
    try {
      const sv = await callStateValue();
      const roll1 = randomDice();
      const e1 = await callEvaluate(roll1, 2);
      const roll2 = applyMask(e1.optimal_mask, roll1);
      const e2 = await callEvaluate(roll2, 1);
      const roll3 = applyMask(e2.optimal_mask, roll2);
      const e3 = await callEvaluate(roll3, 0);
      seq = { stateEv: sv.expected_final_score, rolls: [roll1, roll2, roll3], evals: [e1, e2, e3] };
      explainStep = 0; // start with score highlighted
      loading = false;
      render();
    } catch (err) {
      console.error('Explain failed:', err);
      loading = false; render();
    }
  }

  // ── Play ─────────────────────────────────────────────────────

  function resetPlay() {
    playPhase = 'idle';
    dice = [1, 1, 1, 1, 1];
    keptIndices = new Set();
    evalResp = null;
    playHistory = [];
    playCategory = null;
    playScore = null;
  }

  async function doRoll() {
    const rerolls = playPhase === 'idle' ? 2 : playPhase === 'keep1' ? 1 : 0;

    if (playPhase !== 'idle') {
      // Freeze current keep decision into history
      const mask = keptSetToMask(keptIndices);
      playHistory.push({
        label: playPhase === 'keep1' ? 'Keep 1' : 'Keep 2',
        type: 'decision',
        dice: [...dice],
        kept: new Set(keptIndices),
        ev: evalResp.mask_evs[mask],
        optEv: evalResp.optimal_mask_ev,
        optMask: evalResp.optimal_mask,
      });
      dice = applyMask(mask, dice);
    } else {
      dice = randomDice();
    }

    // Freeze the roll into history
    playHistory.push({
      label: playPhase === 'idle' ? 'Roll 1' : rerolls === 1 ? 'Roll 2' : 'Roll 3',
      type: 'chance',
      dice: [...dice],
      kept: null,
    });

    keptIndices = new Set();
    loading = true; render();
    try {
      evalResp = await callEvaluate(dice, rerolls);
      playPhase = rerolls === 2 ? 'rolled' : rerolls === 1 ? 'rerolled1' : 'rerolled2';
    } catch (e) { console.error('Evaluate failed:', e); }
    loading = false; render();
  }

  function doScore(catId) {
    if (playPhase !== 'rerolled2' || !evalResp) return;
    const c = evalResp.categories[catId];
    if (!c.available) return;
    playCategory = catId;
    playScore = c.score;
    playPhase = 'done';
    render();
  }

  // ── Render ───────────────────────────────────────────────────

  function render() {
    btnExplain.className = mode === 'explain' ? 'active' : '';
    btnPlay.className = mode === 'play' ? 'active' : '';
    flowEl.innerHTML = '';
    explainPanelEl.innerHTML = ''; explainPanelEl.style.display = 'none';
    actionBarEl.innerHTML = ''; actionBarEl.style.display = 'none';
    fallbackEl.style.display = 'none';

    if (!apiAvailable) {
      stateInfoEl.innerHTML = 'Upper Score = 18, Categories Scored = [Ones, Twos, Threes] (turn 4)';
      fallbackEl.style.display = '';
      fallbackEl.innerHTML = 'Start the backend with <code>just dev-backend</code> for the interactive solver.';
      return;
    }
    if (loading) {
      stateInfoEl.innerHTML = 'Upper Score = 18, Categories Scored = [Ones, Twos, Threes] (turn 4)';
      flowEl.innerHTML = '<div class="widget-loading"><div class="spinner"></div>Loading...</div>';
      return;
    }

    if (mode === 'explain') renderExplain();
    else renderPlay();
  }

  // ── Render: Explain (reversed: score at top, initial roll at bottom) ──

  function renderExplain() {
    if (!seq) return;
    stateInfoEl.innerHTML = `Upper Score = 18, Categories Scored = [Ones, Twos, Threes] &middot; <span class="ev-value">Expected score ${seq.stateEv.toFixed(1)}</span>`;

    const kept1 = maskToKeptSet(seq.evals[0].optimal_mask);
    const kept2 = maskToKeptSet(seq.evals[1].optimal_mask);

    // Row 0 (top): Score Category — explainStep 0
    addCategoryStage(flowEl, seq.evals[2], explainStep === 0, false, () => { explainStep = 0; render(); });

    // Row 1: Final Roll
    addRow(flowEl, 'Final Roll', 'chance', false, null, row => {
      appendDicePositional(row, seq.rolls[2], null);
    });

    // Row 2: Keep Decision 2 — explainStep 1
    addRow(flowEl, 'Keep 2', 'decision', explainStep === 1, () => { explainStep = 1; render(); }, row => {
      appendDicePositional(row, seq.rolls[1], kept2);
      evBadge(row, seq.evals[1].optimal_mask_ev, true);
    });

    // Row 3: Second Roll
    addRow(flowEl, 'Roll 2', 'chance', false, null, row => {
      appendDicePositional(row, seq.rolls[1], null);
    });

    // Row 4: Keep Decision 1 — explainStep 2
    addRow(flowEl, 'Keep 1', 'decision', explainStep === 2, () => { explainStep = 2; render(); }, row => {
      appendDicePositional(row, seq.rolls[0], kept1);
      evBadge(row, seq.evals[0].optimal_mask_ev, true);
    });

    // Row 5 (bottom): Initial Roll — explainStep 3
    addRow(flowEl, 'Initial Roll', 'chance', explainStep === 3, () => { explainStep = 3; render(); }, row => {
      appendDicePositional(row, seq.rolls[0], null);
      evBadge(row, seq.stateEv, false, 'avg');
    });

    // Explain text
    explainPanelEl.style.display = '';
    explainPanelEl.textContent = EXPLAIN_TEXTS[explainStep];

    // Action bar
    actionBarEl.style.display = '';
    const pb = mk('button', 'primary'); pb.textContent = 'Your Turn';
    pb.addEventListener('click', () => switchMode('play'));
    const rb = mk('button'); rb.textContent = 'New Example';
    rb.addEventListener('click', () => startExplain());
    actionBarEl.append(pb, rb);
  }

  // ── Render: Play (accumulates rows top-to-bottom) ────────────

  function renderPlay() {
    stateInfoEl.innerHTML = 'Upper Score = 18, Categories Scored = [Ones, Twos, Threes] (turn 4)';
    if (evalResp && evalResp.state_ev) {
      stateInfoEl.innerHTML += ` &middot; <span class="ev-value">Expected score ${evalResp.state_ev.toFixed(1)}</span>`;
    }

    if (playPhase === 'idle') {
      actionBarEl.style.display = '';
      const b = mk('button', 'primary'); b.textContent = 'Roll Dice';
      b.addEventListener('click', doRoll);
      actionBarEl.appendChild(b);
      return;
    }

    // Render frozen history rows
    for (const h of playHistory) {
      if (h.type === 'chance') {
        addRow(flowEl, h.label, 'chance', false, null, row => {
          appendDicePositional(row, h.dice, null);
        });
      } else {
        // Keep decision (frozen)
        addRow(flowEl, h.label, 'decision', false, null, row => {
          appendDicePositional(row, h.dice, h.kept);
          const isOpt = keptSetToMask(h.kept) === h.optMask;
          evBadge(row, h.ev, isOpt);
          if (!isOpt) {
            const d = mk('span');
            d.style.cssText = 'font-size:0.7rem;color:var(--color-danger);margin-left:0.3rem;';
            d.textContent = `(${(h.ev - h.optEv).toFixed(1)})`;
            row.appendChild(d);
          }
        });
      }
    }

    // Current interactive step
    if (playPhase === 'rolled' || playPhase === 'rerolled1') {
      // Interactive keep selection on current dice
      addRow(flowEl, 'Select keeps', 'decision', false, null, row => {
        appendDiceInteractive(row, dice, keptIndices);
        if (evalResp && evalResp.mask_evs) {
          const mask = keptSetToMask(keptIndices);
          const ev = evalResp.mask_evs[mask];
          const isOpt = mask === evalResp.optimal_mask;
          evBadge(row, ev, isOpt);
          if (!isOpt) {
            const d = mk('span');
            d.style.cssText = 'font-size:0.7rem;color:var(--color-danger);margin-left:0.3rem;';
            d.textContent = `(${(ev - evalResp.optimal_mask_ev).toFixed(1)})`;
            row.appendChild(d);
          }
        }
      });

      const rerolls = playPhase === 'rolled' ? 2 : 1;
      actionBarEl.style.display = '';
      const rb = mk('button', 'primary');
      rb.textContent = `Reroll (${rerolls} left)`;
      rb.addEventListener('click', () => {
        playPhase = playPhase === 'rolled' ? 'keep1' : 'keep2';
        doRoll();
      });
      actionBarEl.appendChild(rb);

      if (evalResp) {
        const hb = mk('button'); hb.textContent = 'Show Optimal';
        hb.addEventListener('click', () => { keptIndices = maskToKeptSet(evalResp.optimal_mask); render(); });
        actionBarEl.appendChild(hb);
      }

    } else if (playPhase === 'rerolled2') {
      // Choose category
      addCategoryStage(flowEl, evalResp, false, true, null);

    } else if (playPhase === 'done') {
      // Show full category list with the chosen one highlighted
      addCategoryStage(flowEl, evalResp, false, false, null, playCategory);

      actionBarEl.style.display = '';
      const nb = mk('button', 'primary'); nb.textContent = 'New Turn';
      nb.addEventListener('click', () => { resetPlay(); render(); });
      const eb = mk('button'); eb.textContent = 'Explain';
      eb.addEventListener('click', () => switchMode('explain'));
      actionBarEl.append(nb, eb);
    }
  }

  // ── Row builders ─────────────────────────────────────────────

  function addRow(parent, label, type, highlighted, onClick, contentFn) {
    const row = mk('div', `widget-stage ${type}`);
    if (highlighted) row.classList.add('highlighted');
    if (onClick) { row.style.cursor = 'pointer'; row.addEventListener('click', onClick); }
    const lbl = mk('span', 'widget-stage-label');
    lbl.textContent = label;
    row.appendChild(lbl);
    contentFn(row);
    parent.appendChild(row);
    return row;
  }

  function addCategoryStage(parent, evalData, highlighted, interactive, onClick, selectedCat) {
    const stage = mk('div', 'widget-stage decision categories');
    if (highlighted) stage.classList.add('highlighted');
    if (onClick) { stage.style.cursor = 'pointer'; stage.addEventListener('click', onClick); }

    const header = mk('div');
    header.style.cssText = 'display:flex;align-items:center;width:100%;';
    const lbl = mk('span', 'widget-stage-label');
    lbl.textContent = 'Score';
    header.appendChild(lbl);
    stage.appendChild(header);

    const list = mk('div', 'widget-category-list');
    const cats = evalData.categories;
    const optCat = evalData.optimal_category;

    const available = cats
      .map((c, i) => ({ ...c, idx: i }))
      .filter(c => c.available)
      .sort((a, b) => b.ev_if_scored - a.ev_if_scored);
    const shown = interactive || selectedCat != null ? available : available.filter(c => c.score > 0).slice(0, 4);

    shown.forEach(c => {
      const row = mk('div', 'widget-category-row');
      if (c.idx === optCat) row.classList.add('optimal');
      if (selectedCat != null && c.idx === selectedCat) row.classList.add('selected');
      if (interactive) row.classList.add('interactive');
      row.innerHTML = `<span class="widget-category-name">${CATEGORY_NAMES[c.idx]}</span>`
        + `<span class="widget-category-score">${c.score} pts</span>`
        + `<span class="widget-category-ev">Exp. ${c.ev_if_scored.toFixed(1)}</span>`
        + (c.idx === optCat ? '<span class="widget-optimal-star">\u2605</span>' : '')
        + (selectedCat != null && c.idx === selectedCat ? '<span class="widget-selected-marker">\u2190 your pick</span>' : '');
      if (interactive) row.addEventListener('click', () => doScore(c.idx));
      list.appendChild(row);
    });

    stage.appendChild(list);
    parent.appendChild(stage);
  }

  // ── Dice helpers ─────────────────────────────────────────────
  // All dice renderers keep positional order (index 0..4).
  // `kept` set determines styling only, not ordering.

  function appendDicePositional(row, values, kept) {
    const wrap = mk('div', 'widget-dice-row');
    values.forEach((v, i) => {
      if (kept) {
        wrap.appendChild(createDie(v, { selected: kept.has(i), rerolling: !kept.has(i) }));
      } else {
        wrap.appendChild(createDie(v));
      }
    });
    row.appendChild(wrap);
  }

  function appendDiceInteractive(row, values, kept) {
    const wrap = mk('div', 'widget-dice-row');
    values.forEach((v, i) => {
      const d = createDie(v, { selected: kept.has(i), clickable: true });
      d.addEventListener('click', () => {
        if (keptIndices.has(i)) keptIndices.delete(i); else keptIndices.add(i);
        render();
      });
      wrap.appendChild(d);
    });
    row.appendChild(wrap);
  }

  function evBadge(row, ev, optimal, suffix) {
    const b = mk('span', 'widget-ev-badge' + (optimal ? ' optimal' : ''));
    b.textContent = `Exp. ${ev.toFixed(1)}`;
    if (optimal) b.textContent += ' \u2605';
    if (suffix) b.textContent += ` (${suffix})`;
    row.appendChild(b);
  }

  // ── Init ─────────────────────────────────────────────────────

  apiAvailable = await apiHealth();
  render();
  if (apiAvailable && mode === 'explain') await startExplain();
}
