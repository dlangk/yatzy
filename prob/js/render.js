/**
 * Live simulation trace UI: three rows of dice, per-row holds and randomizers,
 * live "you" vs "best" target probabilities.
 * @module render
 */
import { makeSolver } from './engine.js';
import { CATEGORIES, categoryById, exactTarget } from './targets.js';
import { createDieSVG } from '/yatzy/shared/dice.js';

const REROLLS = [2, 1, 0]; // rerolls left per row index

/** Deterministic default opening hand from the user's example. */
function defaultHand() {
  return [1, 2, 3, 4, 6];
}

function randomDie() {
  return 1 + Math.floor(Math.random() * 6);
}

function pct(p) {
  return (p * 100).toFixed(1) + '%';
}

export function initProbTool(root) {
  const state = {
    mode: 'category',
    categoryId: 'small_straight',
    exactHand: [1, 2, 3, 4, 5],
    rows: [
      { dice: defaultHand(), keep: [true, true, true, true, false] },
      { dice: null, keep: [true, true, true, true, true] },
      { dice: null, keep: [true, true, true, true, true] },
    ],
  };

  root.innerHTML = '';

  // ── Target selector ───────────────────────────────────────────────
  const targetBar = document.createElement('div');
  targetBar.className = 'target-bar';
  root.appendChild(targetBar);

  const modeSel = document.createElement('select');
  modeSel.className = 'mode-select';
  modeSel.setAttribute('aria-label', 'Target mode');
  for (const [val, label] of [['category', 'Category'], ['exact', 'Exact hand']]) {
    const o = document.createElement('option');
    o.value = val;
    o.textContent = label;
    modeSel.appendChild(o);
  }
  modeSel.value = state.mode;
  targetBar.appendChild(modeSel);

  const catSel = document.createElement('select');
  catSel.className = 'cat-select';
  catSel.setAttribute('aria-label', 'Target category');
  for (const c of CATEGORIES) {
    const o = document.createElement('option');
    o.value = c.id;
    o.textContent = c.label;
    catSel.appendChild(o);
  }
  catSel.value = state.categoryId;
  targetBar.appendChild(catSel);

  // Exact-hand editor (five clickable dice)
  const exactWrap = document.createElement('div');
  exactWrap.className = 'exact-wrap';
  const exactLabel = document.createElement('span');
  exactLabel.className = 'exact-label';
  exactLabel.textContent = 'Target hand (click to change):';
  exactWrap.appendChild(exactLabel);
  const exactDice = document.createElement('div');
  exactDice.className = 'exact-dice';
  exactWrap.appendChild(exactDice);
  targetBar.appendChild(exactWrap);

  // ── Rows ──────────────────────────────────────────────────────────
  const rowsWrap = document.createElement('div');
  rowsWrap.className = 'rows';
  root.appendChild(rowsWrap);

  const rowEls = REROLLS.map(() => {
    const el = document.createElement('div');
    el.className = 'prob-row';
    rowsWrap.appendChild(el);
    return el;
  });

  // ── Helpers ───────────────────────────────────────────────────────
  function currentTarget() {
    if (state.mode === 'exact') return exactTarget(state.exactHand);
    return categoryById(state.categoryId).test;
  }

  function keptDice(row) {
    const out = [];
    for (let i = 0; i < 5; i++) if (row.keep[i]) out.push(row.dice[i]);
    return out.sort((a, b) => a - b);
  }

  function clearBelow(index) {
    for (let i = index + 1; i < 3; i++) {
      state.rows[i].dice = null;
      state.rows[i].keep = [true, true, true, true, true];
    }
  }

  function rollInto(index) {
    const src = state.rows[index];
    if (!src.dice) return;
    const next = src.dice.map((v, i) => (src.keep[i] ? v : randomDie()));
    state.rows[index + 1] = { dice: next, keep: [true, true, true, true, true] };
    for (let i = index + 2; i < 3; i++) {
      state.rows[i].dice = null;
      state.rows[i].keep = [true, true, true, true, true];
    }
  }

  // ── Render ────────────────────────────────────────────────────────
  function render() {
    const solver = makeSolver(currentTarget());

    catSel.style.display = state.mode === 'category' ? '' : 'none';
    exactWrap.style.display = state.mode === 'exact' ? '' : 'none';

    // Exact target dice
    exactDice.innerHTML = '';
    if (state.mode === 'exact') {
      state.exactHand.forEach((v, i) => {
        const d = createDieSVG(v, { size: 34, clickable: true });
        d.addEventListener('click', () => {
          state.exactHand[i] = (state.exactHand[i] % 6) + 1;
          render();
        });
        exactDice.appendChild(d);
      });
    }

    REROLLS.forEach((rLeft, index) => {
      const row = state.rows[index];
      const el = rowEls[index];
      el.innerHTML = '';
      el.classList.toggle('empty', !row.dice);

      const label = document.createElement('div');
      label.className = 'row-label';
      label.textContent = rLeft > 0 ? `${rLeft} reroll${rLeft > 1 ? 's' : ''} left` : 'final';
      el.appendChild(label);

      const diceEl = document.createElement('div');
      diceEl.className = 'row-dice';
      el.appendChild(diceEl);

      if (!row.dice) {
        const ph = document.createElement('div');
        ph.className = 'row-placeholder';
        ph.textContent = 'roll the row above';
        diceEl.appendChild(ph);
        return;
      }

      row.dice.forEach((v, i) => {
        const dieState = row.keep[i] ? 'kept' : 'will-reroll';
        const d = createDieSVG(v, { size: 44, state: dieState, clickable: true });
        d.addEventListener('click', () => {
          row.dice[i] = (row.dice[i] % 6) + 1;
          clearBelow(index);
          render();
        });
        const cell = document.createElement('div');
        cell.className = 'die-cell';
        cell.appendChild(d);

        const holdBtn = document.createElement('button');
        holdBtn.className = 'hold-btn' + (row.keep[i] ? ' held' : '');
        holdBtn.textContent = row.keep[i] ? 'keep' : 'reroll';
        holdBtn.addEventListener('click', () => {
          row.keep[i] = !row.keep[i];
          clearBelow(index);
          render();
        });
        cell.appendChild(holdBtn);
        diceEl.appendChild(cell);
      });

      // Stats column
      const stats = document.createElement('div');
      stats.className = 'row-stats';
      el.appendChild(stats);

      if (rLeft > 0) {
        const you = solver.pYou(row.dice, rLeft, keptDice(row));
        const best = solver.pOpt(row.dice, rLeft);
        stats.innerHTML =
          `<span class="stat you">you <b>${pct(you)}</b></span>` +
          `<span class="stat best">best <b>${pct(best)}</b></span>`;

        // Footer: optimal-hold hint + roll button
        const footer = document.createElement('div');
        footer.className = 'row-footer';

        const bestK = solver.bestKeep(row.dice, rLeft);
        const hint = document.createElement('span');
        hint.className = 'best-hint';
        hint.textContent = bestK.length ? `optimal hold: keep ${bestK.join(' ')}` : 'optimal hold: reroll all';
        footer.appendChild(hint);

        const roll = document.createElement('button');
        roll.className = 'roll-btn';
        roll.textContent = 'roll un-held';
        roll.addEventListener('click', () => {
          rollInto(index);
          render();
        });
        footer.appendChild(roll);
        el.appendChild(footer);
      } else {
        const met = currentTarget()(row.dice);
        stats.innerHTML = met
          ? `<span class="result hit">target reached</span>`
          : `<span class="result miss">target missed</span>`;
      }
    });
  }

  // ── Events ────────────────────────────────────────────────────────
  modeSel.addEventListener('change', () => {
    state.mode = modeSel.value;
    render();
  });
  catSel.addEventListener('change', () => {
    state.categoryId = catSel.value;
    render();
  });

  render();
}
