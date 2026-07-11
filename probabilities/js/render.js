/**
 * Roll-transition calculator: dice set 1 -> dice set 2.
 * You set your dice (set 1) and keep some, set the outcome (set 2), and see the
 * one-reroll transition probability. This mirrors the treatise transition
 * matrix (keep -> outcome), using order-independent multiset semantics: keeping
 * a value means the outcome must contain it somewhere, else the step is
 * impossible (0%).
 * @module render
 */
import { createDieSVG } from '/yatzy/shared/dice.js';
import { outcomeProbability, formatOneInN } from '/yatzy/shared/path-prob.js';

const pct = (p) => (p * 100).toFixed(1) + '%';

/** One die with ▲ [die] ▼ (arrows set value; click die toggles keep). */
function makeDie(value, { kept, showKeep, onToggle, onInc, onDec }) {
  const c = document.createElement('div');
  c.className = 'prob-die';

  const up = document.createElement('button');
  up.className = 'prob-die-arrow';
  up.innerHTML = '&#9650;';
  up.setAttribute('aria-label', 'increase value');
  up.addEventListener('click', onInc);

  const die = createDieSVG(value, { size: 44, state: kept ? 'kept' : 'normal', clickable: showKeep });
  if (showKeep) die.addEventListener('click', onToggle);

  const down = document.createElement('button');
  down.className = 'prob-die-arrow';
  down.innerHTML = '&#9660;';
  down.setAttribute('aria-label', 'decrease value');
  down.addEventListener('click', onDec);

  c.appendChild(up);
  c.appendChild(die);
  c.appendChild(down);
  return c;
}

export function initProbTool(root) {
  const state = {
    // row 0 = dice set 1 (you keep some), row 1 = dice set 2 (the outcome)
    rows: [
      [1, 2, 3, 4, 6],
      [1, 2, 3, 4, 5],
    ],
    kept: new Set([0, 1, 2, 3]), // which of set 1's dice you hold
  };

  const ROW_LABELS = ['Dice set 1', 'Dice set 2'];
  const ROW_SUBS = [
    'Your dice. Click a die to keep it; the rest are rerolled.',
    'The outcome you want after the reroll.',
  ];

  root.innerHTML = '';

  const flow = document.createElement('div');
  flow.className = 'prob-flow';
  root.appendChild(flow);

  const row0El = document.createElement('div');
  row0El.className = 'prob-row';
  flow.appendChild(row0El);

  const transEl = document.createElement('div');
  transEl.className = 'prob-transition';
  flow.appendChild(transEl);

  const row1El = document.createElement('div');
  row1El.className = 'prob-row';
  flow.appendChild(row1El);

  const rowEls = [row0El, row1El];

  function keptValues() {
    const out = [];
    for (let i = 0; i < 5; i++) if (state.kept.has(i)) out.push(state.rows[0][i]);
    return out;
  }

  function render() {
    for (let r = 0; r < 2; r++) {
      const rowEl = rowEls[r];
      rowEl.innerHTML = '';

      const head = document.createElement('div');
      head.className = 'prob-row-head';
      head.innerHTML =
        `<span class="prob-row-label">${ROW_LABELS[r]}</span>` +
        `<span class="prob-row-sub">${ROW_SUBS[r]}</span>`;
      rowEl.appendChild(head);

      const diceEl = document.createElement('div');
      diceEl.className = 'prob-row-dice';
      rowEl.appendChild(diceEl);

      const showKeep = r === 0;
      for (let i = 0; i < 5; i++) {
        const die = makeDie(state.rows[r][i], {
          kept: showKeep && state.kept.has(i),
          showKeep,
          onToggle: () => {
            if (state.kept.has(i)) state.kept.delete(i);
            else state.kept.add(i);
            render();
          },
          onInc: () => {
            state.rows[r][i] = (state.rows[r][i] % 6) + 1;
            render();
          },
          onDec: () => {
            state.rows[r][i] = ((state.rows[r][i] - 2 + 6) % 6) + 1;
            render();
          },
        });
        diceEl.appendChild(die);
      }

      if (showKeep) {
        const info = document.createElement('div');
        info.className = 'prob-keep-info';
        const k = state.kept.size;
        info.textContent = k > 0 ? `keeping ${k}, rerolling ${5 - k}` : 'click a die to keep it';
        rowEl.appendChild(info);
      }
    }

    const p = outcomeProbability(keptValues(), state.rows[1]);
    const impossible = p === 0;
    transEl.classList.toggle('impossible', impossible);
    transEl.innerHTML =
      `<span class="prob-trans-arrow">↓</span>` +
      `<span class="prob-trans-label">P(set 1 → set 2)</span>` +
      `<span class="prob-trans-value">${pct(p)}</span>` +
      (impossible
        ? `<span class="prob-trans-note">impossible: you kept a die the outcome does not contain</span>`
        : `<span class="prob-trans-onein">${formatOneInN(p)}</span>`);
  }

  render();
}
