/**
 * Probabilities tab: three always-visible dice rows (first roll, second roll,
 * target) with per-die up/down arrows and keep toggles. Shows the transition
 * probability first -> second, second -> target, and the overall product.
 * A kept die whose value changes in the next row makes that step impossible (0%).
 * @module render
 */
import { createDieSVG } from '/yatzy/shared/dice.js';
import { transitionProb, formatOneInN } from '/yatzy/shared/path-prob.js';

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
    // row 0 = first roll, row 1 = second roll, row 2 = target
    rows: [
      [1, 2, 3, 4, 6],
      [1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5],
    ],
    // keeps for row 0 and row 1 (target has no keeps)
    kept: [new Set([0, 1, 2, 3]), new Set([0, 1, 2, 3])],
  };

  const ROW_LABELS = ['First roll', 'Second roll', 'Target'];
  const ROW_SUBS = [
    'Set your opening dice. Click a die to keep it.',
    'Set the second roll. Click a die to keep it.',
    'Set the outcome you are aiming for.',
  ];

  root.innerHTML = '';

  const flow = document.createElement('div');
  flow.className = 'prob-flow';
  root.appendChild(flow);

  // Build fixed skeleton: row, transition, row, transition, row.
  const rowEls = [];
  const transEls = [];
  for (let r = 0; r < 3; r++) {
    const rowEl = document.createElement('div');
    rowEl.className = 'prob-row' + (r === 2 ? ' target' : '');
    flow.appendChild(rowEl);
    rowEls.push(rowEl);

    if (r < 2) {
      const t = document.createElement('div');
      t.className = 'prob-transition';
      flow.appendChild(t);
      transEls.push(t);
    }
  }

  const overallEl = document.createElement('div');
  overallEl.className = 'prob-overall';
  root.appendChild(overallEl);

  function render() {
    // Rows
    for (let r = 0; r < 3; r++) {
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

      const showKeep = r < 2;
      for (let i = 0; i < 5; i++) {
        const die = makeDie(state.rows[r][i], {
          kept: showKeep && state.kept[r].has(i),
          showKeep,
          onToggle: () => {
            if (state.kept[r].has(i)) state.kept[r].delete(i);
            else state.kept[r].add(i);
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
        const k = state.kept[r].size;
        info.textContent = k > 0 ? `keeping ${k}, rerolling ${5 - k}` : 'click a die to keep it';
        rowEl.appendChild(info);
      }
    }

    // Transitions
    const t0 = transitionProb(state.rows[0], state.rows[1], state.kept[0]);
    const t1 = transitionProb(state.rows[1], state.rows[2], state.kept[1]);
    renderTransition(transEls[0], 'first → second', t0);
    renderTransition(transEls[1], 'second → target', t1);

    // Overall
    const overall = t0 * t1;
    overallEl.innerHTML =
      `<div class="prob-overall-label">Overall: first → second → target</div>` +
      `<div class="prob-overall-eq">` +
      `<span class="prob-term">${pct(t0)}</span>` +
      `<span class="prob-op">×</span>` +
      `<span class="prob-term">${pct(t1)}</span>` +
      `<span class="prob-op">=</span>` +
      `<span class="prob-term prob-total">${pct(overall)}</span>` +
      `</div>` +
      `<div class="prob-overall-onein">${formatOneInN(overall)}</div>`;
  }

  function renderTransition(el, label, p) {
    const impossible = p === 0;
    el.classList.toggle('impossible', impossible);
    el.innerHTML =
      `<span class="prob-trans-arrow">↓</span>` +
      `<span class="prob-trans-label">P(${label})</span>` +
      `<span class="prob-trans-value">${pct(p)}</span>` +
      (impossible
        ? `<span class="prob-trans-note">impossible: a kept die changes value</span>`
        : `<span class="prob-trans-onein">${formatOneInN(p)}</span>`);
  }

  render();
}
