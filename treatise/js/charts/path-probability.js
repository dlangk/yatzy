/**
 * Path Probability Visualizer: interactive three-roll Yatzy path builder
 * with a live probability chain showing multiplicative compounding.
 */

// ── Math helpers ────────────────────────────────────────────────

function factorial(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function frequencies(dice) {
  const freq = [0, 0, 0, 0, 0, 0];
  for (const d of dice) freq[d - 1]++;
  return freq;
}

function multinomial(n, freqs) {
  let denom = 1;
  for (const f of freqs) denom *= factorial(f);
  return factorial(n) / denom;
}

function rollProbability(dice) {
  if (dice.length === 0) return 1;
  const freqs = frequencies(dice);
  return multinomial(dice.length, freqs) / Math.pow(6, dice.length);
}

function formatFraction(dice) {
  if (dice.length === 0) return { num: '1', den: '1' };
  const freqs = frequencies(dice);
  const num = multinomial(dice.length, freqs);
  const den = Math.pow(6, dice.length);
  const g = gcd(num, den);
  return { num: String(num / g), den: String(den / g) };
}

function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

function formatOneInN(p) {
  if (p >= 1) return '1 in 1';
  if (p <= 0) return 'impossible';
  const n = Math.round(1 / p);
  return `1 in ${n.toLocaleString()}`;
}

import { createDieSVG } from '/yatzy/shared/dice.js';

// ── Die component (matches Play UI pattern: ▲ [die] ▼) ────────

function makeDie(value, { kept = false, valueFixed = false, noKeep = false, onToggle, onInc, onDec }) {
  const container = document.createElement('div');
  container.className = 'pp-die-container';

  const upBtn = document.createElement('button');
  upBtn.className = 'pp-die-arrow';
  upBtn.innerHTML = '&#9650;';

  const state = kept ? 'kept' : 'normal';
  const die = createDieSVG(value, { size: 40, state, clickable: !valueFixed || !noKeep });

  const downBtn = document.createElement('button');
  downBtn.className = 'pp-die-arrow';
  downBtn.innerHTML = '&#9660;';

  if (valueFixed) {
    upBtn.style.visibility = 'hidden';
    downBtn.style.visibility = 'hidden';
  } else {
    upBtn.addEventListener('click', onInc);
    downBtn.addEventListener('click', onDec);
  }

  if (!noKeep) {
    die.addEventListener('click', onToggle);
  }

  container.appendChild(upBtn);
  container.appendChild(die);
  container.appendChild(downBtn);
  return container;
}

// ── Main export ─────────────────────────────────────────────────

export async function initPathProbability() {
  const container = document.getElementById('chart-path-probability');
  if (!container) return;

  const caption = container.querySelector('.chart-caption');
  container.innerHTML = '';
  if (caption) container.appendChild(caption);

  // ── State ──
  const d = () => Math.floor(Math.random() * 6) + 1;
  const state = {
    dice: [
      [d(), d(), d(), d(), d()],
      [d(), d(), d(), d(), d()],
      [d(), d(), d(), d(), d()],
    ],
    kept: [
      new Set(),
      new Set(),
    ],
  };

  // ── Build DOM ──
  const widget = document.createElement('div');
  widget.className = 'path-prob-widget';
  container.appendChild(widget);

  const stepsContainer = document.createElement('div');
  stepsContainer.className = 'pp-steps';
  widget.appendChild(stepsContainer);

  const chainContainer = document.createElement('div');
  chainContainer.className = 'pp-chain';
  widget.appendChild(chainContainer);

  const stepEls = [];
  const stepLabels = ['Roll 1', 'Roll 2', 'Roll 3'];
  const stepSubs = [
    'Use arrows to set values, click to toggle keep',
    'Kept dice are locked; set and keep the rerolled dice',
    'Set the final outcome for the rerolled dice',
  ];

  for (let r = 0; r < 3; r++) {
    const step = document.createElement('div');
    step.className = 'pp-step';

    const header = document.createElement('div');
    header.className = 'pp-step-header';
    header.textContent = stepLabels[r];
    step.appendChild(header);

    const sub = document.createElement('div');
    sub.className = 'pp-step-sub';
    sub.textContent = stepSubs[r];
    step.appendChild(sub);

    const diceArea = document.createElement('div');
    diceArea.className = 'pp-step-dice';
    step.appendChild(diceArea);

    const keepInfo = document.createElement('div');
    keepInfo.className = 'pp-keep-info';
    if (r >= 2) keepInfo.style.visibility = 'hidden';
    step.appendChild(keepInfo);

    const probArea = document.createElement('div');
    probArea.className = 'pp-step-prob';
    step.appendChild(probArea);

    stepsContainer.appendChild(step);
    stepEls.push({ step, diceArea, keepInfo, probArea });
  }

  // ── Helpers ──

  function getFreeDiceValues(rollIndex) {
    if (rollIndex === 0) return [...state.dice[0]];
    const prevKept = state.kept[rollIndex - 1];
    const free = [];
    for (let i = 0; i < 5; i++) {
      if (!prevKept.has(i)) free.push(state.dice[rollIndex][i]);
    }
    return free;
  }

  function syncKeptForward(fromRoll) {
    const nextRoll = fromRoll + 1;
    if (nextRoll > 2) return;
    for (const i of state.kept[fromRoll]) {
      state.dice[nextRoll][i] = state.dice[fromRoll][i];
    }
    syncKeptForward(nextRoll);
  }

  function computeRollProb(rollIndex) {
    return rollProbability(getFreeDiceValues(rollIndex));
  }

  function computeRollFraction(rollIndex) {
    return formatFraction(getFreeDiceValues(rollIndex));
  }

  // ── Render ──

  function renderAll() {
    for (let r = 0; r < 3; r++) renderStep(r);
    renderChain();
  }

  function renderStep(r) {
    const { diceArea, keepInfo, probArea } = stepEls[r];
    diceArea.innerHTML = '';

    const prevKept = r > 0 ? state.kept[r - 1] : new Set();

    for (let i = 0; i < 5; i++) {
      const fromPrevKeep = r > 0 && prevKept.has(i);
      const isKept = r < 2 && state.kept[r].has(i);
      const isFinalRoll = r === 2;

      const die = makeDie(state.dice[r][i], {
        kept: isKept,
        valueFixed: fromPrevKeep,
        noKeep: isFinalRoll,
        onToggle: () => {
          if (state.kept[r].has(i)) {
            state.kept[r].delete(i);
          } else {
            state.kept[r].add(i);
          }
          syncKeptForward(r);
          if (r === 0) state.kept[1].clear();
          renderAll();
        },
        onInc: () => {
          state.dice[r][i] = (state.dice[r][i] % 6) + 1;
          if (r < 2 && state.kept[r].has(i)) syncKeptForward(r);
          renderAll();
        },
        onDec: () => {
          state.dice[r][i] = ((state.dice[r][i] - 2 + 6) % 6) + 1;
          if (r < 2 && state.kept[r].has(i)) syncKeptForward(r);
          renderAll();
        },
      });
      diceArea.appendChild(die);
    }

    // Keep info label
    if (r < 2) {
      const keptCount = state.kept[r].size;
      const freeCount = 5 - keptCount;
      keepInfo.textContent = keptCount > 0
        ? `Keeping ${keptCount}, rerolling ${freeCount}`
        : 'Click a die to keep it';
    }

    // Probability
    const free = getFreeDiceValues(r);
    if (free.length === 0) {
      probArea.innerHTML = '<span class="pp-prob-value">P = 1 (all kept)</span>';
    } else {
      const frac = computeRollFraction(r);
      const p = computeRollProb(r);
      probArea.innerHTML =
        `<span class="pp-prob-frac">${frac.num} / ${frac.den}</span>` +
        `<span class="pp-prob-onein">${formatOneInN(p)}</span>`;
    }
  }

  function renderChain() {
    chainContainer.innerHTML = '';

    const p1 = computeRollProb(0);
    const p2 = computeRollProb(1);
    const p3 = computeRollProb(2);
    const pTotal = p1 * p2 * p3;

    const f1 = computeRollFraction(0);
    const f2 = computeRollFraction(1);
    const f3 = computeRollFraction(2);

    const expr = document.createElement('div');
    expr.className = 'pp-chain-expr';

    const parts = [
      { label: 'P(Roll 1)', frac: f1 },
      { label: 'P(Roll 2 | kept)', frac: f2 },
      { label: 'P(Roll 3 | kept)', frac: f3 },
    ];

    parts.forEach((part, i) => {
      if (i > 0) {
        const op = document.createElement('span');
        op.className = 'pp-chain-op';
        op.textContent = '\u00D7';
        expr.appendChild(op);
      }
      const term = document.createElement('span');
      term.className = 'pp-chain-term';
      term.innerHTML =
        `<span class="pp-chain-label">${part.label}</span>` +
        `<span class="pp-chain-frac">${part.frac.num}/${part.frac.den}</span>`;
      expr.appendChild(term);
    });

    const eq = document.createElement('span');
    eq.className = 'pp-chain-op';
    eq.textContent = '=';
    expr.appendChild(eq);

    const totalNum = Number(f1.num) * Number(f2.num) * Number(f3.num);
    const totalDen = Number(f1.den) * Number(f2.den) * Number(f3.den);
    const g = gcd(totalNum, totalDen);
    const totalFracStr = `${(totalNum / g).toLocaleString()} / ${(totalDen / g).toLocaleString()}`;

    const total = document.createElement('span');
    total.className = 'pp-chain-term pp-chain-total';
    total.innerHTML =
      `<span class="pp-chain-label">P(path)</span>` +
      `<span class="pp-chain-frac pp-chain-frac-total">${totalFracStr}</span>` +
      `<span class="pp-prob-onein">${formatOneInN(pTotal)}</span>`;
    expr.appendChild(total);

    chainContainer.appendChild(expr);

    const anchor = document.createElement('div');
    anchor.className = 'pp-chain-anchor';
    if (pTotal >= 1) {
      anchor.textContent = 'Every player traces this path (all dice were kept).';
    } else {
      // Pick a population large enough that at least ~1 person would trace the path
      const oneInN = Math.round(1 / pTotal);
      if (oneInN <= 1_000_000) {
        const expected = Math.round(pTotal * 1_000_000);
        anchor.textContent =
          `If 1,000,000 people played this exact sequence of keeps, roughly ${expected.toLocaleString()} would trace this exact path.`;
      } else {
        // Path is so rare that even 1M people wouldn't see it; scale up
        anchor.textContent =
          `You would need roughly ${oneInN.toLocaleString()} people to expect a single one to trace this exact path.`;
      }
    }
    chainContainer.appendChild(anchor);
  }

  // ── Initial render ──
  renderAll();

  // ── Scoped styles ──
  if (!document.getElementById('pp-styles')) {
    const style = document.createElement('style');
    style.id = 'pp-styles';
    style.textContent = `
      .path-prob-widget {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
      }
      .pp-steps {
        display: flex;
        gap: 0;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
      }
      .pp-step {
        flex: 1;
        padding: 0.75rem 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.4rem;
        align-items: center;
      }
      .pp-step + .pp-step {
        border-left: 1px solid var(--border);
      }
      .pp-step-header {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--text);
      }
      .pp-step-sub {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-align: center;
        line-height: 1.3;
        min-height: 2.2em;
      }
      .pp-step-dice {
        display: flex;
        gap: 4px;
        justify-content: center;
      }
      .pp-die-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
      }
      .pp-die-arrow {
        width: 40px;
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
      .pp-die-arrow:hover {
        background: var(--accent-light);
      }
      .pp-keep-info {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-align: center;
        min-height: 1.2em;
      }
      .pp-step-prob {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.1rem;
        padding-top: 0.3rem;
        border-top: 1px solid var(--border);
        width: 100%;
      }
      .pp-prob-frac {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text);
      }
      .pp-prob-onein {
        font-size: 0.7rem;
        color: var(--text-muted);
      }
      .pp-prob-value {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-style: italic;
      }
      .pp-chain {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem;
        background: var(--bg-alt);
        border: 1px solid var(--border);
        border-radius: 8px;
      }
      .pp-chain-expr {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        justify-content: center;
      }
      .pp-chain-term {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.1rem;
      }
      .pp-chain-label {
        font-size: 0.6rem;
        color: var(--text-muted);
        white-space: nowrap;
      }
      .pp-chain-frac {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--text);
      }
      .pp-chain-op {
        font-size: 1.1rem;
        color: var(--text-muted);
        padding: 0 0.1rem;
      }
      .pp-chain-total {
        padding: 0.25rem 0.5rem;
        background: var(--accent-light);
        border-radius: 6px;
      }
      .pp-chain-frac-total {
        font-weight: 700;
        font-size: 1rem;
        color: var(--accent);
      }
      .pp-chain-anchor {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-align: center;
        line-height: 1.4;
      }
      @media (max-width: 720px) {
        .pp-steps { flex-direction: column; }
        .pp-step + .pp-step { border-left: none; border-top: 1px solid var(--border); }
      }
    `;
    document.head.appendChild(style);
  }
}
