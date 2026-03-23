import { dieSVGString } from '/yatzy/shared/dice.js';

function dieSVG(value, { kept = false, ghosted = false } = {}) {
  const state = ghosted ? 'reroll' : (kept ? 'kept' : 'normal');
  return dieSVGString(value, { size: 26, state });
}

function diceRow(values, opts) {
  return values.map(v => `<div class="keep-die">${dieSVG(v, opts)}</div>`).join('');
}

export async function initKeepEquivalence() {
  const container = document.getElementById('chart-keep-equivalence');
  if (!container) return;

  // Mask A: keep dice 0,1 (both 3s), reroll dice 2,3,4 (3, 5, 5)
  // Mask B: keep dice 0,2 (both 3s), reroll dice 1,3,4 (3, 5, 5)
  // Both produce the same keep: two 3s

  const maskA = {
    dice: [
      { v: 3, kept: true },
      { v: 3, kept: true },
      { v: 3, kept: false },
      { v: 5, kept: false },
      { v: 5, kept: false },
    ],
  };

  const maskB = {
    dice: [
      { v: 3, kept: true },
      { v: 3, kept: false },
      { v: 3, kept: true },
      { v: 5, kept: false },
      { v: 5, kept: false },
    ],
  };

  const result = [3, 3]; // same keep either way

  function renderRow(label, mask) {
    const dice = mask.dice.map(d =>
      `<div class="keep-die">${dieSVG(d.v, { kept: d.kept, ghosted: !d.kept })}</div>`
    ).join('');
    const res = result.map(v =>
      `<div class="keep-die">${dieSVG(v, { kept: true })}</div>`
    ).join('');
    return `
      <div class="keep-equiv-row">
        <span class="keep-equiv-label">${label}</span>
        <div class="keep-equiv-dice">${dice}</div>
        <span class="keep-equiv-arrow">&rarr;</span>
        <div class="keep-equiv-result">${res}</div>
      </div>`;
  }

  container.innerHTML = `
    <div class="keep-equiv">
      ${renderRow('Mask A', maskA)}
      ${renderRow('Mask B', maskB)}
      <span class="keep-equiv-result-label">Same keep</span>
    </div>`;
}
