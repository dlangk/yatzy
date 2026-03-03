import {
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';
import { PIPS } from '../utils/dice-interactive.js';

function createFanDieSVG(value, isKept) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 48 48');
  svg.setAttribute('width', '24');
  svg.setAttribute('height', '24');
  svg.classList.add('fan-die-svg');
  if (isKept) svg.classList.add('fan-die-kept');

  const stroke = isKept ? 'var(--accent)' : 'var(--border)';
  const strokeWidth = isKept ? 3 : 2;
  const fill = 'var(--bg-alt)';
  const pipOpacity = isKept ? 1 : 0.6;

  let html = `<rect x="1" y="1" width="46" height="46" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
  (PIPS[value] || []).forEach(p => {
    html += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)" opacity="${pipOpacity}"/>`;
  });
  svg.innerHTML = html;
  return svg;
}

/**
 * Render a probability fan showing outcome distribution after a keep decision.
 * Called by widget-interactive.js, not directly by index.html.
 */
export function renderProbabilityFan(container, keepOption) {
  container.innerHTML = '';

  if (!keepOption || !keepOption.second_rolls || keepOption.second_rolls.length === 0) {
    container.innerHTML = '<div style="color:var(--text-muted);font-size:0.85rem;padding:1rem;">No reroll data for this keep option.</div>';
    return;
  }

  const rolls = keepOption.second_rolls;
  const keptDice = keepOption.kept_dice || [];
  const freeCount = 5 - keptDice.length;

  // Header
  const header = document.createElement('div');
  header.className = 'fan-header';
  header.innerHTML = `
    <div class="fan-title">After keeping <strong>${keepOption.label}</strong></div>
    <div class="fan-subtitle">${freeCount} ${freeCount === 1 ? 'die' : 'dice'} to reroll</div>
  `;
  container.appendChild(header);

  // Outcome cards
  const grid = document.createElement('div');
  grid.className = 'fan-grid';

  rolls.forEach((roll) => {
    const card = document.createElement('div');
    card.className = 'fan-card';

    // Dice display
    const diceRow = document.createElement('div');
    diceRow.className = 'fan-dice-row';
    roll.dice.forEach((v, idx) => {
      const isKept = idx < keptDice.length;
      diceRow.appendChild(createFanDieSVG(v, isKept));
    });
    card.appendChild(diceRow);

    // Description
    const desc = document.createElement('div');
    desc.className = 'fan-desc';
    desc.textContent = roll.description;
    card.appendChild(desc);

    grid.appendChild(card);
  });

  container.appendChild(grid);
}
