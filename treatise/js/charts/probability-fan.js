import {
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

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
    <div class="fan-subtitle">${freeCount} ${freeCount === 1 ? 'die' : 'dice'} to reroll \u2014 sample outcomes</div>
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
      const die = document.createElement('span');
      die.className = 'fan-die';
      const isKept = idx < keptDice.length;
      if (isKept) die.classList.add('fan-die-kept');
      die.textContent = v;
      diceRow.appendChild(die);
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
