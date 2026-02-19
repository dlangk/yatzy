import { DataLoader } from '../data-loader.js';
import { renderDice, COLORS, getTextColor, getMutedColor } from '../yatzy-viz.js';

const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
];

export async function initGreedyVsOptimal() {
  const data = await DataLoader.greedyVsOptimal();
  const container = document.getElementById('chart-greedy-vs-optimal');
  if (!container) return;

  const inner = document.getElementById('chart-greedy-vs-optimal-inner');
  const prevBtn = document.getElementById('gvo-prev');
  const nextBtn = document.getElementById('gvo-next');
  const label = document.getElementById('gvo-label');

  let currentTurn = 0;

  function render() {
    const turn = data.turns[currentTurn];
    const g = turn.greedy;
    const o = turn.optimal;
    const diverges = turn.diverges;

    label.textContent = `Turn ${currentTurn + 1} / 15`;

    inner.innerHTML = '';

    // Dice row
    const diceRow = document.createElement('div');
    diceRow.className = 'dice-row';
    renderDice(diceRow, turn.dice);
    inner.appendChild(diceRow);

    // Comparison table
    const table = document.createElement('div');
    table.className = 'gvo-comparison';
    table.innerHTML = `
      <div class="gvo-side gvo-greedy">
        <div class="gvo-header" style="color: ${COLORS.accent}">Greedy</div>
        <div class="gvo-category ${diverges ? 'gvo-diverge' : ''}">${g.category}</div>
        <div class="gvo-score-row">
          <span class="gvo-score-label">Score</span>
          <span class="gvo-score-value">+${g.score}</span>
        </div>
        <div class="gvo-score-row">
          <span class="gvo-score-label">Running Total</span>
          <span class="gvo-score-value">${g.total}</span>
        </div>
        <div class="gvo-progress">
          <div class="gvo-progress-label">Upper: ${g.upper}/63</div>
          <div class="gvo-progress-bar">
            <div class="gvo-progress-fill" style="width: ${Math.min(100, (g.upper / 63) * 100)}%; background: ${COLORS.accent}"></div>
          </div>
        </div>
      </div>
      <div class="gvo-divider"></div>
      <div class="gvo-side gvo-optimal">
        <div class="gvo-header" style="color: ${COLORS.optimal}">Optimal</div>
        <div class="gvo-category ${diverges ? 'gvo-diverge' : ''}">${o.category}</div>
        <div class="gvo-score-row">
          <span class="gvo-score-label">Score</span>
          <span class="gvo-score-value">+${o.score}</span>
        </div>
        <div class="gvo-score-row">
          <span class="gvo-score-label">Running Total</span>
          <span class="gvo-score-value">${o.total}</span>
        </div>
        <div class="gvo-progress">
          <div class="gvo-progress-label">Upper: ${o.upper}/63</div>
          <div class="gvo-progress-bar">
            <div class="gvo-progress-fill" style="width: ${Math.min(100, (o.upper / 63) * 100)}%; background: ${COLORS.optimal}"></div>
          </div>
        </div>
      </div>
    `;
    inner.appendChild(table);

    // Divergence indicator
    if (diverges) {
      const div = document.createElement('div');
      div.className = 'gvo-diverge-note';
      div.textContent = 'Strategies diverge on this turn';
      inner.appendChild(div);
    }

    // Final summary on last turn
    if (currentTurn === 14) {
      const summary = document.createElement('div');
      summary.className = 'gvo-final';
      summary.innerHTML = `
        <div class="gvo-final-row">
          <span style="color: ${COLORS.accent}">Greedy: ${data.greedy_total}${data.greedy_bonus ? ' (incl. bonus)' : ' (no bonus)'}</span>
          <span class="gvo-final-gap">Gap: ${data.optimal_total - data.greedy_total}</span>
          <span style="color: ${COLORS.optimal}">Optimal: ${data.optimal_total}${data.optimal_bonus ? ' (incl. bonus)' : ''}</span>
        </div>
      `;
      inner.appendChild(summary);
    }
  }

  prevBtn.addEventListener('click', () => {
    currentTurn = Math.max(0, currentTurn - 1);
    render();
  });

  nextBtn.addEventListener('click', () => {
    currentTurn = Math.min(14, currentTurn + 1);
    render();
  });

  render();
}
