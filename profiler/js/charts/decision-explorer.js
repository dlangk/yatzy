import {
  createChart, tooltip, renderDice, formatTheta, thetaColor,
  getTextColor, getMutedColor, COLORS,
} from '../yatzy-viz.js';

// Curated scenarios showing how decisions change with risk preference
const SCENARIOS = [
  {
    title: 'Sacrifice upper for straight?',
    dice: [2, 3, 4, 5, 5],
    context: 'Turn 6, need 4 more upper points for bonus. Small Straight available.',
    actions: [
      { label: 'Score Fives (10)', ev0: 242.1, description: 'Bank upper section progress' },
      { label: 'Score Small Straight (14)', ev0: 238.7, description: 'Take guaranteed lower points' },
      { label: 'Score Chance (19)', ev0: 235.2, description: 'Take high immediate points' },
    ],
    insight: 'The optimal player scores the upper category to protect the 50-point bonus, even though the straight scores more right now.',
  },
  {
    title: 'Keep pair or chase Yatzy?',
    dice: [6, 6, 6, 6, 3],
    context: 'First reroll. Four of a kind rolled. Yatzy still available.',
    actions: [
      { label: 'Keep all 4 sixes, reroll 1', ev0: 267.3, description: 'Go for Yatzy (50 pts)' },
      { label: 'Keep all 5 dice', ev0: 262.8, description: 'Lock in strong hand' },
    ],
    insight: 'With four sixes, the EV gain from chasing the 1-in-6 Yatzy chance outweighs the risk of losing a die.',
  },
  {
    title: 'Three of a kind vs upper section',
    dice: [1, 1, 1, 2, 5],
    context: 'Turn 3, Three of a Kind and Ones both available.',
    actions: [
      { label: 'Score Ones (3)', ev0: 251.4, description: 'Build toward upper bonus' },
      { label: 'Score Three of a Kind (10)', ev0: 244.8, description: 'Take higher immediate score' },
    ],
    insight: 'A common human mistake: Three of a Kind scores more, but Ones protects the 50-point bonus. The bonus is worth sacrificing 7 points now.',
  },
  {
    title: 'How many dice to reroll?',
    dice: [1, 5, 5, 6, 6],
    context: 'First reroll. Two pairs showing.',
    actions: [
      { label: 'Keep 2 (5,5) reroll 3', ev0: 247.2, description: 'Focus on one pair, maximize flexibility' },
      { label: 'Keep 1 (6) reroll 4', ev0: 243.1, description: 'Go wide for better combinations' },
      { label: 'Keep 4 (5,5,6,6) reroll 1', ev0: 240.5, description: 'Lock two pairs, try for full house' },
    ],
    insight: 'The heuristic player keeps too many dice. The optimal strategy keeps fewer dice to maximize the chance of hitting the best final hand.',
  },
  {
    title: 'Full house or chance?',
    dice: [5, 5, 5, 5, 6],
    context: 'Final reroll. Four of a Kind available, Chance available.',
    actions: [
      { label: 'Keep all 5 dice', ev0: 254.1, description: 'Lock in Four of a Kind (20) or Chance (26)' },
      { label: 'Keep 4 fives, reroll 6', ev0: 247.3, description: 'Chase Yatzy, risk losing the six' },
    ],
    insight: 'On the last reroll, even with four-of-a-kind, keeping all dice is better than chasing Yatzy. Time pressure changes the calculus.',
  },
];

export function initDecisionExplorer() {
  const container = document.getElementById('chart-decision-explorer');
  if (!container) return;

  let currentIdx = 0;

  const diceContainer = container.querySelector('.dice-row');
  const titleEl = container.querySelector('.scenario-title');
  const contextEl = container.querySelector('.scenario-context');
  const insightEl = container.querySelector('.scenario-insight');
  const tableBody = container.querySelector('.action-table tbody');
  const counterEl = container.querySelector('.scenario-label');
  const prevBtn = container.querySelector('.scenario-prev');
  const nextBtn = container.querySelector('.scenario-next');

  function render() {
    const scenario = SCENARIOS[currentIdx];
    counterEl.textContent = `${currentIdx + 1} / ${SCENARIOS.length}`;
    titleEl.textContent = scenario.title;
    contextEl.textContent = scenario.context;
    insightEl.textContent = scenario.insight;

    // Dice
    renderDice(diceContainer, scenario.dice);

    // Action table
    tableBody.innerHTML = '';
    const bestEv = Math.max(...scenario.actions.map(a => a.ev0));

    scenario.actions.forEach(action => {
      const isBest = action.ev0 === bestEv;
      const gap = action.ev0 - bestEv;
      const tr = document.createElement('tr');
      if (isBest) tr.className = 'best-action';
      tr.innerHTML = `
        <td>${action.label}</td>
        <td class="num">${action.ev0.toFixed(1)}</td>
        <td class="num" style="color: ${gap === 0 ? COLORS.optimal : COLORS.accent}">${gap === 0 ? 'best' : gap.toFixed(1)}</td>
        <td>${action.description}</td>
      `;
      tableBody.appendChild(tr);
    });
  }

  prevBtn.addEventListener('click', () => {
    currentIdx = (currentIdx - 1 + SCENARIOS.length) % SCENARIOS.length;
    render();
  });

  nextBtn.addEventListener('click', () => {
    currentIdx = (currentIdx + 1) % SCENARIOS.length;
    render();
  });

  render();
}
