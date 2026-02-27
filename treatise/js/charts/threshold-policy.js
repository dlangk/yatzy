import { DataLoader } from '../data-loader.js';
import { getTextColor, getMutedColor, COLORS } from '../yatzy-viz.js';

export async function initThresholdPolicy() {
  const data = await DataLoader.thresholdPolicy();
  const container = document.getElementById('chart-threshold-policy');
  if (!container) return;

  const scenarios = data.scenarios || data;
  if (!scenarios || scenarios.length === 0) return;

  // Clear previous content
  container.innerHTML = '';

  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.gap = '16px';
  wrapper.style.flexWrap = 'wrap';
  wrapper.style.justifyContent = 'center';
  container.appendChild(wrapper);

  scenarios.slice(0, 3).forEach(scenario => {
    const card = document.createElement('div');
    card.style.cssText = `
      flex: 1 1 200px;
      max-width: 280px;
      border: 1px solid var(--border, #ddd);
      border-radius: 8px;
      padding: 16px;
      background: var(--bg-alt, #fafaf8);
      font-family: 'Newsreader', Georgia, serif;
    `;

    const title = document.createElement('h4');
    title.style.cssText = `
      margin: 0 0 8px 0;
      font-size: 14px;
      color: ${COLORS.accent};
      font-weight: 700;
    `;
    title.textContent = scenario.title || 'Endgame Scenario';
    card.appendChild(title);

    // Categories remaining
    if (scenario.categories_remaining) {
      const cats = document.createElement('div');
      cats.style.cssText = `
        font-size: 11px;
        color: var(--text-muted, #666);
        margin-bottom: 10px;
        line-height: 1.5;
      `;
      cats.textContent = `Remaining: ${scenario.categories_remaining.join(', ')}`;
      card.appendChild(cats);
    }

    // Threshold rules
    if (scenario.rules && scenario.rules.length > 0) {
      const ruleList = document.createElement('ul');
      ruleList.style.cssText = `
        margin: 0 0 12px 0;
        padding-left: 16px;
        font-size: 12px;
        line-height: 1.6;
        color: var(--text, #050505);
      `;
      scenario.rules.forEach(rule => {
        const li = document.createElement('li');
        li.textContent = rule;
        ruleList.appendChild(li);
      });
      card.appendChild(ruleList);
    }

    // EV comparison
    const evBox = document.createElement('div');
    evBox.style.cssText = `
      display: flex;
      justify-content: space-between;
      border-top: 1px solid var(--border, #ddd);
      padding-top: 8px;
      font-size: 12px;
    `;

    const optEV = scenario.optimal_ev || 0;
    const greedyEV = scenario.greedy_ev || 0;

    evBox.innerHTML = `
      <div>
        <div style="font-weight:700; color:${COLORS.optimal}">Optimal</div>
        <div style="font-size:16px; font-weight:700">${optEV.toFixed(1)}</div>
      </div>
      <div style="text-align:center; color:var(--text-muted,#666)">
        <div style="font-size:10px">\u0394</div>
        <div style="font-size:14px; font-weight:600">+${(optEV - greedyEV).toFixed(1)}</div>
      </div>
      <div style="text-align:right">
        <div style="font-weight:700; color:${COLORS.accent}">Greedy</div>
        <div style="font-size:16px; font-weight:700">${greedyEV.toFixed(1)}</div>
      </div>
    `;
    card.appendChild(evBox);

    wrapper.appendChild(card);
  });
}
