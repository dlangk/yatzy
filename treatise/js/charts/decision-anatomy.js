import { DataLoader } from '../data-loader.js';
import { getTextColor, getMutedColor, COLORS } from '../yatzy-viz.js';

export async function initDecisionAnatomy() {
  const container = document.getElementById('chart-decision-anatomy');
  if (!container) return;

  const data = await DataLoader.decisionAnatomy();
  const contentEl = document.getElementById('chart-decision-anatomy-content');
  if (!contentEl) return;

  const depth = () => document.body.getAttribute('data-depth') || '1';

  render();

  function render() {
    contentEl.innerHTML = '';

    // Dice display
    const diceSection = document.createElement('div');
    diceSection.className = 'anatomy-dice-section';

    const diceLabel = document.createElement('div');
    diceLabel.className = 'anatomy-dice-label';
    diceLabel.textContent = 'Current dice:';
    diceSection.appendChild(diceLabel);

    const diceRow = document.createElement('div');
    diceRow.className = 'dice-row';
    data.state.dice.forEach((v) => {
      const die = document.createElement('span');
      die.className = 'die';
      die.textContent = v;
      diceRow.appendChild(die);
    });
    diceSection.appendChild(diceRow);

    const stateLabel = document.createElement('div');
    stateLabel.className = 'anatomy-state-label';
    stateLabel.innerHTML = `No rerolls left | Upper: ${data.state.upper_score} | Scored: ${data.state.scored_labels.join(', ')}`;
    diceSection.appendChild(stateLabel);

    contentEl.appendChild(diceSection);

    // Split container: human vs optimal
    const split = document.createElement('div');
    split.className = 'split-container';

    // Human side
    const humanSide = document.createElement('div');
    humanSide.className = 'split-left';
    humanSide.innerHTML = `
      <div class="anatomy-side-header human-header">Most Players</div>
      <div class="anatomy-action">${data.human_choice.action}</div>
      <div class="anatomy-score">+${data.human_choice.score} points</div>
      <div class="anatomy-why">${data.human_choice.why}</div>
      <div class="anatomy-ev">EV remaining: <strong>${data.human_choice.ev_remaining.toFixed(1)}</strong></div>
    `;

    // Depth 2: bonus rate
    if (depth() >= '2' && data.downstream) {
      const hd = data.downstream.find((d) => d.branch === 'human');
      if (hd) {
        humanSide.innerHTML += `
          <div class="anatomy-detail">
            <div>Bonus rate: ${(hd.expected_bonus_rate * 100).toFixed(0)}%</div>
            <div>Expected total: ${hd.expected_total.toFixed(1)}</div>
          </div>
        `;
      }
    }
    split.appendChild(humanSide);

    // Optimal side
    const optSide = document.createElement('div');
    optSide.className = 'split-right';
    optSide.innerHTML = `
      <div class="anatomy-side-header optimal-header">Solver</div>
      <div class="anatomy-action">${data.optimal_choice.action}</div>
      <div class="anatomy-score">+${data.optimal_choice.score} points</div>
      <div class="anatomy-why">${data.optimal_choice.why}</div>
      <div class="anatomy-ev">EV remaining: <strong>${data.optimal_choice.ev_remaining.toFixed(1)}</strong></div>
    `;

    if (depth() >= '2' && data.downstream) {
      const od = data.downstream.find((d) => d.branch === 'optimal');
      if (od) {
        optSide.innerHTML += `
          <div class="anatomy-detail">
            <div>Bonus rate: ${(od.expected_bonus_rate * 100).toFixed(0)}%</div>
            <div>Expected total: ${od.expected_total.toFixed(1)}</div>
          </div>
        `;
      }
    }
    split.appendChild(optSide);
    contentEl.appendChild(split);

    // Gap indicator
    const gapDiv = document.createElement('div');
    gapDiv.className = 'anatomy-gap';
    gapDiv.innerHTML = `Solver advantage: <span style="color:${COLORS.accent}">+${data.gap.toFixed(1)} EV</span>`;
    contentEl.appendChild(gapDiv);

    // Depth 2: breakdown
    if (depth() >= '2' && data.breakdown) {
      const breakdown = document.createElement('div');
      breakdown.className = 'anatomy-breakdown';
      breakdown.innerHTML = `
        <div class="anatomy-breakdown-title">EV Decomposition</div>
        <div class="anatomy-breakdown-row">
          <span>Immediate sacrifice:</span>
          <span>${data.breakdown.immediate_sacrifice} pts</span>
        </div>
        <div class="anatomy-breakdown-row">
          <span>Upper bonus gain:</span>
          <span>+${data.breakdown.upper_bonus_gain.toFixed(1)} pts</span>
        </div>
        <div class="anatomy-breakdown-row">
          <span>Flexibility value:</span>
          <span>+${data.breakdown.flexibility_value.toFixed(1)} pts</span>
        </div>
      `;
      contentEl.appendChild(breakdown);
    }

    // Depth 3: state tuple
    if (depth() >= '3') {
      const code = document.createElement('pre');
      code.innerHTML = `<code>state = (upper=${data.state.upper_score}, scored=0x${data.state.scored_categories.toString(16)})
human:   V(${data.state.upper_score}, 0x${data.state.scored_categories.toString(16)}) + score(FullHouse) = ${data.human_choice.score} + ${data.human_choice.ev_remaining.toFixed(1)}
optimal: V(${data.state.upper_score + data.optimal_choice.score}, 0x${(data.state.scored_categories | (1 << 4)).toString(16)}) + score(Fives) = ${data.optimal_choice.score} + ${data.optimal_choice.ev_remaining.toFixed(1)}</code>`;
      contentEl.appendChild(code);
    }
  }

  // Re-render on depth change
  const observer = new MutationObserver(() => render());
  observer.observe(document.body, { attributes: true, attributeFilter: ['data-depth'] });
}
