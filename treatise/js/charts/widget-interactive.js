import { DataLoader } from '../data-loader.js';
import {
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';
import { renderDiceSelectable } from '../utils/dice-interactive.js';
import { renderProbabilityFan } from './probability-fan.js';

export async function initWidgetInteractive() {
  const container = document.getElementById('chart-widget-interactive');
  if (!container) return;

  const data = await DataLoader.widgetScenarios();
  const scenarios = data.scenarios;

  const select = document.getElementById('widget-scenario-select');
  const solverBtn = document.getElementById('widget-solver-btn');
  const resetBtn = document.getElementById('widget-reset-btn');
  const flowEl = document.getElementById('widget-flow');
  const fanPanel = document.getElementById('widget-fan-panel');

  let currentScenario = 0;
  let selectedKeep = -1; // index into keep_options
  let phase = 'keep'; // 'keep' | 'rolled' | 'category'

  const depth = () => document.body.getAttribute('data-depth') || '1';

  function render() {
    const scenario = scenarios[currentScenario];
    flowEl.innerHTML = '';
    fanPanel.innerHTML = '';

    // Phase 1: Initial dice
    const col1 = makeColumn('Initial Roll', 'chance');
    const diceRow = document.createElement('div');
    diceRow.className = 'dice-row';
    scenario.initial_dice.forEach((v) => {
      const die = document.createElement('span');
      die.className = 'die';
      die.textContent = v;
      diceRow.appendChild(die);
    });
    col1.appendChild(diceRow);

    // State info
    const stateInfo = document.createElement('div');
    stateInfo.className = 'widget-state-info';
    stateInfo.innerHTML = `<span>Upper: ${scenario.upper_score}</span>`;
    if (depth() >= '2') {
      stateInfo.innerHTML += ` <span>Scored: 0x${scenario.scored_categories.toString(16)}</span>`;
    }
    col1.appendChild(stateInfo);
    flowEl.appendChild(col1);

    // Arrow
    flowEl.appendChild(makeArrow());

    // Phase 2: Keep options
    const col2 = makeColumn('Keep Decision', 'decision');
    scenario.keep_options.forEach((opt, idx) => {
      const card = document.createElement('div');
      card.className = 'keep-card';
      if (idx === selectedKeep) card.classList.add('selected');
      if (opt.optimal) card.classList.add('optimal');

      // Keep dice
      const keptRow = document.createElement('div');
      keptRow.className = 'keep-dice-row';
      opt.kept_dice.forEach((v) => {
        const die = document.createElement('span');
        die.className = 'die die-small';
        die.textContent = v;
        keptRow.appendChild(die);
      });
      card.appendChild(keptRow);

      // Label + EV
      const labelEl = document.createElement('div');
      labelEl.className = 'keep-label';
      labelEl.textContent = opt.label;
      card.appendChild(labelEl);

      const evEl = document.createElement('div');
      evEl.className = 'ev-label';
      evEl.textContent = `EV ${opt.ev.toFixed(1)}`;
      if (opt.optimal) {
        evEl.innerHTML += ' <span class="optimal-badge">\u2605</span>';
      }
      card.appendChild(evEl);

      if (depth() >= '2') {
        const detail = document.createElement('div');
        detail.className = 'keep-detail';
        detail.textContent = `Keep indices: [${opt.indices.join(',')}]`;
        card.appendChild(detail);
      }

      card.addEventListener('click', () => {
        selectedKeep = idx;
        phase = 'rolled';
        render();
      });

      col2.appendChild(card);
    });
    flowEl.appendChild(col2);

    // Phase 3: Show fan + second rolls if a keep is selected
    if (selectedKeep >= 0) {
      flowEl.appendChild(makeArrow());

      const opt = scenario.keep_options[selectedKeep];

      // Second roll column
      const col3 = makeColumn('Reroll Outcomes', 'chance');
      if (opt.second_rolls && opt.second_rolls.length > 0) {
        opt.second_rolls.forEach((roll) => {
          const rollCard = document.createElement('div');
          rollCard.className = 'roll-card';

          const dRow = document.createElement('div');
          dRow.className = 'keep-dice-row';
          roll.dice.forEach((v) => {
            const die = document.createElement('span');
            die.className = 'die die-small';
            die.textContent = v;
            dRow.appendChild(die);
          });
          rollCard.appendChild(dRow);

          const desc = document.createElement('div');
          desc.className = 'roll-desc';
          desc.textContent = roll.description;
          rollCard.appendChild(desc);

          col3.appendChild(rollCard);
        });
      } else {
        const noRoll = document.createElement('div');
        noRoll.className = 'roll-desc';
        noRoll.textContent = 'Keeping all dice.';
        col3.appendChild(noRoll);
      }
      flowEl.appendChild(col3);

      // Phase 4: Category options
      if (scenario.category_examples && scenario.category_examples.length > 0) {
        flowEl.appendChild(makeArrow());

        const col4 = makeColumn('Score Category', 'decision');
        scenario.category_examples.forEach((cat) => {
          const catCard = document.createElement('div');
          catCard.className = 'keep-card';

          const catLabel = document.createElement('div');
          catLabel.className = 'keep-label';
          catLabel.textContent = cat.category;
          catCard.appendChild(catLabel);

          const catScore = document.createElement('div');
          catScore.className = 'ev-label';
          catScore.textContent = `Score: ${cat.score} | EV after: ${cat.ev_after.toFixed(1)}`;
          catCard.appendChild(catScore);

          col4.appendChild(catCard);
        });
        flowEl.appendChild(col4);
      }

      // Probability fan in panel
      renderProbabilityFan(fanPanel, opt);
    }
  }

  function makeColumn(title, type) {
    const col = document.createElement('div');
    col.className = `widget-column ${type}`;

    const header = document.createElement('div');
    header.className = 'widget-col-header';
    header.textContent = title;
    col.appendChild(header);

    const typeLabel = document.createElement('div');
    typeLabel.className = 'widget-col-type';
    typeLabel.textContent = type === 'decision' ? '(max)' : '(\u03A3 P\u00B7x)';
    col.appendChild(typeLabel);

    return col;
  }

  function makeArrow() {
    const arrow = document.createElement('div');
    arrow.className = 'widget-arrow';
    arrow.textContent = '\u2192';
    return arrow;
  }

  // Solver auto-play
  solverBtn.addEventListener('click', () => {
    const scenario = scenarios[currentScenario];
    const optimalIdx = scenario.keep_options.findIndex((o) => o.optimal);
    selectedKeep = optimalIdx >= 0 ? optimalIdx : 0;
    phase = 'rolled';
    render();
  });

  // Reset
  resetBtn.addEventListener('click', () => {
    selectedKeep = -1;
    phase = 'keep';
    render();
  });

  // Scenario select
  select.addEventListener('change', () => {
    currentScenario = +select.value;
    selectedKeep = -1;
    phase = 'keep';
    render();
  });

  // Populate select options
  select.innerHTML = '';
  scenarios.forEach((s, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = `Scenario: ${s.title}`;
    select.appendChild(opt);
  });

  render();
}
