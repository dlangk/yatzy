import { initActionBar } from './components/ActionBar.ts';
import { initDiceBar } from './components/DiceBar.ts';
import { initDiceLegend } from './components/DiceLegend.ts';
import { initEvalPanel } from './components/EvalPanel.ts';
import { initTrajectoryChart } from './components/TrajectoryChart.ts';
import { initScorecard } from './components/Scorecard.ts';
import { initDebugPanel } from './components/DebugPanel.ts';

/** Build the app DOM skeleton and initialize all component modules. */
export function initApp(container: HTMLElement): void {
  container.innerHTML = '';
  const app = document.createElement('div');
  app.className = 'app';

  const h1 = document.createElement('h1');
  h1.textContent = 'Delta Yatzy';
  app.appendChild(h1);

  // Two-column layout
  const columns = document.createElement('div');
  columns.className = 'app-columns';
  app.appendChild(columns);

  // Left column: scorecard
  const leftCol = document.createElement('div');
  leftCol.className = 'app-col-left';
  columns.appendChild(leftCol);

  const scorecardEl = document.createElement('div');
  leftCol.appendChild(scorecardEl);
  initScorecard(scorecardEl);

  // Right column: dice + action bar + graphs
  const rightCol = document.createElement('div');
  rightCol.className = 'app-col-right';
  columns.appendChild(rightCol);

  const actionBarEl = document.createElement('div');
  rightCol.appendChild(actionBarEl);
  initActionBar(actionBarEl);

  const diceBarEl = document.createElement('div');
  rightCol.appendChild(diceBarEl);
  initDiceBar(diceBarEl);

  const diceLegendEl = document.createElement('div');
  rightCol.appendChild(diceLegendEl);
  initDiceLegend(diceLegendEl);

  const evalPanelEl = document.createElement('div');
  rightCol.appendChild(evalPanelEl);
  initEvalPanel(evalPanelEl);

  const chartEl = document.createElement('div');
  rightCol.appendChild(chartEl);
  initTrajectoryChart(chartEl);

  // Debug below both columns
  const debugToggleEl = document.createElement('div');
  debugToggleEl.className = 'debug-toggle';
  app.appendChild(debugToggleEl);

  const debugPanelEl = document.createElement('div');
  app.appendChild(debugPanelEl);

  initDebugPanel(debugToggleEl, debugPanelEl);

  container.appendChild(app);
}
