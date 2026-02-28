/**
 * App DOM skeleton.
 *
 * Four semantic sections ordered for narrow-screen stacking priority:
 *   1. Dice    — ActionBar + DiceBar + DiceLegend  (most interactive)
 *   2. Scorecard — full scoring table
 *   3. Analysis — EvalPanel + TrajectoryChart
 *   4. Log     — DecisionLog (least critical on mobile)
 *
 * CSS Grid places these into columns on wider viewports:
 *   Desktop (>1100px):  Log | Scorecard | Dice + Analysis (stacked in col 3)
 *   Tablet  (721–1100): Dice full-width, Scorecard + Analysis side-by-side, Log below
 *   Narrow  (≤720px):  Single column in DOM order
 */
import { initActionBar } from './components/ActionBar.ts';
import { initDiceBar } from './components/DiceBar.ts';
import { initDiceLegend } from './components/DiceLegend.ts';
import { initEvalPanel } from './components/EvalPanel.ts';
import { initTrajectoryChart } from './components/TrajectoryChart.ts';
import { initScorecard } from './components/Scorecard.ts';
import { initDebugPanel } from './components/DebugPanel.ts';
import { initDecisionLog } from './components/DecisionLog.ts';

/** Build the app DOM skeleton and initialize all component modules. */
export function initApp(container: HTMLElement): void {
  container.innerHTML = '';
  const app = document.createElement('div');
  app.className = 'app';

  const h1 = document.createElement('h1');
  h1.textContent = 'Delta Yatzy';
  app.appendChild(h1);

  // CSS Grid container — placement handled by grid rules in style.css
  const columns = document.createElement('div');
  columns.className = 'app-columns';
  app.appendChild(columns);

  // Section 1: Dice — desktop col 3 row 1, narrow stacks first
  const diceSection = document.createElement('div');
  diceSection.className = 'app-section-dice';
  columns.appendChild(diceSection);

  const actionBarEl = document.createElement('div');
  diceSection.appendChild(actionBarEl);
  initActionBar(actionBarEl);

  const diceBarEl = document.createElement('div');
  diceSection.appendChild(diceBarEl);
  initDiceBar(diceBarEl);

  const diceLegendEl = document.createElement('div');
  diceSection.appendChild(diceLegendEl);
  initDiceLegend(diceLegendEl);

  // Section 2: Scorecard — desktop col 2, narrow stacks second
  const scorecardSection = document.createElement('div');
  scorecardSection.className = 'app-section-scorecard';
  columns.appendChild(scorecardSection);

  const scorecardEl = document.createElement('div');
  scorecardSection.appendChild(scorecardEl);
  initScorecard(scorecardEl);

  // Section 3: Analysis — desktop col 3 row 2, narrow stacks third
  const analysisSection = document.createElement('div');
  analysisSection.className = 'app-section-analysis';
  columns.appendChild(analysisSection);

  const evalPanelEl = document.createElement('div');
  analysisSection.appendChild(evalPanelEl);
  initEvalPanel(evalPanelEl);

  const chartEl = document.createElement('div');
  analysisSection.appendChild(chartEl);
  initTrajectoryChart(chartEl);

  // Section 4: Decision Log — desktop col 1, narrow stacks last
  const logSection = document.createElement('div');
  logSection.className = 'app-section-log';
  columns.appendChild(logSection);

  const decisionLogEl = document.createElement('div');
  logSection.appendChild(decisionLogEl);
  initDecisionLog(decisionLogEl);

  // Debug below grid
  const debugToggleEl = document.createElement('div');
  debugToggleEl.className = 'debug-toggle';
  app.appendChild(debugToggleEl);

  const debugPanelEl = document.createElement('div');
  app.appendChild(debugPanelEl);

  initDebugPanel(debugToggleEl, debugPanelEl);

  container.appendChild(app);
}
