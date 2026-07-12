/**
 * App DOM skeleton.
 *
 * Three grid sections:
 *   1. Right   — Dice (ActionBar + DiceBar + DiceLegend) + Analysis (EvalPanel + TrajectoryChart)
 *   2. Scorecard — full scoring table
 *   3. Log     — DecisionLog
 *
 * CSS Grid places these into columns on wider viewports:
 *   Desktop (>1100px):  Log | Scorecard | Right (dice + analysis stacked)
 *   Tablet  (721–1100): Right full-width, Scorecard + Log below
 *   Narrow  (≤720px):  Single column in DOM order
 */
import { initActionBar } from './components/ActionBar.ts';
import { initAutoPlay } from './autoplay.ts';
import { initDiceBar } from './components/DiceBar.ts';
import { initDiceLegend } from './components/DiceLegend.ts';
import { initEvalPanel } from './components/EvalPanel.ts';
import { initTrajectoryChart } from './components/TrajectoryChart.ts';
import { initScorecard } from './components/Scorecard.ts';
import { initDebugPanel } from './components/DebugPanel.ts';
import { initDecisionLog } from './components/DecisionLog.ts';
import { initHowTo } from './components/HowTo.ts';

/** A short, always-on caption describing what a panel shows. */
function panelCaption(text: string): HTMLDivElement {
  const el = document.createElement('div');
  el.className = 'panel-caption';
  el.textContent = text;
  return el;
}

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

  // Right column: Dice + Analysis stacked in a single grid cell (desktop col 3)
  const rightSection = document.createElement('div');
  rightSection.className = 'app-section-right';
  columns.appendChild(rightSection);

  const diceSection = document.createElement('div');
  rightSection.appendChild(diceSection);

  // How-to intro box, above the action buttons (dismissible, remembered).
  const howToEl = document.createElement('div');
  diceSection.appendChild(howToEl);
  initHowTo(howToEl);

  const actionBarEl = document.createElement('div');
  diceSection.appendChild(actionBarEl);
  initActionBar(actionBarEl);

  const autoplayBarEl = document.createElement('div');
  diceSection.appendChild(autoplayBarEl);
  initAutoPlay(autoplayBarEl);

  const diceBarEl = document.createElement('div');
  diceSection.appendChild(diceBarEl);
  initDiceBar(diceBarEl);

  const diceLegendEl = document.createElement('div');
  diceSection.appendChild(diceLegendEl);
  initDiceLegend(diceLegendEl);

  diceSection.appendChild(panelCaption('Your five dice. Click a die to keep it; the arrows set values.'));

  // Section 2: Scorecard — desktop col 2, narrow stacks second
  const scorecardSection = document.createElement('div');
  scorecardSection.className = 'app-section-scorecard';
  columns.appendChild(scorecardSection);

  scorecardSection.appendChild(panelCaption(
    'The 15 categories. E[final] is your projected final score for scoring each one now; the best choice is highlighted.',
  ));
  const scorecardEl = document.createElement('div');
  scorecardSection.appendChild(scorecardEl);
  initScorecard(scorecardEl);

  // Analysis — stacked below dice in the right column
  const analysisSection = document.createElement('div');
  rightSection.appendChild(analysisSection);

  analysisSection.appendChild(panelCaption(
    'Live analysis: the best play this turn, and how your whole game is projected to finish.',
  ));
  const evalPanelEl = document.createElement('div');
  analysisSection.appendChild(evalPanelEl);
  initEvalPanel(evalPanelEl);

  analysisSection.appendChild(panelCaption(
    'Your projected final score across the game, with the range it will most likely land in.',
  ));
  const chartEl = document.createElement('div');
  analysisSection.appendChild(chartEl);
  initTrajectoryChart(chartEl);

  // Section 4: Decision Log — desktop col 1, narrow stacks last
  const logSection = document.createElement('div');
  logSection.className = 'app-section-log';
  columns.appendChild(logSection);

  logSection.appendChild(panelCaption(
    'Every decision so far and how many points it cost versus optimal play.',
  ));
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
