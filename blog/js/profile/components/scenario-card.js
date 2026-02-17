/**
 * Scenario card: renders dice, scorecard, and action buttons for one quiz question.
 * Reroll decisions use clickable dice (toggle keep/reroll) + confirm button.
 * Category decisions use action buttons.
 * Supports navigation between questions and shows previous answers.
 */
import { subscribe, getState, dispatch } from '../store.js';
import { createProfileScorecard } from './profile-scorecard.js';

export function initScenarioCard(container) {
  const el = document.createElement('div');
  el.className = 'profile-scenario-card';
  container.appendChild(el);

  const scorecard = createProfileScorecard();

  // Persistent DOM elements
  const diceRow = document.createElement('div');
  diceRow.className = 'game-dice-row';

  const diceLegend = document.createElement('div');
  diceLegend.className = 'game-dice-legend';
  const legendItems = [
    { bg: 'var(--bg)', border: '1px solid var(--border)', label: 'Held' },
    { bg: 'var(--bg-alt)', border: '1px solid var(--border)', label: 'Reroll' },
  ];
  for (const item of legendItems) {
    const span = document.createElement('span');
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    swatch.style.background = item.bg;
    swatch.style.border = item.border;
    span.appendChild(swatch);
    span.appendChild(document.createTextNode(item.label));
    diceLegend.appendChild(span);
  }

  const actionsEl = document.createElement('div');
  actionsEl.className = 'profile-actions';

  const feedbackEl = document.createElement('div');
  feedbackEl.className = 'profile-feedback';

  const navEl = document.createElement('div');
  navEl.className = 'profile-nav';

  const rerollIndicator = document.createElement('div');
  rerollIndicator.className = 'profile-reroll-indicator';

  // Two-column layout
  const layout = document.createElement('div');
  layout.className = 'profile-scenario-layout';

  const leftCol = document.createElement('div');
  leftCol.className = 'profile-scenario-left';

  const resetBtn = document.createElement('button');
  resetBtn.className = 'game-btn-secondary profile-reset-btn';
  resetBtn.textContent = 'Reset';
  resetBtn.addEventListener('click', () => {
    if (window.confirm('Reset quiz? All answers will be lost.')) {
      dispatch({ type: 'RESET' });
    }
  });

  const rightCol = document.createElement('div');
  rightCol.className = 'profile-scenario-right';
  rightCol.append(rerollIndicator, scorecard.el, resetBtn);

  leftCol.append(diceRow, diceLegend, actionsEl, feedbackEl, navEl);
  layout.append(leftCol, rightCol);
  el.append(layout);

  let advanceTimeout = null;
  let currentScenarioId = null;

  // Reroll dice state
  let held = [true, true, true, true, true];
  let dieButtons = [];
  let isRerollDecision = false;
  let answered = false;

  function buildRerollMask() {
    let mask = 0;
    for (let i = 0; i < 5; i++) {
      if (!held[i]) mask |= (1 << i);
    }
    return mask;
  }

  function updateDieVisuals() {
    dieButtons.forEach((btn, i) => {
      btn.className = 'die--interactive';
      if (held[i]) {
        btn.classList.add('die--held');
        btn.title = 'Held (click to reroll)';
      } else {
        btn.classList.add('die--will-reroll');
        btn.title = 'Will reroll (click to hold)';
      }
    });
  }

  function showOptimalMask(scenario) {
    const optimalMask = scenario.optimal_action_id;
    dieButtons.forEach((btn, i) => {
      if (optimalMask & (1 << i)) {
        btn.classList.add('die--optimal-reroll');
      } else {
        btn.classList.add('die--optimal-keep');
      }
    });
  }

  function applyRerollMask(mask) {
    for (let i = 0; i < 5; i++) {
      held[i] = !(mask & (1 << i));
    }
    updateDieVisuals();
  }

  function render(state) {
    if ((state.phase !== 'answering' && state.phase !== 'complete') || !state.scenarios.length) {
      el.style.visibility = 'hidden';
      return;
    }
    el.style.visibility = 'visible';

    const scenario = state.scenarios[state.currentIndex];
    if (!scenario) return;
    if (scenario.id === currentScenarioId) return;
    currentScenarioId = scenario.id;

    if (advanceTimeout) { clearTimeout(advanceTimeout); advanceTimeout = null; }

    // Check if this scenario was already answered
    const prevAnswer = state.answers.find(a => a.scenarioId === scenario.id);
    answered = !!prevAnswer;

    scorecard.update(scenario);
    isRerollDecision = scenario.decision_type !== 'category';

    // Reroll indicator — bar fills based on remaining rerolls
    const rr = scenario.rerolls_remaining;
    const pct = rr === 2 ? 100 : rr === 1 ? 50 : 0;
    rerollIndicator.innerHTML = `
      <div class="profile-reroll-bar">
        <div class="profile-reroll-bar-fill" style="width:${pct}%"></div>
        <span class="profile-reroll-bar-label">${rr} reroll${rr !== 1 ? 's' : ''} left</span>
      </div>
    `;

    // Dice legend — show for reroll decisions, hide for category
    diceLegend.style.opacity = isRerollDecision ? '1' : '0';

    // Dice
    held = [true, true, true, true, true];
    dieButtons = [];
    diceRow.innerHTML = '';

    const startTime = performance.now();

    scenario.dice.forEach((v, i) => {
      const btn = document.createElement('button');
      btn.className = 'die--interactive die--held';
      btn.textContent = v;
      btn.title = 'Held (click to reroll)';

      if (isRerollDecision && !answered) {
        btn.addEventListener('click', () => {
          if (answered) return;
          held[i] = !held[i];
          updateDieVisuals();
        });
      } else if (!isRerollDecision) {
        btn.disabled = true;
        btn.classList.add('die--faded');
        btn.title = '';
      }

      diceRow.appendChild(btn);
      dieButtons.push(btn);
    });

    // Actions
    actionsEl.innerHTML = '';
    feedbackEl.textContent = '';
    feedbackEl.className = 'profile-feedback';

    if (answered) {
      // Show previous answer state
      if (isRerollDecision) {
        applyRerollMask(prevAnswer.actionId);
        dieButtons.forEach(b => { b.disabled = true; });
        showOptimalMask(scenario);

        const changeBtn = document.createElement('button');
        changeBtn.className = 'game-btn-secondary profile-confirm-btn';
        changeBtn.textContent = 'Change Answer';
        changeBtn.addEventListener('click', () => {
          currentScenarioId = null; // force re-render
          dispatch({ type: 'CLEAR_ANSWER', scenarioId: scenario.id });
        });
        actionsEl.appendChild(changeBtn);
      } else {
        scenario.actions.forEach(a => {
          const btn = document.createElement('button');
          btn.className = 'profile-action-btn';
          btn.textContent = a.label;
          btn.disabled = true;
          if (a.id === prevAnswer.actionId) {
            btn.classList.add('profile-btn-selected');
          }
          if (a.id === scenario.optimal_action_id) {
            btn.classList.add('profile-btn-optimal');
          }
          actionsEl.appendChild(btn);
        });

        const changeBtn = document.createElement('button');
        changeBtn.className = 'game-btn-secondary profile-change-btn';
        changeBtn.textContent = 'Change Answer';
        changeBtn.addEventListener('click', () => {
          currentScenarioId = null;
          dispatch({ type: 'CLEAR_ANSWER', scenarioId: scenario.id });
        });
        actionsEl.appendChild(changeBtn);
      }

      const isOptimal = prevAnswer.actionId === scenario.optimal_action_id;
      if (isOptimal) {
        feedbackEl.textContent = 'Optimal!';
        feedbackEl.className = 'profile-feedback profile-feedback-correct';
      } else {
        feedbackEl.textContent = `EV gap: ${scenario.gap.toFixed(1)} pts`;
        feedbackEl.className = 'profile-feedback profile-feedback-wrong';
      }
    } else if (isRerollDecision) {
      const confirmBtn = document.createElement('button');
      confirmBtn.className = 'game-btn-primary profile-confirm-btn';
      confirmBtn.textContent = 'Confirm Choice';
      confirmBtn.addEventListener('click', () => {
        if (answered) return;
        answered = true;
        const mask = buildRerollMask();

        dispatch({
          type: 'ANSWER',
          answer: {
            scenarioId: scenario.id,
            actionId: mask,
            responseTimeMs: Math.round(performance.now() - startTime),
          },
        });

        const isOptimal = mask === scenario.optimal_action_id;
        confirmBtn.disabled = true;
        dieButtons.forEach(b => { b.disabled = true; });
        showOptimalMask(scenario);

        const matchedAction = scenario.actions.find(a => a.id === mask);
        if (!matchedAction && !isOptimal) {
          showFeedback(false, scenario, true);
        } else {
          showFeedback(isOptimal, scenario, false);
        }
      });
      actionsEl.appendChild(confirmBtn);
    } else {
      scenario.actions.forEach(a => {
        const btn = document.createElement('button');
        btn.className = 'profile-action-btn';
        btn.textContent = a.label;
        btn.dataset.actionId = a.id;
        btn.dataset.optimal = a.id === scenario.optimal_action_id ? 'true' : 'false';
        btn.addEventListener('click', () => {
          if (answered) return;
          answered = true;
          dispatch({
            type: 'ANSWER',
            answer: {
              scenarioId: scenario.id,
              actionId: a.id,
              responseTimeMs: Math.round(performance.now() - startTime),
            },
          });
          showFeedback(btn.dataset.optimal === 'true', scenario);
        });
        actionsEl.appendChild(btn);
      });
    }

    // Navigation
    renderNav(state);
  }

  function renderNav(state) {
    navEl.innerHTML = '';
    const total = state.scenarios.length;
    const idx = state.currentIndex;

    const prevBtn = document.createElement('button');
    prevBtn.className = 'game-btn-secondary profile-nav-btn';
    prevBtn.textContent = '\u2190 Previous';
    prevBtn.disabled = idx === 0;
    prevBtn.addEventListener('click', () => {
      dispatch({ type: 'GO_TO', index: idx - 1 });
    });

    const nextBtn = document.createElement('button');
    nextBtn.className = 'game-btn-secondary profile-nav-btn';
    nextBtn.textContent = 'Next \u2192';
    nextBtn.disabled = idx >= total - 1;
    nextBtn.addEventListener('click', () => {
      dispatch({ type: 'GO_TO', index: idx + 1 });
    });

    const label = document.createElement('span');
    label.className = 'profile-nav-label';
    label.textContent = `${idx + 1} / ${total}`;

    navEl.append(prevBtn, label, nextBtn);
  }

  function showFeedback(isOptimal, scenario, unknownMask = false) {
    if (advanceTimeout) clearTimeout(advanceTimeout);

    // For category decisions, disable buttons and highlight optimal
    if (!isRerollDecision) {
      actionsEl.querySelectorAll('.profile-action-btn').forEach(btn => {
        btn.disabled = true;
        if (btn.dataset.optimal === 'true') {
          btn.classList.add('profile-btn-optimal');
        }
      });
    }

    if (isOptimal) {
      feedbackEl.textContent = 'Optimal!';
      feedbackEl.className = 'profile-feedback profile-feedback-correct';
    } else if (unknownMask) {
      feedbackEl.textContent = 'Not optimal \u2014 see green/red hints';
      feedbackEl.className = 'profile-feedback profile-feedback-wrong';
    } else {
      feedbackEl.textContent = `EV gap: ${scenario.gap.toFixed(1)} pts`;
      feedbackEl.className = 'profile-feedback profile-feedback-wrong';
    }

    // Auto-advance after feedback (skip if quiz already complete — user is reviewing)
    if (getState().phase !== 'complete') {
      advanceTimeout = setTimeout(() => {
        advanceTimeout = null;
        currentScenarioId = null;
        dispatch({ type: 'ADVANCE' });
      }, 1500);
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
