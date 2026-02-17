/**
 * Question list: clickable sidebar showing all quiz questions with outcomes.
 * Each row shows: question number, decision type icon, and result (optimal / EV gap).
 * Clicking a row dispatches GO_TO to navigate to that question.
 */
import { subscribe, getState, dispatch } from '../store.js';

const TYPE_LABELS = { category: 'Cat', reroll1: 'R1', reroll2: 'R2' };

export function initQuestionList(container) {
  const el = document.createElement('div');
  el.className = 'profile-question-list';
  container.appendChild(el);

  function render(state) {
    if ((state.phase !== 'answering' && state.phase !== 'complete') || !state.scenarios.length) {
      el.style.display = 'none';
      return;
    }
    el.style.display = '';

    const rows = state.scenarios.map((s, i) => {
      const answer = state.answers.find(a => a.scenarioId === s.id);
      const isCurrent = i === state.currentIndex;
      const isOptimal = answer && answer.actionId === s.optimal_action_id;

      let statusText = '';
      let statusClass = 'ql-status-pending';
      if (answer) {
        if (isOptimal) {
          statusText = '\u2713';
          statusClass = 'ql-status-optimal';
        } else {
          statusText = `\u2212${s.gap.toFixed(1)}`;
          statusClass = 'ql-status-gap';
        }
      }

      const typeLabel = TYPE_LABELS[s.decision_type] || '?';

      return `<tr class="ql-row${isCurrent ? ' ql-row-active' : ''}${answer ? ' ql-row-answered' : ''}" data-index="${i}">
        <td class="ql-num">${i + 1}</td>
        <td class="ql-type">${typeLabel}</td>
        <td class="${statusClass}">${statusText}</td>
      </tr>`;
    }).join('');

    el.innerHTML = `<table class="ql-table">
      <thead><tr><th>#</th><th>Type</th><th>Result</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

    // Click handlers
    el.querySelectorAll('.ql-row').forEach(row => {
      row.addEventListener('click', () => {
        const idx = parseInt(row.dataset.index, 10);
        dispatch({ type: 'GO_TO', index: idx });
      });
    });

    // Auto-scroll active row into view
    const activeRow = el.querySelector('.ql-row-active');
    if (activeRow) {
      activeRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  render(getState());
  subscribe((state) => render(state));

  return el;
}
