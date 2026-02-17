import { dispatch } from '../store.js';

function sparkGradient(fraction, color) {
  const pct = `${(fraction * 100).toFixed(1)}%`;
  return `linear-gradient(to right, ${color} ${pct}, transparent ${pct})`;
}

export function createScorecardRow(catId) {
  const tr = document.createElement('tr');

  const nameCell = document.createElement('td');
  nameCell.className = 'scorecard-cell';

  const scoreCell = document.createElement('td');
  scoreCell.className = 'scorecard-cell';
  scoreCell.style.textAlign = 'center';

  const input = document.createElement('input');
  input.type = 'number';
  input.className = 'score-input';
  input.value = '0';
  scoreCell.appendChild(input);

  input.addEventListener('change', (e) => {
    const v = parseInt(e.target.value, 10);
    if (!isNaN(v) && v >= 0) {
      dispatch({ type: 'SET_CATEGORY_SCORE', categoryId: catId, score: v });
    }
  });

  const evCell = document.createElement('td');
  evCell.className = 'scorecard-cell';
  evCell.style.textAlign = 'center';
  evCell.style.fontSize = '12px';

  const actionCell = document.createElement('td');
  actionCell.className = 'scorecard-cell';
  actionCell.style.textAlign = 'center';

  const actionBtn = document.createElement('button');
  actionBtn.className = 'scorecard-action-btn';
  actionBtn.textContent = 'Score';
  actionBtn.style.visibility = 'hidden';
  actionCell.appendChild(actionBtn);

  actionBtn.addEventListener('click', () => {
    if (actionBtn._isScored) {
      dispatch({ type: 'UNSET_CATEGORY', categoryId: catId });
    } else {
      dispatch({ type: 'SCORE_CATEGORY', categoryId: catId });
    }
  });

  tr.append(nameCell, scoreCell, evCell, actionCell);

  return {
    el: tr,
    update(cat, isOptimal, canScore, scoreFraction, evFraction) {
      nameCell.textContent = cat.name;

      const displayedScore = cat.isScored ? cat.score : cat.suggestedScore;
      const isZero = !cat.isScored && displayedScore === 0;
      const dimmed = isZero && !isOptimal;

      // Row background
      tr.className = '';
      if (cat.isScored) {
        tr.classList.add('scorecard-row--scored');
      } else if (isOptimal) {
        tr.classList.add('scorecard-row--optimal');
      }
      if (dimmed) tr.classList.add('scorecard-row--dimmed');

      // Score input
      input.value = displayedScore;
      input.style.color = dimmed ? 'var(--text-muted)' : 'inherit';

      // Score spark bar
      const scoreBarColor = isOptimal
        ? 'rgba(40, 167, 69, 0.18)'
        : cat.isScored
          ? 'rgba(128, 128, 128, 0.1)'
          : 'rgba(59, 130, 246, 0.15)';
      scoreCell.style.background = scoreFraction > 0 ? sparkGradient(scoreFraction, scoreBarColor) : '';

      // EV
      const hasEv = !cat.isScored && cat.available;
      evCell.textContent = hasEv ? cat.evIfScored.toFixed(1) : '';
      evCell.style.color = dimmed ? 'var(--text-muted)' : 'inherit';

      const evBarColor = isOptimal ? 'rgba(40, 167, 69, 0.18)' : 'rgba(59, 130, 246, 0.15)';
      evCell.style.background = evFraction != null && evFraction > 0 ? sparkGradient(evFraction, evBarColor) : '';

      // Action button
      const showAction = canScore && !cat.isScored && cat.available;
      actionBtn._isScored = cat.isScored;

      if (cat.isScored) {
        actionBtn.textContent = '\u2713';
        actionBtn.className = 'scorecard-action-btn scorecard-action-btn--scored';
        actionBtn.disabled = false;
        actionBtn.style.visibility = 'visible';
        actionBtn.title = 'Click to un-score';
      } else if (showAction) {
        actionBtn.textContent = 'Score';
        actionBtn.className = 'scorecard-action-btn';
        actionBtn.disabled = false;
        actionBtn.style.visibility = 'visible';
        actionBtn.title = '';
      } else {
        actionBtn.textContent = 'Score';
        actionBtn.className = 'scorecard-action-btn';
        actionBtn.disabled = true;
        actionBtn.style.visibility = 'hidden';
        actionBtn.title = '';
      }
    },
  };
}
