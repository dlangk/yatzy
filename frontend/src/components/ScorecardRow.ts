import type { CategoryState } from '../types.ts';

function sparkGradient(fraction: number, color: string): string {
  const pct = `${(fraction * 100).toFixed(1)}%`;
  return `linear-gradient(to right, ${color} ${pct}, transparent ${pct})`;
}

/** DOM handles and update function returned by {@link createScorecardRow}. */
export interface RowElements {
  tr: HTMLTableRowElement;
  update: (opts: {
    category: CategoryState;
    isOptimal: boolean;
    canScore: boolean;
    mustScore: boolean;
    scoreFraction: number;
    evFraction: number | null;
    totalScore: number;
    showHints: boolean;
    bonusAchieved: boolean;
  }) => void;
}

/**
 * Creates a single scorecard row with name, score input, EV display, and action button.
 *
 * Callbacks: onScore (score current roll), onSetScore (manual override), onUnsetCategory (undo).
 * Returned `update()` refreshes display based on category state, optimal hints,
 * spark-bar gradients, and scoring availability.
 */
export function createScorecardRow(
  cat: CategoryState,
  onScore: () => void,
  onSetScore: (score: number) => void,
  onUnsetCategory: () => void,
): RowElements {
  const tr = document.createElement('tr');

  const nameTd = document.createElement('td');
  nameTd.textContent = cat.name;
  tr.appendChild(nameTd);

  const scoreTd = document.createElement('td');
  scoreTd.style.textAlign = 'center';
  const scoreInput = document.createElement('input');
  scoreInput.type = 'number';
  scoreInput.className = 'score-input';
  scoreInput.addEventListener('change', (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    if (!isNaN(v) && v >= 0) onSetScore(v);
  });
  scoreTd.appendChild(scoreInput);
  tr.appendChild(scoreTd);

  const evTd = document.createElement('td');
  evTd.style.textAlign = 'center';
  evTd.style.fontSize = '12px';
  tr.appendChild(evTd);

  const actionTd = document.createElement('td');
  actionTd.style.textAlign = 'center';
  const actionBtn = document.createElement('button');
  actionBtn.addEventListener('click', () => {
    // re-check current state at click time
    if (actionBtn.dataset.action === 'unset') {
      onUnsetCategory();
    } else if (actionBtn.dataset.action === 'score') {
      onScore();
    }
  });
  actionTd.appendChild(actionBtn);
  tr.appendChild(actionTd);

  function update(opts: {
    category: CategoryState;
    isOptimal: boolean;
    canScore: boolean;
    mustScore: boolean;
    scoreFraction: number;
    evFraction: number | null;
    totalScore: number;
    showHints: boolean;
    bonusAchieved: boolean;
  }) {
    const { category, isOptimal, canScore, mustScore, scoreFraction, evFraction, totalScore, showHints, bonusAchieved } = opts;
    const displayedScore = category.isScored ? category.score : category.suggestedScore;
    const isZero = !category.isScored && displayedScore === 0;
    const dimmed = isZero && !isOptimal;

    // Row background
    if (category.isScored) {
      tr.style.background = 'var(--bg-alt)';
    } else if (isOptimal) {
      tr.style.background = 'rgba(44, 160, 44, 0.12)';
    } else {
      tr.style.background = 'transparent';
    }

    // Name color
    nameTd.style.color = dimmed ? 'var(--text-muted)' : 'inherit';

    // Score cell — upper categories show +/- vs par until bonus achieved
    const isUpper = category.id >= 0 && category.id <= 5;
    if (category.isScored && isUpper && !bonusAchieved) {
      const par = 3 * (category.id + 1);
      const delta = category.score - par;
      scoreInput.type = 'text';
      scoreInput.value = delta > 0 ? `+${delta}` : delta < 0 ? `${delta}` : '±0';
      scoreInput.style.color = delta > 0 ? 'var(--color-success)' : delta < 0 ? 'var(--color-danger)' : 'var(--text-muted)';
    } else {
      scoreInput.type = 'number';
      scoreInput.value = String(displayedScore);
      scoreInput.style.color = dimmed ? 'var(--text-muted)' : 'inherit';
    }

    const scoreBarColor = isOptimal
      ? 'rgba(44, 160, 44, 0.18)'
      : category.isScored
        ? 'rgba(0, 0, 0, 0.06)'
        : 'rgba(59, 76, 192, 0.15)';
    scoreTd.style.background = scoreFraction > 0 ? sparkGradient(scoreFraction, scoreBarColor) : '';

    // EV cell (cumulative expected final score)
    // totalScore here is the raw scored sum (no bonus) — the solver's evIfScored
    // = cat_score + V(successor) already includes the terminal bonus.
    // Only show when we have fresh eval data (canScore implies turnPhase=rolled).
    const hasEv = showHints && canScore && !category.isScored && category.available;
    evTd.textContent = hasEv ? (totalScore + category.evIfScored).toFixed(0) : '';
    evTd.style.color = dimmed ? 'var(--text-muted)' : 'inherit';
    const evBarColor = isOptimal ? 'rgba(44, 160, 44, 0.18)' : 'rgba(59, 76, 192, 0.15)';
    evTd.style.background = hasEv && evFraction != null && evFraction > 0 ? sparkGradient(evFraction, evBarColor) : '';

    // Action button
    const showAction = canScore && !category.isScored && category.available;
    if (category.isScored) {
      actionBtn.textContent = '\u2713';
      actionBtn.dataset.action = 'unset';
      actionBtn.disabled = false;
      actionBtn.className = 'scorecard-action-btn';
      actionBtn.style.border = 'none';
      actionBtn.style.background = 'transparent';
      actionBtn.style.cursor = 'pointer';
      actionBtn.style.visibility = 'visible';
      actionBtn.title = 'Click to un-score';
    } else if (showAction) {
      actionBtn.textContent = 'Score';
      actionBtn.dataset.action = 'score';
      actionBtn.disabled = false;
      if (mustScore) {
        actionBtn.className = 'scorecard-action-btn game-btn-primary';
      } else {
        actionBtn.className = 'scorecard-action-btn';
      }
      actionBtn.style.border = '';
      actionBtn.style.background = '';
      actionBtn.style.cursor = 'pointer';
      actionBtn.style.visibility = 'visible';
      actionBtn.title = '';
    } else {
      actionBtn.textContent = 'Score';
      actionBtn.dataset.action = '';
      actionBtn.disabled = true;
      actionBtn.className = 'scorecard-action-btn';
      actionBtn.style.border = '';
      actionBtn.style.background = '';
      actionBtn.style.cursor = 'default';
      actionBtn.style.visibility = 'hidden';
      actionBtn.title = '';
    }
  }

  return { tr, update };
}
