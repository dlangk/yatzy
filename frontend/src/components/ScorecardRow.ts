import type { CategoryState } from '../types.ts';
import { COLORS } from '../constants.ts';

function sparkGradient(fraction: number, color: string): string {
  const pct = `${(fraction * 100).toFixed(1)}%`;
  return `linear-gradient(to right, ${color} ${pct}, transparent ${pct})`;
}

export interface RowElements {
  tr: HTMLTableRowElement;
  update: (opts: {
    category: CategoryState;
    isOptimal: boolean;
    canScore: boolean;
    scoreFraction: number;
    evFraction: number | null;
  }) => void;
}

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
  actionBtn.style.fontSize = '12px';
  actionBtn.style.padding = '2px 8px';
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
    scoreFraction: number;
    evFraction: number | null;
  }) {
    const { category, isOptimal, canScore, scoreFraction, evFraction } = opts;
    const displayedScore = category.isScored ? category.score : category.suggestedScore;
    const isZero = !category.isScored && displayedScore === 0;
    const dimmed = isZero && !isOptimal;

    // Row background
    if (category.isScored) {
      tr.style.background = COLORS.bgAlt;
    } else if (isOptimal) {
      tr.style.background = 'rgba(44, 160, 44, 0.12)';
    } else {
      tr.style.background = 'transparent';
    }

    // Name color
    nameTd.style.color = dimmed ? COLORS.textMuted : 'inherit';

    // Score cell
    scoreInput.value = String(displayedScore);
    scoreInput.style.color = dimmed ? COLORS.textMuted : 'inherit';

    const scoreBarColor = isOptimal
      ? 'rgba(44, 160, 44, 0.18)'
      : category.isScored
        ? 'rgba(0, 0, 0, 0.06)'
        : 'rgba(59, 76, 192, 0.15)';
    scoreTd.style.background = scoreFraction > 0 ? sparkGradient(scoreFraction, scoreBarColor) : '';

    // EV cell
    const hasEv = !category.isScored && category.available;
    evTd.textContent = hasEv ? category.evIfScored.toFixed(1) : '';
    evTd.style.color = dimmed ? COLORS.textMuted : 'inherit';
    const evBarColor = isOptimal ? 'rgba(44, 160, 44, 0.18)' : 'rgba(59, 76, 192, 0.15)';
    evTd.style.background = evFraction != null && evFraction > 0 ? sparkGradient(evFraction, evBarColor) : '';

    // Action button
    const showAction = canScore && !category.isScored && category.available;
    if (category.isScored) {
      actionBtn.textContent = '\u2713';
      actionBtn.dataset.action = 'unset';
      actionBtn.disabled = false;
      actionBtn.style.border = 'none';
      actionBtn.style.background = 'transparent';
      actionBtn.style.cursor = 'pointer';
      actionBtn.style.visibility = 'visible';
      actionBtn.title = 'Click to un-score';
    } else if (showAction) {
      actionBtn.textContent = 'Score';
      actionBtn.dataset.action = 'score';
      actionBtn.disabled = false;
      actionBtn.style.border = '';
      actionBtn.style.background = '';
      actionBtn.style.cursor = 'pointer';
      actionBtn.style.visibility = 'visible';
      actionBtn.title = '';
    } else {
      actionBtn.textContent = 'Score';
      actionBtn.dataset.action = '';
      actionBtn.disabled = true;
      actionBtn.style.cursor = 'default';
      actionBtn.style.visibility = 'hidden';
      actionBtn.title = '';
    }
  }

  return { tr, update };
}
