import { getState, dispatch, subscribe } from '../store.ts';
import { UPPER_CATEGORIES } from '../constants.ts';
import { createScorecardRow, type RowElements } from './ScorecardRow.ts';

function normalize(value: number, min: number, max: number): number {
  if (max <= min) return value > 0 ? 1 : 0;
  return (value - min) / (max - min);
}

/** Render the 15-category scorecard with score inputs, E[final] column, and action buttons. */
export function initScorecard(container: HTMLElement): void {
  const table = document.createElement('table');
  table.className = 'scorecard';

  // Fixed colgroup
  const colgroup = document.createElement('colgroup');
  for (const cls of ['col-name', 'col-score', 'col-ev', 'col-action']) {
    const col = document.createElement('col');
    col.className = cls;
    colgroup.appendChild(col);
  }
  table.appendChild(colgroup);

  // Header
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  for (const [text, center] of [['Category', false], ['Score', true], ['E[final]', true], ['', true]] as [string, boolean][]) {
    const th = document.createElement('th');
    th.textContent = text;
    if (center) th.className = 'center';
    th.style.textAlign = center ? 'center' : 'left';
    if (text === 'EV') th.style.fontSize = '12px';
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Body
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);

  const s = getState();
  const rows: RowElements[] = [];

  // Upper categories
  for (let i = 0; i < UPPER_CATEGORIES; i++) {
    const cat = s.categories[i];
    const row = createScorecardRow(
      cat,
      () => dispatch({ type: 'SCORE_CATEGORY', categoryId: cat.id }),
      (score) => dispatch({ type: 'SET_CATEGORY_SCORE', categoryId: cat.id, score }),
      () => dispatch({ type: 'UNSET_CATEGORY', categoryId: cat.id }),
    );
    rows.push(row);
    tbody.appendChild(row.tr);
  }

  // Upper Total row
  const upperTotalRow = document.createElement('tr');
  upperTotalRow.className = 'upper-summary';
  const upperTotalNameTd = document.createElement('td');
  upperTotalNameTd.textContent = 'Upper Total';
  const upperTotalScoreTd = document.createElement('td');
  upperTotalScoreTd.style.textAlign = 'center';
  const upperTotalRestTd = document.createElement('td');
  upperTotalRestTd.colSpan = 2;
  upperTotalRow.appendChild(upperTotalNameTd);
  upperTotalRow.appendChild(upperTotalScoreTd);
  upperTotalRow.appendChild(upperTotalRestTd);
  tbody.appendChild(upperTotalRow);

  // Bonus row
  const bonusRow = document.createElement('tr');
  bonusRow.className = 'upper-summary';
  const bonusNameTd = document.createElement('td');
  bonusNameTd.textContent = 'Bonus';
  const bonusScoreTd = document.createElement('td');
  bonusScoreTd.style.textAlign = 'center';
  const bonusRestTd = document.createElement('td');
  bonusRestTd.colSpan = 2;
  bonusRow.appendChild(bonusNameTd);
  bonusRow.appendChild(bonusScoreTd);
  bonusRow.appendChild(bonusRestTd);
  tbody.appendChild(bonusRow);

  // Lower categories
  for (let i = UPPER_CATEGORIES; i < s.categories.length; i++) {
    const cat = s.categories[i];
    const row = createScorecardRow(
      cat,
      () => dispatch({ type: 'SCORE_CATEGORY', categoryId: cat.id }),
      (score) => dispatch({ type: 'SET_CATEGORY_SCORE', categoryId: cat.id, score }),
      () => dispatch({ type: 'UNSET_CATEGORY', categoryId: cat.id }),
    );
    rows.push(row);
    tbody.appendChild(row.tr);
  }

  // Total row
  const totalRow = document.createElement('tr');
  totalRow.className = 'total-row';
  const totalNameTd = document.createElement('td');
  totalNameTd.textContent = 'Total';
  const totalScoreTd = document.createElement('td');
  totalScoreTd.style.textAlign = 'center';
  const totalRestTd = document.createElement('td');
  totalRestTd.colSpan = 2;
  totalRow.appendChild(totalNameTd);
  totalRow.appendChild(totalScoreTd);
  totalRow.appendChild(totalRestTd);
  tbody.appendChild(totalRow);

  container.appendChild(table);

  function render() {
    const s = getState();
    const canScore = s.turnPhase === 'rolled';
    const mustScore = canScore && s.rerollsRemaining <= 0;
    const optimalCategoryId = s.showHints ? (s.lastEvalResponse?.optimal_category ?? null) : null;

    // Compute normalization ranges
    const scoreValues = s.categories.map(c => c.isScored ? c.score : c.suggestedScore);
    const scoreMin = Math.min(...scoreValues);
    const scoreMax = Math.max(...scoreValues);
    // Raw scored sum without bonus — the solver's evIfScored already includes
    // the terminal bonus, so using totalScore (which includes bonus) would double-count.
    const rawScoredSum = s.categories.reduce((sum, c) => c.isScored ? sum + c.score : sum, 0);
    const unscoredAvailable = s.categories.filter(c => !c.isScored && c.available);
    const evCumulativeValues = unscoredAvailable.map(c => rawScoredSum + c.evIfScored);
    const evMin = evCumulativeValues.length > 0 ? Math.min(...evCumulativeValues) : 0;
    const evMax = evCumulativeValues.length > 0 ? Math.max(...evCumulativeValues) : 0;

    for (let i = 0; i < s.categories.length; i++) {
      const cat = s.categories[i];
      const scoreVal = cat.isScored ? cat.score : cat.suggestedScore;
      const scoreFraction = normalize(scoreVal, scoreMin, scoreMax);
      const cumulativeEv = rawScoredSum + cat.evIfScored;
      const evFraction = (canScore && !cat.isScored && cat.available) ? normalize(cumulativeEv, evMin, evMax) : null;

      rows[i].update({
        category: cat,
        isOptimal: cat.id === optimalCategoryId && canScore,
        canScore,
        mustScore,
        scoreFraction,
        evFraction,
        totalScore: rawScoredSum,
        showHints: s.showHints,
        bonusAchieved: s.bonus > 0,
      });
    }

    // Upper Total
    upperTotalScoreTd.textContent = String(s.upperScore);

    // Bonus — show cumulative +/- par until bonus achieved
    if (s.bonus > 0) {
      bonusScoreTd.textContent = `+${s.bonus}`;
      bonusScoreTd.style.color = 'var(--color-success)';
    } else {
      const parDelta = s.categories.slice(0, UPPER_CATEGORIES).reduce(
        (sum, c) => c.isScored ? sum + (c.score - 3 * (c.id + 1)) : sum, 0,
      );
      bonusScoreTd.textContent = parDelta > 0 ? `+${parDelta}` : parDelta < 0 ? `${parDelta}` : '±0';
      bonusScoreTd.style.color = parDelta > 0 ? 'var(--color-success)' : parDelta < 0 ? 'var(--color-danger)' : 'var(--text-muted)';
    }

    // Total
    totalScoreTd.textContent = String(s.totalScore);
  }

  render();
  subscribe(render);
}
