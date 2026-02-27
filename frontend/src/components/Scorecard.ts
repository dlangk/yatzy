import { getState, dispatch, subscribe } from '../store.ts';
import { UPPER_CATEGORIES, BONUS_THRESHOLD } from '../constants.ts';
import { createScorecardRow, type RowElements } from './ScorecardRow.ts';

function normalize(value: number, min: number, max: number): number {
  if (max <= min) return value > 0 ? 1 : 0;
  return (value - min) / (max - min);
}

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
  for (const [text, center] of [['Category', false], ['Score', true], ['EV', true], ['', true]] as [string, boolean][]) {
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

  // Upper summary row
  const upperRow = document.createElement('tr');
  upperRow.className = 'upper-summary';
  const upperNameTd = document.createElement('td');
  const upperScoreTd = document.createElement('td');
  upperScoreTd.style.textAlign = 'center';
  const upperRestTd = document.createElement('td');
  upperRestTd.colSpan = 2;
  upperRow.appendChild(upperNameTd);
  upperRow.appendChild(upperScoreTd);
  upperRow.appendChild(upperRestTd);
  tbody.appendChild(upperRow);

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
    console.log('Scorecard render:', { turnPhase: s.turnPhase, rerollsRemaining: s.rerollsRemaining, canScore, mustScore });
    const optimalCategoryId = s.lastEvalResponse?.optimal_category ?? null;

    // Compute normalization ranges
    const scoreValues = s.categories.map(c => c.isScored ? c.score : c.suggestedScore);
    const scoreMin = Math.min(...scoreValues);
    const scoreMax = Math.max(...scoreValues);
    const unscoredAvailable = s.categories.filter(c => !c.isScored && c.available);
    const evValues = unscoredAvailable.map(c => c.evIfScored);
    const evMin = evValues.length > 0 ? Math.min(...evValues) : 0;
    const evMax = evValues.length > 0 ? Math.max(...evValues) : 0;

    for (let i = 0; i < s.categories.length; i++) {
      const cat = s.categories[i];
      const scoreVal = cat.isScored ? cat.score : cat.suggestedScore;
      const scoreFraction = normalize(scoreVal, scoreMin, scoreMax);
      const evFraction = (!cat.isScored && cat.available) ? normalize(cat.evIfScored, evMin, evMax) : null;

      rows[i].update({
        category: cat,
        isOptimal: cat.id === optimalCategoryId && canScore,
        canScore,
        mustScore,
        scoreFraction,
        evFraction,
      });
    }

    // Upper summary
    upperNameTd.textContent = `Upper (${s.upperScore}/${BONUS_THRESHOLD})`;
    upperScoreTd.textContent = s.bonus > 0 ? `+${s.bonus}` : '\u2014';

    // Total
    totalScoreTd.textContent = String(s.totalScore);
  }

  render();
  subscribe(render);
}
