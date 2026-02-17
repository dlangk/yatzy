import { subscribe, getState } from '../store.js';
import { createScorecardRow } from './scorecard-row.js';
import { UPPER_CATEGORIES, BONUS_THRESHOLD, CATEGORY_COUNT } from '../constants.js';

function normalize(value, min, max) {
  if (max <= min) return value > 0 ? 1 : 0;
  return (value - min) / (max - min);
}

export function initScorecard(container) {
  const table = document.createElement('table');
  table.className = 'scorecard';

  const colgroup = document.createElement('colgroup');
  for (let i = 0; i < 4; i++) {
    colgroup.appendChild(document.createElement('col'));
  }
  table.appendChild(colgroup);

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  ['Category', 'Score', 'EV', ''].forEach((text, i) => {
    const th = document.createElement('th');
    th.textContent = text;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  table.appendChild(tbody);

  // Create upper category rows
  const upperRows = [];
  for (let i = 0; i < UPPER_CATEGORIES; i++) {
    const row = createScorecardRow(i);
    tbody.appendChild(row.el);
    upperRows.push(row);
  }

  // Upper bonus separator row
  const bonusTr = document.createElement('tr');
  bonusTr.className = 'scorecard-separator';
  const bonusNameTd = document.createElement('td');
  bonusNameTd.className = 'scorecard-cell';
  const bonusValueTd = document.createElement('td');
  bonusValueTd.className = 'scorecard-cell';
  bonusValueTd.style.textAlign = 'center';
  const bonusEmptyTd = document.createElement('td');
  bonusEmptyTd.colSpan = 2;
  bonusEmptyTd.className = 'scorecard-cell';
  bonusTr.append(bonusNameTd, bonusValueTd, bonusEmptyTd);
  tbody.appendChild(bonusTr);

  // Create lower category rows
  const lowerRows = [];
  for (let i = UPPER_CATEGORIES; i < CATEGORY_COUNT; i++) {
    const row = createScorecardRow(i);
    tbody.appendChild(row.el);
    lowerRows.push(row);
  }

  // Total row
  const totalTr = document.createElement('tr');
  totalTr.className = 'scorecard-total';
  const totalNameTd = document.createElement('td');
  totalNameTd.textContent = 'Total';
  const totalValueTd = document.createElement('td');
  totalValueTd.style.textAlign = 'center';
  const totalEmptyTd = document.createElement('td');
  totalEmptyTd.colSpan = 2;
  totalTr.append(totalNameTd, totalValueTd, totalEmptyTd);
  tbody.appendChild(totalTr);

  container.appendChild(table);

  const allRows = [...upperRows, ...lowerRows];

  function render(state) {
    const canScore = state.turnPhase === 'rolled';
    const optimalCategoryId = state.lastEvalResponse?.optimal_category ?? null;

    // Normalization ranges
    const scoreValues = state.categories.map(c => c.isScored ? c.score : c.suggestedScore);
    const scoreMin = Math.min(...scoreValues);
    const scoreMax = Math.max(...scoreValues);

    const unscoredAvailable = state.categories.filter(c => !c.isScored && c.available);
    const evValues = unscoredAvailable.map(c => c.evIfScored);
    const evMin = evValues.length > 0 ? Math.min(...evValues) : 0;
    const evMax = evValues.length > 0 ? Math.max(...evValues) : 0;

    for (let i = 0; i < CATEGORY_COUNT; i++) {
      const cat = state.categories[i];
      const isOptimal = cat.id === optimalCategoryId && canScore;
      const scoreVal = cat.isScored ? cat.score : cat.suggestedScore;
      const scoreFraction = normalize(scoreVal, scoreMin, scoreMax);
      const evFraction = (!cat.isScored && cat.available) ? normalize(cat.evIfScored, evMin, evMax) : null;
      allRows[i].update(cat, isOptimal, canScore, scoreFraction, evFraction);
    }

    // Upper bonus
    bonusNameTd.textContent = `Upper (${state.upperScore}/${BONUS_THRESHOLD})`;
    bonusValueTd.textContent = state.bonus > 0 ? `+${state.bonus}` : '\u2014';

    // Total
    totalValueTd.textContent = state.totalScore;
  }

  render(getState());
  subscribe((state) => render(state));
}
