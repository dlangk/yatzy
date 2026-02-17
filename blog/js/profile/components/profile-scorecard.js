/**
 * Read-only scorecard for profiling scenarios.
 * Reuses the same CSS classes as the game scorecard (game.css).
 */

const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
];
const UPPER_CATEGORIES = 6;
const BONUS_THRESHOLD = 63;

// Typical lower-category scores for display (cosmetic only, not used in estimation)
const TYPICAL_LOWER = {
  6: 10,   // One Pair
  7: 16,   // Two Pairs
  8: 15,   // Three of a Kind
  9: 20,   // Four of a Kind
  10: 15,  // Small Straight
  11: 20,  // Large Straight
  12: 24,  // Full House
  13: 26,  // Chance
  14: 50,  // Yatzy
};

/**
 * Generate plausible fake scores for scored categories.
 * Upper categories: distribute upper_score proportionally by face value.
 * Lower categories: use typical values.
 * These are cosmetic — they don't affect parameter estimation.
 */
function fakeScores(scored, upperScore) {
  const scores = new Array(15).fill(null);

  // Upper: distribute proportionally by face value (1-6)
  let weightSum = 0;
  for (let i = 0; i < UPPER_CATEGORIES; i++) {
    if (scored & (1 << i)) weightSum += (i + 1);
  }
  if (weightSum > 0) {
    let assigned = 0;
    const scoredUpper = [];
    for (let i = 0; i < UPPER_CATEGORIES; i++) {
      if (scored & (1 << i)) scoredUpper.push(i);
    }
    for (let j = 0; j < scoredUpper.length; j++) {
      const i = scoredUpper[j];
      if (j === scoredUpper.length - 1) {
        // Last one gets the remainder to ensure exact sum
        scores[i] = upperScore - assigned;
      } else {
        const val = Math.round(upperScore * (i + 1) / weightSum);
        scores[i] = val;
        assigned += val;
      }
    }
  }

  // Lower: typical values
  for (let i = UPPER_CATEGORIES; i < 15; i++) {
    if (scored & (1 << i)) scores[i] = TYPICAL_LOWER[i];
  }

  return scores;
}

/**
 * Create a read-only scorecard element.
 * Call update(scenario) to refresh it for each new scenario.
 */
export function createProfileScorecard() {
  const table = document.createElement('table');
  table.className = 'scorecard';

  const colgroup = document.createElement('colgroup');
  const col1 = document.createElement('col');
  col1.style.width = '70%';
  const col2 = document.createElement('col');
  col2.style.width = '30%';
  colgroup.append(col1, col2);
  table.appendChild(colgroup);

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  ['Category', ''].forEach(text => {
    const th = document.createElement('th');
    th.textContent = text;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  table.appendChild(tbody);

  // Build rows for all 15 categories
  const rows = [];
  let bonusRow;

  for (let i = 0; i < 15; i++) {
    const tr = document.createElement('tr');
    const nameCell = document.createElement('td');
    nameCell.className = 'scorecard-cell';
    nameCell.textContent = CATEGORY_NAMES[i];
    const statusCell = document.createElement('td');
    statusCell.className = 'scorecard-cell profile-scorecard-status';
    tr.append(nameCell, statusCell);
    tbody.appendChild(tr);
    rows.push({ tr, nameCell, statusCell });

    // Bonus separator after last upper category
    if (i === UPPER_CATEGORIES - 1) {
      const sep = document.createElement('tr');
      sep.className = 'scorecard-separator';
      const sepName = document.createElement('td');
      sepName.className = 'scorecard-cell';
      const sepVal = document.createElement('td');
      sepVal.className = 'scorecard-cell profile-scorecard-status';
      sep.append(sepName, sepVal);
      tbody.appendChild(sep);
      bonusRow = { sepName, sepVal };
    }
  }

  function update(scenario) {
    const scored = scenario.scored_categories;
    const upperScore = scenario.upper_score;
    const scores = fakeScores(scored, upperScore);

    for (let i = 0; i < 15; i++) {
      const isScored = (scored & (1 << i)) !== 0;
      const row = rows[i];

      row.tr.className = '';
      if (isScored) {
        row.tr.classList.add('scorecard-row--scored');
        row.statusCell.textContent = scores[i] != null ? scores[i] : '\u2713';
        row.statusCell.className = 'scorecard-cell profile-scorecard-status profile-scorecard-check';
      } else {
        row.statusCell.textContent = '';
        row.statusCell.className = 'scorecard-cell profile-scorecard-status';
      }

      // Highlight actionable categories for category decisions
      if (scenario.decision_type === 'category' && !isScored) {
        const isAction = scenario.actions.some(a => a.id === i);
        if (isAction) {
          row.statusCell.textContent = '?';
          row.statusCell.className = 'scorecard-cell profile-scorecard-status profile-scorecard-action';
        }
      }
    }

    // Bonus row — just show upper score progress
    if (bonusRow) {
      bonusRow.sepName.textContent = 'Bonus';
      bonusRow.sepVal.textContent = `${upperScore}/${BONUS_THRESHOLD}`;
      bonusRow.sepVal.className = upperScore >= BONUS_THRESHOLD
        ? 'scorecard-cell profile-scorecard-status profile-scorecard-bonus-yes'
        : 'scorecard-cell profile-scorecard-status profile-scorecard-bonus-no';
    }
  }

  return { el: table, update };
}
