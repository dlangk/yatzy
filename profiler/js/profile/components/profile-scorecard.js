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
const UPPER_SCORE_CAP = 63;

// Par = 3 × face value (3 of each gives 63 total)
const UPPER_PAR = [3, 6, 9, 12, 15, 18];

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
 * Generate valid scores for scored upper categories.
 * Each upper category i scores a multiple of (i+1), range [0, 5*(i+1)].
 * The scores must sum to exactly upperScore.
 *
 * Strategy: assign each category its par value (3×face), then adjust
 * to hit the exact upperScore sum using valid increments/decrements.
 */
function fakeScores(scored, upperScore) {
  const scores = new Array(15).fill(null);

  // Collect which upper categories are scored
  const scoredUpper = [];
  for (let i = 0; i < UPPER_CATEGORIES; i++) {
    if (scored & (1 << i)) scoredUpper.push(i);
  }

  if (scoredUpper.length > 0) {
    // Start at par (3 × face value)
    const vals = scoredUpper.map(i => UPPER_PAR[i]);
    let currentSum = vals.reduce((a, b) => a + b, 0);
    let deficit = upperScore - currentSum;

    // Adjust values to hit exact sum, respecting valid range [0, 5*(i+1)]
    // Iterate in random-ish order to spread adjustments
    let attempts = 0;
    while (deficit !== 0 && attempts < 100) {
      for (let j = 0; j < scoredUpper.length && deficit !== 0; j++) {
        const i = scoredUpper[j];
        const face = i + 1;
        if (deficit > 0 && vals[j] + face <= 5 * face) {
          vals[j] += face;
          deficit -= face;
        } else if (deficit < 0 && vals[j] - face >= 0) {
          vals[j] -= face;
          deficit += face;
        }
      }
      attempts++;
    }

    // If we still can't hit exact sum (rare edge case), clamp
    if (deficit !== 0) {
      // Force last category to absorb remainder (may be slightly off-multiple)
      vals[vals.length - 1] += deficit;
      vals[vals.length - 1] = Math.max(0, vals[vals.length - 1]);
    }

    for (let j = 0; j < scoredUpper.length; j++) {
      scores[scoredUpper[j]] = vals[j];
    }
  }

  // Lower: typical values
  for (let i = UPPER_CATEGORIES; i < 15; i++) {
    if (scored & (1 << i)) scores[i] = TYPICAL_LOWER[i];
  }

  return scores;
}

/**
 * Format a delta value as "+N" or "−N" or "±0".
 */
function fmtDelta(d) {
  if (d > 0) return `+${d}`;
  if (d < 0) return `\u2212${Math.abs(d)}`;
  return '\u00b10';
}

/**
 * Create a read-only scorecard element.
 * Call update(scenario) to refresh it for each new scenario.
 */
export function createProfileScorecard() {
  const table = document.createElement('table');
  table.className = 'scorecard profile-scorecard';

  const colgroup = document.createElement('colgroup');
  const col1 = document.createElement('col');
  col1.style.width = '55%';
  const col2 = document.createElement('col');
  col2.style.width = '25%';
  const col3 = document.createElement('col');
  col3.style.width = '20%';
  colgroup.append(col1, col2, col3);
  table.appendChild(colgroup);

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  ['Category', '', ''].forEach(text => {
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
    const deltaCell = document.createElement('td');
    deltaCell.className = 'scorecard-cell profile-scorecard-delta';
    tr.append(nameCell, statusCell, deltaCell);
    tbody.appendChild(tr);
    rows.push({ tr, nameCell, statusCell, deltaCell });

    // Bonus separator after last upper category
    if (i === UPPER_CATEGORIES - 1) {
      const sep = document.createElement('tr');
      sep.className = 'scorecard-separator';
      const sepName = document.createElement('td');
      sepName.className = 'scorecard-cell';
      const sepVal = document.createElement('td');
      sepVal.className = 'scorecard-cell profile-scorecard-status';
      const sepDelta = document.createElement('td');
      sepDelta.className = 'scorecard-cell profile-scorecard-delta';
      sep.append(sepName, sepVal, sepDelta);
      tbody.appendChild(sep);
      bonusRow = { sepName, sepVal, sepDelta };
    }
  }

  function update(scenario) {
    const scored = scenario.scored_categories;
    const upperScore = scenario.upper_score;
    const scores = fakeScores(scored, upperScore);

    // Compute total par for scored upper categories
    let parSum = 0;
    for (let i = 0; i < UPPER_CATEGORIES; i++) {
      if (scored & (1 << i)) parSum += UPPER_PAR[i];
    }
    const totalDelta = upperScore - parSum;

    for (let i = 0; i < 15; i++) {
      const isScored = (scored & (1 << i)) !== 0;
      const row = rows[i];

      row.tr.className = '';
      row.deltaCell.textContent = '';
      row.deltaCell.className = 'scorecard-cell profile-scorecard-delta';

      if (isScored) {
        row.tr.classList.add('scorecard-row--scored');
        row.statusCell.textContent = scores[i] != null ? scores[i] : '\u2713';
        row.statusCell.className = 'scorecard-cell profile-scorecard-status profile-scorecard-check';

        // Show delta vs par for upper categories
        if (i < UPPER_CATEGORIES && scores[i] != null) {
          const delta = scores[i] - UPPER_PAR[i];
          row.deltaCell.textContent = `(${fmtDelta(delta)})`;
          row.deltaCell.className = 'scorecard-cell profile-scorecard-delta'
            + (delta > 0 ? ' delta-pos' : delta < 0 ? ' delta-neg' : '');
        }
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

    // Bonus row — upper score progress + total delta
    if (bonusRow) {
      bonusRow.sepName.textContent = 'Bonus';
      bonusRow.sepVal.textContent = `${upperScore}/${UPPER_SCORE_CAP}`;
      bonusRow.sepVal.className = upperScore >= UPPER_SCORE_CAP
        ? 'scorecard-cell profile-scorecard-status profile-scorecard-bonus-yes'
        : 'scorecard-cell profile-scorecard-status profile-scorecard-bonus-no';

      // Show total delta vs par for all scored upper cats
      if (parSum > 0) {
        bonusRow.sepDelta.textContent = `(${fmtDelta(totalDelta)})`;
        bonusRow.sepDelta.className = 'scorecard-cell profile-scorecard-delta'
          + (totalDelta > 0 ? ' delta-pos' : totalDelta < 0 ? ' delta-neg' : '');
      } else {
        bonusRow.sepDelta.textContent = '';
      }
    }
  }

  return { el: table, update };
}
