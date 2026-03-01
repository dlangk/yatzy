import { getState, subscribe } from '../store.ts';
import { computeRerollMask, mapMaskToSorted, unmapMask } from '../mask.ts';

const DASH = '\u2014';

/** Optimal-play final score percentiles (θ=0, 1M games, scores.bin).
 *  Each entry: [score_cutoff, percentile_rank]. Descending by score. */
const OPTIMAL_PCTLS: [number, number][] = [
  [343, 99.99], [336, 99.9],
  [325, 99], [319, 98], [315, 97], [312, 96],
  [309, 95], [307, 94], [305, 93], [303, 92],
  [301, 91], [299, 90],
  [282, 80], [269, 70], [258, 60], [249, 50],
  [239, 40], [230, 30], [219, 20], [203, 10],
  [200, 9], [197, 8], [192, 7], [186, 6],
  [179, 5], [172, 4], [164, 3], [155, 2],
  [145, 1],
];

function optimalPercentile(ev: number): string {
  for (const [cutoff, pct] of OPTIMAL_PCTLS) {
    if (ev >= cutoff) {
      return pct >= 100 ? `p${pct.toFixed(2)}` : `p${pct}`;
    }
  }
  return 'p0';
}

/** Build a label/value grid column with a header. Returns refs to value spans. */
function buildColumn(
  parent: HTMLElement,
  header: string,
  labels: string[],
): HTMLSpanElement[] {
  const col = document.createElement('div');
  col.className = 'eval-col';

  const hdr = document.createElement('div');
  hdr.className = 'eval-col-header';
  hdr.textContent = header;
  col.appendChild(hdr);

  const grid = document.createElement('div');
  grid.className = 'eval-col-grid';
  col.appendChild(grid);

  const vals: HTMLSpanElement[] = [];
  for (const label of labels) {
    const lbl = document.createElement('span');
    lbl.textContent = label;
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = DASH;
    grid.appendChild(lbl);
    grid.appendChild(val);
    vals.push(val);
  }

  parent.appendChild(col);
  return vals;
}

/** Render the two-column evaluation panel: turn-specific + game-level stats. */
export function initEvalPanel(container: HTMLElement): void {
  container.className = 'eval-panel';

  const columns = document.createElement('div');
  columns.className = 'eval-columns';
  container.appendChild(columns);

  // Left column: This Turn
  const turnVals = buildColumn(columns, 'This Turn', [
    'Best keep',
    'Keep EV',
    'Your keep EV',
    'Best score',
    'Score EV',
  ]);

  // Right column: Game
  const gameVals = buildColumn(columns, 'Game', [
    'Expected final',
    '80% Finish range',
    'Current score',
    'Optimal percentile',
    'Turn',
  ]);

  function render() {
    const s = getState();
    const hasData = s.turnPhase === 'rolled' && s.lastEvalResponse !== null;
    const rerolls = s.rerollsRemaining;
    const hints = s.showHints;

    // --- Turn column ---
    const showKeep = hasData && hints && rerolls > 0;
    const showScore = hasData && hints;

    // Best keep (dice values held by optimal mask)
    if (showKeep && s.lastEvalResponse?.optimal_mask != null && s.sortMap) {
      const originalMask = unmapMask(s.lastEvalResponse.optimal_mask, s.sortMap);
      const kept: number[] = [];
      for (let i = 0; i < 5; i++) {
        if (!(originalMask & (1 << i))) {
          kept.push(s.dice[i].value);
        }
      }
      turnVals[0].textContent = kept.length === 5 ? 'All' : kept.length === 0 ? 'None' : kept.join(', ');
    } else {
      turnVals[0].textContent = DASH;
    }

    // Keep EV (optimal mask EV)
    const optMaskEv = s.lastEvalResponse?.optimal_mask_ev ?? null;
    turnVals[1].textContent = showKeep && optMaskEv !== null ? optMaskEv.toFixed(2) : DASH;

    // Your keep EV
    let currentMaskEv: number | null = null;
    if (s.lastEvalResponse?.mask_evs && s.sortMap) {
      const originalMask = computeRerollMask(s.dice.map(d => d.held));
      const sortedMask = mapMaskToSorted(originalMask, s.sortMap);
      currentMaskEv = s.lastEvalResponse.mask_evs[sortedMask] ?? null;
    }
    turnVals[2].textContent = showKeep && currentMaskEv !== null ? currentMaskEv.toFixed(2) : DASH;

    // Best score (category name + score)
    const optCat = s.lastEvalResponse?.optimal_category ?? null;
    const optCatEv = s.lastEvalResponse?.optimal_category_ev ?? null;
    const optCatName = optCat !== null && optCat >= 0
      ? s.categories[optCat]?.name ?? '?'
      : null;
    if (showScore && optCatName) {
      const catInfo = s.lastEvalResponse!.categories.find(c => c.id === optCat);
      const score = catInfo?.score;
      turnVals[3].textContent = score != null ? `${optCatName} (${score})` : optCatName;
    } else {
      turnVals[3].textContent = DASH;
    }

    // Score EV
    turnVals[4].textContent = showScore && optCatEv !== null ? optCatEv.toFixed(2) : DASH;

    // --- Game column ---

    // [0] Expected final score = accumulated + remaining EV
    const stateEv = s.lastEvalResponse?.state_ev ?? null;
    const rawScoredSum = s.categories.reduce((sum, c) => c.isScored ? sum + c.score : sum, 0);
    const latestScored = [...s.trajectory].reverse().find(p => p.event === 'score');
    const expectedFinal = hasData && stateEv !== null
      ? rawScoredSum + stateEv
      : latestScored?.expectedFinal ?? null;
    gameVals[0].textContent = expectedFinal !== null ? String(Math.round(expectedFinal)) : DASH;

    // [1] Finish range: "80% to finish in X–Y" (p10–p90 from latest density)
    const latestWithPct = [...s.trajectory].reverse().find(
      p => p.percentiles && p.event !== 'score' || (p.event === 'score' && p.turn < 15),
    );
    if (latestWithPct?.percentiles) {
      const p10 = latestWithPct.percentiles.p10;
      const p90 = latestWithPct.percentiles.p90;
      if (p10 != null && p90 != null) {
        gameVals[1].textContent = `${Math.round(p10)}–${Math.round(p90)}`;
      } else {
        gameVals[1].textContent = DASH;
      }
    } else {
      gameVals[1].textContent = DASH;
    }

    // [2] Current score
    gameVals[2].textContent = String(s.totalScore);

    // [3] Optimal percentile
    if (expectedFinal !== null) {
      gameVals[3].textContent = optimalPercentile(expectedFinal);
    } else {
      gameVals[3].textContent = DASH;
    }

    // [4] Turn X / 15
    const scoredCount = s.categories.filter(c => c.isScored).length;
    gameVals[4].textContent = `${scoredCount} / 15`;
  }

  render();
  subscribe(render);
}
