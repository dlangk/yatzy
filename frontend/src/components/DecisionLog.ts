import { getState, subscribe } from '../store.ts';
import type { TrajectoryPoint } from '../types.ts';

const MAX_BAR_WIDTH = 60;

/**
 * Scrollable decision log table showing per-turn reroll/score choices.
 *
 * Reads trajectory points from store (turn, event, label, delta, expectedFinal)
 * and re-renders the full table on every state change via subscribe().
 */
export function initDecisionLog(container: HTMLElement): void {
  container.className = 'decision-log';

  const header = document.createElement('div');
  header.className = 'decision-log-header';
  const title = document.createElement('span');
  title.textContent = 'Decision Log';
  const totalDelta = document.createElement('span');
  totalDelta.className = 'decision-log-total';
  header.appendChild(title);
  header.appendChild(totalDelta);
  container.appendChild(header);

  const table = document.createElement('table');
  table.className = 'decision-log-table';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const text of ['#', 'Decision', 'Delta', 'E[final]']) {
    const th = document.createElement('th');
    th.textContent = text;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  container.appendChild(table);

  function render() {
    const s = getState();
    tbody.innerHTML = '';

    // Filter to reroll + score entries with labels
    const decisions = s.trajectory.filter(
      (p: TrajectoryPoint) =>
        (p.event === 'reroll' || p.event === 'score') && p.label != null,
    );

    if (decisions.length === 0) {
      totalDelta.textContent = '';
      return;
    }

    // Compute running total delta
    let runningDelta = 0;
    // Find max |delta| for scaling bars
    let maxAbsDelta = 0;
    for (const d of decisions) {
      if (d.delta != null) maxAbsDelta = Math.max(maxAbsDelta, Math.abs(d.delta));
    }
    if (maxAbsDelta === 0) maxAbsDelta = 1;

    for (const d of [...decisions].reverse()) {
      const tr = document.createElement('tr');

      // Turn #
      const turnTd = document.createElement('td');
      turnTd.className = 'decision-log-turn';
      turnTd.textContent = String(d.turn);
      tr.appendChild(turnTd);

      // Decision label
      const labelTd = document.createElement('td');
      labelTd.className = 'decision-log-label';
      labelTd.textContent = d.label ?? '';
      tr.appendChild(labelTd);

      // Delta with spark bar
      const deltaTd = document.createElement('td');
      deltaTd.className = 'decision-log-delta';
      if (s.showHints && d.delta != null) {
        runningDelta += d.delta;
        const bar = document.createElement('span');
        bar.className = 'spark-bar';
        const pct = Math.abs(d.delta) / maxAbsDelta;
        const barWidth = Math.max(2, pct * MAX_BAR_WIDTH);
        if (Math.abs(d.delta) < 0.01) {
          // Optimal
          bar.style.background = 'var(--color-success)';
          bar.style.width = `${barWidth}px`;
          bar.style.marginLeft = `${MAX_BAR_WIDTH}px`;
        } else if (d.delta < 0) {
          // Suboptimal (loss)
          bar.style.background = 'var(--color-danger)';
          bar.style.width = `${barWidth}px`;
          bar.style.marginLeft = `${MAX_BAR_WIDTH - barWidth}px`;
        } else {
          // Shouldn't happen (delta > 0 means better than optimal?)
          bar.style.background = 'var(--color-success)';
          bar.style.width = `${barWidth}px`;
          bar.style.marginLeft = `${MAX_BAR_WIDTH}px`;
        }
        const value = document.createElement('span');
        value.className = 'spark-value';
        value.textContent = d.delta.toFixed(2);
        deltaTd.appendChild(bar);
        deltaTd.appendChild(value);
      } else {
        deltaTd.textContent = '\u2014';
      }
      tr.appendChild(deltaTd);

      // E[final]
      const evTd = document.createElement('td');
      evTd.className = 'decision-log-ev';
      evTd.textContent = d.expectedFinal.toFixed(0);
      tr.appendChild(evTd);

      tbody.appendChild(tr);
    }

    // Update header total
    if (s.showHints) {
      totalDelta.textContent = `\u0394 ${runningDelta.toFixed(2)}`;
      totalDelta.style.color = Math.abs(runningDelta) < 0.01
        ? 'var(--color-success)'
        : runningDelta < 0
          ? 'var(--color-danger)'
          : 'var(--color-success)';
    } else {
      totalDelta.textContent = '';
    }
  }

  render();
  subscribe(render);
}
