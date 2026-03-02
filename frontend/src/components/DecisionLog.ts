import { getState, subscribe } from '../store.ts';
import type { TrajectoryPoint } from '../types.ts';
import { emitHover, onHover } from '../hoverBus.ts';

const MAX_BAR_WIDTH = 60;

/**
 * Scrollable decision log table showing per-turn reroll/score choices.
 *
 * Append-only: rows are created once per decision and prepended (newest first).
 * Delta bars and header are updated when showHints changes.
 * Full clear only on RESET_GAME (trajectory becomes empty).
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

  // Map trajectory index → row element for external highlight
  const rowMap = new Map<number, HTMLTableRowElement>();
  let currentHighlight: HTMLTableRowElement | null = null;

  // Track rendered state to avoid unnecessary work
  let renderedDecisionCount = 0;
  let prevShowHints: boolean | null = null;
  let prevTrajectoryRef: TrajectoryPoint[] | null = null;

  // Keep references to delta cells for showHints updates
  interface RowData { tr: HTMLTableRowElement; deltaTd: HTMLTableCellElement; d: TrajectoryPoint }
  const renderedRows: RowData[] = []; // in chronological order

  function createRow(d: TrajectoryPoint): RowData {
    const tr = document.createElement('tr');
    tr.dataset.trajIndex = String(d.index);
    rowMap.set(d.index, tr);

    tr.addEventListener('mouseenter', () => {
      emitHover(d.index, 'log');
      tr.classList.add('decision-log-hover');
    });
    tr.addEventListener('mouseleave', () => {
      emitHover(null, 'log');
      tr.classList.remove('decision-log-hover');
    });

    const turnTd = document.createElement('td');
    turnTd.className = 'decision-log-turn';
    turnTd.textContent = String(d.turn);
    tr.appendChild(turnTd);

    const labelTd = document.createElement('td');
    labelTd.className = 'decision-log-label';
    labelTd.textContent = d.label ?? '';
    tr.appendChild(labelTd);

    const deltaTd = document.createElement('td');
    deltaTd.className = 'decision-log-delta';
    tr.appendChild(deltaTd);

    const evTd = document.createElement('td');
    evTd.className = 'decision-log-ev';
    evTd.textContent = d.expectedFinal.toFixed(0);
    tr.appendChild(evTd);

    return { tr, deltaTd, d };
  }

  function updateDeltaCells(showHints: boolean) {
    let maxAbsDelta = 0;
    for (const { d } of renderedRows) {
      if (d.delta != null) maxAbsDelta = Math.max(maxAbsDelta, Math.abs(d.delta));
    }
    if (maxAbsDelta === 0) maxAbsDelta = 1;

    let runningDelta = 0;
    for (const { deltaTd, d } of renderedRows) {
      deltaTd.innerHTML = '';
      if (showHints && d.delta != null) {
        runningDelta += d.delta;
        const bar = document.createElement('span');
        bar.className = 'spark-bar';
        const pct = Math.abs(d.delta) / maxAbsDelta;
        const barWidth = Math.max(2, pct * MAX_BAR_WIDTH);
        if (Math.abs(d.delta) < 0.01) {
          bar.style.background = 'var(--color-success)';
          bar.style.width = `${barWidth}px`;
          bar.style.marginLeft = `${MAX_BAR_WIDTH}px`;
        } else if (d.delta < 0) {
          bar.style.background = 'var(--color-danger)';
          bar.style.width = `${barWidth}px`;
          bar.style.marginLeft = `${MAX_BAR_WIDTH - barWidth}px`;
        } else {
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
    }

    if (showHints && renderedRows.length > 0) {
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

  function render() {
    const s = getState();

    // Skip if nothing relevant changed
    if (s.trajectory === prevTrajectoryRef && s.showHints === prevShowHints) return;

    const decisions = s.trajectory.filter(
      (p: TrajectoryPoint) =>
        (p.event === 'reroll' || p.event === 'score') && p.label != null,
    );

    // Reset: trajectory shrunk (game reset)
    if (decisions.length < renderedDecisionCount) {
      tbody.innerHTML = '';
      rowMap.clear();
      renderedRows.length = 0;
      renderedDecisionCount = 0;
    }

    // Append new decisions (prepend to tbody since display is newest-first)
    if (decisions.length > renderedDecisionCount) {
      const newDecisions = decisions.slice(renderedDecisionCount);
      for (const d of newDecisions) {
        const rowData = createRow(d);
        renderedRows.push(rowData);
        // Prepend: newest row goes first
        tbody.insertBefore(rowData.tr, tbody.firstChild);
      }
      renderedDecisionCount = decisions.length;
    }

    // Update delta cells (needed on new rows or showHints change)
    if (s.showHints !== prevShowHints || decisions.length !== renderedRows.length) {
      updateDeltaCells(s.showHints);
    } else if (decisions.length > 0 && renderedRows.length > 0) {
      // New rows were added — always update deltas for rescaling
      updateDeltaCells(s.showHints);
    }

    prevTrajectoryRef = s.trajectory;
    prevShowHints = s.showHints;
  }

  render();
  subscribe(render);

  // Listen for hover events from other components (e.g. trajectory chart)
  onHover((trajectoryIndex, source) => {
    if (source === 'log') return;
    if (currentHighlight) {
      currentHighlight.classList.remove('decision-log-hover');
      currentHighlight = null;
    }
    if (trajectoryIndex === null) return;
    const row = rowMap.get(trajectoryIndex);
    if (row) {
      row.classList.add('decision-log-hover');
      currentHighlight = row;
    }
  });
}
