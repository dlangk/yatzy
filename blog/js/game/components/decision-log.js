import { subscribe, getState } from '../store.js';

export function initDecisionLog(container) {
  const panel = document.createElement('div');
  panel.className = 'decision-log-panel';

  const header = document.createElement('div');
  header.className = 'decision-log-header';
  const headerTitle = document.createElement('span');
  headerTitle.textContent = 'Decision Log';
  const headerDelta = document.createElement('span');
  headerDelta.className = 'decision-log-total-delta';
  headerDelta.textContent = '';
  header.append(headerTitle, headerDelta);
  panel.appendChild(header);

  const scroll = document.createElement('div');
  scroll.className = 'decision-log-scroll';

  const table = document.createElement('table');
  table.className = 'decision-log-table';
  table.innerHTML = `
    <colgroup>
      <col style="width:28px"><col><col style="width:130px"><col style="width:56px">
    </colgroup>
    <thead><tr>
      <th>#</th><th>Decision</th><th>Delta</th><th>E[final]</th>
    </tr></thead>`;
  const tbody = document.createElement('tbody');
  table.appendChild(tbody);
  scroll.appendChild(table);
  panel.appendChild(scroll);
  container.appendChild(panel);

  // Track how many rows we've rendered so we can append incrementally
  let renderedCount = 0;

  function updateHeaderDelta(entries) {
    const totalDelta = entries.reduce((sum, e) => sum + (e.delta ?? 0), 0);
    if (entries.length === 0) {
      headerDelta.textContent = '';
      headerDelta.className = 'decision-log-total-delta';
      return;
    }
    const sign = totalDelta > 0.005 ? '+' : '';
    headerDelta.textContent = sign + totalDelta.toFixed(1);
    headerDelta.className = 'decision-log-total-delta ' +
      (Math.abs(totalDelta) < 0.05 ? 'dl-val--optimal' : 'dl-val--suboptimal');
  }

  function renderRow(entry) {
    const tr = document.createElement('tr');
    tr.className = 'dl-row' + (entry.type === 'score' || entry.type === 'game_over' ? ' dl-row--score' : '');

    const tdNum = document.createElement('td');
    tdNum.className = 'dl-cell-num';
    tdNum.textContent = entry.turn;

    const tdLabel = document.createElement('td');
    tdLabel.className = 'dl-cell-label';
    tdLabel.textContent = entry.label ?? '';

    const tdDelta = document.createElement('td');
    tdDelta.className = 'dl-cell-delta';
    tdDelta.appendChild(buildSparkBar(entry.delta ?? 0));

    const tdEf = document.createElement('td');
    tdEf.className = 'dl-cell-ef';
    tdEf.textContent = entry.expectedFinal != null ? Math.round(entry.expectedFinal) : '\u2014';

    tr.append(tdNum, tdLabel, tdDelta, tdEf);
    tbody.appendChild(tr);
  }

  function buildSparkBar(delta) {
    const wrap = document.createElement('div');
    wrap.className = 'dl-spark-wrap';

    const isOptimal = Math.abs(delta) < 0.05;
    const clamped = Math.max(-10, Math.min(10, delta));
    const pct = Math.abs(clamped) / 10 * 50;

    const track = document.createElement('div');
    track.className = 'dl-spark-track';

    const center = document.createElement('div');
    center.className = 'dl-spark-center';
    track.appendChild(center);

    if (pct > 0.5) {
      const fill = document.createElement('div');
      fill.className = 'dl-spark-fill';
      if (delta < 0) {
        fill.style.right = '50%';
        fill.style.width = pct + '%';
        fill.style.background = 'var(--color-danger)';
      } else {
        fill.style.left = '50%';
        fill.style.width = pct + '%';
        fill.style.background = 'var(--color-success)';
      }
      track.appendChild(fill);
    }

    const val = document.createElement('span');
    val.className = 'dl-spark-val ' + (isOptimal ? 'dl-val--optimal' : 'dl-val--suboptimal');
    const sign = delta > 0.005 ? '+' : '';
    val.textContent = sign + delta.toFixed(1);

    wrap.append(track, val);
    return wrap;
  }

  function updateEfCell(rowIndex, expectedFinal) {
    const row = tbody.children[rowIndex];
    if (!row) return;
    const efCell = row.querySelector('.dl-cell-ef');
    if (efCell) efCell.textContent = expectedFinal != null ? Math.round(expectedFinal) : '\u2014';
  }

  function render(state) {
    // Filter to decision entries (reroll + score + game_over with non-null delta/label)
    const entries = state.history.filter(e =>
      (e.type === 'reroll' || e.type === 'score' || e.type === 'game_over') && e.label != null
    );

    if (entries.length < renderedCount) {
      // History shrank (reset) — clear and re-render
      tbody.innerHTML = '';
      renderedCount = 0;
    }

    // Append new rows
    for (let i = renderedCount; i < entries.length; i++) {
      renderRow(entries[i]);
    }

    // Update E[final] for all rendered rows (backfill may have changed them)
    for (let i = 0; i < entries.length; i++) {
      updateEfCell(i, entries[i].expectedFinal);
    }

    renderedCount = entries.length;
    updateHeaderDelta(entries);

    if (entries.length > renderedCount - 1) {
      scroll.scrollTop = scroll.scrollHeight;
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
