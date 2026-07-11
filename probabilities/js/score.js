/**
 * Score-probability module: "how likely was my score?"
 * Set a target score and a risk preference (theta); see the exact score
 * distribution with the target marked, and P(score >= target).
 *
 * Data: /yatzy/data/kde_curves.json (exact forward-DP PMFs per theta, the same
 * file the treatise risk-theta chart uses). Math: /yatzy/shared/score-prob.js.
 * Rendering uses the vendored global d3 (loaded via a <script> in index.html).
 * @module score
 */
import { pAtLeast } from '/yatzy/shared/score-prob.js';

const MAX_SCORE = 374;
const X_DOMAIN = [0, 400];
const Y_DOMAIN = [0, 0.015];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

function fmtTheta(t) {
  const s = t.toFixed(2);
  if (Math.abs(t) < 0.001) return '0.00 (neutral)';
  return (t > 0 ? '+' + s + ' (risk-seeking)' : s + ' (risk-averse)');
}

export async function initScoreTool(root) {
  root.innerHTML = '<div class="score-loading">Loading score distribution…</div>';

  let raw;
  try {
    const resp = await fetch('/yatzy/data/kde_curves.json', { cache: 'no-cache' });
    raw = await resp.json();
  } catch (e) {
    root.innerHTML = '<div class="score-loading">Could not load score distribution data.</div>';
    return;
  }

  // Build curves for theta in [-0.4, 0.4], sorted ascending.
  const curves = raw
    .filter((e) => e.theta >= -0.401 && e.theta <= 0.401)
    .sort((a, b) => a.theta - b.theta)
    .map((e) => ({ theta: e.theta, points: e.score.map((s, i) => ({ x: s, y: e.density[i] })) }));

  const refIdx = Math.max(0, curves.findIndex((c) => Math.abs(c.theta) < 0.001));

  const state = { target: 200, idx: refIdx };

  // ── DOM ──
  root.innerHTML = '';

  const controls = document.createElement('div');
  controls.className = 'score-controls';
  root.appendChild(controls);

  // Score input
  const scoreWrap = document.createElement('label');
  scoreWrap.className = 'score-field';
  scoreWrap.innerHTML = '<span>Target score</span>';
  const scoreInput = document.createElement('input');
  scoreInput.type = 'number';
  scoreInput.min = '0';
  scoreInput.max = String(MAX_SCORE);
  scoreInput.step = '1';
  scoreInput.value = String(state.target);
  scoreWrap.appendChild(scoreInput);
  controls.appendChild(scoreWrap);

  // Theta slider
  const thetaWrap = document.createElement('label');
  thetaWrap.className = 'score-field score-theta';
  const thetaHead = document.createElement('span');
  const thetaLabel = document.createElement('span');
  thetaLabel.innerHTML = 'Risk preference &theta; = ';
  const thetaValEl = document.createElement('b');
  thetaValEl.className = 'score-theta-val';
  thetaHead.appendChild(thetaLabel);
  thetaHead.appendChild(thetaValEl);
  thetaWrap.appendChild(thetaHead);
  const thetaSlider = document.createElement('input');
  thetaSlider.type = 'range';
  thetaSlider.min = '0';
  thetaSlider.max = String(curves.length - 1);
  thetaSlider.step = '1';
  thetaSlider.value = String(state.idx);
  thetaWrap.appendChild(thetaSlider);
  const thetaHint = document.createElement('span');
  thetaHint.className = 'score-theta-hint';
  thetaHint.textContent = 'risk-averse ← → risk-seeking';
  thetaWrap.appendChild(thetaHint);
  controls.appendChild(thetaWrap);

  // Chart
  const chartEl = document.createElement('div');
  chartEl.className = 'score-chart';
  root.appendChild(chartEl);

  // Readout
  const readout = document.createElement('div');
  readout.className = 'score-readout';
  root.appendChild(readout);

  // ── Render ──
  // currentCur is the active theta curve; chartRefs holds the mutable chart
  // elements so a target drag can update in place without rebuilding the SVG
  // (a rebuild mid-gesture would destroy the element being dragged).
  let currentCur = curves[state.idx];
  let chartRefs = null;

  /** Percentage with precision that scales to the magnitude, so rare targets
   *  (e.g. 340+) show real digits instead of collapsing to "0.0%". */
  function fmtPct(p) {
    const pc = p * 100;
    if (pc <= 0) return '0%';
    if (pc >= 10) return pc.toFixed(1) + '%';
    if (pc >= 1) return pc.toFixed(2) + '%';
    return pc.toPrecision(2) + '%'; // 0.030%, 0.0012%, 3.0e-7% for the deep tail
  }

  /** ~1 in N games needed to reach the target on average at this theta. */
  function oneInGames(p) {
    if (p <= 0) return 'unreachable';
    const n = 1 / p;
    const nStr = n < 10 ? n.toFixed(1) : Math.round(n).toLocaleString();
    return `~1 in ${nStr} games`;
  }

  function bestThetaFor(target) {
    let best = curves[0], bestP = -1;
    for (const c of curves) {
      const cp = pAtLeast(c.points, target);
      if (cp > bestP) { bestP = cp; best = c; }
    }
    return { best, bestP };
  }

  function updateReadout() {
    const p = pAtLeast(currentCur.points, state.target);
    const { best, bestP } = bestThetaFor(state.target);
    const atBest = Math.abs(best.theta - currentCur.theta) < 1e-9;
    readout.innerHTML =
      `<span class="score-readout-label">P(score ≥ ${state.target})</span>` +
      `<span class="score-readout-value">${fmtPct(p)}</span>` +
      `<span class="score-readout-onein">${oneInGames(p)}</span>` +
      `<span class="score-readout-sub">at θ = ${currentCur.theta.toFixed(2)}</span>` +
      `<span class="score-readout-best">` +
        (atBest
          ? `✓ this θ maximizes the chance`
          : `best at θ = ${best.theta > 0 ? '+' : ''}${best.theta.toFixed(2)} (${fmtPct(bestP)})`) +
      `</span>`;
  }

  /** Move the target line, handle, tail shade, and label in place. */
  function updateTargetVisuals() {
    if (!chartRefs) return;
    const { x, iw, targetLine, targetLabel, handle, tailRect } = chartRefs;
    const tx = x(Math.min(state.target, X_DOMAIN[1]));
    targetLine.attr('x1', tx).attr('x2', tx);
    targetLabel.attr('x', tx + 5).text(`target = ${state.target}`);
    handle.attr('cx', tx);
    tailRect.attr('x', tx).attr('width', Math.max(0, iw - tx));
    updateReadout();
  }

  function render() {
    currentCur = curves[state.idx];
    thetaValEl.textContent = fmtTheta(currentCur.theta);
    renderChart(currentCur);
    updateReadout();
  }

  function renderChart(cur) {
    chartEl.innerHTML = '';
    const width = Math.max(280, chartEl.clientWidth || 640);
    const height = Math.round(width * 0.44);
    const m = { top: 16, right: 14, bottom: 34, left: 40 };
    const iw = width - m.left - m.right;
    const ih = height - m.top - m.bottom;

    const accent = cssVar('--accent');
    const text = cssVar('--text');
    const muted = cssVar('--text-muted');
    const border = cssVar('--border');

    const svg = d3.select(chartEl).append('svg')
      .attr('width', '100%')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain(X_DOMAIN).range([0, iw]);
    const y = d3.scaleLinear().domain(Y_DOMAIN).range([ih, 0]);

    // Grid
    g.append('g').selectAll('line').data(y.ticks(5)).join('line')
      .attr('x1', 0).attr('x2', iw)
      .attr('y1', (d) => y(d)).attr('y2', (d) => y(d))
      .attr('stroke', border).attr('stroke-dasharray', '2,3');

    const area = d3.area().x((d) => x(d.x)).y0(ih).y1((d) => y(d.y)).curve(d3.curveBasis);
    const line = d3.line().x((d) => x(d.x)).y((d) => y(d.y)).curve(d3.curveBasis);

    // Full area (faint)
    g.append('path').datum(cur.points).attr('d', area).attr('fill', accent).attr('opacity', 0.12);

    // Shaded tail (score >= target) via clip
    const clipId = 'score-tail-clip';
    const tailRect = svg.append('defs').append('clipPath').attr('id', clipId)
      .append('rect')
      .attr('x', x(state.target)).attr('y', 0)
      .attr('width', Math.max(0, iw - x(state.target))).attr('height', ih);
    g.append('path').datum(cur.points).attr('d', area)
      .attr('fill', accent).attr('opacity', 0.32)
      .attr('clip-path', `url(#${clipId})`);

    // Curve line
    g.append('path').datum(cur.points).attr('d', line)
      .attr('fill', 'none').attr('stroke', accent).attr('stroke-width', 2.5);

    // Target vertical line + draggable handle
    const tx = x(Math.min(state.target, X_DOMAIN[1]));
    const targetLine = g.append('line').attr('x1', tx).attr('x2', tx).attr('y1', 0).attr('y2', ih)
      .attr('stroke', text).attr('stroke-width', 2);
    const targetLabel = g.append('text').attr('x', tx + 5).attr('y', 12)
      .attr('fill', text).style('font-size', '11px').text(`target = ${state.target}`);
    const handle = g.append('circle').attr('cx', tx).attr('cy', 0).attr('r', 6)
      .attr('fill', accent).attr('stroke', text).attr('stroke-width', 1).style('cursor', 'ew-resize');

    // Axes
    const xAxis = g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6));
    xAxis.selectAll('text').attr('fill', muted).style('font-size', '10px');
    xAxis.selectAll('line').attr('stroke', border);
    xAxis.select('.domain').attr('stroke', border);
    g.append('text').attr('x', iw / 2).attr('y', ih + 30)
      .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '11px').text('Score');

    const yAxis = g.append('g').call(d3.axisLeft(y).ticks(0).tickSize(0));
    yAxis.select('.domain').attr('stroke', border);
    g.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -ih / 2).attr('y', -28)
      .attr('text-anchor', 'middle').attr('fill', muted).style('font-size', '11px').text('Density');

    // Drag anywhere on the plot to move the target line (desktop + touch).
    const overlay = g.append('rect').attr('width', iw).attr('height', ih)
      .attr('fill', 'none').attr('pointer-events', 'all').style('cursor', 'ew-resize');
    const overlayNode = overlay.node();
    function applyDrag(event) {
      const [mx] = d3.pointer(event, overlayNode);
      let s = Math.round(x.invert(Math.max(0, Math.min(iw, mx))));
      s = Math.max(0, Math.min(MAX_SCORE, s));
      if (s === state.target) return;
      state.target = s;
      scoreInput.value = String(s);
      updateTargetVisuals();
    }
    overlay.call(d3.drag().on('start drag', applyDrag));

    chartRefs = { x, iw, targetLine, targetLabel, handle, tailRect };
  }

  // ── Events ──
  scoreInput.addEventListener('input', () => {
    let v = parseInt(scoreInput.value, 10);
    if (Number.isNaN(v)) return;
    v = Math.max(0, Math.min(MAX_SCORE, v));
    state.target = v;
    updateTargetVisuals();
  });
  scoreInput.addEventListener('blur', () => {
    let v = parseInt(scoreInput.value, 10);
    if (Number.isNaN(v)) v = 0;
    v = Math.max(0, Math.min(MAX_SCORE, v));
    scoreInput.value = String(v);
    state.target = v;
    updateTargetVisuals();
  });
  thetaSlider.addEventListener('input', () => {
    state.idx = parseInt(thetaSlider.value, 10);
    render();
  });

  // Re-render on theme toggle (colors are resolved at draw time) and resize.
  new MutationObserver(() => render()).observe(document.documentElement, {
    attributes: true, attributeFilter: ['class'],
  });
  window.addEventListener('resize', () => render());

  render();
}
