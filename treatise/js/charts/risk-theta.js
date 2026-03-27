import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

/** Integrate KDE density above/below a score threshold via trapezoidal rule. */
function integrateTail(points, threshold, above) {
  let sum = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const x0 = points[i].x, x1 = points[i + 1].x;
    const y0 = points[i].y, y1 = points[i + 1].y;
    if (above) {
      if (x1 <= threshold) continue;
      const lo = Math.max(x0, threshold);
      const frac = (x1 === x0) ? 1 : (x1 - lo) / (x1 - x0);
      const yLo = y0 + (y1 - y0) * (1 - frac);
      sum += (yLo + y1) / 2 * (x1 - lo);
    } else {
      if (x0 >= threshold) continue;
      const hi = Math.min(x1, threshold);
      const frac = (x1 === x0) ? 1 : (hi - x0) / (x1 - x0);
      const yHi = y0 + (y1 - y0) * frac;
      sum += (y0 + yHi) / 2 * (hi - x0);
    }
  }
  return sum;
}

export async function initRiskTheta() {
  const [kdeCurves, summaryData] = await Promise.all([
    DataLoader.kdeCurves(),
    DataLoader.sweepSummary(),
  ]);

  const container = document.getElementById('chart-risk-theta');
  if (!container) return;

  // Build lookup: theta -> { points: [{x,y}], mean, std, min, max }
  // Filter to ±0.4 range for clean simulation data
  const curves = kdeCurves
    .filter(entry => entry.theta >= -0.401 && entry.theta <= 0.401)
    .map(entry => {
      const points = entry.score.map((s, i) => ({ x: s, y: entry.density[i] }));
      const summary = summaryData.find(d => Math.abs(d.theta - entry.theta) < 0.001);
      return {
        theta: entry.theta,
        points,
        mean: summary ? summary.mean : 0,
        std: summary ? summary.std : 0,
        scoreMin: summary ? summary.min : 0,
        scoreMax: summary ? summary.max : 0,
      };
    });

  const refIdx = curves.findIndex(c => Math.abs(c.theta) < 0.001);

  // Slider
  const controls = container.querySelector('.chart-controls');
  const slider = controls.querySelector('.chart-slider');
  const valueDisplay = controls.querySelector('.slider-value');

  slider.min = 0;
  slider.max = curves.length - 1;
  const defaultIdx = refIdx >= 0 ? refIdx : 0;
  slider.value = defaultIdx;

  // Frontier data: match slider's theta range
  const thetaMin = curves[0].theta;
  const thetaMax = curves[curves.length - 1].theta;
  const frontierData = summaryData
    .filter(d => d.theta >= thetaMin - 0.01 && d.theta <= thetaMax + 0.01)
    .sort((a, b) => a.theta - b.theta);

  // Fixed scales for distribution
  const xDomain = [0, 400];
  const yDomain = [0, 0.015];

  // Precompute reference values
  const thresholds = [
    { score: 200, above: false, label: 'Below 200', bad: true },
    { score: 248.4, above: true, label: 'Above EV (248)', bad: false },
    { score: 310, above: true,  label: 'Above 310', bad: false },
  ];
  const refProbs = thresholds.map(t =>
    integrateTail(curves[refIdx].points, t.score, t.above));
  const refMin = curves[refIdx].scoreMin;
  const refMax = curves[refIdx].scoreMax;

  function render(idx) {
    const cur = curves[idx];
    valueDisplay.textContent = formatTheta(cur.theta);

    renderDistribution(cur, curves, idx);
    renderTailStats(cur);
    renderFrontier(cur);
  }

  // ── Distribution chart (top) ──────────────────────────────────────

  function renderDistribution(cur, allCurves, idx) {
    const chart = createChart('chart-risk-theta-distribution', { aspectRatio: 0.5 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear().domain(xDomain).range([0, width]);
    const y = d3.scaleLinear().domain(yDomain).range([height, 0]);

    // Grid
    g.append('g').attr('class', 'grid')
      .selectAll('line').data(y.ticks(5)).join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,3');

    const line = d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveBasis);

    // Background curves
    for (let i = 0; i < allCurves.length; i++) {
      if (i === idx) continue;
      g.append('path')
        .datum(allCurves[i].points)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', thetaColor(allCurves[i].theta))
        .attr('stroke-width', 1)
        .attr('opacity', 0.15);
    }

    // Selected curve: filled area + bold line
    const color = thetaColor(cur.theta);

    g.append('path')
      .datum(cur.points)
      .attr('d', d3.area().x(d => x(d.x)).y0(height).y1(d => y(d.y)).curve(d3.curveBasis))
      .attr('fill', color)
      .attr('opacity', 0.12);

    g.append('path')
      .datum(cur.points)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2.5);

    // ±1σ lines
    const loSigma = cur.mean - cur.std;
    const hiSigma = cur.mean + cur.std;
    for (const sx of [loSigma, hiSigma]) {
      if (sx > 0 && sx < 400) {
        g.append('line')
          .attr('x1', x(sx)).attr('x2', x(sx))
          .attr('y1', 0).attr('y2', height)
          .attr('stroke', color)
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '3,4')
          .attr('opacity', 0.4);
      }
    }

    // Mean line
    g.append('line')
      .attr('x1', x(cur.mean)).attr('x2', x(cur.mean))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', color)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', x(cur.mean) + 5).attr('y', 14)
      .attr('fill', color)
      .style('font-size', '11px')
      .text(`mean = ${cur.mean.toFixed(1)}`);

    // σ labels
    if (loSigma > 20) {
      g.append('text')
        .attr('x', x(loSigma) + 4).attr('y', height - 6)
        .attr('fill', color)
        .attr('opacity', 0.5)
        .style('font-size', '9px')
        .text('\u22121\u03c3');
    }
    if (hiSigma < 390) {
      g.append('text')
        .attr('x', x(hiSigma) + 4).attr('y', height - 6)
        .attr('fill', color)
        .attr('opacity', 0.5)
        .style('font-size', '9px')
        .text('+1\u03c3');
    }

    // min / max lines
    for (const { val, label, anchor, dx } of [
      { val: cur.scoreMin, label: `min = ${cur.scoreMin}`, anchor: 'start', dx: 4 },
      { val: cur.scoreMax, label: `max = ${cur.scoreMax}`, anchor: 'end', dx: -4 },
    ]) {
      if (val > 0 && val < 400) {
        g.append('line')
          .attr('x1', x(val)).attr('x2', x(val))
          .attr('y1', 0).attr('y2', height)
          .attr('stroke', getMutedColor())
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '2,3')
          .attr('opacity', 0.6);
        g.append('text')
          .attr('x', x(val) + dx).attr('y', 26)
          .attr('text-anchor', anchor)
          .attr('fill', getMutedColor())
          .style('font-size', '9px')
          .text(label);
      }
    }

    // Axes
    drawAxis(g.append('g').attr('transform', `translate(0,${height})`), x, 'bottom', 'Score');
    // Y-axis: label only, no tick values
    const yG = g.append('g').call(d3.axisLeft(y).ticks(0).tickSize(0));
    yG.select('.domain').attr('stroke', getGridColor());
    yG.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -35)
      .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
      .style('font-size', '12px').text('Density');

    // Crosshair tooltip
    const tt = tooltip(container);
    const bisect = d3.bisector(d => d.x).left;

    const overlay = g.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'none').attr('pointer-events', 'all');

    const crosshairLine = g.append('line')
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0);

    overlay
      .on('mousemove', (event) => {
        const [mx] = d3.pointer(event);
        const xVal = x.invert(mx);
        const i = bisect(cur.points, xVal);
        const d = cur.points[Math.min(i, cur.points.length - 1)];
        if (!d) return;
        crosshairLine.attr('x1', x(d.x)).attr('x2', x(d.x)).attr('opacity', 0.6);
        tt.show(
          `<div class="tt-label">Score: ${d.x.toFixed(0)}</div>
           <div>Density: <span class="tt-value">${d.y.toFixed(4)}</span></div>
           <div>Mean: ${cur.mean.toFixed(1)}, Std: ${cur.std.toFixed(1)}</div>`,
          event
        );
      })
      .on('mouseleave', () => {
        crosshairLine.attr('opacity', 0);
        tt.hide();
      });
  }

  // ── Threshold probability stats (middle) ──────────────────────────

  function renderTailStats(cur) {
    const el = document.getElementById('chart-risk-theta-stats');
    if (!el) return;

    const isAtRef = Math.abs(cur.theta) < 0.001;

    function scoreStat(val, refVal, label, higherIsBetter) {
      const delta = val - refVal;
      const isBetter = higherIsBetter ? delta > 0.5 : delta < -0.5;
      const isWorse = higherIsBetter ? delta < -0.5 : delta > 0.5;
      const color = isWorse ? COLORS.riskSeeking : isBetter ? '#2ca02c' : getMutedColor();
      const sign = delta > 0 ? '+' : '';
      return `<div class="tail-stat">
        <div class="tail-stat-value">${val.toFixed(0)}</div>
        <div class="tail-stat-change" style="color:${color}">${sign}${delta.toFixed(0)} vs &theta;=0</div>
        <div class="tail-stat-label">${label}</div>
      </div>`;
    }

    function probStat(idx) {
      const pct = integrateTail(cur.points, thresholds[idx].score, thresholds[idx].above) * 100;
      const refPct = refProbs[idx] * 100;
      const relChange = refPct > 0.01 ? ((pct - refPct) / refPct) * 100 : 0;
      const isWorse = thresholds[idx].bad ? relChange > 0.5 : relChange < -0.5;
      const isBetter = thresholds[idx].bad ? relChange < -0.5 : relChange > 0.5;
      const color = isWorse ? COLORS.riskSeeking : isBetter ? '#2ca02c' : getMutedColor();
      const sign = relChange > 0 ? '+' : '';
      return `<div class="tail-stat">
        <div class="tail-stat-value">${pct.toFixed(1)}%</div>
        <div class="tail-stat-change" style="color:${color}">${sign}${relChange.toFixed(0)}% vs &theta;=0</div>
        <div class="tail-stat-label">${thresholds[idx].label}</div>
      </div>`;
    }

    el.innerHTML =
      scoreStat(cur.scoreMin, refMin, 'Min', true) +
      probStat(0) +
      probStat(1) +
      probStat(2) +
      scoreStat(cur.scoreMax, refMax, 'Max', true);
  }

  // ── Mean-variance frontier (bottom) ───────────────────────────────

  function renderFrontier(cur) {
    const chart = createChart('chart-risk-theta-frontier', { aspectRatio: 0.45 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear()
      .domain([d3.min(frontierData, d => d.std) - 1, d3.max(frontierData, d => d.std) + 1])
      .range([0, width]);
    const y = d3.scaleLinear()
      .domain([d3.min(frontierData, d => d.mean) - 5, d3.max(frontierData, d => d.mean) + 5])
      .range([height, 0]);

    // Grid
    g.append('g').selectAll('line').data(y.ticks(5)).join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

    g.append('g').selectAll('line').data(x.ticks(6)).join('line')
      .attr('x1', d => x(d)).attr('x2', d => x(d))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

    // Path segments
    for (let i = 0; i < frontierData.length - 1; i++) {
      const d0 = frontierData[i], d1 = frontierData[i + 1];
      g.append('line')
        .attr('x1', x(d0.std)).attr('y1', y(d0.mean))
        .attr('x2', x(d1.std)).attr('y2', y(d1.mean))
        .attr('stroke', thetaColor((d0.theta + d1.theta) / 2))
        .attr('stroke-width', 2)
        .attr('stroke-linecap', 'round')
        .attr('opacity', 0.35);
    }

    // All dots (muted)
    g.selectAll('.mv-dot')
      .data(frontierData)
      .join('circle')
      .attr('cx', d => x(d.std)).attr('cy', d => y(d.mean))
      .attr('r', 3)
      .attr('fill', d => thetaColor(d.theta))
      .attr('opacity', 0.3);

    // θ=0 reference marker
    const t0 = frontierData.find(d => Math.abs(d.theta) < 0.001);
    if (t0) {
      g.append('circle')
        .attr('cx', x(t0.std)).attr('cy', y(t0.mean))
        .attr('r', 5)
        .attr('fill', 'none')
        .attr('stroke', getMutedColor())
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '2,2');
      g.append('text')
        .attr('x', x(t0.std) + 8).attr('y', y(t0.mean) - 8)
        .attr('fill', getMutedColor())
        .style('font-size', '10px')
        .text('θ = 0');
    }

    // Current θ highlight
    const curPt = frontierData.find(d => Math.abs(d.theta - cur.theta) < 0.001);
    if (curPt) {
      const color = thetaColor(cur.theta);
      g.append('circle')
        .attr('cx', x(curPt.std)).attr('cy', y(curPt.mean))
        .attr('r', 8)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2);
      g.append('circle')
        .attr('cx', x(curPt.std)).attr('cy', y(curPt.mean))
        .attr('r', 4)
        .attr('fill', color)
        .attr('stroke', 'white')
        .attr('stroke-width', 1.5);
      g.append('text')
        .attr('x', x(curPt.std) + 12).attr('y', y(curPt.mean) + 4)
        .attr('fill', color)
        .style('font-size', '11px')
        .style('font-weight', '600')
        .text(`θ = ${formatTheta(cur.theta)}`);
    }

    // Branch labels
    const ra = frontierData.find(d => Math.abs(d.theta + 0.3) < 0.01) || frontierData[0];
    g.append('text')
      .attr('x', x(ra.std) - 8).attr('y', y(ra.mean) + 16)
      .attr('text-anchor', 'end')
      .attr('fill', COLORS.riskAverse)
      .style('font-size', '10px')
      .text('risk-averse \u2190');

    const rs = frontierData.find(d => Math.abs(d.theta - 0.3) < 0.01) || frontierData[frontierData.length - 1];
    g.append('text')
      .attr('x', x(rs.std) + 8).attr('y', y(rs.mean) + 16)
      .attr('fill', COLORS.riskSeeking)
      .style('font-size', '10px')
      .text('\u2192 risk-seeking');

    // Axes
    drawAxis(
      g.append('g').attr('transform', `translate(0,${height})`),
      x, 'bottom', 'Standard deviation (risk)'
    );
    drawAxis(g, y, 'left', 'Mean score (return)');

    // Tooltip on hover
    const tt = tooltip(container);
    g.selectAll('.mv-hover')
      .data(frontierData)
      .join('circle')
      .attr('cx', d => x(d.std)).attr('cy', d => y(d.mean))
      .attr('r', 10)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .attr('cursor', 'pointer')
      .on('mousemove', (event, d) => {
        tt.show(
          `<div class="tt-label">\u03b8 = ${formatTheta(d.theta)}</div>
           <div>Mean: <span class="tt-value">${d.mean.toFixed(1)}</span></div>
           <div>Std dev: <span class="tt-value">${d.std.toFixed(1)}</span></div>
           <div>p5: ${d.p5}, p50: ${d.p50}, p95: ${d.p95}</div>`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());
  }

  render(+slider.value);
  slider.addEventListener('input', () => render(+slider.value));
}
