import { DataLoader } from '../data-loader.js';
import {
  createChart, drawAxis, formatTheta, thetaColor,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initRiskReward() {
  const sweep = await DataLoader.sweepSummary();

  const container = document.getElementById('chart-risk-reward');
  if (!container) return;

  // Filter to ±0.3 range, sorted
  const data = sweep
    .filter(d => d.theta >= -0.301 && d.theta <= 0.301)
    .sort((a, b) => a.theta - b.theta);

  const percentileKeys = [
    { key: 'p5',  label: 'p5',     color: COLORS.percentiles.p5 },
    { key: 'p25', label: 'p25',    color: COLORS.percentiles.p25 },
    { key: 'p50', label: 'median', color: COLORS.percentiles.p50 },
    { key: 'p75', label: 'p75',    color: COLORS.percentiles.p75 },
    { key: 'p95', label: 'p95',    color: COLORS.percentiles.p95 },
    { key: 'p99', label: 'p99',    color: COLORS.percentiles.p99 },
  ];

  // Parabolic interpolation to find the smooth-curve peak between data points.
  // curveMonotoneX can visually peak between samples, especially with uneven spacing.
  function parabolicPeak(key) {
    let bi = 0;
    for (let i = 1; i < data.length; i++) {
      if (data[i][key] > data[bi][key]) bi = i;
    }
    if (bi === 0 || bi === data.length - 1) return data[bi].theta;
    const t0 = data[bi - 1].theta, v0 = data[bi - 1][key];
    const t1 = data[bi].theta,     v1 = data[bi][key];
    const t2 = data[bi + 1].theta, v2 = data[bi + 1][key];
    const A = v0 / ((t0 - t1) * (t0 - t2));
    const B = v1 / ((t1 - t0) * (t1 - t2));
    const C = v2 / ((t2 - t0) * (t2 - t1));
    const denom = 2 * (A + B + C);
    if (Math.abs(denom) < 1e-12) return t1;
    const tStar = (A * (t1 + t2) + B * (t0 + t2) + C * (t0 + t1)) / denom;
    return Math.max(t0, Math.min(t2, tStar));
  }

  // Precompute peak θ for each percentile (where it is maximized)
  const peaks = percentileKeys.map(pk => ({
    ...pk, peakTheta: parabolicPeak(pk.key),
  })).sort((a, b) => a.peakTheta - b.peakTheta);

  // Slider
  const controls = container.querySelector('.chart-controls');
  const slider = controls.querySelector('.chart-slider');
  const valueDisplay = controls.querySelector('.slider-value');

  slider.min = 0;
  slider.max = data.length - 1;
  const defaultIdx = data.findIndex(d => Math.abs(d.theta) < 0.001);
  slider.value = defaultIdx >= 0 ? defaultIdx : Math.floor(data.length / 2);

  function render(idx) {
    const cur = data[idx];
    valueDisplay.textContent = formatTheta(cur.theta);
    renderChart(cur);
    updateStatsPanel(cur);
  }

  function renderChart(cur) {
    const chart = createChart('chart-risk-reward-svg', { aspectRatio: 0.55 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear()
      .domain([data[0].theta, data[data.length - 1].theta])
      .range([0, width]);

    const yMin = d3.min(data, d => d.p5) - 10;
    const yMax = d3.max(data, d => d.p99) + 10;
    const y = d3.scaleLinear().domain([yMin, yMax]).range([height, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(y.ticks(6)).join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,3');

    // p5–p95 band
    const bandArea = d3.area()
      .x(d => x(d.theta))
      .y0(d => y(d.p5))
      .y1(d => y(d.p95))
      .curve(d3.curveMonotoneX);
    g.append('path').datum(data).attr('d', bandArea)
      .attr('fill', getMutedColor()).attr('opacity', 0.08);

    // Percentile lines
    percentileKeys.forEach(pk => {
      const line = d3.line()
        .x(d => x(d.theta))
        .y(d => y(d[pk.key]))
        .curve(d3.curveMonotoneX);
      g.append('path').datum(data).attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', pk.color)
        .attr('stroke-width', pk.key === 'p50' ? 2.5 : 1.8)
        .attr('stroke-dasharray', pk.key === 'p50' ? null : '5,3');
    });

    // Peak lines
    peaks.forEach(pk => {
      const px = x(pk.peakTheta);
      g.append('line')
        .attr('x1', px).attr('x2', px)
        .attr('y1', 0).attr('y2', height)
        .attr('stroke', pk.color)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,4')
        .attr('opacity', 0.35);
    });

    // Zero reference line
    g.append('line')
      .attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.4);

    // Cursor line
    const cx = x(cur.theta);
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', thetaColor(cur.theta))
      .attr('stroke-width', 2);

    // Dots on curves at cursor
    percentileKeys.forEach(pk => {
      g.append('circle')
        .attr('cx', cx).attr('cy', y(cur[pk.key]))
        .attr('r', 3.5)
        .attr('fill', pk.color)
        .attr('stroke', 'white').attr('stroke-width', 1.5);
    });

    // Tooltip box
    const boxW = 116;
    const rowH = 16;
    const padV = 7;
    const padH = 8;
    const boxH = percentileKeys.length * rowH + padV * 2;
    const boxOnRight = cx < boxW + 10;
    const boxX = boxOnRight ? cx + 10 : cx - 10 - boxW;
    const medY = y(cur.p50);
    const boxY = Math.max(0, Math.min(height - boxH, medY - boxH / 2));

    const tipG = g.append('g');
    tipG.append('rect')
      .attr('x', boxX).attr('y', boxY)
      .attr('width', boxW).attr('height', boxH)
      .attr('fill', 'var(--bg)')
      .attr('stroke', 'var(--border)')
      .attr('rx', 4)
      .style('filter', 'drop-shadow(0 1px 4px rgba(0,0,0,0.12))');

    [...percentileKeys].reverse().forEach((pk, i) => {
      const ry = boxY + padV + i * rowH + rowH / 2;
      tipG.append('circle')
        .attr('cx', boxX + padH).attr('cy', ry)
        .attr('r', 3).attr('fill', pk.color);
      tipG.append('text')
        .attr('x', boxX + padH + 8).attr('y', ry + 3.5)
        .attr('fill', getTextColor())
        .style('font-size', '10px')
        .text(pk.label);
      tipG.append('text')
        .attr('x', boxX + boxW - padH).attr('y', ry + 3.5)
        .attr('text-anchor', 'end')
        .attr('fill', pk.color)
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text(cur[pk.key].toFixed(0));
    });

    // Axes
    drawAxis(
      g.append('g').attr('transform', `translate(0,${height})`),
      x, 'bottom', 'Risk parameter \u03b8',
      { tickFormat: d => d === 0 ? '0' : d > 0 ? `+${d}` : `${d}` }
    );
    drawAxis(g, y, 'left', 'Score');
  }

  function updateStatsPanel(cur) {
    const panel = container.querySelector('.chart-stats-panel');
    if (!panel) return;
    panel.innerHTML = `
      <div class="chart-stat">
        <div class="chart-stat-value">${formatTheta(cur.theta)}</div>
        <div class="chart-stat-label">theta</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${cur.mean.toFixed(1)}</div>
        <div class="chart-stat-label">mean</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${cur.std.toFixed(1)}</div>
        <div class="chart-stat-label">std dev</div>
      </div>
    `;
  }

  render(+slider.value);
  slider.addEventListener('input', () => render(+slider.value));
}
