import {
  createChart, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor,
} from '../yatzy-viz.js';
import { DataLoader } from '../data-loader.js';

export async function initUtilityCurves() {
  const container = document.getElementById('chart-utility-curves');
  if (!container) return;

  // Use the same theta values as the main risk-theta chart (±0.4)
  const kdeCurves = await DataLoader.kdeCurves();
  const thetas = kdeCurves
    .map(e => e.theta)
    .filter(t => t >= -0.401 && t <= 0.401);
  const refIdx = thetas.findIndex(t => Math.abs(t) < 0.001);

  const slider = container.querySelector('.chart-slider');
  const valueDisplay = container.querySelector('.slider-value');
  slider.min = 0;
  slider.max = thetas.length - 1;
  slider.value = refIdx >= 0 ? refIdx : 0;

  const xRange = [0, 374];
  const nPts = 200;
  const step = (xRange[1] - xRange[0]) / (nPts - 1);
  const scores = d3.range(nPts).map(i => xRange[0] + i * step);

  // Precompute normalized marginal value curves for all thetas
  const allCurves = thetas.map(theta => {
    const raw = Math.abs(theta) < 0.001
      ? scores.map(() => 1)
      : scores.map(s => theta * Math.exp(theta * s));
    const rMin = d3.min(raw), rMax = d3.max(raw);
    const span = rMax - rMin;
    const norm = span > 1e-12
      ? raw.map(v => (v - rMin) / span)
      : raw.map(() => 0.5);
    return norm;
  });

  // EV baseline (constant, normalized to 0.5)
  const evNorm = scores.map(() => 0.5);

  function render(idx) {
    const theta = thetas[idx];
    valueDisplay.textContent = formatTheta(theta);

    const chart = createChart('chart-utility-curves-svg', { aspectRatio: 0.4 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear().domain(xRange).range([0, width]);
    const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    // Grid
    g.append('g').selectAll('line').data(y.ticks(4)).join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

    const line = d3.line().curve(d3.curveBasis);

    // Background curves (all other thetas, pale)
    for (let i = 0; i < thetas.length; i++) {
      if (i === idx) continue;
      const path = scores.map((s, j) => [x(s), y(allCurves[i][j])]);
      g.append('path')
        .attr('d', line(path))
        .attr('fill', 'none')
        .attr('stroke', thetaColor(thetas[i]))
        .attr('stroke-width', 1)
        .attr('opacity', 0.15);
    }

    // EV: constant marginal value (flat dashed line)
    const evPath = scores.map((s, i) => [x(s), y(evNorm[i])]);
    g.append('path')
      .attr('d', line(evPath))
      .attr('fill', 'none')
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4');

    g.append('text')
      .attr('x', width - 4).attr('y', y(evNorm[0]) + 14)
      .attr('text-anchor', 'end')
      .attr('fill', getMutedColor())
      .style('font-size', '10px')
      .text('EV: every point equal');

    // Active curve: bold
    const color = thetaColor(theta);
    const activePath = scores.map((s, j) => [x(s), y(allCurves[idx][j])]);
    g.append('path')
      .attr('d', line(activePath))
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2.5);

    // Shade between EV and active
    const area = d3.area()
      .x((d, i) => x(scores[i]))
      .y0((d, i) => y(evNorm[i]))
      .y1((d, i) => y(allCurves[idx][i]))
      .curve(d3.curveBasis);

    g.append('path')
      .datum(scores)
      .attr('d', area)
      .attr('fill', color)
      .attr('opacity', 0.08);

    // Label
    const labelTheta = Math.abs(theta) < 0.001
      ? 'same as EV'
      : `θ·e^(${formatTheta(theta).trim()}x)`;
    g.append('text')
      .attr('x', width / 2).attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', color)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(labelTheta);

    // Axes
    drawAxis(g.append('g').attr('transform', `translate(0,${height})`), x, 'bottom', 'Score');
    drawAxis(g, y, 'left', 'Marginal value');
  }

  render(+slider.value);
  slider.addEventListener('input', () => render(+slider.value));
}
