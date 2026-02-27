import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMeanVarianceFrontier() {
  const data = await DataLoader.sweepSummary();
  const container = document.getElementById('chart-mean-variance-frontier');
  if (!container) return;

  // Filter to interesting range and sort by theta
  const plotData = data
    .filter(d => d.theta >= -1.0 && d.theta <= 1.0)
    .sort((a, b) => a.theta - b.theta);

  const chart = createChart('chart-mean-variance-frontier-svg', { aspectRatio: 0.6 });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain([d3.min(plotData, d => d.std) - 1, d3.max(plotData, d => d.std) + 1])
    .range([0, width]);

  const y = d3.scaleLinear()
    .domain([d3.min(plotData, d => d.mean) - 5, d3.max(plotData, d => d.mean) + 5])
    .range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  g.append('g').selectAll('line')
    .data(x.ticks(6))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Draw the frontier curve (connected by theta order)
  const line = d3.line()
    .x(d => x(d.std))
    .y(d => y(d.mean))
    .curve(d3.curveMonotoneX);

  // Path with gradient-like coloring via segments
  for (let i = 0; i < plotData.length - 1; i++) {
    const d0 = plotData[i];
    const d1 = plotData[i + 1];
    const avgTheta = (d0.theta + d1.theta) / 2;
    g.append('line')
      .attr('x1', x(d0.std)).attr('y1', y(d0.mean))
      .attr('x2', x(d1.std)).attr('y2', y(d1.mean))
      .attr('stroke', thetaColor(avgTheta))
      .attr('stroke-width', 2.5)
      .attr('stroke-linecap', 'round');
  }

  // Points
  const tt = tooltip(container);

  g.selectAll('.mv-dot')
    .data(plotData)
    .join('circle')
    .attr('cx', d => x(d.std))
    .attr('cy', d => y(d.mean))
    .attr('r', d => Math.abs(d.theta) < 0.001 ? 6 : 4)
    .attr('fill', d => thetaColor(d.theta))
    .attr('stroke', 'white')
    .attr('stroke-width', 1.5)
    .attr('cursor', 'pointer')
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">\u03b8 = ${formatTheta(d.theta)}</div>
         <div>Mean: <span class="tt-value">${d.mean.toFixed(1)}</span></div>
         <div>Std dev: <span class="tt-value">${d.std.toFixed(1)}</span></div>
         <div>Skewness: ${d.skewness.toFixed(3)}</div>
         <div>p5: ${d.p5}, p50: ${d.p50}, p95: ${d.p95}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Label theta=0
  const t0 = plotData.find(d => Math.abs(d.theta) < 0.001);
  if (t0) {
    g.append('text')
      .attr('x', x(t0.std) + 8)
      .attr('y', y(t0.mean) - 8)
      .attr('fill', getTextColor())
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text('\u03b8 = 0 (EV-optimal)');
  }

  // Label extreme risk-averse
  const riskAverse = plotData.find(d => Math.abs(d.theta + 0.3) < 0.01) || plotData[0];
  g.append('text')
    .attr('x', x(riskAverse.std) - 8)
    .attr('y', y(riskAverse.mean) + 16)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.riskAverse)
    .style('font-size', '10px')
    .text('risk-averse \u2190');

  // Label extreme risk-seeking
  const riskSeeking = plotData.find(d => Math.abs(d.theta - 0.3) < 0.01) || plotData[plotData.length - 1];
  g.append('text')
    .attr('x', x(riskSeeking.std) + 8)
    .attr('y', y(riskSeeking.mean) + 16)
    .attr('fill', COLORS.riskSeeking)
    .style('font-size', '10px')
    .text('\u2192 risk-seeking');

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Standard deviation (risk)'
  );
  drawAxis(g, y, 'left', 'Mean score (return)');
}
