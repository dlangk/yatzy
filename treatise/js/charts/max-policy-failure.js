import { DataLoader } from '../data-loader.js';
import {
  createChart, drawAxis, normalPDF,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMaxPolicyFailure() {
  const data = await DataLoader.maxPolicy();
  const container = document.getElementById('chart-max-policy-failure');
  if (!container) return;

  const chart = createChart('chart-max-policy-failure', {
    aspectRatio: 0.5,
    marginLeft: 55,
    marginBottom: 50,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Use density_comparison if present, otherwise generate from summary stats
  const maxMean = data.max_mean || 118.7;
  const maxStd = data.max_std || 35;
  const optMean = data.optimal_mean || 248.4;
  const optStd = data.optimal_std || 42;

  let maxCurve, optCurve;
  if (Array.isArray(data.density_comparison) && data.density_comparison.length > 0) {
    // JSON format: [{score, max_density, optimal_density}, ...]
    maxCurve = data.density_comparison.map(d => ({ x: d.score, y: d.max_density }));
    optCurve = data.density_comparison.map(d => ({ x: d.score, y: d.optimal_density }));
  } else if (data.density_comparison && data.density_comparison.max_policy) {
    maxCurve = data.density_comparison.max_policy;
    optCurve = data.density_comparison.optimal;
  } else {
    // Generate synthetic density curves
    const nPoints = 200;
    maxCurve = [];
    optCurve = [];
    for (let i = 0; i <= nPoints; i++) {
      const x = (i / nPoints) * 400;
      maxCurve.push({ x, y: normalPDF(x, maxMean, maxStd) });
      optCurve.push({ x, y: normalPDF(x, optMean, optStd) });
    }
  }

  const x = d3.scaleLinear().domain([0, 400]).range([0, width]);
  const yMax = d3.max([...maxCurve, ...optCurve], d => d.y) * 1.15;
  const y = d3.scaleLinear().domain([0, yMax]).range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const line = d3.line()
    .x(d => x(d.x))
    .y(d => y(d.y))
    .curve(d3.curveBasis);

  const area = d3.area()
    .x(d => x(d.x))
    .y0(height)
    .y1(d => y(d.y))
    .curve(d3.curveBasis);

  // Max-policy distribution (red)
  const maxColor = COLORS.riskSeeking;
  g.append('path').datum(maxCurve).attr('d', area)
    .attr('fill', maxColor).attr('opacity', 0.12);
  g.append('path').datum(maxCurve).attr('d', line)
    .attr('fill', 'none').attr('stroke', maxColor).attr('stroke-width', 2.5);

  // Optimal distribution (green/orange)
  const optColor = COLORS.optimal;
  g.append('path').datum(optCurve).attr('d', area)
    .attr('fill', optColor).attr('opacity', 0.12);
  g.append('path').datum(optCurve).attr('d', line)
    .attr('fill', 'none').attr('stroke', optColor).attr('stroke-width', 2.5);

  // Vertical mean lines
  g.append('line')
    .attr('x1', x(maxMean)).attr('x2', x(maxMean))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', maxColor).attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '5,3');

  g.append('text')
    .attr('x', x(maxMean) - 5).attr('y', 14)
    .attr('text-anchor', 'end')
    .attr('fill', maxColor)
    .style('font-size', '11px')
    .text(`max-policy: ${maxMean.toFixed(1)}`);

  g.append('line')
    .attr('x1', x(optMean)).attr('x2', x(optMean))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', optColor).attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '5,3');

  g.append('text')
    .attr('x', x(optMean) + 5).attr('y', 14)
    .attr('fill', optColor)
    .style('font-size', '11px')
    .text(`optimal: ${optMean.toFixed(1)}`);

  // Gap annotation
  const gap = optMean - maxMean;
  const midX = (maxMean + optMean) / 2;
  g.append('line')
    .attr('x1', x(maxMean)).attr('x2', x(optMean))
    .attr('y1', y(yMax * 0.85)).attr('y2', y(yMax * 0.85))
    .attr('stroke', getTextColor()).attr('stroke-width', 1.5)
    .attr('marker-start', 'url(#arrow-left)')
    .attr('marker-end', 'url(#arrow-right)');

  g.append('text')
    .attr('x', x(midX))
    .attr('y', y(yMax * 0.85) - 6)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '600')
    .text(`\u2190 ${gap.toFixed(1)}-point gap \u2192`);

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Score'
  );
  drawAxis(g, y, 'left', 'Density');
}
