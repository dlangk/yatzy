import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initAdaptiveWinrate() {
  const data = await DataLoader.winrate();
  const container = document.getElementById('chart-adaptive-winrate');
  if (!container) return;

  // Include all rows, fill in theta=0 self-play draw_rate
  const plotData = data
    .filter(d => d.win_rate !== null && d.win_rate !== undefined)
    .map(d => ({ ...d, draw_rate: d.draw_rate ?? 0, loss_rate: d.loss_rate ?? 0.5 }))
    .sort((a, b) => a.theta - b.theta);

  const chart = createChart('chart-adaptive-winrate-svg', { aspectRatio: 0.5 });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain([d3.min(plotData, d => d.theta), d3.max(plotData, d => d.theta)])
    .range([0, width]);

  const y = d3.scaleLinear()
    .domain([0.3, 0.55])
    .range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // 50% reference line
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(0.5)).attr('y2', y(0.5))
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '6,3');

  g.append('text')
    .attr('x', width - 4).attr('y', y(0.5) - 6)
    .attr('text-anchor', 'end')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text('50% (break even)');

  // Shade advantage/disadvantage
  const areaAbove = d3.area()
    .x(d => x(d.theta))
    .y0(y(0.5))
    .y1(d => d.win_rate >= 0.5 ? y(d.win_rate) : y(0.5))
    .curve(d3.curveMonotoneX);

  const areaBelow = d3.area()
    .x(d => x(d.theta))
    .y0(y(0.5))
    .y1(d => d.win_rate < 0.5 ? y(d.win_rate) : y(0.5))
    .curve(d3.curveMonotoneX);

  g.append('path')
    .datum(plotData)
    .attr('d', areaAbove)
    .attr('fill', COLORS.riskAverse)
    .attr('opacity', 0.08);

  g.append('path')
    .datum(plotData)
    .attr('d', areaBelow)
    .attr('fill', COLORS.riskSeeking)
    .attr('opacity', 0.08);

  // Win rate line with gradient coloring via segments
  for (let i = 0; i < plotData.length - 1; i++) {
    const d0 = plotData[i];
    const d1 = plotData[i + 1];
    const avgTheta = (d0.theta + d1.theta) / 2;

    g.append('line')
      .attr('x1', x(d0.theta)).attr('y1', y(d0.win_rate))
      .attr('x2', x(d1.theta)).attr('y2', y(d1.win_rate))
      .attr('stroke', thetaColor(avgTheta))
      .attr('stroke-width', 2.5)
      .attr('stroke-linecap', 'round');
  }

  // Points
  const tt = tooltip(container);

  g.selectAll('.wr-dot')
    .data(plotData)
    .join('circle')
    .attr('cx', d => x(d.theta))
    .attr('cy', d => y(d.win_rate))
    .attr('r', 4)
    .attr('fill', d => thetaColor(d.theta))
    .attr('stroke', 'white')
    .attr('stroke-width', 1.5)
    .attr('cursor', 'pointer')
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">\u03b8 = ${formatTheta(d.theta)}</div>
         <div>Win rate: <span class="tt-value">${(d.win_rate * 100).toFixed(1)}%</span></div>
         <div>Draw rate: ${(d.draw_rate * 100).toFixed(2)}%</div>
         <div>Mean score: ${d.mean.toFixed(1)}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Annotation at peak
  const peak = plotData.reduce((a, b) => b.win_rate > a.win_rate ? b : a);
  if (Math.abs(peak.theta) < 0.02) {
    g.append('text')
      .attr('x', x(peak.theta))
      .attr('y', y(peak.win_rate) - 12)
      .attr('text-anchor', 'middle')
      .attr('fill', getTextColor())
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(`Peak: ${(peak.win_rate * 100).toFixed(1)}%`);
  }

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Risk parameter \u03b8 (opponent plays \u03b8 = 0)',
    { tickFormat: d => d === 0 ? '0' : d > 0 ? `+${d}` : d }
  );
  drawAxis(g, y, 'left', 'Win rate vs EV-optimal',
    { tickFormat: d3.format('.0%') }
  );
}
