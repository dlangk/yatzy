import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor,
  getTextColor, getMutedColor, getGridColor,
} from '../yatzy-viz.js';

export async function initFreeEnergyPhase() {
  const data = await DataLoader.sweepSummary();
  const container = document.getElementById('chart-free-energy-phase');
  if (!container) return;

  if (!data || data.length === 0) return;

  // Sort by theta
  const sorted = [...data].sort((a, b) => a.theta - b.theta);

  // Compute free energy proxy: V(theta) = mean - theta * variance / 2
  const plotData = sorted.map(d => ({
    theta: d.theta,
    mean: d.mean,
    std: d.std,
    freeEnergy: d.mean - d.theta * (d.std * d.std) / 2,
  }));

  const chart = createChart('chart-free-energy-phase', {
    aspectRatio: 0.55,
    marginLeft: 65,
    marginBottom: 50,
    marginTop: 20,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain([d3.min(plotData, d => d.theta), d3.max(plotData, d => d.theta)])
    .range([0, width]);

  const yExtent = d3.extent(plotData, d => d.freeEnergy);
  const yPad = (yExtent[1] - yExtent[0]) * 0.1;
  const y = d3.scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad])
    .range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Draw curve segments colored by theta
  for (let i = 0; i < plotData.length - 1; i++) {
    const d0 = plotData[i];
    const d1 = plotData[i + 1];
    const avgTheta = (d0.theta + d1.theta) / 2;
    g.append('line')
      .attr('x1', x(d0.theta)).attr('y1', y(d0.freeEnergy))
      .attr('x2', x(d1.theta)).attr('y2', y(d1.freeEnergy))
      .attr('stroke', thetaColor(avgTheta))
      .attr('stroke-width', 2.5)
      .attr('stroke-linecap', 'round');
  }

  // Points
  const tt = tooltip(container);

  g.selectAll('.fe-dot')
    .data(plotData)
    .join('circle')
    .attr('cx', d => x(d.theta))
    .attr('cy', d => y(d.freeEnergy))
    .attr('r', 3.5)
    .attr('fill', d => thetaColor(d.theta))
    .attr('stroke', 'white')
    .attr('stroke-width', 1)
    .attr('cursor', 'pointer')
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">\u03b8 = ${d.theta.toFixed(3)}</div>
         <div>Free energy: <span class="tt-value">${d.freeEnergy.toFixed(1)}</span></div>
         <div>Mean: ${d.mean.toFixed(1)}, Std: ${d.std.toFixed(1)}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Domain boundaries at |theta| = 0.15
  [-0.15, 0.15].forEach(boundary => {
    if (boundary < plotData[0].theta || boundary > plotData[plotData.length - 1].theta) return;
    g.append('line')
      .attr('x1', x(boundary)).attr('x2', x(boundary))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6);
  });

  // Domain labels
  g.append('text')
    .attr('x', x(0)).attr('y', height - 8)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '9px')
    .text('utility domain');

  if (x.domain()[1] > 0.2) {
    g.append('text')
      .attr('x', x(Math.min(0.6, x.domain()[1] * 0.7)))
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text('LSE domain');
  }

  if (x.domain()[0] < -0.2) {
    g.append('text')
      .attr('x', x(Math.max(-0.6, x.domain()[0] * 0.7)))
      .attr('y', height - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text('LSE domain');
  }

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Risk parameter \u03b8',
    { tickFormat: d => d === 0 ? '0' : d > 0 ? `+${d}` : d }
  );
  drawAxis(g, y, 'left', 'Free energy proxy V(\u03b8)');
}
