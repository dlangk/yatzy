import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initScoreDistribution() {
  const [kdeCurves, summaryData] = await Promise.all([
    DataLoader.kdeCurves(),
    DataLoader.sweepSummary(),
  ]);

  const container = document.getElementById('chart-score-distribution');
  if (!container) return;

  // Build lookup: theta -> { points: [{x,y}], mean, std }
  const curves = kdeCurves.map(entry => {
    const points = entry.score.map((s, i) => ({ x: s, y: entry.density[i] }));
    const summary = summaryData.find(d => Math.abs(d.theta - entry.theta) < 0.001);
    return {
      theta: entry.theta,
      points,
      mean: summary ? summary.mean : 0,
      std: summary ? summary.std : 0,
    };
  });

  // Slider
  const controls = container.querySelector('.chart-controls');
  const slider = controls.querySelector('.chart-slider');
  const valueDisplay = controls.querySelector('.slider-value');

  slider.min = 0;
  slider.max = curves.length - 1;
  const defaultIdx = curves.findIndex(c => Math.abs(c.theta) < 0.001);
  slider.value = defaultIdx >= 0 ? defaultIdx : 0;

  // Fixed scales
  const xDomain = [0, 400];
  const yDomain = [0, 0.015];

  function render(idx) {
    const cur = curves[idx];
    valueDisplay.textContent = formatTheta(cur.theta);

    const chart = createChart('chart-score-distribution-svg', { aspectRatio: 0.5 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear().domain(xDomain).range([0, width]);
    const y = d3.scaleLinear().domain(yDomain).range([height, 0]);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .selectAll('line')
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

    // Draw all curves at low opacity (background)
    for (let i = 0; i < curves.length; i++) {
      if (i === idx) continue;
      g.append('path')
        .datum(curves[i].points)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', thetaColor(curves[i].theta))
        .attr('stroke-width', 1)
        .attr('opacity', 0.15);
    }

    // Selected curve: filled area + bold line
    const color = thetaColor(cur.theta);

    const area = d3.area()
      .x(d => x(d.x))
      .y0(height)
      .y1(d => y(d.y))
      .curve(d3.curveBasis);

    g.append('path')
      .datum(cur.points)
      .attr('d', area)
      .attr('fill', color)
      .attr('opacity', 0.12);

    g.append('path')
      .datum(cur.points)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2.5);

    // Mean line
    g.append('line')
      .attr('x1', x(cur.mean)).attr('x2', x(cur.mean))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', color)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', x(cur.mean) + 5)
      .attr('y', 14)
      .attr('fill', color)
      .style('font-size', '11px')
      .text(`mean = ${cur.mean.toFixed(1)}`);

    // Axes
    drawAxis(g.append('g').attr('transform', `translate(0,${height})`),
      x, 'bottom', 'Score');
    drawAxis(g, y, 'left', 'Density');

    // Crosshair tooltip
    const tt = tooltip(container);
    const bisect = d3.bisector(d => d.x).left;

    const overlay = g.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all');

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

        crosshairLine
          .attr('x1', x(d.x)).attr('x2', x(d.x))
          .attr('opacity', 0.6);

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

  render(+slider.value);
  slider.addEventListener('input', () => render(+slider.value));
}
