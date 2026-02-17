import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, densityFromPercentiles, COLORS,
} from '../yatzy-viz.js';

export async function initScoreDistribution() {
  const data = await DataLoader.sweepSummary();
  const container = document.getElementById('chart-score-distribution');
  if (!container) return;

  // Find theta=0 index
  const theta0Idx = data.findIndex(d => Math.abs(d.theta) < 0.001);

  // Slider
  const controls = container.querySelector('.chart-controls');
  const slider = controls.querySelector('.chart-slider');
  const valueDisplay = controls.querySelector('.slider-value');

  // Filter to useful range
  const usable = data.filter(d => d.theta >= -0.5 && d.theta <= 0.5);
  slider.min = 0;
  slider.max = usable.length - 1;
  const defaultIdx = usable.findIndex(d => Math.abs(d.theta) < 0.001);
  slider.value = defaultIdx;

  function render(idx) {
    const row = usable[idx];
    const theta0Row = data[theta0Idx];
    valueDisplay.textContent = formatTheta(row.theta);

    const chart = createChart('chart-score-distribution-svg', { aspectRatio: 0.5 });
    if (!chart) return;
    const { g, width, height } = chart;

    const density = densityFromPercentiles(row);
    const refDensity = densityFromPercentiles(theta0Row);

    const x = d3.scaleLinear()
      .domain([80, 370])
      .range([0, width]);

    const yMax = Math.max(
      d3.max(density, d => d.y),
      d3.max(refDensity, d => d.y)
    ) * 1.1;
    const y = d3.scaleLinear()
      .domain([0, yMax])
      .range([height, 0]);

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

    // Reference line (theta=0) if different
    if (Math.abs(row.theta) > 0.001) {
      const refLine = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveBasis);

      g.append('path')
        .datum(refDensity.filter(d => d.x >= 80 && d.x <= 370))
        .attr('d', refLine)
        .attr('fill', 'none')
        .attr('stroke', getMutedColor())
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '4,3')
        .attr('opacity', 0.5);
    }

    // Main density
    const area = d3.area()
      .x(d => x(d.x))
      .y0(height)
      .y1(d => y(d.y))
      .curve(d3.curveBasis);

    const line = d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
      .curve(d3.curveBasis);

    const filtered = density.filter(d => d.x >= 80 && d.x <= 370);
    const color = thetaColor(row.theta);

    g.append('path')
      .datum(filtered)
      .attr('d', area)
      .attr('fill', color)
      .attr('opacity', 0.15);

    g.append('path')
      .datum(filtered)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2.5);

    // Mean line
    g.append('line')
      .attr('x1', x(row.mean)).attr('x2', x(row.mean))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', color)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', x(row.mean) + 5)
      .attr('y', 14)
      .attr('fill', color)
      .style('font-size', '11px')
      .text(`mean = ${row.mean.toFixed(1)}`);

    // Axes
    const xAxisG = drawAxis(g.append('g').attr('transform', `translate(0,${height})`),
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
        const i = bisect(filtered, xVal);
        const d = filtered[Math.min(i, filtered.length - 1)];
        if (!d) return;

        crosshairLine
          .attr('x1', x(d.x)).attr('x2', x(d.x))
          .attr('opacity', 0.6);

        tt.show(
          `<div class="tt-label">Score: ${d.x.toFixed(0)}</div>
           <div>Density: <span class="tt-value">${d.y.toFixed(4)}</span></div>
           <div>Mean: ${row.mean.toFixed(1)}, Std: ${row.std.toFixed(1)}</div>`,
          event
        );
      })
      .on('mouseleave', () => {
        crosshairLine.attr('opacity', 0);
        tt.hide();
      });
  }

  render(defaultIdx);
  slider.addEventListener('input', () => render(+slider.value));
}
