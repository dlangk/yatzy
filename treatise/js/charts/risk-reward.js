import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, formatTheta, thetaColor, lerp,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initRiskReward() {
  const sweep = await DataLoader.sweepSummary();

  // percentilePeaks may not exist in treatise data-loader — handle gracefully
  let peaks = [];
  try {
    if (typeof DataLoader.percentilePeaks === 'function') {
      peaks = await DataLoader.percentilePeaks();
    }
  } catch (e) {
    // No peak data available — skip peak markers
  }

  const container = document.getElementById('chart-risk-reward');
  if (!container) return;

  // Filter to useful range
  const data = sweep.filter(d => d.theta >= -0.3 && d.theta <= 0.3);

  const percentileKeys = [
    { key: 'p5', label: 'p5', color: COLORS.percentiles.p5 },
    { key: 'p25', label: 'p25', color: COLORS.percentiles.p25 },
    { key: 'p50', label: 'p50 (median)', color: COLORS.percentiles.p50 },
    { key: 'p75', label: 'p75', color: COLORS.percentiles.p75 },
    { key: 'p95', label: 'p95', color: COLORS.percentiles.p95 },
    { key: 'p99', label: 'p99', color: COLORS.percentiles.p99 },
  ];

  let cursorTheta = 0;

  function render() {
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
      .data(y.ticks(6))
      .join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,3');

    // p5-p95 band
    const bandArea = d3.area()
      .x(d => x(d.theta))
      .y0(d => y(d.p5))
      .y1(d => y(d.p95))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('d', bandArea)
      .attr('fill', getMutedColor())
      .attr('opacity', 0.08);

    // Percentile lines
    percentileKeys.forEach(pk => {
      const line = d3.line()
        .x(d => x(d.theta))
        .y(d => y(d[pk.key]))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(data)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', pk.color)
        .attr('stroke-width', pk.key === 'p50' ? 2.5 : 1.8)
        .attr('stroke-dasharray', pk.key === 'p50' ? null : '5,3');

      // End label
      const lastD = data[data.length - 1];
      g.append('text')
        .attr('x', width + 4)
        .attr('y', y(lastD[pk.key]) + 4)
        .attr('fill', pk.color)
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text(pk.label);
    });

    // Peak stars (only if peaks data is available)
    if (peaks && peaks.length > 0) {
      peaks.filter(p => ['p5', 'p50', 'p95', 'p99'].includes(p.percentile))
        .filter(p => p.theta_star >= data[0].theta && p.theta_star <= data[data.length - 1].theta)
        .forEach(p => {
          const pKey = p.percentile;
          const yVal = lerp(data, 'theta', pKey, p.theta_star);
          g.append('text')
            .attr('x', x(p.theta_star))
            .attr('y', y(yVal) - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', COLORS.percentiles[pKey] || COLORS.accent)
            .style('font-size', '14px')
            .text('\u2605');
        });
    }

    // Vertical zero line
    g.append('line')
      .attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '3,3')
      .attr('opacity', 0.4);

    // Draggable cursor
    const cursorLine = g.append('line')
      .attr('x1', x(cursorTheta)).attr('x2', x(cursorTheta))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', thetaColor(cursorTheta))
      .attr('stroke-width', 2)
      .attr('cursor', 'ew-resize');

    const cursorDots = g.append('g');
    function updateCursorDots() {
      cursorDots.selectAll('circle').remove();
      percentileKeys.forEach(pk => {
        const yVal = lerp(data, 'theta', pk.key, cursorTheta);
        cursorDots.append('circle')
          .attr('cx', x(cursorTheta))
          .attr('cy', y(yVal))
          .attr('r', 4)
          .attr('fill', pk.color)
          .attr('stroke', 'white')
          .attr('stroke-width', 1.5);
      });
    }
    updateCursorDots();

    // Drag behavior
    const drag = d3.drag()
      .on('drag', (event) => {
        const theta = x.invert(Math.max(0, Math.min(width, event.x)));
        cursorTheta = Math.round(theta * 1000) / 1000;
        cursorLine
          .attr('x1', x(cursorTheta))
          .attr('x2', x(cursorTheta))
          .attr('stroke', thetaColor(cursorTheta));
        updateCursorDots();
        updateStatsPanel();
      });

    // Invisible wide drag target
    g.append('rect')
      .attr('x', x(cursorTheta) - 8)
      .attr('y', 0)
      .attr('width', 16)
      .attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .attr('cursor', 'ew-resize')
      .call(drag);

    // Also allow clicking anywhere to move cursor
    g.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .style('cursor', 'crosshair')
      .lower()
      .on('click', (event) => {
        const theta = x.invert(d3.pointer(event)[0]);
        cursorTheta = Math.round(theta * 1000) / 1000;
        render();
        updateStatsPanel();
      });

    // Axes
    drawAxis(
      g.append('g').attr('transform', `translate(0,${height})`),
      x, 'bottom', 'Risk parameter \u03b8',
      { tickFormat: d => d === 0 ? '0' : d > 0 ? `+${d}` : d }
    );
    drawAxis(g, y, 'left', 'Score');
  }

  function updateStatsPanel() {
    const panel = container.querySelector('.chart-stats-panel');
    if (!panel) return;

    const stats = {};
    percentileKeys.forEach(pk => {
      stats[pk.key] = lerp(sweep, 'theta', pk.key, cursorTheta);
    });
    const mean = lerp(sweep, 'theta', 'mean', cursorTheta);
    const std = lerp(sweep, 'theta', 'std', cursorTheta);

    panel.innerHTML = `
      <div class="chart-stat">
        <div class="chart-stat-value">${formatTheta(cursorTheta)}</div>
        <div class="chart-stat-label">theta</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${mean.toFixed(1)}</div>
        <div class="chart-stat-label">mean</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${std.toFixed(1)}</div>
        <div class="chart-stat-label">std dev</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${stats.p5.toFixed(0)}</div>
        <div class="chart-stat-label">5th %ile</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${stats.p50.toFixed(0)}</div>
        <div class="chart-stat-label">median</div>
      </div>
      <div class="chart-stat">
        <div class="chart-stat-value">${stats.p95.toFixed(0)}</div>
        <div class="chart-stat-label">95th %ile</div>
      </div>
    `;
  }

  render();
  updateStatsPanel();

  // Legend
  const legend = container.querySelector('.chart-legend');
  if (legend) {
    legend.innerHTML = percentileKeys.map(pk =>
      `<div class="chart-legend-item">
        <span class="legend-swatch" style="background:${pk.color}"></span>
        <span>${pk.label}</span>
      </div>`
    ).join('');
  }
}
