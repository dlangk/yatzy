import { subscribe, getState } from '../store.js';
import {
  createChart,
  drawAxis,
  tooltip,
  getTextColor,
  getMutedColor,
  getGridColor,
  COLORS,
} from '../../yatzy-viz.js';

const CONTAINER_ID = 'trajectory-chart-svg';

// DOM references
let chartContainer = null;
let tt = null;

/**
 * Build band path data from points that have density.
 * Returns array of {x, lower, upper} using array index as x.
 * Points without density are interpolated by carrying forward the
 * last known density values so bands span the full chart.
 */
function bandData(points, pKeyLower, pKeyUpper) {
  if (!points.some(p => p.density)) return null;

  const result = [];
  let lastLower = null;
  let lastUpper = null;

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (p.density) {
      lastLower = p.density[pKeyLower] ?? p.expectedFinal;
      lastUpper = p.density[pKeyUpper] ?? p.expectedFinal;
    }
    if (lastLower != null) {
      result.push({ x: i, lower: lastLower, upper: lastUpper });
    }
  }
  return result.length > 0 ? result : null;
}

/**
 * Render the D3 fan chart from state.history.
 */
function renderChart(history) {
  if (!chartContainer) return;

  // Only render points that have expectedFinal
  const points = history.filter(p => p.expectedFinal != null);

  if (points.length === 0) {
    chartContainer.querySelector(`#${CONTAINER_ID}`)?.querySelectorAll('svg').forEach(s => s.remove());
    return;
  }

  const chart = createChart(CONTAINER_ID, {
    marginTop: 25,
    marginRight: 15,
    marginBottom: 40,
    marginLeft: 50,
    aspectRatio: 0.45,
  });
  if (!chart) return;

  const { g, width, height } = chart;

  // Scales — x is array index within renderable points
  const xExtent = [0, Math.max(points.length - 1, 1)];
  const x = d3.scaleLinear().domain(xExtent).range([0, width]);

  // Y domain: encompass all density bands + center line
  let yMin = Infinity, yMax = -Infinity;
  for (const p of points) {
    const ef = p.expectedFinal;
    yMin = Math.min(yMin, ef, p.accumulatedScore);
    yMax = Math.max(yMax, ef);
    if (p.density) {
      yMin = Math.min(yMin, p.density.p10 ?? ef);
      yMax = Math.max(yMax, p.density.p90 ?? ef);
    }
  }
  const yPad = Math.max((yMax - yMin) * 0.1, 10);
  const y = d3.scaleLinear()
    .domain([Math.max(0, yMin - yPad), yMax + yPad])
    .range([height, 0]);

  // Grid lines
  const gridColor = getGridColor();
  g.append('g')
    .attr('class', 'grid')
    .selectAll('line')
    .data(y.ticks(5))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', gridColor)
    .attr('stroke-dasharray', '2,3')
    .attr('opacity', 0.5);

  // Axes
  const bottomG = g.append('g').attr('transform', `translate(0,${height})`);
  drawAxis(bottomG, x, 'bottom', 'Event', { tickCount: Math.min(points.length, 8), tickFormat: d3.format('d') });
  drawAxis(g, y, 'left', 'Score', { tickCount: 6 });

  const curve = d3.curveMonotoneX;

  // Check if any point has density
  const hasDensity = points.some(p => p.density);

  // Fan bands — drawn first (behind lines)
  if (hasDensity) {
    const outerBand = bandData(points, 'p10', 'p90');
    const innerBand = bandData(points, 'p25', 'p75');

    const areaGen = d3.area()
      .x(d => x(d.x))
      .y0(d => y(d.lower))
      .y1(d => y(d.upper))
      .curve(curve);

    if (outerBand && outerBand.length >= 1) {
      g.append('path')
        .datum(outerBand)
        .attr('d', areaGen)
        .attr('fill', COLORS.accentLight)
        .attr('opacity', 0.35);
    }

    if (innerBand && innerBand.length >= 1) {
      g.append('path')
        .datum(innerBand)
        .attr('d', areaGen)
        .attr('fill', COLORS.accent)
        .attr('opacity', 0.2);
    }
  }

  // Banked score step-line (behind center line)
  const stepData = points.map((p, i) => ({ i, acc: p.accumulatedScore }));
  const stepLine = d3.line()
    .x(d => x(d.i))
    .y(d => y(d.acc))
    .curve(d3.curveStepAfter);

  g.append('path')
    .datum(stepData)
    .attr('d', stepLine)
    .attr('fill', 'none')
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3');

  // Center line (expected final) — most prominent
  const lineData = points.map((p, i) => ({ i, ef: p.expectedFinal }));
  const line = d3.line()
    .x(d => x(d.i))
    .y(d => y(d.ef))
    .curve(curve);

  g.append('path')
    .datum(lineData)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 2.5);

  // Dots (on top)
  const textColor = getTextColor();
  g.selectAll('.traj-dot')
    .data(points)
    .join('circle')
    .attr('class', 'traj-dot')
    .attr('cx', (d, i) => x(i))
    .attr('cy', d => y(d.expectedFinal))
    .attr('r', d => d.type === 'score' || d.type === 'start' || d.type === 'game_over' ? 4 : 2.5)
    .attr('fill', d => d.type === 'score' || d.type === 'game_over' ? COLORS.accent : getMutedColor())
    .attr('stroke', d => d.type === 'score' ? textColor : 'none')
    .attr('stroke-width', 1);

  // Tooltip
  if (!tt) tt = tooltip(chartContainer.querySelector(`#${CONTAINER_ID}`));

  g.selectAll('.traj-dot')
    .on('mouseenter', (event, d) => {
      const pctRange = d.density
        ? `<br>p10\u2013p90: ${d.density.p10}\u2013${d.density.p90}`
        : '';
      tt.show(
        `<div class="tt-label">Turn ${d.turn + 1} \u2014 ${d.type}</div>` +
        `<div>Expected: <span class="tt-value">${d.expectedFinal.toFixed(1)}</span></div>` +
        `<div>Banked: ${d.accumulatedScore}</div>` +
        (d.stateEv != null ? `<div>State EV: ${d.stateEv.toFixed(1)}</div>` : '') +
        pctRange,
        event
      );
    })
    .on('mousemove', (event) => tt.move(event))
    .on('mouseleave', () => tt.hide());

  // Legend
  let legend = chartContainer.querySelector('.trajectory-legend');
  if (!legend) {
    legend = document.createElement('div');
    legend.className = 'chart-legend trajectory-legend';
    chartContainer.querySelector(`#${CONTAINER_ID}`).after(legend);
  }
  legend.innerHTML = '';

  const items = [
    { color: COLORS.accent, label: 'Expected final' },
    { color: getMutedColor(), label: 'Banked score', dashed: true },
  ];
  if (hasDensity) {
    items.push({ color: COLORS.accent, label: 'p25\u2013p75', opacity: 0.2 });
    items.push({ color: COLORS.accentLight, label: 'p10\u2013p90', opacity: 0.35 });
  }

  for (const item of items) {
    const el = document.createElement('span');
    el.className = 'chart-legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'legend-swatch';
    if (item.dashed) {
      swatch.style.background = 'none';
      swatch.style.borderBottom = `2px dashed ${item.color}`;
      swatch.style.height = '0';
      swatch.style.alignSelf = 'center';
    } else if (item.opacity) {
      swatch.style.background = item.color;
      swatch.style.opacity = item.opacity;
    } else {
      swatch.style.background = item.color;
    }
    el.appendChild(swatch);
    el.appendChild(document.createTextNode(item.label));
    legend.appendChild(el);
  }
}

/**
 * Re-render on theme toggle.
 */
function watchTheme() {
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', () => {
      setTimeout(() => renderChart(getState().history), 50);
    });
  }
}

/**
 * Initialize the trajectory chart component.
 */
export function initTrajectoryChart(container) {
  chartContainer = document.createElement('div');
  chartContainer.className = 'trajectory-chart-wrapper';

  const title = document.createElement('div');
  title.className = 'chart-title';
  title.textContent = 'Expected Score Trajectory';
  chartContainer.appendChild(title);

  const svgContainer = document.createElement('div');
  svgContainer.id = CONTAINER_ID;
  svgContainer.className = 'chart-container';
  chartContainer.appendChild(svgContainer);

  container.appendChild(chartContainer);

  // Render on every state change
  subscribe((state) => renderChart(state.history));
  watchTheme();

  // Initial render
  renderChart(getState().history);
}
