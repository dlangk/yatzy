/**
 * Trajectory Chart — D3 SVG visualization of expected score over turns.
 *
 * Layout:
 *   SVG (viewBox 568×230) with margins { top:20, right:16, bottom:44, left:48 }
 *   HTML legend below SVG (flex-wrap, always visible)
 *   Hover tooltip with crosshair + nearest-point snapping (desktop only)
 *
 * Margins: bottom=44 gives room for x-axis labels + breathing space.
 * HEIGHT=230 adds space so the chart area isn't cramped.
 *
 * ResizeObserver re-renders on container width changes to keep
 * hover overlay dimensions and legend layout correct.
 */
import * as d3 from 'd3';
import { getState, subscribe } from '../store.ts';
import type { TrajectoryPoint } from '../types.ts';
import { COLORS } from '../constants.ts';

const WIDTH = 568;
const HEIGHT = 230;
const MARGIN = { top: 20, right: 16, bottom: 44, left: 48 };

const Y_MIN = 0;
const Y_MAX = 400;
const X_MIN = 0;
const X_MAX = 15;

const EVENT_COLORS: Record<string, string> = {
  start: COLORS.textMuted,
  roll: COLORS.blue,
  reroll: COLORS.orange,
  score: COLORS.success,
};

const EVENT_LABELS: Record<string, string> = {
  start: 'Start',
  roll: 'Roll',
  reroll: 'Reroll',
  score: 'Score',
};

/** Compute x-positions for trajectory points:
 *  - start → 0
 *  - score → turn (integer)
 *  - roll/reroll → evenly spaced fractions between previous score and next score
 */
function computeXPositions(trajectory: TrajectoryPoint[]): number[] {
  const positions = new Array<number>(trajectory.length);

  // First pass: assign fixed positions for start and score events
  for (let i = 0; i < trajectory.length; i++) {
    const p = trajectory[i];
    if (p.event === 'start') {
      positions[i] = 0;
    } else if (p.event === 'score') {
      positions[i] = p.turn;
    } else {
      positions[i] = -1; // placeholder
    }
  }

  // Second pass: place roll/reroll events as fractions between anchors
  let prevAnchorIdx = 0;
  for (let i = 1; i < trajectory.length; i++) {
    if (positions[i] >= 0) {
      // This is an anchor (start or score). Fill in any gaps.
      const gapStart = prevAnchorIdx;
      const gapEnd = i;
      const count = gapEnd - gapStart; // total intervals
      if (count > 1) {
        const xFrom = positions[gapStart];
        const xTo = positions[gapEnd];
        for (let j = gapStart + 1; j < gapEnd; j++) {
          const frac = (j - gapStart) / count;
          positions[j] = xFrom + frac * (xTo - xFrom);
        }
      }
      prevAnchorIdx = i;
    }
  }

  // Handle trailing roll/reroll points after last score (game in progress)
  if (prevAnchorIdx < trajectory.length - 1) {
    const lastAnchorX = positions[prevAnchorIdx];
    const nextTurn = lastAnchorX + 1; // project to next integer turn
    const remaining = trajectory.length - prevAnchorIdx;
    for (let j = prevAnchorIdx + 1; j < trajectory.length; j++) {
      const frac = (j - prevAnchorIdx) / remaining;
      positions[j] = lastAnchorX + frac * (nextTurn - lastAnchorX);
    }
  }

  return positions;
}

/** Build the HTML legend below the chart SVG. */
function buildLegend(container: HTMLElement): void {
  const legend = document.createElement('div');
  legend.className = 'trajectory-legend';

  const items: { svg: string; label: string }[] = [
    {
      svg: `<svg width="16" height="12" viewBox="0 0 16 12"><line x1="0" y1="6" x2="16" y2="6" stroke="${COLORS.blue}" stroke-width="2"/></svg>`,
      label: 'Expected final',
    },
    {
      svg: `<svg width="16" height="12" viewBox="0 0 16 12"><rect x="0" y="1" width="16" height="10" fill="rgba(44,160,44,0.15)" stroke="rgba(44,160,44,0.3)" stroke-width="1"/></svg>`,
      label: 'Accumulated',
    },
    {
      svg: `<svg width="16" height="12" viewBox="0 0 16 12"><rect x="0" y="1" width="16" height="10" fill="rgba(59,76,192,0.06)"/></svg>`,
      label: 'p1–p99',
    },
    {
      svg: `<svg width="16" height="12" viewBox="0 0 16 12"><rect x="0" y="1" width="16" height="10" fill="rgba(59,76,192,0.12)"/></svg>`,
      label: 'p10–p90',
    },
    {
      svg: `<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="3.5" fill="${COLORS.textMuted}" stroke="${COLORS.bg}" stroke-width="1"/></svg>`,
      label: 'Start',
    },
    {
      svg: `<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="3.5" fill="${COLORS.blue}" stroke="${COLORS.bg}" stroke-width="1"/></svg>`,
      label: 'Roll',
    },
    {
      svg: `<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="3.5" fill="${COLORS.orange}" stroke="${COLORS.bg}" stroke-width="1"/></svg>`,
      label: 'Reroll',
    },
    {
      svg: `<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="3.5" fill="${COLORS.success}" stroke="${COLORS.bg}" stroke-width="1"/></svg>`,
      label: 'Score',
    },
  ];

  for (const item of items) {
    const el = document.createElement('span');
    el.className = 'trajectory-legend-item';
    el.innerHTML = `<span class="trajectory-legend-swatch">${item.svg}</span>${item.label}`;
    legend.appendChild(el);
  }

  container.appendChild(legend);
}

/** Render the D3 trajectory chart: expected score over turns with percentile bands. */
export function initTrajectoryChart(container: HTMLElement): void {
  const wrapper = document.createElement('div');
  wrapper.className = 'trajectory-chart';
  wrapper.style.position = 'relative'; // anchor for tooltip
  container.appendChild(wrapper);

  const svg = d3.select(wrapper)
    .append('svg')
    .attr('viewBox', `0 0 ${WIDTH} ${HEIGHT}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Background
  svg.append('rect')
    .attr('width', WIDTH)
    .attr('height', HEIGHT)
    .attr('fill', COLORS.bgPanel);

  // Clip path for plot area
  svg.append('defs')
    .append('clipPath')
    .attr('id', 'plot-clip')
    .append('rect')
    .attr('x', MARGIN.left)
    .attr('y', MARGIN.top)
    .attr('width', WIDTH - MARGIN.left - MARGIN.right)
    .attr('height', HEIGHT - MARGIN.top - MARGIN.bottom);

  const plotG = svg.append('g').attr('clip-path', 'url(#plot-clip)');

  // Groups for layered drawing
  const gridG = svg.append('g').attr('class', 'grid');
  const bandG = plotG.append('g').attr('class', 'bands');
  const floorG = plotG.append('g').attr('class', 'floor');
  const lineG = plotG.append('g').attr('class', 'line');
  const pointsG = plotG.append('g').attr('class', 'points');
  const xAxisG = svg.append('g').attr('class', 'x-labels');
  const yAxisG = svg.append('g').attr('class', 'y-labels');

  // Hover elements — crosshair line and highlight circle (hidden by default)
  const hoverG = svg.append('g').attr('class', 'hover').style('display', 'none');
  const crosshairLine = hoverG.append('line')
    .attr('stroke', COLORS.textMuted)
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '3,3')
    .attr('y1', MARGIN.top)
    .attr('y2', HEIGHT - MARGIN.bottom);
  const highlightCircle = hoverG.append('circle')
    .attr('r', 6)
    .attr('fill', 'none')
    .attr('stroke-width', 2);

  // Hover overlay rect (captures mouse events over the plot area)
  const hoverOverlay = svg.append('rect')
    .attr('x', MARGIN.left)
    .attr('y', MARGIN.top)
    .attr('width', WIDTH - MARGIN.left - MARGIN.right)
    .attr('height', HEIGHT - MARGIN.top - MARGIN.bottom)
    .attr('fill', 'transparent')
    .attr('cursor', 'crosshair');

  // Tooltip element (HTML, positioned absolute over the chart)
  const tooltip = document.createElement('div');
  tooltip.className = 'trajectory-tooltip';
  tooltip.style.display = 'none';
  wrapper.appendChild(tooltip);

  // Placeholder text
  const placeholder = svg.append('text')
    .attr('x', WIDTH / 2)
    .attr('y', HEIGHT / 2)
    .attr('text-anchor', 'middle')
    .attr('fill', COLORS.textMuted)
    .attr('font-size', 12)
    .attr('font-family', 'monospace')
    .text('Expected Score Trajectory');

  // Legend below the SVG
  buildLegend(wrapper);

  // Module-level data refs updated each render so hover handlers can access current data
  let currentTrajectory: TrajectoryPoint[] = [];
  let currentXPositions: number[] = [];
  let currentXScale: d3.ScaleLinear<number, number> | null = null;
  let currentYScale: d3.ScaleLinear<number, number> | null = null;

  /**
   * Hover handler: find nearest trajectory point by x-position.
   * Uses bisect on xPositions for O(log n) lookup.
   */
  hoverOverlay.on('mousemove', function (event: MouseEvent) {
    if (!currentXScale || !currentYScale || currentTrajectory.length === 0) return;

    // Convert mouse position from screen to SVG viewBox coords
    const svgNode = svg.node()!;
    const pt = svgNode.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const svgPt = pt.matrixTransform(svgNode.getScreenCTM()!.inverse());

    const mouseXValue = currentXScale.invert(svgPt.x);

    // Find nearest point by x
    let nearestIdx = 0;
    let nearestDist = Infinity;
    for (let i = 0; i < currentTrajectory.length; i++) {
      const dist = Math.abs(currentXPositions[i] - mouseXValue);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIdx = i;
      }
    }

    const p = currentTrajectory[nearestIdx];
    const px = currentXScale(currentXPositions[nearestIdx]);
    const py = currentYScale(p.expectedFinal);

    // Position crosshair and highlight
    hoverG.style('display', null);
    crosshairLine.attr('x1', px).attr('x2', px);
    highlightCircle
      .attr('cx', px)
      .attr('cy', py)
      .attr('stroke', EVENT_COLORS[p.event] || COLORS.text);

    // Build tooltip content
    const eventLabel = EVENT_LABELS[p.event] || p.event;
    let html = `<div><b>Turn ${p.turn} — ${eventLabel}</b></div>`;
    html += `<div>E[final]: ${p.expectedFinal.toFixed(1)}</div>`;
    html += `<div>Accumulated: ${p.accumulatedScore}</div>`;
    if (p.delta !== undefined) {
      const sign = p.delta >= 0 ? '+' : '';
      const color = p.delta >= 0 ? COLORS.success : COLORS.danger;
      html += `<div style="color:${color}">Delta: ${sign}${p.delta.toFixed(1)}</div>`;
    }
    if (p.label) {
      html += `<div style="color:${COLORS.textMuted}">${p.label}</div>`;
    }
    if (p.percentiles) {
      const pct = p.percentiles;
      html += `<div style="color:${COLORS.textMuted};margin-top:2px;font-size:0.9em">`;
      html += `p10–p90: ${Math.round(pct.p10)}–${Math.round(pct.p90)}`;
      html += `<br>p1–p99: ${Math.round(pct.p1)}–${Math.round(pct.p99)}`;
      html += `</div>`;
    }
    tooltip.innerHTML = html;
    tooltip.style.display = '';

    // Position tooltip in wrapper-relative coords.
    // Convert SVG viewBox px → wrapper px using wrapper's actual width.
    const wrapperRect = wrapper.getBoundingClientRect();
    const svgRect = svgNode.getBoundingClientRect();
    const scaleX = svgRect.width / WIDTH;
    const scaleY = svgRect.height / HEIGHT;
    const tipX = (px * scaleX) + (svgRect.left - wrapperRect.left);
    const tipY = (py * scaleY) + (svgRect.top - wrapperRect.top);

    // Flip horizontally when near right edge
    const tooltipWidth = tooltip.offsetWidth;
    if (tipX + tooltipWidth + 12 > wrapperRect.width) {
      tooltip.style.left = `${tipX - tooltipWidth - 8}px`;
    } else {
      tooltip.style.left = `${tipX + 8}px`;
    }
    tooltip.style.top = `${tipY - 12}px`;
  });

  hoverOverlay.on('mouseleave', () => {
    hoverG.style('display', 'none');
    tooltip.style.display = 'none';
  });

  function render() {
    const trajectory = getState().trajectory;

    gridG.selectAll('*').remove();
    bandG.selectAll('*').remove();
    floorG.selectAll('*').remove();
    lineG.selectAll('*').remove();
    pointsG.selectAll('*').remove();
    xAxisG.selectAll('*').remove();
    yAxisG.selectAll('*').remove();

    // Fixed scales
    const x = d3.scaleLinear()
      .domain([X_MIN, X_MAX])
      .range([MARGIN.left, WIDTH - MARGIN.right]);

    const y = d3.scaleLinear()
      .domain([Y_MIN, Y_MAX])
      .range([HEIGHT - MARGIN.bottom, MARGIN.top]);

    // Update module-level refs for hover handlers
    currentXScale = x;
    currentYScale = y;

    // Grid lines (Y)
    const yStep = niceStep(Y_MIN, Y_MAX, 5);
    const yTicks: number[] = [];
    for (let v = Math.ceil(Y_MIN / yStep) * yStep; v <= Y_MAX; v += yStep) {
      yTicks.push(v);
    }

    gridG.selectAll('line')
      .data(yTicks)
      .join('line')
      .attr('x1', MARGIN.left)
      .attr('x2', WIDTH - MARGIN.right)
      .attr('y1', d => y(d))
      .attr('y2', d => y(d))
      .attr('stroke', COLORS.borderPanel)
      .attr('stroke-width', 0.5);

    // Y-axis labels
    yAxisG.selectAll('text')
      .data(yTicks)
      .join('text')
      .attr('x', MARGIN.left - 4)
      .attr('y', d => y(d) + 3)
      .attr('text-anchor', 'end')
      .attr('fill', COLORS.textMuted)
      .attr('font-size', 10)
      .attr('font-family', 'monospace')
      .text(d => String(Math.round(d)));

    // X-axis: fixed integer labels (0, 3, 6, 9, 12, 15)
    const xTicks = [0, 3, 6, 9, 12, 15];
    xAxisG.selectAll('text')
      .data(xTicks)
      .join('text')
      .attr('x', d => x(d))
      .attr('y', HEIGHT - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', COLORS.textMuted)
      .attr('font-size', 10)
      .attr('font-family', 'monospace')
      .text(d => String(d));

    if (trajectory.length === 0) {
      placeholder.style('display', null);
      currentTrajectory = [];
      currentXPositions = [];
      return;
    }
    placeholder.style('display', 'none');

    // Compute fractional x positions
    const xPositions = computeXPositions(trajectory);
    const getX = (d: TrajectoryPoint) => x(xPositions[d.index]);

    // Update module-level data refs for hover
    currentTrajectory = trajectory;
    currentXPositions = xPositions;

    // Percentile bands — include start point if it has percentiles
    const bandPoints = trajectory.filter(p => (p.event === 'score' || p.event === 'start') && p.percentiles);
    if (bandPoints.length >= 2) {
      drawBand(bandG, bandPoints, 'p1', 'p99', 'rgba(59, 76, 192, 0.06)', xPositions, x, y);
      drawBand(bandG, bandPoints, 'p10', 'p90', 'rgba(59, 76, 192, 0.12)', xPositions, x, y);
    } else if (bandPoints.length === 1 && bandPoints[0].percentiles) {
      drawWhisker(bandG, bandPoints[0], xPositions, x, y);
    }

    // Accumulated score floor (shaded area)
    const areaGen = d3.area<TrajectoryPoint>()
      .x(d => getX(d))
      .y0(y(Y_MIN))
      .y1(d => y(d.accumulatedScore));

    floorG.append('path')
      .datum(trajectory)
      .attr('d', areaGen)
      .attr('fill', 'rgba(44, 160, 44, 0.08)');

    // Accumulated score line
    const accLine = d3.line<TrajectoryPoint>()
      .x(d => getX(d))
      .y(d => y(d.accumulatedScore));

    floorG.append('path')
      .datum(trajectory)
      .attr('d', accLine)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(44, 160, 44, 0.3)')
      .attr('stroke-width', 1);

    // EV trajectory line
    const evLine = d3.line<TrajectoryPoint>()
      .x(d => getX(d))
      .y(d => y(d.expectedFinal));

    lineG.append('path')
      .datum(trajectory)
      .attr('d', evLine)
      .attr('fill', 'none')
      .attr('stroke', COLORS.blue)
      .attr('stroke-width', 2);

    // Points
    pointsG.selectAll('circle')
      .data(trajectory)
      .join('circle')
      .attr('cx', d => getX(d))
      .attr('cy', d => y(d.expectedFinal))
      .attr('r', d => d.event === 'score' ? 4 : 3)
      .attr('fill', d => EVENT_COLORS[d.event] || COLORS.text)
      .attr('stroke', COLORS.bg)
      .attr('stroke-width', 1);
  }

  render();
  subscribe(render);

  // Re-render on container resize so hover overlay and legend stay correct
  const ro = new ResizeObserver(() => render());
  ro.observe(wrapper);
}

function drawBand(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  points: TrajectoryPoint[],
  lowKey: string,
  highKey: string,
  color: string,
  xPositions: number[],
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  const valid = points.filter(
    p => p.percentiles && p.percentiles[lowKey] !== undefined && p.percentiles[highKey] !== undefined,
  );
  if (valid.length < 2) return;

  const area = d3.area<TrajectoryPoint>()
    .x(d => x(xPositions[d.index]))
    .y0(d => y(d.percentiles![lowKey]))
    .y1(d => y(d.percentiles![highKey]));

  g.append('path')
    .datum(valid)
    .attr('d', area)
    .attr('fill', color);
}

function drawWhisker(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  p: TrajectoryPoint,
  xPositions: number[],
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  if (!p.percentiles) return;
  const px = x(xPositions[p.index]);
  for (const [lowKey, highKey, alpha] of [
    ['p1', 'p99', 0.12],
    ['p10', 'p90', 0.25],
  ] as [string, string, number][]) {
    const lo = p.percentiles[lowKey];
    const hi = p.percentiles[highKey];
    if (lo !== undefined && hi !== undefined) {
      g.append('line')
        .attr('x1', px)
        .attr('x2', px)
        .attr('y1', y(lo))
        .attr('y2', y(hi))
        .attr('stroke', `rgba(59, 76, 192, ${alpha})`)
        .attr('stroke-width', 3);
    }
  }
}

function niceStep(min: number, max: number, targetTicks: number): number {
  const range = max - min || 50;
  const rough = range / targetTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  if (norm <= 2) return 2 * mag;
  if (norm <= 5) return 5 * mag;
  return 10 * mag;
}
