import * as d3 from 'd3';
import { getState, subscribe } from '../store.ts';
import type { TrajectoryPoint } from '../types.ts';
import { COLORS } from '../constants.ts';

const WIDTH = 568;
const HEIGHT = 200;
const MARGIN = { top: 20, right: 16, bottom: 28, left: 48 };

const EVENT_COLORS: Record<string, string> = {
  start: COLORS.textMuted,
  roll: COLORS.blue,
  reroll: COLORS.orange,
  score: COLORS.success,
};

export function initTrajectoryChart(container: HTMLElement): void {
  const wrapper = document.createElement('div');
  wrapper.className = 'trajectory-chart';
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

  // Placeholder text
  const placeholder = svg.append('text')
    .attr('x', WIDTH / 2)
    .attr('y', HEIGHT / 2)
    .attr('text-anchor', 'middle')
    .attr('fill', COLORS.textMuted)
    .attr('font-size', 12)
    .attr('font-family', 'monospace')
    .text('Expected Score Trajectory');

  function render() {
    const trajectory = getState().trajectory;

    gridG.selectAll('*').remove();
    bandG.selectAll('*').remove();
    floorG.selectAll('*').remove();
    lineG.selectAll('*').remove();
    pointsG.selectAll('*').remove();
    xAxisG.selectAll('*').remove();
    yAxisG.selectAll('*').remove();

    if (trajectory.length === 0) {
      placeholder.style('display', null);
      drawEmptyGrid();
      return;
    }
    placeholder.style('display', 'none');

    // Compute Y range
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const p of trajectory) {
      yMin = Math.min(yMin, p.accumulatedScore, p.expectedFinal);
      yMax = Math.max(yMax, p.expectedFinal);
      if (p.percentiles) {
        const p90 = p.percentiles['p90'];
        const p10 = p.percentiles['p10'];
        if (p90 !== undefined) yMax = Math.max(yMax, p90);
        if (p10 !== undefined) yMin = Math.min(yMin, p10);
      }
    }
    const yRange = yMax - yMin || 50;
    yMin = Math.max(0, Math.floor((yMin - yRange * 0.1) / 10) * 10);
    yMax = Math.ceil((yMax + yRange * 0.1) / 10) * 10;

    const xMax = Math.max(trajectory.length - 1, 1);

    const x = d3.scaleLinear()
      .domain([0, xMax])
      .range([MARGIN.left, WIDTH - MARGIN.right]);

    const y = d3.scaleLinear()
      .domain([yMin, yMax])
      .range([HEIGHT - MARGIN.bottom, MARGIN.top]);

    // Grid lines
    const yStep = niceStep(yMin, yMax, 5);
    const yTicks: number[] = [];
    for (let v = Math.ceil(yMin / yStep) * yStep; v <= yMax; v += yStep) {
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

    // Percentile bands
    const scorePoints = trajectory.filter(p => p.event === 'score' && p.percentiles);
    if (scorePoints.length >= 2) {
      drawBand(bandG, scorePoints, 'p10', 'p90', 'rgba(59, 76, 192, 0.08)', x, y);
      drawBand(bandG, scorePoints, 'p25', 'p75', 'rgba(59, 76, 192, 0.15)', x, y);
    } else if (scorePoints.length === 1 && scorePoints[0].percentiles) {
      drawWhisker(bandG, scorePoints[0], x, y);
    }

    // Accumulated score floor (shaded area)
    const areaGen = d3.area<TrajectoryPoint>()
      .x(d => x(d.index))
      .y0(y(yMin))
      .y1(d => y(d.accumulatedScore));

    floorG.append('path')
      .datum(trajectory)
      .attr('d', areaGen)
      .attr('fill', 'rgba(44, 160, 44, 0.08)');

    // Accumulated score line
    const accLine = d3.line<TrajectoryPoint>()
      .x(d => x(d.index))
      .y(d => y(d.accumulatedScore));

    floorG.append('path')
      .datum(trajectory)
      .attr('d', accLine)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(44, 160, 44, 0.3)')
      .attr('stroke-width', 1);

    // EV trajectory line
    const evLine = d3.line<TrajectoryPoint>()
      .x(d => x(d.index))
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
      .attr('cx', d => x(d.index))
      .attr('cy', d => y(d.expectedFinal))
      .attr('r', d => d.event === 'score' ? 4 : 3)
      .attr('fill', d => EVENT_COLORS[d.event] || COLORS.text)
      .attr('stroke', COLORS.bg)
      .attr('stroke-width', 1);

    // X-axis: turn labels at score/start events
    const xLabels = trajectory.filter(p => p.event === 'score' || p.event === 'start');
    xAxisG.selectAll('text')
      .data(xLabels)
      .join('text')
      .attr('x', d => x(d.index))
      .attr('y', HEIGHT - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', COLORS.textMuted)
      .attr('font-size', 10)
      .attr('font-family', 'monospace')
      .text(d => String(d.turn));

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
  }

  function drawEmptyGrid() {
    const yStep = niceStep(100, 350, 5);
    const y = d3.scaleLinear().domain([100, 350]).range([HEIGHT - MARGIN.bottom, MARGIN.top]);
    const ticks: number[] = [];
    for (let v = Math.ceil(100 / yStep) * yStep; v <= 350; v += yStep) {
      ticks.push(v);
    }
    gridG.selectAll('line')
      .data(ticks)
      .join('line')
      .attr('x1', MARGIN.left)
      .attr('x2', WIDTH - MARGIN.right)
      .attr('y1', d => y(d))
      .attr('y2', d => y(d))
      .attr('stroke', COLORS.borderPanel)
      .attr('stroke-width', 0.5);
  }

  render();
  subscribe(render);
}

function drawBand(
  g: d3.Selection<SVGGElement, unknown, null, undefined>,
  points: TrajectoryPoint[],
  lowKey: string,
  highKey: string,
  color: string,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  const valid = points.filter(
    p => p.percentiles && p.percentiles[lowKey] !== undefined && p.percentiles[highKey] !== undefined,
  );
  if (valid.length < 2) return;

  const area = d3.area<TrajectoryPoint>()
    .x(d => x(d.index))
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
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  if (!p.percentiles) return;
  const px = x(p.index);
  for (const [lowKey, highKey, alpha] of [
    ['p10', 'p90', 0.15],
    ['p25', 'p75', 0.3],
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
