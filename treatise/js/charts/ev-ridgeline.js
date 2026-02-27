import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initEvRidgeline() {
  const raw = await DataLoader.evFunnel();
  const container = document.getElementById('chart-ev-ridgeline');
  if (!container) return;

  // Group by turn
  const turns = [...new Set(raw.map(d => d.turn))].sort((a, b) => a - b);
  const byTurn = new Map();
  turns.forEach(t => {
    const entries = raw.filter(d => d.turn === t).sort((a, b) => a.ev_bin - b.ev_bin);
    byTurn.set(t, entries);
  });

  const numTurns = turns.length;
  const rowOverlap = 0.2;

  const chart = createChart('chart-ev-ridgeline', {
    aspectRatio: 0.9,
    marginTop: 20,
    marginRight: 30,
    marginBottom: 45,
    marginLeft: 55,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // X scale: shared EV range
  const allBins = raw.map(d => d.ev_bin);
  const xDomain = [d3.min(allBins) - 5, d3.max(allBins) + 5];
  const x = d3.scaleLinear().domain(xDomain).range([0, width]);

  // Row layout
  const effectiveRows = numTurns - 1 + (1 - rowOverlap);
  const rowHeight = height / effectiveRows;
  const curveHeight = rowHeight * 1.8; // allow curves to extend upward

  // Global max density for normalization
  const globalMaxMass = d3.max(raw, d => d.mass);

  // Color: from light slate (turn 0) to accent orange (turn 15)
  const isDark = document.documentElement.classList.contains('dark');
  const bgColor = isDark ? '#1a1a1a' : '#faf9f6';
  const turnColor = d3.scaleLinear()
    .domain([0, numTurns - 1])
    .range([isDark ? '#5a6a7a' : '#b0bec5', COLORS.accent])
    .interpolate(d3.interpolateRgb);

  const tt = tooltip(container);

  // Render from turn 0 (back) to turn 15 (front) so later turns occlude earlier
  turns.forEach((turn, i) => {
    const entries = byTurn.get(turn);
    const baseline = i * rowHeight * (1 - rowOverlap);

    // Local y scale for this turn's density
    const yLocal = d3.scaleLinear()
      .domain([0, globalMaxMass])
      .range([0, curveHeight]);

    const color = turnColor(i);
    const strokeColor = d3.color(color).darker(0.4).toString();

    // Build area path
    const areaGen = d3.area()
      .x(d => x(d.ev_bin))
      .y0(height - baseline)
      .y1(d => height - baseline - yLocal(d.mass))
      .curve(d3.curveBasis);

    const lineGen = d3.line()
      .x(d => x(d.ev_bin))
      .y(d => height - baseline - yLocal(d.mass))
      .curve(d3.curveBasis);

    // Fill — use solid background color to occlude
    g.append('path')
      .datum(entries)
      .attr('d', areaGen)
      .attr('fill', bgColor)
      .attr('stroke', 'none');

    // Colored fill on top
    g.append('path')
      .datum(entries)
      .attr('d', areaGen)
      .attr('fill', color)
      .attr('opacity', 0.7);

    // Stroke
    g.append('path')
      .datum(entries)
      .attr('d', lineGen)
      .attr('fill', 'none')
      .attr('stroke', strokeColor)
      .attr('stroke-width', 1.2);

    // Turn label
    g.append('text')
      .attr('x', -8)
      .attr('y', height - baseline - 2)
      .attr('text-anchor', 'end')
      .attr('fill', getMutedColor())
      .style('font-size', '10px')
      .text(turn);
  });

  // Hover overlay
  const overlay = g.append('rect')
    .attr('width', width).attr('height', height)
    .attr('fill', 'none')
    .attr('pointer-events', 'all');

  overlay
    .on('mousemove', (event) => {
      const [mx, my] = d3.pointer(event);
      const evVal = x.invert(mx);

      // Find which turn row the mouse is in
      const yFromBottom = height - my;
      const rowIdx = Math.round(yFromBottom / (rowHeight * (1 - rowOverlap)));
      const turn = Math.max(0, Math.min(numTurns - 1, rowIdx));
      const entries = byTurn.get(turns[turn]);

      if (entries && entries.length > 0) {
        const evRange = `${entries[0].ev_bin}–${entries[entries.length - 1].ev_bin}`;
        tt.show(
          `<div class="tt-label">Turn ${turns[turn]}</div>
           <div>EV range: <span class="tt-value">${evRange}</span></div>`,
          event
        );
      }
    })
    .on('mouseleave', () => tt.hide());

  // X axis
  const xAxisG = g.append('g').attr('transform', `translate(0,${height})`);
  const xAxis = d3.axisBottom(x).ticks(10);
  xAxisG.call(xAxis);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");

  xAxisG.append('text')
    .attr('x', width / 2)
    .attr('y', 35)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text('Expected final score');

  // Y axis label
  g.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -40)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text('Turn');
}
