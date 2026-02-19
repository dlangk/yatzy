import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor,
} from '../yatzy-viz.js';

export async function initStateHeatmap() {
  const data = await DataLoader.stateHeatmap();
  const container = document.getElementById('chart-state-heatmap');
  if (!container) return;

  const rows = data.states;

  // Build lookup: [num_scored][upper_score] -> ev
  const grid = [];
  for (let ns = 0; ns <= 15; ns++) {
    grid[ns] = new Array(64).fill(null);
  }
  let evMin = Infinity, evMax = -Infinity;
  for (const r of rows) {
    grid[r.num_scored][r.upper_score] = r.ev;
    if (r.ev < evMin) evMin = r.ev;
    if (r.ev > evMax) evMax = r.ev;
  }

  render();

  function render() {
    const chart = createChart('chart-state-heatmap-svg', {
      aspectRatio: 0.5,
      marginLeft: 65,
      marginBottom: 50,
      marginTop: 20,
      marginRight: 50,
    });
    if (!chart) return;
    const { g, width, height } = chart;

    const numRows = 16; // 0..15 scored categories
    const numCols = 64; // 0..63 upper score
    const cellW = width / numCols;
    const cellH = height / numRows;

    // Color scale — coolwarm palette
    const colorScale = d3.scaleLinear()
      .domain([evMin, evMin + (evMax - evMin) * 0.25, evMin + (evMax - evMin) * 0.5, evMin + (evMax - evMin) * 0.75, evMax])
      .range(['#3b4cc0', '#8db0fe', '#F37021', '#f4987a', '#b40426'])
      .interpolate(d3.interpolateRgb);

    // Draw cells
    for (let ns = 0; ns <= 15; ns++) {
      for (let up = 0; up < 64; up++) {
        const ev = grid[ns][up];
        if (ev === null) continue;
        g.append('rect')
          .attr('x', up * cellW)
          .attr('y', ns * cellH)
          .attr('width', cellW + 0.5)
          .attr('height', cellH + 0.5)
          .attr('fill', colorScale(ev))
          .attr('class', 'heatmap-cell')
          .attr('data-ns', ns)
          .attr('data-up', up);
      }
    }

    // Bonus ridge indicator
    const ridgeX = 63 * cellW;
    g.append('line')
      .attr('x1', ridgeX + cellW)
      .attr('x2', ridgeX + cellW)
      .attr('y1', 0)
      .attr('y2', height)
      .attr('stroke', getTextColor())
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.5);

    g.append('text')
      .attr('x', ridgeX - 2)
      .attr('y', -6)
      .attr('fill', getMutedColor())
      .attr('text-anchor', 'end')
      .style('font-size', '10px')
      .text('bonus →');

    // X axis
    const xScale = d3.scaleLinear().domain([0, 63]).range([0, width]);
    const xAxisG = g.append('g').attr('transform', `translate(0,${height})`);
    drawAxis(xAxisG, xScale, 'bottom', 'Upper Section Score', { tickCount: 8 });

    // Y axis (custom — 0 at top, 15 at bottom)
    const yScale = d3.scaleLinear().domain([0, 15]).range([0, height]);
    const yAxis = d3.axisLeft(yScale).ticks(8);
    const yAxisG = g.append('g').call(yAxis);
    yAxisG.selectAll('line').attr('stroke', getGridColor());
    yAxisG.selectAll('path').attr('stroke', getGridColor());
    yAxisG.selectAll('text')
      .attr('fill', getMutedColor())
      .style('font-size', '11px')
      .style('font-family', "'Newsreader', Georgia, serif");
    yAxisG.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -40)
      .attr('fill', getMutedColor())
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Categories Scored');

    // Color legend
    const legendW = 15;
    const legendH = height;
    const legendX = width + 5;
    const defs = chart.svg.append('defs');
    const gradId = 'heatmap-grad';
    const grad = defs.append('linearGradient')
      .attr('id', gradId)
      .attr('x1', '0%').attr('y1', '100%')
      .attr('x2', '0%').attr('y2', '0%');
    const nStops = 10;
    for (let i = 0; i <= nStops; i++) {
      const t = i / nStops;
      const val = evMin + t * (evMax - evMin);
      grad.append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', colorScale(val));
    }
    const lg = g.append('g').attr('transform', `translate(${legendX}, 0)`);
    lg.append('rect')
      .attr('width', legendW)
      .attr('height', legendH)
      .attr('fill', `url(#${gradId})`);
    lg.append('text')
      .attr('x', legendW + 4)
      .attr('y', 10)
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text(evMax.toFixed(0));
    lg.append('text')
      .attr('x', legendW + 4)
      .attr('y', legendH)
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text(evMin.toFixed(0));

    // Tooltip
    const tt = tooltip(container);
    g.selectAll('.heatmap-cell')
      .on('mousemove', function(event) {
        const ns = +this.getAttribute('data-ns');
        const up = +this.getAttribute('data-up');
        const ev = grid[ns][up];
        tt.show(
          `<div class="tt-label">${ns} categories scored, upper = ${up}</div>
           <div>Expected Value: <span class="tt-value">${ev.toFixed(1)}</span></div>
           ${up >= 63 ? '<div style="color:var(--accent)">Bonus active (+50)</div>' : ''}`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());
  }
}
