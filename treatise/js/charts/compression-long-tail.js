import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initCompressionLongTail() {
  const data = await DataLoader.heuristicGap();
  const container = document.getElementById('chart-compression-long-tail');
  if (!container) return;

  const patterns = data.top_patterns.slice(0, 12);
  const dtypeColors = {
    category: COLORS.category,
    reroll1: COLORS.reroll1,
    reroll2: COLORS.reroll2,
  };

  function shortLabel(pattern) {
    return pattern
      .replace('reroll1:', 'R1: ')
      .replace('reroll2:', 'R2: ')
      .replace('cat:', 'Cat: ')
      .replace('wrong_keep', 'wrong keep')
      .replace('wasted_upper', 'wasted upper')
      .replace('missed_upper', 'missed upper')
      .replace('should_keep_all', 'should keep all');
  }

  const chart = createChart('chart-compression-long-tail-svg', {
    aspectRatio: 0.7,
    marginLeft: 200,
    marginRight: 60,
    marginBottom: 40,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain([0, d3.max(patterns, d => d.ev_per_game) * 1.1])
    .range([0, width]);

  const y = d3.scaleBand()
    .domain(patterns.map((_, i) => i))
    .range([0, height])
    .padding(0.25);

  // Grid
  g.append('g').selectAll('line')
    .data(x.ticks(5))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Bars
  const tt = tooltip(container);

  g.selectAll('.bar')
    .data(patterns)
    .join('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', (_, i) => y(i))
    .attr('width', d => x(d.ev_per_game))
    .attr('height', y.bandwidth())
    .attr('fill', d => dtypeColors[d.decision_type] || COLORS.accent)
    .attr('opacity', 0.8)
    .attr('rx', 3)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.pattern}</div>
         <div>EV loss/game: <span class="tt-value">${d.ev_per_game.toFixed(2)}</span></div>
         <div>Occurrences/game: ${(d.count / 100000).toFixed(1)}</div>
         <div>Max EV gap: ${d.max_ev_gap.toFixed(1)}</div>
         <div>Example: ${d.example_dice}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Value labels
  g.selectAll('.val-label')
    .data(patterns)
    .join('text')
    .attr('x', d => x(d.ev_per_game) + 4)
    .attr('y', (_, i) => y(i) + y.bandwidth() / 2 + 4)
    .attr('fill', getTextColor())
    .style('font-size', '11px')
    .style('font-weight', '600')
    .text(d => d.ev_per_game.toFixed(1));

  // Y-axis labels
  g.selectAll('.bar-label')
    .data(patterns)
    .join('text')
    .attr('x', -6)
    .attr('y', (_, i) => y(i) + y.bandwidth() / 2 + 4)
    .attr('fill', getTextColor())
    .attr('text-anchor', 'end')
    .style('font-size', '11px')
    .text(d => shortLabel(d.pattern));

  // Cumulative line
  let cumSum = 0;
  const cumData = patterns.map((d, i) => {
    cumSum += d.ev_per_game;
    return { i, cum: cumSum };
  });

  const x2 = d3.scaleLinear()
    .domain([0, cumData[cumData.length - 1].cum])
    .range([0, width]);

  const cumLine = d3.line()
    .x(d => x2(d.cum))
    .y((d) => y(d.i) + y.bandwidth() / 2)
    .curve(d3.curveMonotoneY);

  g.append('path')
    .datum(cumData)
    .attr('d', cumLine)
    .attr('fill', 'none')
    .attr('stroke', getTextColor())
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3')
    .attr('opacity', 0.5);

  // Bottom axis
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'EV loss per game (points)'
  );

  // Legend
  const legend = container.querySelector('.chart-legend');
  if (legend) {
    legend.innerHTML = Object.entries(dtypeColors).map(([k, c]) =>
      `<div class="chart-legend-item">
        <span class="legend-swatch" style="background:${c}"></span>
        <span>${k === 'reroll1' ? '1st reroll' : k === 'reroll2' ? '2nd reroll' : 'Category'}</span>
      </div>`
    ).join('');
  }
}
