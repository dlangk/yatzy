import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initFilterGrammar() {
  const data = await DataLoader.filterGrammar();
  const container = document.getElementById('chart-filter-grammar');
  if (!container) return;

  const actions = data.actions || data;
  if (!actions || actions.length === 0) return;

  // Normalize field names (JSON uses reroll1_rules, reroll2_rules)
  const normalized = actions.map(a => ({
    ...a,
    action: a.action || a.name,
    reroll1: a.reroll1 ?? a.reroll1_rules ?? 0,
    reroll2: a.reroll2 ?? a.reroll2_rules ?? 0,
  }));

  // Sort by total rules descending
  const sorted = [...normalized].sort((a, b) =>
    (b.reroll1 + b.reroll2) - (a.reroll1 + a.reroll2)
  );

  const chart = createChart('chart-filter-grammar', {
    aspectRatio: 0.65,
    marginLeft: 130,
    marginBottom: 50,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain([0, d3.max(sorted, d => d.reroll1 + d.reroll2) * 1.15])
    .range([0, width]);

  const y = d3.scaleBand()
    .domain(sorted.map(d => d.action))
    .range([0, height])
    .padding(0.25);

  const barH = y.bandwidth() / 2;

  // Grid
  g.append('g').selectAll('line')
    .data(x.ticks(5))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const tt = tooltip(container);

  // Reroll1 bars (green)
  g.selectAll('.bar-r1')
    .data(sorted)
    .join('rect')
    .attr('class', 'bar-r1')
    .attr('x', 0)
    .attr('y', d => y(d.action))
    .attr('width', d => x(d.reroll1))
    .attr('height', barH)
    .attr('fill', COLORS.reroll1)
    .attr('opacity', 0.8)
    .attr('rx', 2)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.action}</div>
         <div>Reroll 1 rules: <span class="tt-value">${d.reroll1}</span></div>
         <div>Reroll 2 rules: <span class="tt-value">${d.reroll2}</span></div>
         <div>Total: ${d.reroll1 + d.reroll2}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Reroll2 bars (purple)
  g.selectAll('.bar-r2')
    .data(sorted)
    .join('rect')
    .attr('class', 'bar-r2')
    .attr('x', 0)
    .attr('y', d => y(d.action) + barH)
    .attr('width', d => x(d.reroll2))
    .attr('height', barH)
    .attr('fill', COLORS.reroll2)
    .attr('opacity', 0.8)
    .attr('rx', 2)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.action}</div>
         <div>Reroll 1 rules: <span class="tt-value">${d.reroll1}</span></div>
         <div>Reroll 2 rules: <span class="tt-value">${d.reroll2}</span></div>
         <div>Total: ${d.reroll1 + d.reroll2}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Y-axis labels
  g.selectAll('.action-label')
    .data(sorted)
    .join('text')
    .attr('x', -6)
    .attr('y', d => y(d.action) + y.bandwidth() / 2 + 4)
    .attr('fill', getTextColor())
    .attr('text-anchor', 'end')
    .style('font-size', '10px')
    .text(d => d.action);

  // EV improvement annotation
  g.append('text')
    .attr('x', width / 2)
    .attr('y', -6)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '600')
    .text('+15.7 EV improvement over greedy');

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Number of rules'
  );

  // Legend
  const legend = g.append('g').attr('transform', `translate(${width - 120}, ${height - 36})`);
  legend.append('rect').attr('width', 10).attr('height', 10).attr('fill', COLORS.reroll1).attr('rx', 2);
  legend.append('text').attr('x', 14).attr('y', 9).attr('fill', getMutedColor()).style('font-size', '10px').text('Reroll 1');
  legend.append('rect').attr('y', 14).attr('width', 10).attr('height', 10).attr('fill', COLORS.reroll2).attr('rx', 2);
  legend.append('text').attr('x', 14).attr('y', 23).attr('fill', getMutedColor()).style('font-size', '10px').text('Reroll 2');
}
