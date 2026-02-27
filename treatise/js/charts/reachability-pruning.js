import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initReachabilityPruning() {
  const data = await DataLoader.reachability();
  const container = document.getElementById('chart-reachability-pruning');
  if (!container) return;

  const rows = (data.by_popcount || data).map(d => ({
    ...d,
    popcount: d.popcount ?? d.categories_scored,
  }));
  if (!rows || rows.length === 0) return;

  const chart = createChart('chart-reachability-pruning', {
    aspectRatio: 0.55,
    marginLeft: 70,
    marginBottom: 50,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const popcounts = rows.map(d => d.popcount);
  const x = d3.scaleBand()
    .domain(popcounts)
    .range([0, width])
    .padding(0.2);

  const barWidth = x.bandwidth() / 2;

  const yMax = d3.max(rows, d => d.total) * 1.1;
  const y = d3.scaleLinear().domain([0, yMax]).range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const tt = tooltip(container);

  // Total bars (gray)
  g.selectAll('.bar-total')
    .data(rows)
    .join('rect')
    .attr('class', 'bar-total')
    .attr('x', d => x(d.popcount))
    .attr('y', d => y(d.total))
    .attr('width', barWidth)
    .attr('height', d => height - y(d.total))
    .attr('fill', '#999')
    .attr('opacity', 0.5)
    .attr('rx', 2)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">Popcount ${d.popcount}</div>
         <div>Total: <span class="tt-value">${d.total.toLocaleString()}</span></div>
         <div>Reachable: <span class="tt-value">${d.reachable.toLocaleString()}</span></div>
         <div>Pruned: ${((1 - d.reachable / d.total) * 100).toFixed(1)}%</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Reachable bars (orange)
  g.selectAll('.bar-reach')
    .data(rows)
    .join('rect')
    .attr('class', 'bar-reach')
    .attr('x', d => x(d.popcount) + barWidth)
    .attr('y', d => y(d.reachable))
    .attr('width', barWidth)
    .attr('height', d => height - y(d.reachable))
    .attr('fill', COLORS.accent)
    .attr('opacity', 0.85)
    .attr('rx', 2)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">Popcount ${d.popcount}</div>
         <div>Total: <span class="tt-value">${d.total.toLocaleString()}</span></div>
         <div>Reachable: <span class="tt-value">${d.reachable.toLocaleString()}</span></div>
         <div>Pruned: ${((1 - d.reachable / d.total) * 100).toFixed(1)}%</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // 31.8% pruning annotation
  g.append('text')
    .attr('x', width / 2)
    .attr('y', -6)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '600')
    .text('31.8% of states pruned as unreachable');

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    d3.scaleLinear().domain([0, 15]).range([0, width]),
    'bottom', 'Categories Scored'
  );
  drawAxis(g, y, 'left', 'State Count');

  // Legend
  const legend = g.append('g').attr('transform', `translate(${width - 140}, 10)`);
  legend.append('rect').attr('width', 12).attr('height', 12).attr('fill', '#999').attr('opacity', 0.5);
  legend.append('text').attr('x', 16).attr('y', 10).attr('fill', getMutedColor()).style('font-size', '11px').text('Total states');
  legend.append('rect').attr('y', 18).attr('width', 12).attr('height', 12).attr('fill', COLORS.accent).attr('opacity', 0.85);
  legend.append('text').attr('x', 16).attr('y', 28).attr('fill', getMutedColor()).style('font-size', '11px').text('Reachable states');
}
