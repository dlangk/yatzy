import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initCategoryLandscape() {
  const data = await DataLoader.categoryLandscape();
  const container = document.getElementById('chart-category-landscape');
  if (!container) return;

  const chart = createChart('chart-category-landscape', {
    aspectRatio: 0.65,
    marginTop: 20,
    marginRight: 30,
    marginBottom: 50,
    marginLeft: 60,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Scales
  const x = d3.scaleLinear()
    .domain([1, 15])
    .range([0, width]);

  const y = d3.scaleLinear()
    .domain([0, 1])
    .range([height, 0]);

  const maxVar = d3.max(data, d => d.variance_contribution);
  const r = d3.scaleSqrt()
    .domain([0, maxVar])
    .range([4, 32]);

  const upperColor = '#4e79a7';
  const lowerColor = '#e15759';

  // Grid
  g.append('g').selectAll('line.hgrid')
    .data(y.ticks(5))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  g.append('g').selectAll('line.vgrid')
    .data(x.ticks(15))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Bubbles
  const tt = tooltip(container);

  const bubbles = g.selectAll('circle.bubble')
    .data(data)
    .join('circle')
    .attr('class', 'bubble')
    .attr('cx', d => x(d.mean_fill_turn))
    .attr('cy', d => y(d.score_pct_ceiling))
    .attr('r', d => r(d.variance_contribution))
    .attr('fill', d => d.section === 'upper' ? upperColor : lowerColor)
    .attr('opacity', 0.6)
    .attr('stroke', d => d.section === 'upper' ? upperColor : lowerColor)
    .attr('stroke-width', 1.5);

  // Labels
  g.selectAll('text.cat-label')
    .data(data)
    .join('text')
    .attr('class', 'cat-label')
    .attr('x', d => x(d.mean_fill_turn))
    .attr('y', d => y(d.score_pct_ceiling) - r(d.variance_contribution) - 4)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', d => d.name === 'Yatzy' ? '10px' : '9px')
    .style('font-weight', d => d.name === 'Yatzy' ? '600' : '400')
    .text(d => d.name);

  // Interaction
  bubbles
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.name}</div>
         <div>Mean score: <span class="tt-value">${d.mean_score.toFixed(1)}</span> / ${d.ceiling}</div>
         <div>Score % of ceiling: <span class="tt-value">${(d.score_pct_ceiling * 100).toFixed(0)}%</span></div>
         <div>Mean fill turn: <span class="tt-value">${d.mean_fill_turn.toFixed(1)}</span></div>
         <div>Zero rate: <span class="tt-value">${(d.zero_rate * 100).toFixed(1)}%</span></div>
         <div>Variance contribution: <span class="tt-value">${d.variance_contribution.toFixed(1)}</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Axes
  drawAxis(g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Mean fill turn', { tickCount: 15, tickFormat: d3.format('d') });
  drawAxis(g, y, 'left', 'Score as % of ceiling', { tickFormat: d3.format('.0%') });

  // Legend
  const legendG = g.append('g')
    .attr('transform', `translate(${width - 140}, 10)`);

  [{ label: 'Upper section', color: upperColor }, { label: 'Lower section', color: lowerColor }].forEach((item, i) => {
    legendG.append('circle')
      .attr('cx', 6).attr('cy', i * 18)
      .attr('r', 5)
      .attr('fill', item.color)
      .attr('opacity', 0.6);
    legendG.append('text')
      .attr('x', 16).attr('y', i * 18 + 4)
      .attr('fill', getMutedColor())
      .style('font-size', '10px')
      .text(item.label);
  });

  // Bubble size legend
  const sizeLegG = g.append('g')
    .attr('transform', `translate(${width - 140}, 50)`);

  sizeLegG.append('text')
    .attr('x', 0).attr('y', 0)
    .attr('fill', getMutedColor())
    .style('font-size', '9px')
    .text('Bubble size = variance contribution');
}
