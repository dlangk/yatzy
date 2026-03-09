import { DataLoader } from '../data-loader.js';
import {
  createChart,
  getMutedColor, getGridColor, COLORS,
  tooltip,
} from '../yatzy-viz.js';

export async function initBonusDependency() {
  const raw = await DataLoader.categoryLandscape();
  const container = document.getElementById('chart-bonus-dependency');
  if (!container) return;

  // Sort by bonus_dependency descending (largest positive gap at top)
  const data = [...raw].sort((a, b) => b.bonus_dependency - a.bonus_dependency);

  const chart = createChart('chart-bonus-dependency', {
    aspectRatio: 0.8,
    marginTop: 20,
    marginRight: 30,
    marginBottom: 45,
    marginLeft: 120,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Move caption below SVG (must be after createChart which appends the SVG)
  const caption = container.querySelector('.chart-caption');
  if (caption) container.appendChild(caption);

  const names = data.map(d => d.name);

  // X domain: span all conditional means with padding
  const allVals = data.flatMap(d => [d.mean_score_bonus, d.mean_score_no_bonus]);
  const xMin = d3.min(allVals);
  const xMax = d3.max(allVals);
  const xPad = (xMax - xMin) * 0.1;

  const y = d3.scaleBand().domain(names).range([0, height]).padding(0.3);
  const x = d3.scaleLinear().domain([Math.max(0, xMin - xPad), xMax + xPad]).range([0, width]);

  const noBonusColor = '#4e79a7';
  const bonusColor = COLORS.accent;

  // Tooltip
  const tt = tooltip(container);

  // Grid lines
  g.append('g').selectAll('line')
    .data(x.ticks(6))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Connecting lines
  g.selectAll('line.connector')
    .data(data)
    .join('line')
    .attr('class', 'connector')
    .attr('x1', d => x(d.mean_score_no_bonus))
    .attr('x2', d => x(d.mean_score_bonus))
    .attr('y1', d => y(d.name) + y.bandwidth() / 2)
    .attr('y2', d => y(d.name) + y.bandwidth() / 2)
    .attr('stroke', getGridColor())
    .attr('stroke-width', 2);

  // No-bonus dots (blue)
  g.selectAll('circle.no-bonus')
    .data(data)
    .join('circle')
    .attr('class', 'no-bonus')
    .attr('cx', d => x(d.mean_score_no_bonus))
    .attr('cy', d => y(d.name) + y.bandwidth() / 2)
    .attr('r', 5)
    .attr('fill', noBonusColor)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.name}</div>
         <div>No bonus: <span class="tt-value">${d.mean_score_no_bonus.toFixed(1)}</span></div>
         <div>Bonus: <span class="tt-value">${d.mean_score_bonus.toFixed(1)}</span></div>
         <div>Gap: <span class="tt-value">${d.bonus_dependency > 0 ? '+' : ''}${d.bonus_dependency.toFixed(2)}</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Bonus dots (accent/orange)
  g.selectAll('circle.bonus')
    .data(data)
    .join('circle')
    .attr('class', 'bonus')
    .attr('cx', d => x(d.mean_score_bonus))
    .attr('cy', d => y(d.name) + y.bandwidth() / 2)
    .attr('r', 5)
    .attr('fill', bonusColor)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.name}</div>
         <div>No bonus: <span class="tt-value">${d.mean_score_no_bonus.toFixed(1)}</span></div>
         <div>Bonus: <span class="tt-value">${d.mean_score_bonus.toFixed(1)}</span></div>
         <div>Gap: <span class="tt-value">${d.bonus_dependency > 0 ? '+' : ''}${d.bonus_dependency.toFixed(2)}</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Y axis: category names
  g.selectAll('text.cat-label')
    .data(names)
    .join('text')
    .attr('class', 'cat-label')
    .attr('x', -8)
    .attr('y', d => y(d) + y.bandwidth() / 2)
    .attr('text-anchor', 'end')
    .attr('dominant-baseline', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text(d => d);

  // X axis
  const xAxisG = g.append('g')
    .attr('transform', `translate(0,${height})`);
  const xAxisFn = d3.axisBottom(x).ticks(6);
  xAxisG.call(xAxisFn);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '10px');

  // X axis label
  g.append('text')
    .attr('x', width / 2)
    .attr('y', height + 35)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '12px')
    .text('Mean category score');

  // Legend (top right)
  const legendG = g.append('g').attr('transform', `translate(${width - 140}, 0)`);
  // No bonus
  legendG.append('circle').attr('cx', 0).attr('cy', 0).attr('r', 5).attr('fill', noBonusColor);
  legendG.append('text').attr('x', 10).attr('y', 4)
    .attr('fill', getMutedColor()).style('font-size', '10px').text('No bonus');
  // Bonus
  legendG.append('circle').attr('cx', 75).attr('cy', 0).attr('r', 5).attr('fill', bonusColor);
  legendG.append('text').attr('x', 85).attr('y', 4)
    .attr('fill', getMutedColor()).style('font-size', '10px').text('Bonus');
}
