import { DataLoader } from '../data-loader.js';
import {
  createChart,
  getMutedColor, COLORS,
  tooltip,
} from '../yatzy-viz.js';

export async function initFillTurnHeatmap() {
  const data = await DataLoader.fillTurnHeatmap();
  const container = document.getElementById('chart-fill-turn-heatmap');
  if (!container) return;

  const categories = data.categories;
  const turns = data.turns;
  const matrix = data.matrix;
  const nRows = categories.length;  // 16 (15 cats + bonus)
  const nCols = turns.length;       // 15

  // Find max probability for color scale
  const maxProb = d3.max(matrix.flat());

  // Compute aspect ratio so cells are square
  const mLeft = 120, mRight = 40, mTop = 20, mBottom = 45;
  const containerWidth = container.clientWidth || 635;
  const innerW = containerWidth - mLeft - mRight;
  const cellSize = innerW / nCols;
  const innerH = cellSize * nRows;
  const totalH = innerH + mTop + mBottom;
  const aspectRatio = totalH / containerWidth;

  const chart = createChart('chart-fill-turn-heatmap', {
    aspectRatio,
    marginTop: mTop,
    marginRight: mRight,
    marginBottom: mBottom,
    marginLeft: mLeft,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Move caption below SVG (must be after createChart which appends the SVG)
  const caption = container.querySelector('.chart-caption');
  if (caption) container.appendChild(caption);

  const x = d3.scaleBand().domain(turns).range([0, width]).padding(0.05);
  const y = d3.scaleBand().domain(categories).range([0, height]).padding(0.05);
  const color = d3.scaleLinear().domain([0, maxProb]).range(['#f5f5f5', COLORS.accent]);

  // Flatten to cell objects
  const cells = [];
  for (let i = 0; i < categories.length; i++) {
    for (let j = 0; j < turns.length; j++) {
      cells.push({ cat: categories[i], turn: turns[j], val: matrix[i][j] });
    }
  }

  // Tooltip
  const tt = tooltip(container);

  // Draw cells
  g.selectAll('rect.fill-cell')
    .data(cells)
    .join('rect')
    .attr('class', 'fill-cell')
    .attr('x', d => x(d.turn))
    .attr('y', d => y(d.cat))
    .attr('width', x.bandwidth())
    .attr('height', y.bandwidth())
    .attr('fill', d => color(d.val))
    .attr('stroke', 'none')
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.cat} on Turn ${d.turn}</div>
         <div>Probability: <span class="tt-value">${(d.val * 100).toFixed(1)}%</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Y axis: category names
  g.selectAll('text.row-label')
    .data(categories)
    .join('text')
    .attr('class', 'row-label')
    .attr('x', -8)
    .attr('y', d => y(d) + y.bandwidth() / 2)
    .attr('text-anchor', 'end')
    .attr('dominant-baseline', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text(d => d);

  // X axis: turn numbers
  const xAxisG = g.append('g')
    .attr('transform', `translate(0,${height})`);
  const xAxis = d3.axisBottom(x).tickSize(0);
  xAxisG.call(xAxis);
  xAxisG.select('.domain').remove();
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '10px');

  // X axis label
  g.append('text')
    .attr('x', width / 2)
    .attr('y', height + 38)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '12px')
    .text('Turn');

  // Color legend: horizontal gradient bar
  const legendW = Math.min(width * 0.4, 150);
  const legendH = 10;
  const legendX = width - legendW;
  const legendY = -12;

  const defs = g.append('defs');
  const gradId = 'fill-turn-gradient';
  const grad = defs.append('linearGradient').attr('id', gradId);
  grad.append('stop').attr('offset', '0%').attr('stop-color', '#f5f5f5');
  grad.append('stop').attr('offset', '100%').attr('stop-color', COLORS.accent);

  g.append('rect')
    .attr('x', legendX).attr('y', legendY)
    .attr('width', legendW).attr('height', legendH)
    .attr('fill', `url(#${gradId})`)
    .attr('rx', 2);

  g.append('text').attr('x', legendX).attr('y', legendY + legendH + 12)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px').text('0%');
  g.append('text').attr('x', legendX + legendW).attr('y', legendY + legendH + 12)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px')
    .text(`${(maxProb * 100).toFixed(0)}%`);
}
