import { DataLoader } from '../data-loader.js';
import {
  createChart,
  getMutedColor,
  tooltip,
} from '../yatzy-viz.js';

export async function initCategoryCorrelations() {
  const data = await DataLoader.categoryCorrelations();
  const container = document.getElementById('chart-category-correlations');
  if (!container) return;

  const categories = data.categories;
  const matrix = data.matrix;
  const n = categories.length;

  // Compute aspect ratio so cells are square
  const mLeft = 120, mRight = 20, mTop = 100, mBottom = 40;
  const containerWidth = container.clientWidth || 635;
  const innerW = containerWidth - mLeft - mRight;
  const totalH = innerW + mTop + mBottom;  // square inner area (n x n)
  const aspectRatio = totalH / containerWidth;

  const chart = createChart('chart-category-correlations', {
    aspectRatio,
    marginTop: mTop,
    marginRight: mRight,
    marginBottom: mBottom,
    marginLeft: mLeft,
  });
  if (!chart) return;

  // Move caption below SVG (must be after createChart which appends the SVG)
  const caption = container.querySelector('.chart-caption');
  if (caption) container.appendChild(caption);
  const { g, width, height } = chart;

  const x = d3.scaleBand().domain(categories).range([0, width]).padding(0.05);
  const y = d3.scaleBand().domain(categories).range([0, height]).padding(0.05);

  // Compute off-diagonal extremes to clamp color scale
  let offMin = 0, offMax = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      if (matrix[i][j] < offMin) offMin = matrix[i][j];
      if (matrix[i][j] > offMax) offMax = matrix[i][j];
    }
  }
  const absMax = Math.max(Math.abs(offMin), Math.abs(offMax));
  // Symmetric domain around 0, clamped to off-diagonal range
  const color = d3.scaleLinear().domain([-absMax, 0, absMax]).range(['#4e79a7', '#f5f5f5', '#e15759']);
  const diagColor = '#e0e0e0';

  // Flatten matrix to cell objects
  const cells = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      cells.push({ i, j, val: matrix[i][j], cat1: categories[i], cat2: categories[j], diag: i === j });
    }
  }

  // Tooltip
  const tt = tooltip(container);

  // Draw cells
  g.selectAll('rect.corr-cell')
    .data(cells)
    .join('rect')
    .attr('class', 'corr-cell')
    .attr('x', d => x(d.cat2))
    .attr('y', d => y(d.cat1))
    .attr('width', x.bandwidth())
    .attr('height', y.bandwidth())
    .attr('fill', d => d.diag ? diagColor : color(d.val))
    .attr('stroke', 'none')
    .on('mousemove', (event, d) => {
      const label = d.diag
        ? `<div class="tt-label">${d.cat1}</div><div>(self-correlation)</div>`
        : `<div class="tt-label">${d.cat1} × ${d.cat2}</div>
           <div>r = <span class="tt-value">${d.val.toFixed(3)}</span></div>`;
      tt.show(label, event);
    })
    .on('mouseleave', () => tt.hide());

  // Row labels (left)
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

  // Column labels (top, vertical)
  g.selectAll('text.col-label')
    .data(categories)
    .join('text')
    .attr('class', 'col-label')
    .attr('text-anchor', 'start')
    .attr('dominant-baseline', 'central')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .attr('transform', d => {
      const cx = x(d) + x.bandwidth() / 2;
      return `translate(${cx}, -10) rotate(-90)`;
    })
    .text(d => d);

  // Color legend: horizontal gradient bar below the matrix
  const legendW = Math.min(width * 0.6, 200);
  const legendH = 10;
  const legendX = (width - legendW) / 2;
  const legendY = height + 4;

  // Gradient
  const defs = g.append('defs');
  const gradId = 'corr-gradient';
  const grad = defs.append('linearGradient').attr('id', gradId);
  grad.append('stop').attr('offset', '0%').attr('stop-color', '#4e79a7');
  grad.append('stop').attr('offset', '50%').attr('stop-color', '#f5f5f5');
  grad.append('stop').attr('offset', '100%').attr('stop-color', '#e15759');

  g.append('rect')
    .attr('x', legendX).attr('y', legendY)
    .attr('width', legendW).attr('height', legendH)
    .attr('fill', `url(#${gradId})`)
    .attr('rx', 2);

  // Legend labels (show actual off-diagonal range)
  const fmtR = v => (v > 0 ? '+' : '') + v.toFixed(2);
  g.append('text').attr('x', legendX).attr('y', legendY + legendH + 12)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px').text(fmtR(-absMax));
  g.append('text').attr('x', legendX + legendW / 2).attr('y', legendY + legendH + 12)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px').text('0');
  g.append('text').attr('x', legendX + legendW).attr('y', legendY + legendH + 12)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px').text(fmtR(absMax));
}
