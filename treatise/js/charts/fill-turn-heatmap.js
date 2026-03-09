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
  const matrices = data.matrices;
  const nRows = categories.length;
  const nCols = turns.length;

  // Compute aspect ratio so cells are square
  const mLeft = 120, mRight = 40, mTop = 40, mBottom = 55;
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

  // Move caption below SVG
  const caption = container.querySelector('.chart-caption');
  if (caption) container.appendChild(caption);

  // Two independent toggle groups
  const controls = document.createElement('div');
  controls.className = 'chart-controls';
  container.insertBefore(controls, container.querySelector('svg'));

  let bonusKey = 'no_bonus';
  let scoreKey = 'zeroed';

  function matrixKey() { return `${bonusKey}__${scoreKey}`; }

  // Bonus group
  const bonusBtns = [
    { key: 'no_bonus', label: 'No Bonus' },
    { key: 'bonus', label: 'Bonus' },
    { key: 'all', label: 'Both' },
  ].map(({ key, label }) => {
    const btn = document.createElement('button');
    btn.className = 'chart-btn' + (key === bonusKey ? ' active' : '');
    btn.textContent = label;
    btn.addEventListener('click', () => {
      bonusKey = key;
      bonusBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateCells();
    });
    controls.appendChild(btn);
    return btn;
  });

  // Separator
  const sep = document.createElement('span');
  sep.style.cssText = 'width:1px;height:20px;background:var(--border);margin:0 0.25rem;';
  controls.appendChild(sep);

  // Score group
  const scoreBtns = [
    { key: 'zeroed', label: 'Zeroed' },
    { key: 'scored', label: 'Scored' },
    { key: 'all', label: 'Both' },
  ].map(({ key, label }) => {
    const btn = document.createElement('button');
    btn.className = 'chart-btn' + (key === scoreKey ? ' active' : '');
    btn.textContent = label;
    btn.addEventListener('click', () => {
      scoreKey = key;
      scoreBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateCells();
    });
    controls.appendChild(btn);
    return btn;
  });

  const x = d3.scaleBand().domain(turns).range([0, width]).padding(0.05);
  const y = d3.scaleBand().domain(categories).range([0, height]).padding(0.05);
  // Sqrt scale: spreads out low values so they aren't all washed out
  const color = d3.scalePow().exponent(0.5).domain([0, 1]).range(['#f5f5f5', COLORS.accent]);

  // Build cell data (values will be updated)
  const initMatrix = matrices[matrixKey()];
  const cells = [];
  for (let i = 0; i < categories.length; i++) {
    for (let j = 0; j < turns.length; j++) {
      cells.push({ cat: categories[i], turn: turns[j], i, j, val: initMatrix[i][j] });
    }
  }

  const tt = tooltip(container);

  const rects = g.selectAll('rect.fill-cell')
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

  function updateCells() {
    const m = matrices[matrixKey()];
    cells.forEach(c => { c.val = m[c.i][c.j]; });
    rects.transition().duration(300)
      .attr('fill', d => color(d.val));
  }

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
  const xAxisFn = d3.axisBottom(x).tickSize(0);
  xAxisG.call(xAxisFn);
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

  // Color legend (in bottom margin, right of axis label)
  const legendW = Math.min(width * 0.3, 120);
  const legendH = 8;
  const legendX = width - legendW;
  const legendY = height + 28;

  const defs = g.append('defs');
  const gradId = 'fill-turn-gradient';
  const grad = defs.append('linearGradient').attr('id', gradId);
  // Approximate sqrt curve with gradient stops
  [0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.5, 1.0].forEach(v => {
    const pct = Math.sqrt(v) * 100;
    grad.append('stop').attr('offset', `${pct}%`).attr('stop-color', color(v));
  });

  g.append('rect')
    .attr('x', legendX).attr('y', legendY)
    .attr('width', legendW).attr('height', legendH)
    .attr('fill', `url(#${gradId})`)
    .attr('rx', 2);

  g.append('text').attr('x', legendX).attr('y', legendY + legendH + 10)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px').text('0%');
  g.append('text').attr('x', legendX + legendW).attr('y', legendY + legendH + 10)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '9px')
    .text('100%');
}
