import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initCategoryPmfGrid() {
  const { categories } = await DataLoader.categoryPmfs();
  const container = document.getElementById('chart-category-pmf-grid');
  if (!container) return;

  // 3 columns x 5 rows grid
  const cols = 3;
  const rows = 5;
  const cellPadding = 8;

  const containerWidth = container.clientWidth || 635;
  const cellWidth = (containerWidth - cellPadding * (cols + 1)) / cols;
  const cellHeight = 85;
  const totalHeight = rows * (cellHeight + cellPadding) + cellPadding;

  // Clear previous
  container.querySelectorAll('svg').forEach(s => s.remove());
  container.querySelectorAll('.pmf-grid').forEach(s => s.remove());

  const grid = document.createElement('div');
  grid.className = 'pmf-grid';
  grid.style.cssText = `display:grid;grid-template-columns:repeat(${cols},1fr);gap:${cellPadding}px;`;

  const tt = tooltip(container);

  // Color by type
  const typeColors = {
    binary: '#e15759',
    discrete: '#4e79a7',
    continuous: '#59a14f',
  };

  categories.forEach(cat => {
    const cell = document.createElement('div');
    cell.style.cssText = 'position:relative;';

    const svg = d3.select(cell).append('svg')
      .attr('viewBox', `0 0 ${cellWidth} ${cellHeight}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .style('width', '100%');

    const margin = { top: 18, right: 6, bottom: 16, left: 6 };
    const w = cellWidth - margin.left - margin.right;
    const h = cellHeight - margin.top - margin.bottom;
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const pmf = cat.pmf;
    const maxScore = d3.max(pmf, d => d.score);
    const maxProb = d3.max(pmf, d => d.prob);

    const x = d3.scaleLinear()
      .domain([0, maxScore])
      .range([0, w]);

    const y = d3.scaleLinear()
      .domain([0, maxProb * 1.1])
      .range([h, 0]);

    const color = typeColors[cat.type] || '#4e79a7';

    // Baseline
    g.append('line')
      .attr('x1', 0).attr('x2', w)
      .attr('y1', h).attr('y2', h)
      .attr('stroke', getGridColor())
      .attr('stroke-width', 0.5);

    if (cat.type === 'binary') {
      // Two thick bars for binary categories
      const barWidth = w * 0.15;
      pmf.forEach(d => {
        g.append('rect')
          .attr('x', x(d.score) - barWidth / 2)
          .attr('y', y(d.prob))
          .attr('width', barWidth)
          .attr('height', h - y(d.prob))
          .attr('fill', color)
          .attr('opacity', 0.7)
          .attr('rx', 1);
      });
    } else {
      // Bars for discrete categories
      const barWidth = Math.max(2, w / (pmf.length * 1.8));
      pmf.forEach(d => {
        g.append('rect')
          .attr('x', x(d.score) - barWidth / 2)
          .attr('y', y(d.prob))
          .attr('width', barWidth)
          .attr('height', h - y(d.prob))
          .attr('fill', color)
          .attr('opacity', 0.7)
          .attr('rx', 1);
      });
    }

    // Category name
    svg.append('text')
      .attr('x', cellWidth / 2)
      .attr('y', 12)
      .attr('text-anchor', 'middle')
      .attr('fill', getTextColor())
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(cat.name);

    // Score range annotation
    const mean = pmf.reduce((s, d) => s + d.score * d.prob, 0);
    svg.append('text')
      .attr('x', cellWidth / 2)
      .attr('y', cellHeight - 2)
      .attr('text-anchor', 'middle')
      .attr('fill', getMutedColor())
      .style('font-size', '8px')
      .text(`avg ${mean.toFixed(1)} / ${cat.ceiling}`);

    // Hover interaction
    svg.on('mousemove', (event) => {
      const zeroRate = pmf.find(d => d.score === 0);
      tt.show(
        `<div class="tt-label">${cat.name}</div>
         <div>Ceiling: <span class="tt-value">${cat.ceiling}</span></div>
         <div>Mean: <span class="tt-value">${mean.toFixed(1)}</span></div>
         <div>Zero rate: <span class="tt-value">${zeroRate ? (zeroRate.prob * 100).toFixed(1) + '%' : '0%'}</span></div>
         <div>Type: <span class="tt-value">${cat.type}</span></div>`,
        event
      );
    });
    svg.on('mouseleave', () => tt.hide());

    grid.appendChild(cell);
  });

  // Insert before caption
  const caption = container.querySelector('.chart-caption');
  if (caption) {
    container.insertBefore(grid, caption);
  } else {
    container.appendChild(grid);
  }
}
