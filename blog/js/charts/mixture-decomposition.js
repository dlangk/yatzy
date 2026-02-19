import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, normalPDF,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMixtureDecomposition() {
  const { populations } = await DataLoader.mixture();
  const container = document.getElementById('chart-mixture');
  if (!container) return;

  // Components indexed as: 0=No bonus/No Yatzy, 1=No bonus/Yatzy,
  //                         2=Bonus/No Yatzy, 3=Bonus+Yatzy
  // Map each component to its bonus/yatzy flags
  const compFlags = [
    { bonus: false, yatzy: false },
    { bonus: false, yatzy: true },
    { bonus: true,  yatzy: false },
    { bonus: true,  yatzy: true },
  ];

  const nPoints = 300;
  const xMin = 0;
  const xMax = 400;
  const step = (xMax - xMin) / nPoints;

  const components = populations.map((pop, i) => {
    const pts = [];
    for (let j = 0; j <= nPoints; j++) {
      const x = xMin + j * step;
      pts.push({ x, y: pop.fraction * normalPDF(x, pop.mean, pop.std) });
    }
    return { ...pop, ...compFlags[i], points: pts, color: COLORS.mixture[i] };
  });

  // Toggle state
  let activeBonus = true;   // true=Yes, false=No
  let activeYatzy = true;

  // Fixed scales matching the score distribution chart
  const xDomain = [0, 400];
  const yDomain = [0, 0.015];

  function render() {
    const chart = createChart('chart-mixture-svg', { aspectRatio: 0.5 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear().domain(xDomain).range([0, width]);
    const y = d3.scaleLinear().domain(yDomain).range([height, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(y.ticks(5))
      .join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,3');

    const line = d3.line()
      .x(d => x(d.x))
      .y(d => y(d.y))
      .curve(d3.curveBasis);

    const area = d3.area()
      .x(d => x(d.x))
      .y0(height)
      .y1(d => y(d.y))
      .curve(d3.curveBasis);

    // Determine which component is highlighted
    const highlightIdx = components.findIndex(
      c => c.bonus === activeBonus && c.yatzy === activeYatzy
    );

    // Draw inactive components first (low opacity)
    components.forEach((comp, i) => {
      if (i === highlightIdx) return;
      g.append('path')
        .datum(comp.points)
        .attr('d', area)
        .attr('fill', comp.color)
        .attr('opacity', 0.08);
      g.append('path')
        .datum(comp.points)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', comp.color)
        .attr('stroke-width', 1)
        .attr('opacity', 0.25);
    });

    // Draw highlighted component
    if (highlightIdx >= 0) {
      const comp = components[highlightIdx];
      g.append('path')
        .datum(comp.points)
        .attr('d', area)
        .attr('fill', comp.color)
        .attr('opacity', 0.35);
      g.append('path')
        .datum(comp.points)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', comp.color)
        .attr('stroke-width', 2.5);

      // Mean + fraction annotation
      g.append('line')
        .attr('x1', x(comp.mean)).attr('x2', x(comp.mean))
        .attr('y1', 0).attr('y2', height)
        .attr('stroke', comp.color)
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3');
      g.append('text')
        .attr('x', x(comp.mean) + 5)
        .attr('y', 14)
        .attr('fill', comp.color)
        .style('font-size', '11px')
        .text(`${comp.label} (${(comp.fraction * 100).toFixed(1)}%, mean ${comp.mean})`);
    }

    // Total outline (sum of all components)
    const totalPts = [];
    for (let j = 0; j <= nPoints; j++) {
      const xv = xMin + j * step;
      let total = 0;
      components.forEach(c => { total += c.points[j].y; });
      totalPts.push({ x: xv, y: total });
    }
    g.append('path')
      .datum(totalPts)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', getTextColor())
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.3);

    // Axes
    drawAxis(g.append('g').attr('transform', `translate(0,${height})`), x, 'bottom', 'Score');
    drawAxis(g, y, 'left', 'Density');

    // Tooltip
    const tt = tooltip(container);
    g.append('rect')
      .attr('width', width).attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .on('mousemove', (event) => {
        const [mx] = d3.pointer(event);
        const xVal = x.invert(mx);
        const idx = Math.round((xVal - xMin) / step);
        if (idx < 0 || idx > nPoints) return;
        let html = `<div class="tt-label">Score: ${(xMin + idx * step).toFixed(0)}</div>`;
        components.forEach(c => {
          const bold = c.bonus === activeBonus && c.yatzy === activeYatzy;
          html += `<div style="color:${c.color}${bold ? ';font-weight:bold' : ''}">` +
            `${c.label}: ${c.points[idx].y.toFixed(4)}</div>`;
        });
        html += `<div><strong>Total: ${totalPts[idx].y.toFixed(4)}</strong></div>`;
        tt.show(html, event);
      })
      .on('mouseleave', () => tt.hide());
  }

  render();

  // Legend (static, shows all four components)
  const legend = container.querySelector('.chart-legend');
  if (legend) {
    legend.innerHTML = '';
    components.forEach(comp => {
      const item = document.createElement('div');
      item.className = 'chart-legend-item';
      item.innerHTML = `<span class="legend-swatch" style="background:${comp.color}"></span>` +
        `<span>${comp.label} (${(comp.fraction * 100).toFixed(1)}%)</span>`;
      legend.appendChild(item);
    });
  }

  // Wire up toggle buttons
  container.querySelectorAll('.mixture-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const dim = btn.dataset.dim;
      const val = btn.dataset.val === 'yes';

      // Deactivate sibling, activate this one
      container.querySelectorAll(`.mixture-toggle[data-dim="${dim}"]`).forEach(
        b => b.classList.remove('active')
      );
      btn.classList.add('active');

      if (dim === 'bonus') activeBonus = val;
      if (dim === 'yatzy') activeYatzy = val;
      render();
    });
  });
}
