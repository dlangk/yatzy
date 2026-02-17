import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, normalPDF,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMixtureDecomposition() {
  const { populations } = await DataLoader.mixture();
  const container = document.getElementById('chart-mixture');
  if (!container) return;

  const nPoints = 300;
  const xMin = 80;
  const xMax = 370;
  const step = (xMax - xMin) / nPoints;

  // Precompute each component's density
  const components = populations.map((pop, i) => {
    const pts = [];
    for (let j = 0; j <= nPoints; j++) {
      const x = xMin + j * step;
      pts.push({ x, y: pop.fraction * normalPDF(x, pop.mean, pop.std) });
    }
    return { ...pop, points: pts, color: COLORS.mixture[i], visible: true };
  });

  let animStep = components.length; // Start fully shown

  function render() {
    const chart = createChart('chart-mixture-svg', { aspectRatio: 0.5 });
    if (!chart) return;
    const { g, width, height } = chart;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, width]);

    // Stack components
    const stackedPoints = [];
    for (let j = 0; j <= nPoints; j++) {
      const xv = xMin + j * step;
      const entry = { x: xv };
      let cumY = 0;
      components.forEach((comp, i) => {
        if (i < animStep && comp.visible) {
          cumY += comp.points[j].y;
        }
        entry[`y${i}`] = cumY;
        entry[`raw${i}`] = comp.points[j].y;
      });
      entry.total = cumY;
      stackedPoints.push(entry);
    }

    const yMax = d3.max(stackedPoints, d => d.total) * 1.1 || 0.02;
    const y = d3.scaleLinear().domain([0, yMax]).range([height, 0]);

    // Grid
    g.append('g').selectAll('line')
      .data(y.ticks(5))
      .join('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', d => y(d)).attr('y2', d => y(d))
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,3');

    // Draw stacked areas bottom-up
    for (let i = Math.min(animStep, components.length) - 1; i >= 0; i--) {
      if (!components[i].visible) continue;

      const area = d3.area()
        .x(d => x(d.x))
        .y0(d => y(i === 0 ? 0 : d[`y${i - 1}`] || 0))
        .y1(d => y(d[`y${i}`]))
        .curve(d3.curveBasis);

      g.append('path')
        .datum(stackedPoints)
        .attr('d', area)
        .attr('fill', components[i].color)
        .attr('opacity', 0.5);

      // Component outline
      const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d[`y${i}`]))
        .curve(d3.curveBasis);

      g.append('path')
        .datum(stackedPoints)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', components[i].color)
        .attr('stroke-width', 1.5);
    }

    // Total outline
    const totalLine = d3.line()
      .x(d => x(d.x))
      .y(d => y(d.total))
      .curve(d3.curveBasis);

    g.append('path')
      .datum(stackedPoints)
      .attr('d', totalLine)
      .attr('fill', 'none')
      .attr('stroke', getTextColor())
      .attr('stroke-width', 2);

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
        const sp = stackedPoints[idx];
        let html = `<div class="tt-label">Score: ${sp.x.toFixed(0)}</div>`;
        components.forEach((c, i) => {
          if (i < animStep && c.visible) {
            html += `<div style="color:${c.color}">${c.label}: ${sp[`raw${i}`].toFixed(4)}</div>`;
          }
        });
        html += `<div><strong>Total: ${sp.total.toFixed(4)}</strong></div>`;
        tt.show(html, event);
      })
      .on('mouseleave', () => tt.hide());
  }

  render();

  // Legend with toggles
  const legend = container.querySelector('.chart-legend');
  if (legend) {
    legend.innerHTML = '';
    components.forEach((comp, i) => {
      const item = document.createElement('div');
      item.className = 'chart-legend-item';
      item.innerHTML = `<span class="legend-swatch" style="background:${comp.color}"></span>
        <span>${comp.label} (${(comp.fraction * 100).toFixed(1)}%)</span>`;
      item.addEventListener('click', () => {
        comp.visible = !comp.visible;
        item.classList.toggle('dimmed', !comp.visible);
        render();
      });
      legend.appendChild(item);
    });
  }

  // Build-up button
  const btn = container.querySelector('.chart-btn');
  if (btn) {
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      animStep = 0;
      render();
      for (let i = 1; i <= components.length; i++) {
        await new Promise(r => setTimeout(r, 700));
        animStep = i;
        render();
      }
      btn.disabled = false;
    });
  }
}
