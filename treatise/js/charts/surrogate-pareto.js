import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initSurrogatePareto() {
  const models = await DataLoader.gameEval();
  const container = document.getElementById('chart-surrogate-pareto');
  if (!container) return;

  const optimalMean = 248.4;

  // Classify models
  const classified = models.map(m => ({
    ...m,
    type: m.name === 'heuristic' ? 'heuristic'
      : m.name.startsWith('dt_') ? 'dt'
      : 'mlp',
  }));

  const chart = createChart('chart-surrogate-pareto-svg', { aspectRatio: 0.55 });
  if (!chart) return;
  const { g, width, height } = chart;

  // x: log scale for params
  const xMin = 0.8;
  const xMax = d3.max(classified, d => d.total_params) * 2;
  const x = d3.scaleLog()
    .domain([xMin, xMax])
    .range([0, width]);

  // Heuristic at x=1 (0 params → show at 1)
  const xVal = d => d.total_params === 0 ? 1 : d.total_params;

  const y = d3.scaleLinear()
    .domain([100, 260])
    .range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Human range band
  g.append('rect')
    .attr('x', 0).attr('width', width)
    .attr('y', y(230)).attr('height', y(220) - y(230))
    .attr('fill', COLORS.humanRange);

  g.append('text')
    .attr('x', width - 4).attr('y', y(225) + 4)
    .attr('text-anchor', 'end')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .style('font-style', 'italic')
    .text('Human range');

  // Optimal line
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(optimalMean)).attr('y2', y(optimalMean))
    .attr('stroke', COLORS.optimal)
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '6,3');

  g.append('text')
    .attr('x', 4).attr('y', y(optimalMean) - 6)
    .attr('fill', COLORS.optimal)
    .style('font-size', '10px')
    .text(`Optimal (${optimalMean})`);

  // Pareto line through DTs
  const dts = classified.filter(m => m.type === 'dt').sort((a, b) => xVal(a) - xVal(b));
  // Build Pareto frontier
  const pareto = [];
  let bestY = 0;
  dts.forEach(d => {
    if (d.mean > bestY) {
      pareto.push(d);
      bestY = d.mean;
    }
  });
  if (pareto.length > 1) {
    const paretoLine = d3.line()
      .x(d => x(xVal(d)))
      .y(d => y(d.mean))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(pareto)
      .attr('d', paretoLine)
      .attr('fill', 'none')
      .attr('stroke', COLORS.dt)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.6);
  }

  // Points
  const tt = tooltip(container);

  const colorMap = { heuristic: COLORS.heuristic, dt: COLORS.dt, mlp: COLORS.mlp };
  const shapeMap = { heuristic: d3.symbolDiamond, dt: d3.symbolCircle, mlp: d3.symbolSquare };

  classified.forEach(m => {
    const sym = d3.symbol().type(shapeMap[m.type]).size(m.type === 'heuristic' ? 120 : 80);

    g.append('path')
      .attr('d', sym)
      .attr('transform', `translate(${x(xVal(m))},${y(m.mean)})`)
      .attr('fill', colorMap[m.type])
      .attr('stroke', 'white')
      .attr('stroke-width', 1)
      .attr('cursor', 'pointer')
      .on('mousemove', (event) => {
        tt.show(
          `<div class="tt-label">${m.name}</div>
           <div>Params: <span class="tt-value">${m.total_params.toLocaleString()}</span></div>
           <div>Mean score: <span class="tt-value">${m.mean.toFixed(1)}</span></div>
           <div>Bonus rate: ${(m.bonus_rate * 100).toFixed(1)}%</div>
           <div>Std: ${m.std.toFixed(1)}</div>`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());
  });

  // Key labels
  const labelled = ['heuristic', 'dt_d5', 'dt_d10', 'dt_d15', 'dt_d20', 'mlp_64'];
  classified.filter(m => labelled.includes(m.name)).forEach(m => {
    const xPos = x(xVal(m));
    const yPos = y(m.mean);
    const offset = m.name === 'heuristic' ? { dx: 8, dy: -8 } : { dx: 6, dy: -8 };
    g.append('text')
      .attr('x', xPos + offset.dx)
      .attr('y', yPos + offset.dy)
      .attr('fill', colorMap[m.type])
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(m.name);
  });

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Parameters (log scale)',
    { tickFormat: d => d >= 1000 ? `${d / 1000}k` : d }
  );
  drawAxis(g, y, 'left', 'Mean score (10K games)');

  // Legend
  const legend = container.querySelector('.chart-legend');
  if (legend) {
    legend.innerHTML = [
      { label: 'Decision Tree', color: COLORS.dt },
      { label: 'MLP', color: COLORS.mlp },
      { label: 'Heuristic', color: COLORS.heuristic },
    ].map(l =>
      `<div class="chart-legend-item">
        <span class="legend-swatch" style="background:${l.color}"></span>
        <span>${l.label}</span>
      </div>`
    ).join('');
  }
}
