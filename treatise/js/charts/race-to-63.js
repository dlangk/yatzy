import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initRaceTo63() {
  const data = await DataLoader.raceTo63();
  const container = document.getElementById('chart-race-to-63');
  if (!container) return;

  const nodes = data.nodes;
  const links = data.links.filter(l => l.mass >= 0.0001);

  const chart = createChart('chart-race-to-63', {
    aspectRatio: 0.6,
    marginTop: 20,
    marginRight: 30,
    marginBottom: 45,
    marginLeft: 55,
  });
  if (!chart) return;
  const { svg, g, width, height } = chart;

  const x = d3.scaleLinear().domain([0, 15]).range([0, width]);
  const y = d3.scaleLinear().domain([0, 63]).range([height, 0]);

  // Mass extents for scaling
  const linkMassExtent = d3.extent(links, d => d.mass);
  const nodeMassExtent = d3.extent(nodes, d => d.mass);

  const linkWidth = d3.scaleSqrt()
    .domain(linkMassExtent)
    .range([0.3, 4]);

  const linkOpacity = d3.scaleLinear()
    .domain(linkMassExtent)
    .range([0.03, 0.25]);

  const nodeRadius = d3.scaleSqrt()
    .domain(nodeMassExtent)
    .range([1, 8]);

  const nodeColor = d3.scaleSequential(d3.interpolateBlues)
    .domain([0, Math.cbrt(nodeMassExtent[1])]);

  // Cube-root compress mass so even low-probability nodes get visible color
  const nodeColorVal = mass => nodeColor(Math.cbrt(mass));

  // Build node lookup
  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  // Grid
  g.append('g').selectAll('line.hgrid')
    .data(y.ticks(8))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Bonus threshold line
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(63)).attr('y2', y(63))
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '6,4');

  g.append('text')
    .attr('x', width - 4)
    .attr('y', y(63) - 6)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.accent)
    .style('font-size', '10px')
    .style('font-weight', '600')
    .text('Bonus threshold (63)');

  // Links — additive blending via mix-blend-mode
  const isDark = document.documentElement.classList.contains('dark');
  const linkColorScale = isDark
    ? d3.scaleSequential(d3.interpolateBlues).domain([0, linkMassExtent[1]])
    : d3.scaleSequential(d3.interpolateBlues).domain([0, linkMassExtent[1]]);

  const linksG = g.append('g')
    .style('mix-blend-mode', isDark ? 'lighten' : 'screen');

  linksG.selectAll('path')
    .data(links)
    .join('path')
    .attr('d', d => {
      const src = nodeMap.get(d.source);
      const tgt = nodeMap.get(d.target);
      if (!src || !tgt) return '';
      return d3.linkHorizontal()({
        source: [x(src.turn), y(src.upper_score)],
        target: [x(tgt.turn), y(tgt.upper_score)],
      });
    })
    .attr('fill', 'none')
    .attr('stroke', d => linkColorScale(d.mass))
    .attr('stroke-width', d => linkWidth(d.mass))
    .attr('opacity', d => linkOpacity(d.mass));

  // Nodes
  const tt = tooltip(container);

  g.selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('cx', d => x(d.turn))
    .attr('cy', d => y(d.upper_score))
    .attr('r', d => nodeRadius(d.mass))
    .attr('fill', d => nodeColorVal(d.mass))
    .attr('stroke', d => d3.color(nodeColorVal(d.mass)).darker(0.5))
    .attr('stroke-width', 0.5)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">Turn ${d.turn}, Upper ${d.upper_score}</div>
         <div>Probability: <span class="tt-value">${(d.mass * 100).toFixed(2)}%</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Axes
  drawAxis(g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Turn', { tickCount: 16, tickFormat: d3.format('d') });
  drawAxis(g, y, 'left', 'Upper section score');
}
