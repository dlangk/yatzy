import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initOptimizationTimeline() {
  const data = await DataLoader.optimizationTimeline();
  const container = document.getElementById('chart-optimization-timeline');
  if (!container) return;

  const steps = data.steps || data;
  if (!steps || steps.length === 0) return;

  const chart = createChart('chart-optimization-timeline', {
    aspectRatio: 0.55,
    marginLeft: 65,
    marginBottom: 80,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleBand()
    .domain(steps.map(d => d.label))
    .range([0, width])
    .padding(0.2);

  const yMin = 0.1;
  const yMax = d3.max(steps, d => d.time) * 2;
  const y = d3.scaleLog().domain([yMin, yMax]).range([height, 0]).clamp(true);

  // Grid lines
  const gridTicks = [0.1, 1, 10, 100, 1000];
  g.append('g').selectAll('line')
    .data(gridTicks.filter(t => t <= yMax))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Color by speedup magnitude
  const speedups = steps.map(d => d.speedup || 1);
  const maxSpeedup = d3.max(speedups);
  const barColor = d3.scaleSequential()
    .domain([1, maxSpeedup])
    .interpolator(d3.interpolateOranges);

  const tt = tooltip(container);

  // Bars
  g.selectAll('.bar')
    .data(steps)
    .join('rect')
    .attr('class', 'bar')
    .attr('x', d => x(d.label))
    .attr('y', d => y(Math.max(yMin, d.time)))
    .attr('width', x.bandwidth())
    .attr('height', d => height - y(Math.max(yMin, d.time)))
    .attr('fill', d => barColor(d.speedup || 1))
    .attr('rx', 3)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.label}</div>
         <div>Time: <span class="tt-value">${d.time.toFixed(2)}s</span></div>
         <div>Technique: ${d.technique || 'baseline'}</div>
         ${d.speedup ? `<div>Speedup: ${d.speedup.toFixed(1)}x</div>` : ''}`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Technique annotations above bars
  g.selectAll('.tech-label')
    .data(steps)
    .join('text')
    .attr('class', 'tech-label')
    .attr('x', d => x(d.label) + x.bandwidth() / 2)
    .attr('y', d => y(Math.max(yMin, d.time)) - 5)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '8px')
    .text(d => d.technique || '');

  // Final time line at 1.1s
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(1.1)).attr('y2', y(1.1))
    .attr('stroke', COLORS.optimal)
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '5,3');

  g.append('text')
    .attr('x', width - 4)
    .attr('y', y(1.1) - 5)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.optimal)
    .style('font-size', '10px')
    .style('font-weight', '600')
    .text('Final: 1.1s');

  // X axis with rotated labels
  const xAxisG = g.append('g').attr('transform', `translate(0,${height})`);
  const xAxis = d3.axisBottom(x);
  xAxisG.call(xAxis);
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '9px')
    .style('font-family', "'Newsreader', Georgia, serif")
    .attr('transform', 'rotate(-35)')
    .attr('text-anchor', 'end');
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());

  // Y axis
  drawAxis(g, y, 'left', 'Time (seconds, log scale)', {
    tickCount: 4,
    tickFormat: d => d >= 1 ? `${d}s` : `${(d * 1000).toFixed(0)}ms`,
  });
}
