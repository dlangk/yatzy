import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initReductionFunnel() {
  const container = document.getElementById('chart-reduction-funnel');
  if (!container) return;

  const stages = [
    { label: 'Possible games', count: 2.09e11, note: 'Ordered action sequences', color: getMutedColor() },
    { label: 'State slots', count: 2_097_152, note: '64 × 2¹⁵ sufficient statistics', color: getMutedColor() },
    { label: 'Reachable states', count: 1_430_000, note: '31.8% pruned by forward DP', color: COLORS.accent },
    { label: 'Per-widget nodes', count: 1_681, note: 'Self-contained 6-layer tree', color: COLORS.accent },
  ];

  const chart = createChart('chart-reduction-funnel', {
    aspectRatio: 0.45,
    marginLeft: 140,
    marginBottom: 30,
    marginTop: 15,
    marginRight: 100,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const y = d3.scaleBand()
    .domain(stages.map(d => d.label))
    .range([0, height])
    .padding(0.25);

  const x = d3.scaleLog()
    .domain([1e2, 3e11])
    .range([0, width])
    .clamp(true);

  // Grid lines
  const gridTicks = [1e3, 1e5, 1e7, 1e9, 1e11];
  g.append('g').selectAll('line')
    .data(gridTicks)
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const tt = tooltip(container);

  // Bars
  g.selectAll('.funnel-bar')
    .data(stages)
    .join('rect')
    .attr('class', 'funnel-bar')
    .attr('x', 0)
    .attr('y', d => y(d.label))
    .attr('width', d => Math.max(20, x(d.count)))
    .attr('height', y.bandwidth())
    .attr('fill', d => d.color)
    .attr('opacity', 0.8)
    .attr('rx', 3)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.label}</div>
         <div>Count: <span class="tt-value">${d.count.toLocaleString()}</span></div>
         <div>${d.note}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Count labels on bars
  g.selectAll('.funnel-count')
    .data(stages)
    .join('text')
    .attr('class', 'funnel-count')
    .attr('x', d => Math.max(20, x(d.count)) + 6)
    .attr('y', d => y(d.label) + y.bandwidth() / 2)
    .attr('dy', '0.35em')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '700')
    .style('font-variant-numeric', 'tabular-nums')
    .text(d => {
      if (d.count >= 1e9) return `${(d.count / 1e9).toFixed(1)}B`;
      if (d.count >= 1e6) return `${(d.count / 1e6).toFixed(1)}M`;
      return d.count.toLocaleString();
    });

  // Y axis labels
  g.selectAll('.funnel-label')
    .data(stages)
    .join('text')
    .attr('class', 'funnel-label')
    .attr('x', -8)
    .attr('y', d => y(d.label) + y.bandwidth() / 2)
    .attr('dy', '0.35em')
    .attr('text-anchor', 'end')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .text(d => d.label);

  // Reduction arrows between bars
  for (let i = 0; i < stages.length - 1; i++) {
    const y1 = y(stages[i].label) + y.bandwidth();
    const y2 = y(stages[i + 1].label);
    const midY = (y1 + y2) / 2;
    const ratio = stages[i].count / stages[i + 1].count;
    let ratioLabel;
    if (ratio >= 1e6) ratioLabel = `${(ratio / 1e6).toFixed(0)}M×`;
    else if (ratio >= 1e3) ratioLabel = `${(ratio / 1e3).toFixed(0)}K×`;
    else ratioLabel = `${ratio.toFixed(0)}×`;

    g.append('text')
      .attr('x', 24)
      .attr('y', midY + 2)
      .attr('fill', COLORS.accent)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(`↓ ${ratioLabel}`);
  }

  // X axis
  const xAxis = d3.axisBottom(x)
    .ticks(5, '.0s')
    .tickFormat(d => {
      if (d >= 1e9) return `10⁹`;
      if (d >= 1e7) return `10⁷`;
      if (d >= 1e5) return `10⁵`;
      if (d >= 1e3) return `10³`;
      return d;
    });

  const xAxisG = g.append('g').attr('transform', `translate(0,${height})`).call(xAxis);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");
}
