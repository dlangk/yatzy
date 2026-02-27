import { DataLoader } from '../data-loader.js';
import {
  createChart,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initBonusCovariance() {
  const data = await DataLoader.bonusCovariance();
  const container = document.getElementById('chart-bonus-covariance');
  if (!container) return;

  // Default decomposition if data is missing fields
  const components = data.components || [
    { label: 'Bonus itself', value: 50, color: COLORS.accent },
    { label: 'Better upper scores', value: 8, color: COLORS.riskAverse },
    { label: 'Better lower scores', value: 14, color: COLORS.optimal },
  ];

  const totalGap = data.total_gap || components.reduce((s, c) => s + c.value, 0);

  const chart = createChart('chart-bonus-covariance', {
    aspectRatio: 0.3,
    marginLeft: 55,
    marginBottom: 40,
    marginTop: 30,
    marginRight: 30,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const colors = [COLORS.accent, COLORS.riskAverse, COLORS.optimal];
  const barH = Math.min(height * 0.5, 40);
  const barY = (height - barH) / 2;

  const x = d3.scaleLinear().domain([0, totalGap]).range([0, width]);

  // Grid
  g.append('g').selectAll('line')
    .data(x.ticks(6))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', barY - 10).attr('y2', barY + barH + 10)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Stacked horizontal bars
  let cumX = 0;
  components.forEach((comp, i) => {
    const barW = x(comp.value);
    const color = comp.color || colors[i % colors.length];

    g.append('rect')
      .attr('x', x(cumX))
      .attr('y', barY)
      .attr('width', barW)
      .attr('height', barH)
      .attr('fill', color)
      .attr('opacity', 0.8)
      .attr('rx', i === 0 ? 4 : 0);

    // Value label inside bar
    if (barW > 30) {
      g.append('text')
        .attr('x', x(cumX) + barW / 2)
        .attr('y', barY + barH / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .style('font-size', '13px')
        .style('font-weight', '700')
        .text(`+${comp.value}`);
    }

    // Label above
    g.append('text')
      .attr('x', x(cumX) + barW / 2)
      .attr('y', barY - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', color)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text(comp.label);

    cumX += comp.value;
  });

  // Total annotation
  g.append('text')
    .attr('x', width)
    .attr('y', barY + barH + 24)
    .attr('text-anchor', 'end')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '600')
    .text(`Total gap: ${totalGap} points`);

  // Bottom axis
  const xAxisG = g.append('g').attr('transform', `translate(0,${barY + barH + 2})`);
  const xAxis = d3.axisBottom(x).ticks(6).tickFormat(d => `+${d}`);
  xAxisG.call(xAxis);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '10px');
}
