import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initRosettaRules() {
  const data = await DataLoader.rosettaRules();
  const container = document.getElementById('chart-rosetta-rules');
  if (!container) return;

  const rawRules = (data.rules || data).slice(0, 15);
  if (!rawRules || rawRules.length === 0) return;
  // Normalize field names (JSON uses 'regret'/'action', chart expects flexible access)
  const rules = rawRules.map(r => ({
    ...r,
    regret_prevented: r.regret_prevented ?? r.regret ?? r.value ?? 0,
    name: r.name || r.action || `Rule ${r.rank}`,
  }));

  const categoryColors = {
    upper: COLORS.riskAverse,
    combo: COLORS.accent,
    jackpot: COLORS.riskSeeking,
    dump: '#999',
    speculative: COLORS.mlp || '#7b3294',
  };

  const chart = createChart('chart-rosetta-rules', {
    aspectRatio: 0.75,
    marginLeft: 200,
    marginRight: 60,
    marginBottom: 45,
    marginTop: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const xMax = d3.max(rules, d => d.regret_prevented || d.value) * 1.15;
  const x = d3.scaleLinear().domain([0, xMax]).range([0, width]);

  const y = d3.scaleBand()
    .domain(rules.map((_, i) => i))
    .range([0, height])
    .padding(0.2);

  // Grid
  g.append('g').selectAll('line')
    .data(x.ticks(5))
    .join('line')
    .attr('x1', d => x(d)).attr('x2', d => x(d))
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const tt = tooltip(container);

  // Bars
  g.selectAll('.bar')
    .data(rules)
    .join('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', (_, i) => y(i))
    .attr('width', d => x(d.regret_prevented || d.value))
    .attr('height', y.bandwidth())
    .attr('fill', d => categoryColors[d.category] || COLORS.accent)
    .attr('opacity', 0.8)
    .attr('rx', 3)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.name || d.label}</div>
         <div>Regret prevented: <span class="tt-value">${(d.regret_prevented || d.value).toFixed(2)}</span></div>
         ${d.condition ? `<div>If: ${d.condition}</div>` : ''}
         ${d.action ? `<div>Then: ${d.action}</div>` : ''}
         <div>Category: ${d.category}</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Value labels
  g.selectAll('.val-label')
    .data(rules)
    .join('text')
    .attr('x', d => x(d.regret_prevented || d.value) + 4)
    .attr('y', (_, i) => y(i) + y.bandwidth() / 2 + 4)
    .attr('fill', getTextColor())
    .style('font-size', '10px')
    .style('font-weight', '600')
    .text(d => (d.regret_prevented || d.value).toFixed(1));

  // Y-axis labels
  g.selectAll('.bar-label')
    .data(rules)
    .join('text')
    .attr('x', -6)
    .attr('y', (_, i) => y(i) + y.bandwidth() / 2 + 4)
    .attr('fill', getTextColor())
    .attr('text-anchor', 'end')
    .style('font-size', '10px')
    .text(d => {
      const label = d.name || d.label || '';
      return label.length > 30 ? label.substring(0, 28) + '\u2026' : label;
    });

  // Cumulative coverage line overlay
  let cumSum = 0;
  const totalRegret = d3.sum(rules, d => d.regret_prevented || d.value);
  const cumData = rules.map((d, i) => {
    cumSum += (d.regret_prevented || d.value);
    return { i, cum: cumSum / totalRegret };
  });

  const x2 = d3.scaleLinear().domain([0, 1]).range([0, width]);

  const cumLine = d3.line()
    .x(d => x2(d.cum))
    .y(d => y(d.i) + y.bandwidth() / 2)
    .curve(d3.curveMonotoneY);

  g.append('path')
    .datum(cumData)
    .attr('d', cumLine)
    .attr('fill', 'none')
    .attr('stroke', getTextColor())
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3')
    .attr('opacity', 0.5);

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Regret prevented (EV points)'
  );

  // Legend
  const legend = g.append('g').attr('transform', `translate(${width - 130}, 0)`);
  const cats = ['upper', 'combo', 'jackpot', 'dump', 'speculative'];
  cats.forEach((cat, i) => {
    legend.append('rect')
      .attr('y', i * 16)
      .attr('width', 10).attr('height', 10)
      .attr('fill', categoryColors[cat])
      .attr('rx', 2);
    legend.append('text')
      .attr('x', 14).attr('y', i * 16 + 9)
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text(cat);
  });
}
