import {
  createChart,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initRlBarrierDiagram() {
  const container = document.getElementById('chart-rl-barrier-diagram');
  if (!container) return;

  const chart = createChart('chart-rl-barrier-diagram', {
    aspectRatio: 0.5,
    marginLeft: 10,
    marginRight: 10,
    marginTop: 10,
    marginBottom: 10,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const barriers = [
    {
      label: 'Delayed Reward',
      detail: 'Bonus only revealed\nat game end',
      x: width * 0.15,
      y: height * 0.15,
    },
    {
      label: 'Combinatorial Actions',
      detail: '462 keeps \u00d7 15 categories\n= huge action space',
      x: width * 0.75,
      y: height * 0.15,
    },
    {
      label: 'No Curriculum',
      detail: 'All-or-nothing bonus\nnot decomposable',
      x: width * 0.45,
      y: height * 0.72,
    },
  ];

  // Central RL Agent node
  const centerX = width * 0.45;
  const centerY = height * 0.42;
  const agentR = 32;

  g.append('circle')
    .attr('cx', centerX)
    .attr('cy', centerY)
    .attr('r', agentR)
    .attr('fill', COLORS.accent)
    .attr('opacity', 0.15)
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 2);

  g.append('text')
    .attr('x', centerX)
    .attr('y', centerY - 4)
    .attr('text-anchor', 'middle')
    .attr('fill', COLORS.accent)
    .style('font-size', '12px')
    .style('font-weight', '700')
    .text('RL Agent');

  g.append('text')
    .attr('x', centerX)
    .attr('y', centerY + 12)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '9px')
    .text('best: ~236');

  // Barrier boxes
  const boxW = 150;
  const boxH = 60;

  barriers.forEach(b => {
    const bx = b.x - boxW / 2;
    const by = b.y - boxH / 2;

    g.append('rect')
      .attr('x', bx)
      .attr('y', by)
      .attr('width', boxW)
      .attr('height', boxH)
      .attr('rx', 6)
      .attr('fill', COLORS.riskSeeking)
      .attr('opacity', 0.08)
      .attr('stroke', COLORS.riskSeeking)
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.4);

    // Label
    g.append('text')
      .attr('x', b.x)
      .attr('y', by + 18)
      .attr('text-anchor', 'middle')
      .attr('fill', getTextColor())
      .style('font-size', '11px')
      .style('font-weight', '700')
      .text(b.label);

    // Detail (multiline)
    const lines = b.detail.split('\n');
    lines.forEach((line, li) => {
      g.append('text')
        .attr('x', b.x)
        .attr('y', by + 32 + li * 13)
        .attr('text-anchor', 'middle')
        .attr('fill', getMutedColor())
        .style('font-size', '9px')
        .text(line);
    });

    // Arrow from barrier to agent
    const dx = centerX - b.x;
    const dy = centerY - b.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const nx = dx / dist;
    const ny = dy / dist;

    // Start from edge of box, end at edge of circle
    const startX = b.x + nx * (boxW / 2 + 2);
    const startY = b.y + ny * (boxH / 2 + 2);
    const endX = centerX - nx * (agentR + 4);
    const endY = centerY - ny * (agentR + 4);

    g.append('line')
      .attr('x1', startX).attr('y1', startY)
      .attr('x2', endX).attr('y2', endY)
      .attr('stroke', COLORS.riskSeeking)
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0.5);

    // Arrowhead
    const arrowLen = 8;
    const angle = Math.atan2(endY - startY, endX - startX);
    g.append('path')
      .attr('d', `M${endX},${endY} L${endX - arrowLen * Math.cos(angle - 0.4)},${endY - arrowLen * Math.sin(angle - 0.4)} L${endX - arrowLen * Math.cos(angle + 0.4)},${endY - arrowLen * Math.sin(angle + 0.4)} Z`)
      .attr('fill', COLORS.riskSeeking)
      .attr('opacity', 0.5);
  });

  // Optimal reference
  g.append('text')
    .attr('x', width - 10)
    .attr('y', height - 8)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.optimal)
    .style('font-size', '11px')
    .style('font-weight', '600')
    .text('DP optimal: 248.4');

  g.append('text')
    .attr('x', width - 10)
    .attr('y', height - 22)
    .attr('text-anchor', 'end')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text('Gap: 12.4 points (\u22125.0%)');
}
