import {
  createChart,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initWidgetStructure() {
  const container = document.getElementById('chart-widget-structure');
  if (!container) return;

  const layers = [
    { label: 'Entry',   count: 1,   type: 'start',    desc: 'State input' },
    { label: 'Roll 1',  count: 252, type: 'chance',   desc: '252 dice outcomes' },
    { label: 'Keep 1',  count: 462, type: 'decision', desc: '462 keep choices' },
    { label: 'Roll 2',  count: 252, type: 'chance',   desc: '252 dice outcomes' },
    { label: 'Keep 2',  count: 462, type: 'decision', desc: '462 keep choices' },
    { label: 'Score',   count: 252, type: 'chance',   desc: '252 → category exit' },
  ];

  const chart = createChart('chart-widget-structure', {
    aspectRatio: 0.65,
    marginLeft: 15,
    marginBottom: 50,
    marginTop: 15,
    marginRight: 15,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const maxLog = Math.log10(462);
  const minBarW = 30;
  const maxBarW = width * 0.85;

  function barWidth(count) {
    if (count <= 1) return minBarW;
    return minBarW + (Math.log10(count) / maxLog) * (maxBarW - minBarW);
  }

  const layerH = 28;
  const gap = (height - 50 - layers.length * layerH) / (layers.length - 1);

  const chanceColor = getGridColor();
  const decisionColor = COLORS.accent;

  // Draw layers
  layers.forEach((layer, i) => {
    const bw = barWidth(layer.count);
    const bx = (width - bw) / 2;
    const by = i * (layerH + gap);

    const isDecision = layer.type === 'decision';
    const fill = isDecision ? decisionColor : (layer.type === 'start' ? getMutedColor() : chanceColor);
    const opacity = isDecision ? 0.25 : 0.35;

    // Bar
    g.append('rect')
      .attr('x', bx)
      .attr('y', by)
      .attr('width', bw)
      .attr('height', layerH)
      .attr('rx', 4)
      .attr('fill', fill)
      .attr('opacity', opacity)
      .attr('stroke', isDecision ? COLORS.accent : getMutedColor())
      .attr('stroke-width', isDecision ? 1.5 : 0.5);

    // Label inside bar
    g.append('text')
      .attr('x', width / 2)
      .attr('y', by + layerH / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle')
      .attr('fill', getTextColor())
      .style('font-size', '11px')
      .style('font-weight', '600')
      .text(`${layer.label} (${layer.count})`);

    // Type label on right
    g.append('text')
      .attr('x', bx + bw + 8)
      .attr('y', by + layerH / 2)
      .attr('dy', '0.35em')
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .style('font-style', 'italic')
      .text(layer.type === 'start' ? '' : layer.type);

    // Connecting edges to next layer
    if (i < layers.length - 1) {
      const nextBw = barWidth(layers[i + 1].count);
      const nextBx = (width - nextBw) / 2;
      const nextBy = (i + 1) * (layerH + gap);

      // Draw a few representative bundled edges
      const nEdges = 5;
      for (let e = 0; e < nEdges; e++) {
        const frac = (e + 0.5) / nEdges;
        const x1 = bx + bw * frac;
        const x2 = nextBx + nextBw * frac;
        const y1 = by + layerH;
        const y2 = nextBy;
        const midY = (y1 + y2) / 2;

        g.append('path')
          .attr('d', `M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}`)
          .attr('fill', 'none')
          .attr('stroke', getMutedColor())
          .attr('stroke-width', 0.5)
          .attr('opacity', 0.3);
      }
    }
  });

  // Exit arrow to E-table
  const lastBy = (layers.length - 1) * (layerH + gap) + layerH;
  const arrowY = lastBy + 12;

  g.append('line')
    .attr('x1', width / 2).attr('x2', width / 2)
    .attr('y1', lastBy).attr('y2', arrowY + 10)
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 1.5)
    .attr('marker-end', 'url(#arrow-widget)');

  // Arrow marker
  g.append('defs').append('marker')
    .attr('id', 'arrow-widget')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 8).attr('refY', 5)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto-start-reverse')
    .append('path')
    .attr('d', 'M 0 0 L 10 5 L 0 10 z')
    .attr('fill', COLORS.accent);

  // E-table box
  const boxW = 100, boxH = 24;
  g.append('rect')
    .attr('x', width / 2 - boxW / 2)
    .attr('y', arrowY + 14)
    .attr('width', boxW)
    .attr('height', boxH)
    .attr('rx', 4)
    .attr('fill', 'none')
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3');

  g.append('text')
    .attr('x', width / 2)
    .attr('y', arrowY + 14 + boxH / 2)
    .attr('dy', '0.35em')
    .attr('text-anchor', 'middle')
    .attr('fill', COLORS.accent)
    .style('font-size', '11px')
    .style('font-weight', '600')
    .text('E-table');

  // Summary annotation
  g.append('text')
    .attr('x', width / 2)
    .attr('y', height - 4)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text('1,681 nodes · ≤21,000 edges · ~3 KB per widget');

  // Legend
  const legY = 0;
  const legX = width - 90;
  [
    { label: 'Chance', color: chanceColor, opacity: 0.35 },
    { label: 'Decision', color: decisionColor, opacity: 0.25 },
  ].forEach((item, i) => {
    g.append('rect')
      .attr('x', legX).attr('y', legY + i * 18)
      .attr('width', 12).attr('height', 12)
      .attr('rx', 2)
      .attr('fill', item.color).attr('opacity', item.opacity)
      .attr('stroke', i === 1 ? COLORS.accent : getMutedColor())
      .attr('stroke-width', 1);
    g.append('text')
      .attr('x', legX + 16).attr('y', legY + i * 18 + 10)
      .attr('fill', getMutedColor())
      .style('font-size', '10px')
      .text(item.label);
  });
}
