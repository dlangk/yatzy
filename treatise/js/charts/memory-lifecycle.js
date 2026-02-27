import {
  createChart, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMemoryLifecycle() {
  const container = document.getElementById('chart-memory-lifecycle');
  if (!container) return;

  const chart = createChart('chart-memory-lifecycle', {
    aspectRatio: 0.5,
    marginLeft: 70,
    marginBottom: 50,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Data: x = widget processing progress (0-100%), y = memory in bytes
  const KB = 1024;
  const MB = 1024 * KB;
  const GB = 1024 * MB;

  const etableSize = 8 * MB;      // persistent E-table (~16MB / 2 for active)
  const widgetMem = 3 * KB;       // per-widget working memory
  const naiveSize = 8 * GB;       // naive: store all intermediate values

  // Generate data points
  const nPoints = 100;
  const points = [];
  for (let i = 0; i <= nPoints; i++) {
    const pct = i / nPoints * 100;
    // E-table is constant; widget memory is a small spike at each widget
    const spike = Math.sin(pct / 100 * Math.PI * 16) > 0.7 ? widgetMem : widgetMem * 0.3;
    points.push({
      pct,
      etable: etableSize,
      widget: spike,
      total: etableSize + spike,
    });
  }

  const x = d3.scaleLinear().domain([0, 100]).range([0, width]);
  const y = d3.scaleLog().domain([1 * KB, 16 * GB]).range([height, 0]).clamp(true);

  // Grid lines
  const gridTicks = [1 * KB, 1 * MB, 1 * GB, 16 * GB];
  g.append('g').selectAll('line')
    .data(gridTicks)
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // E-table area (constant band)
  const etableArea = d3.area()
    .x(d => x(d.pct))
    .y0(height)
    .y1(d => y(d.etable));

  g.append('path')
    .datum(points)
    .attr('d', etableArea)
    .attr('fill', COLORS.accent)
    .attr('opacity', 0.25);

  // E-table line
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(etableSize)).attr('y2', y(etableSize))
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 2);

  // Widget working memory spikes
  const widgetLine = d3.line()
    .x(d => x(d.pct))
    .y(d => y(d.total));

  g.append('path')
    .datum(points)
    .attr('d', widgetLine)
    .attr('fill', 'none')
    .attr('stroke', COLORS.optimal)
    .attr('stroke-width', 1)
    .attr('opacity', 0.6);

  // Naive approach dashed line at 8GB
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(naiveSize)).attr('y2', y(naiveSize))
    .attr('stroke', COLORS.riskAverse)
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '6,4');

  // Labels
  // E-table label
  g.append('text')
    .attr('x', width - 4)
    .attr('y', y(etableSize) + 16)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.accent)
    .style('font-size', '11px')
    .style('font-weight', '600')
    .text('E-table: 8 MB');

  // Widget memory label
  g.append('text')
    .attr('x', width - 4)
    .attr('y', y(etableSize + widgetMem) - 30)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.optimal)
    .style('font-size', '10px')
    .text('Widget working set: ~3 KB');

  // Naive label
  g.append('text')
    .attr('x', width - 4)
    .attr('y', y(naiveSize) - 8)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.riskAverse)
    .style('font-size', '11px')
    .style('font-weight', '600')
    .text('Naive: ~8 GB');

  // 1000:1 ratio annotation
  const midX = width * 0.35;
  const midY = (y(naiveSize) + y(etableSize)) / 2;
  g.append('text')
    .attr('x', midX)
    .attr('y', midY)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '14px')
    .style('font-weight', '700')
    .text('1000 : 1');

  g.append('text')
    .attr('x', midX)
    .attr('y', midY + 16)
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text('memory ratio');

  // Double-headed arrow showing the gap
  g.append('line')
    .attr('x1', midX - 50).attr('x2', midX - 50)
    .attr('y1', y(naiveSize) + 4).attr('y2', y(etableSize) - 4)
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1)
    .attr('marker-start', 'url(#arrow-mem-up)')
    .attr('marker-end', 'url(#arrow-mem-down)');

  // Arrow markers
  const defs = g.append('defs');
  defs.append('marker')
    .attr('id', 'arrow-mem-up')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 5).attr('refY', 10)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M 0 10 L 5 0 L 10 10 z')
    .attr('fill', getMutedColor());

  defs.append('marker')
    .attr('id', 'arrow-mem-down')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 5).attr('refY', 0)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M 0 0 L 5 10 L 10 0 z')
    .attr('fill', getMutedColor());

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Widget Processing Progress (%)'
  );

  const formatBytes = d => {
    if (d >= GB) return `${(d / GB).toFixed(0)} GB`;
    if (d >= MB) return `${(d / MB).toFixed(0)} MB`;
    if (d >= KB) return `${(d / KB).toFixed(0)} KB`;
    return `${d} B`;
  };

  drawAxis(g, y, 'left', 'Memory (log scale)', {
    tickCount: 4,
    tickFormat: formatBytes,
  });
}
