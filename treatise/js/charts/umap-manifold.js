import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initUmapManifold() {
  const data = await DataLoader.umapEmbeddings();
  const container = document.getElementById('chart-umap-manifold');
  if (!container) return;

  const isPlaceholder = data.placeholder === true;

  // Cluster definitions
  const clusterDefs = data.clusters || [
    { name: 'Upper-focused', center: [-3, 2], color: COLORS.riskAverse },
    { name: 'Combo-seekers', center: [2, 3], color: COLORS.accent },
    { name: 'Bonus-chasers', center: [0, -2], color: COLORS.optimal },
    { name: 'Yatzy-gamblers', center: [4, -1], color: COLORS.riskSeeking },
    { name: 'Balanced', center: [-1, -3], color: '#7b3294' },
  ];

  // Generate or use points
  let points;
  if (data.points && data.points.length > 0) {
    points = data.points;
  } else {
    // Generate synthetic points around cluster centers
    points = [];
    const rng = () => (Math.random() - 0.5) * 2;
    clusterDefs.forEach((cluster, ci) => {
      const nPts = 40 + Math.floor(Math.random() * 30);
      for (let i = 0; i < nPts; i++) {
        points.push({
          x: cluster.center[0] + rng() * 1.8,
          y: cluster.center[1] + rng() * 1.8,
          cluster: ci,
          cluster_name: cluster.name,
        });
      }
    });
  }

  const chart = createChart('chart-umap-manifold', {
    aspectRatio: 0.7,
    marginLeft: 50,
    marginBottom: 50,
    marginTop: 25,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const xExtent = d3.extent(points, d => d.x);
  const yExtent = d3.extent(points, d => d.y);
  const xPad = (xExtent[1] - xExtent[0]) * 0.1;
  const yPad = (yExtent[1] - yExtent[0]) * 0.1;

  const x = d3.scaleLinear()
    .domain([xExtent[0] - xPad, xExtent[1] + xPad])
    .range([0, width]);

  const y = d3.scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad])
    .range([height, 0]);

  // Build color map from cluster index
  const clusterColors = {};
  clusterDefs.forEach((c, i) => { clusterColors[i] = c.color; });

  const tt = tooltip(container);

  // Draw points
  g.selectAll('.umap-dot')
    .data(points)
    .join('circle')
    .attr('class', 'umap-dot')
    .attr('cx', d => x(d.x))
    .attr('cy', d => y(d.y))
    .attr('r', 3)
    .attr('fill', d => clusterColors[d.cluster] || COLORS.accent)
    .attr('opacity', 0.6)
    .on('mousemove', (event, d) => {
      tt.show(
        `<div class="tt-label">${d.cluster_name || `Cluster ${d.cluster}`}</div>
         <div>UMAP: (${d.x.toFixed(2)}, ${d.y.toFixed(2)})</div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Cluster name labels at centers
  clusterDefs.forEach((cluster, i) => {
    const cx = cluster.center ? cluster.center[0] : d3.mean(points.filter(p => p.cluster === i), p => p.x);
    const cy = cluster.center ? cluster.center[1] : d3.mean(points.filter(p => p.cluster === i), p => p.y);
    if (cx == null || cy == null) return;

    g.append('text')
      .attr('x', x(cx))
      .attr('y', y(cy) - 12)
      .attr('text-anchor', 'middle')
      .attr('fill', cluster.color)
      .style('font-size', '11px')
      .style('font-weight', '700')
      .text(cluster.name);
  });

  // Placeholder watermark
  if (isPlaceholder) {
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', getMutedColor())
      .attr('opacity', 0.25)
      .style('font-size', '28px')
      .style('font-weight', '700')
      .attr('transform', `rotate(-20, ${width / 2}, ${height / 2})`)
      .text('PLACEHOLDER');
  }

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'UMAP 1'
  );
  drawAxis(g, y, 'left', 'UMAP 2');
}
