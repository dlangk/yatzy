import {
  createChart, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initExponentialUtility() {
  const container = document.getElementById('chart-exponential-utility');
  if (!container) return;

  const chart = createChart('chart-exponential-utility', {
    aspectRatio: 0.55,
    marginLeft: 60,
    marginBottom: 50,
    marginTop: 20,
    marginRight: 100,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const thetas = [
    { theta: -0.3, label: '\u03b8 = \u22120.3 (risk-averse)', color: COLORS.riskAverse },
    { theta: -0.1, label: '\u03b8 = \u22120.1', color: '#8db0fe' },
    { theta: 0, label: '\u03b8 = 0 (linear)', color: COLORS.accent },
    { theta: 0.1, label: '\u03b8 = +0.1', color: '#f4987a' },
    { theta: 0.3, label: '\u03b8 = +0.3 (risk-seeking)', color: COLORS.riskSeeking },
  ];

  const nPoints = 200;
  const xDomain = [0, 400];
  const x = d3.scaleLinear().domain(xDomain).range([0, width]);

  // Compute all curves to find y extent
  const allCurves = thetas.map(({ theta }) => {
    const pts = [];
    for (let i = 0; i <= nPoints; i++) {
      const xv = (i / nPoints) * 400;
      let yv;
      if (Math.abs(theta) < 1e-9) {
        yv = xv; // linear
      } else {
        // Scale theta down for visualization: u(x) = (1 - exp(-theta * x / 100)) / theta
        yv = (1 - Math.exp(-theta * xv / 100)) / theta;
      }
      pts.push({ x: xv, y: yv });
    }
    return pts;
  });

  const yMin = d3.min(allCurves, curve => d3.min(curve, d => d.y));
  const yMax = d3.max(allCurves, curve => d3.max(curve, d => d.y));
  const yPad = (yMax - yMin) * 0.05;
  const y = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([height, 0]);

  // Grid
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  const line = d3.line()
    .x(d => x(d.x))
    .y(d => y(d.y))
    .curve(d3.curveMonotoneX);

  // Draw curves
  allCurves.forEach((curve, i) => {
    const { label, color, theta } = thetas[i];

    g.append('path')
      .datum(curve)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', Math.abs(theta) < 1e-9 ? 2.5 : 2)
      .attr('stroke-dasharray', Math.abs(theta) < 1e-9 ? null : '6,3');

    // End label
    const lastPt = curve[curve.length - 1];
    g.append('text')
      .attr('x', width + 5)
      .attr('y', y(lastPt.y) + 4)
      .attr('fill', color)
      .style('font-size', '10px')
      .style('font-weight', '600')
      .text(label);
  });

  // Concavity annotations
  g.append('text')
    .attr('x', x(80)).attr('y', y(allCurves[0][40].y) - 12)
    .attr('fill', COLORS.riskAverse)
    .style('font-size', '10px')
    .style('font-style', 'italic')
    .text('concave (risk-averse)');

  g.append('text')
    .attr('x', x(250)).attr('y', y(allCurves[4][125].y) + 16)
    .attr('fill', COLORS.riskSeeking)
    .style('font-size', '10px')
    .style('font-style', 'italic')
    .text('convex (risk-seeking)');

  // Axes
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    x, 'bottom', 'Score'
  );
  drawAxis(g, y, 'left', 'Utility u(x)');
}
