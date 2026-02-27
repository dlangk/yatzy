/**
 * Shared D3 utilities for Yatzy blog visualizations.
 */

export const COLORS = {
  accent: 'rgba(243, 112, 33, 0.85)',
  accentLight: 'rgba(243, 112, 33, 0.15)',
  riskAverse: '#3b4cc0',
  riskSeeking: '#b40426',
  neutral: '#636363',
  dt: '#2ca02c',
  mlp: '#7b3294',
  heuristic: '#d95f02',
  optimal: '#2ca02c',
  category: '#e7298a',
  reroll1: '#66a61e',
  reroll2: '#7570b3',
  mixture: ['#3b4cc0', '#8db0fe', '#F37021', '#b40426'],
  percentiles: {
    p5: '#3b4cc0',
    p25: '#8db0fe',
    p50: '#636363',
    p75: '#f4987a',
    p95: '#b40426',
    p99: '#7a0218',
  },
  humanRange: 'rgba(150, 150, 150, 0.15)',
};

const isDark = () => document.documentElement.classList.contains('dark');

export function getTextColor() {
  return isDark() ? '#e0ddd5' : '#050505';
}

export function getMutedColor() {
  return isDark() ? '#999' : '#555';
}

export function getGridColor() {
  return isDark() ? '#333' : '#ddd';
}

/**
 * Create a responsive SVG chart inside a container.
 */
export function createChart(containerId, opts = {}) {
  const {
    marginTop = 30,
    marginRight = 20,
    marginBottom = 45,
    marginLeft = 55,
    aspectRatio = 0.55,
  } = opts;

  const container = document.getElementById(containerId);
  if (!container) return null;

  const containerWidth = container.clientWidth || 635;
  const totalWidth = containerWidth;
  const totalHeight = Math.round(totalWidth * aspectRatio);
  const width = totalWidth - marginLeft - marginRight;
  const height = totalHeight - marginTop - marginBottom;

  // Clear previous
  container.querySelectorAll('svg').forEach(s => s.remove());

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g')
    .attr('transform', `translate(${marginLeft},${marginTop})`);

  return { svg, g, width, height, marginLeft, marginTop, totalWidth, totalHeight };
}

/**
 * Tooltip helper. Returns show/hide/move methods.
 */
export function tooltip(container) {
  let el = container.querySelector('.chart-tooltip');
  if (!el) {
    el = document.createElement('div');
    el.className = 'chart-tooltip';
    container.appendChild(el);
  }

  return {
    show(html, event) {
      el.innerHTML = html;
      el.classList.add('visible');
      this.move(event);
    },
    hide() {
      el.classList.remove('visible');
    },
    move(event) {
      const rect = container.getBoundingClientRect();
      let x = event.clientX - rect.left + 12;
      let y = event.clientY - rect.top - 10;
      // Keep tooltip in bounds
      const ttRect = el.getBoundingClientRect();
      if (x + ttRect.width > rect.width - 10) x = x - ttRect.width - 24;
      if (y + ttRect.height > rect.height - 10) y = y - ttRect.height - 10;
      if (y < 0) y = 10;
      el.style.left = x + 'px';
      el.style.top = y + 'px';
    },
  };
}

/**
 * Draw an axis with a label.
 */
export function drawAxis(g, scale, orient, label, opts = {}) {
  const { tickCount = 6, tickFormat = null, className = '' } = opts;

  let axis;
  if (orient === 'bottom') {
    axis = d3.axisBottom(scale).ticks(tickCount);
  } else if (orient === 'left') {
    axis = d3.axisLeft(scale).ticks(tickCount);
  }

  if (tickFormat) axis.tickFormat(tickFormat);

  const axisG = g.append('g')
    .attr('class', `axis ${className}`)
    .call(axis);

  // Style
  axisG.selectAll('line').attr('stroke', getGridColor());
  axisG.selectAll('path').attr('stroke', getGridColor());
  axisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");

  // Label
  if (label) {
    if (orient === 'bottom') {
      const width = scale.range()[1] - scale.range()[0];
      axisG.append('text')
        .attr('x', width / 2)
        .attr('y', 35)
        .attr('fill', getMutedColor())
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(label);
    } else {
      const height = scale.range()[0] - scale.range()[1];
      axisG.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -40)
        .attr('fill', getMutedColor())
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text(label);
    }
  }

  return axisG;
}

/**
 * Coolwarm_mid color scale for theta: blue → orange (center) → red.
 */
export function thetaColor(theta) {
  const stops = [
    { t: -0.3, color: '#3b4cc0' },
    { t: -0.15, color: '#8db0fe' },
    { t: 0, color: '#F37021' },
    { t: 0.15, color: '#f4987a' },
    { t: 0.3, color: '#b40426' },
  ];
  const clamped = Math.max(-0.3, Math.min(0.3, theta));
  for (let i = 0; i < stops.length - 1; i++) {
    if (clamped <= stops[i + 1].t) {
      const frac = (clamped - stops[i].t) / (stops[i + 1].t - stops[i].t);
      return d3.interpolateRgb(stops[i].color, stops[i + 1].color)(frac);
    }
  }
  return stops[stops.length - 1].color;
}

/**
 * Gaussian PDF.
 */
export function normalPDF(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/**
 * Format theta for display.
 */
export function formatTheta(theta) {
  if (theta === 0) return '0 (EV-optimal)';
  const sign = theta > 0 ? '+' : '\u2212';
  return `${sign}${Math.abs(theta).toFixed(3)}`;
}

/**
 * Linear interpolation in sorted data.
 */
export function lerp(arr, xKey, yKey, targetX) {
  if (arr.length === 0) return 0;
  if (targetX <= arr[0][xKey]) return arr[0][yKey];
  if (targetX >= arr[arr.length - 1][xKey]) return arr[arr.length - 1][yKey];
  for (let i = 0; i < arr.length - 1; i++) {
    if (targetX >= arr[i][xKey] && targetX <= arr[i + 1][xKey]) {
      const t = (targetX - arr[i][xKey]) / (arr[i + 1][xKey] - arr[i][xKey]);
      return arr[i][yKey] + t * (arr[i + 1][yKey] - arr[i][yKey]);
    }
  }
  return arr[arr.length - 1][yKey];
}

/**
 * Render dice as styled div elements.
 */
export function renderDice(container, values) {
  container.innerHTML = '';
  const pips = [
    [], // 0 placeholder
    [{ cx: 24, cy: 24 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 24, cy: 24 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 14 }, { cx: 34, cy: 14 }, { cx: 24, cy: 24 }, { cx: 14, cy: 34 }, { cx: 34, cy: 34 }],
    [{ cx: 14, cy: 12 }, { cx: 34, cy: 12 }, { cx: 14, cy: 24 }, { cx: 34, cy: 24 }, { cx: 14, cy: 36 }, { cx: 34, cy: 36 }],
  ];

  values.forEach(v => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 48 48');
    svg.setAttribute('width', '48');
    svg.setAttribute('height', '48');
    svg.classList.add('die-svg');
    svg.innerHTML = `<rect x="1" y="1" width="46" height="46" rx="8" fill="var(--bg-alt)" stroke="var(--border)" stroke-width="2"/>`;
    (pips[v] || []).forEach(p => {
      svg.innerHTML += `<circle cx="${p.cx}" cy="${p.cy}" r="4.5" fill="var(--text)"/>`;
    });
    container.appendChild(svg);
  });
}

/**
 * Build density from percentile knots via piecewise-linear CDF differentiation.
 */
export function densityFromPercentiles(row, nPoints = 200) {
  // Place a Gaussian kernel at each percentile midpoint, weighted by the
  // probability mass in that interval.  This produces inherently smooth curves.
  const knots = [
    { p: 0.00, v: row.p1  - (row.p5 - row.p1) },  // synthetic lower tail
    { p: 0.01, v: row.p1  },
    { p: 0.05, v: row.p5  },
    { p: 0.10, v: row.p10 },
    { p: 0.25, v: row.p25 },
    { p: 0.50, v: row.p50 },
    { p: 0.75, v: row.p75 },
    { p: 0.90, v: row.p90 },
    { p: 0.95, v: row.p95 },
    { p: 0.99, v: row.p99 },
    { p: 1.00, v: row.p99 + (row.p99 - row.p95) },  // synthetic upper tail
  ];

  // Build mixture components: one Gaussian per interval
  const components = [];
  for (let j = 0; j < knots.length - 1; j++) {
    const weight = knots[j + 1].p - knots[j].p;
    const mu = (knots[j].v + knots[j + 1].v) / 2;
    const span = Math.max(knots[j + 1].v - knots[j].v, 1);
    // Bandwidth: half the interval width, clamped to a reasonable minimum
    const sigma = Math.max(span / 2, 5);
    components.push({ weight, mu, sigma });
  }

  const xMin = 0;
  const xMax = 400;
  const step = (xMax - xMin) / nPoints;
  const density = [];
  const sqrt2pi = Math.sqrt(2 * Math.PI);

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;
    for (const c of components) {
      const z = (x - c.mu) / c.sigma;
      y += c.weight * Math.exp(-0.5 * z * z) / (c.sigma * sqrt2pi);
    }
    density.push({ x, y });
  }

  return density;
}
