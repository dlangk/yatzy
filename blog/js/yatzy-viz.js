/**
 * Shared D3 utilities for Yatzy blog visualizations.
 */

export const COLORS = {
  accent: 'rgba(255, 77, 0, 0.8)',
  accentLight: 'rgba(255, 77, 0, 0.15)',
  riskAverse: '#2171b5',
  riskSeeking: '#cb181d',
  neutral: '#636363',
  dt: '#2ca02c',
  mlp: '#7b3294',
  heuristic: '#d95f02',
  optimal: '#1b9e77',
  category: '#e7298a',
  reroll1: '#66a61e',
  reroll2: '#7570b3',
  mixture: ['#4575b4', '#74add1', '#fdae61', '#f46d43'],
  percentiles: {
    p5: '#2171b5',
    p25: '#6baed6',
    p50: '#333333',
    p75: '#fd8d3c',
    p95: '#cb181d',
    p99: '#99000d',
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
 * Blue → gray → red color scale for theta.
 */
export function thetaColor(theta) {
  if (theta < 0) {
    const t = Math.min(1, -theta / 0.3);
    return d3.interpolateRgb('#636363', '#2171b5')(t);
  } else if (theta > 0) {
    const t = Math.min(1, theta / 0.3);
    return d3.interpolateRgb('#636363', '#cb181d')(t);
  }
  return '#636363';
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
export function densityFromPercentiles(row, nPoints = 300) {
  const knots = [
    { p: 0.01, v: row.p1 },
    { p: 0.05, v: row.p5 },
    { p: 0.10, v: row.p10 },
    { p: 0.25, v: row.p25 },
    { p: 0.50, v: row.p50 },
    { p: 0.75, v: row.p75 },
    { p: 0.90, v: row.p90 },
    { p: 0.95, v: row.p95 },
    { p: 0.99, v: row.p99 },
  ];

  const points = [];
  const xMin = knots[0].v - 20;
  const xMax = knots[knots.length - 1].v + 20;
  const step = (xMax - xMin) / nPoints;

  for (let i = 0; i < nPoints; i++) {
    const x = xMin + i * step;
    // Interpolate CDF
    let cdf;
    if (x <= knots[0].v) {
      cdf = knots[0].p * Math.max(0, 1 - (knots[0].v - x) / 30);
    } else if (x >= knots[knots.length - 1].v) {
      cdf = knots[knots.length - 1].p + (1 - knots[knots.length - 1].p) * Math.min(1, (x - knots[knots.length - 1].v) / 30);
    } else {
      for (let j = 0; j < knots.length - 1; j++) {
        if (x >= knots[j].v && x <= knots[j + 1].v) {
          const t = (x - knots[j].v) / (knots[j + 1].v - knots[j].v);
          cdf = knots[j].p + t * (knots[j + 1].p - knots[j].p);
          break;
        }
      }
    }
    points.push({ x, cdf: cdf || 0 });
  }

  // Differentiate for density
  const density = [];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dcdf = points[i].cdf - points[i - 1].cdf;
    density.push({
      x: (points[i].x + points[i - 1].x) / 2,
      y: Math.max(0, dcdf / dx),
    });
  }

  return density;
}
