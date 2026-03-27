import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

// Shared constants
const TAIL_THETAS = [0, 0.05, 0.1, 0.2, 0.5];
const SCORE_MIN = 200;

/** Format a number as "M × 10<sup>N</sup>" HTML, always 1 decimal on the mantissa. */
function fmtSci(n) {
  if (n <= 0) return '0';
  const exp = Math.floor(Math.log10(n));
  if (exp === 0) return n.toFixed(1);
  const mantissa = (n / Math.pow(10, exp)).toFixed(1);
  return `${mantissa} \u00d7 10<sup style="line-height:0;font-size:0.75em;vertical-align:0.4em">${exp}</sup>`;
}

/** Render a log-scale y-axis with clean power-of-10 labels. */
function drawLogAxis(g, scale, ticks, label) {
  const axis = d3.axisLeft(scale)
    .tickValues(ticks)
    .tickFormat(d => {
      if (d === 1) return '1';
      const exp = Math.round(Math.log10(d));
      return `10${exp < 0 ? '\u207B' : ''}${Math.abs(exp).toString().split('').map(c => '\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079'[+c]).join('')}`;
    });
  const yG = g.append('g').call(axis);
  yG.selectAll('line').attr('stroke', getGridColor());
  yG.selectAll('path').attr('stroke', getGridColor());
  yG.selectAll('text').attr('fill', getMutedColor()).style('font-size', '10px')
    .style('font-family', "'Newsreader', Georgia, serif");
  if (label) {
    const height = scale.range()[0] - scale.range()[1];
    yG.append('text').attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -42)
      .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
      .style('font-size', '12px').text(label);
  }
  return yG;
}

// ── Chart A: Survival curve with observation horizons ────────────────

export async function initTailSurvival() {
  const data = await DataLoader.tailExact();
  const container = document.getElementById('chart-tail-survival');
  if (!container) return;

  const chart = createChart('chart-tail-survival-svg', { aspectRatio: 0.55 });
  if (!chart) return;
  const { g, width, height } = chart;

  const x = d3.scaleLinear().domain([SCORE_MIN, 374]).range([0, width]);
  const y = d3.scaleLog().domain([1e-15, 1]).range([height, 0]).clamp(true);

  // Grid
  const yTicks = [1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15];
  g.append('g').selectAll('line').data(yTicks).join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

  // Observation horizon lines
  const horizons = [
    { n: 2000, label: '1 lifetime (2,000 games)', dash: '8,4' },
    { n: 1e6, label: '1M simulations', dash: '5,3' },
    { n: 1e9, label: '1 billion games', dash: '3,2' },
  ];
  for (const h of horizons) {
    const prob = 1 / h.n;
    if (prob < 1e-15) continue;
    g.append('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', y(prob)).attr('y2', y(prob))
      .attr('stroke', 'var(--accent)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', h.dash)
      .attr('opacity', 0.5);
    g.append('text')
      .attr('x', width - 4).attr('y', y(prob) - 5)
      .attr('text-anchor', 'end')
      .attr('fill', 'var(--accent)')
      .style('font-size', '9px')
      .text(h.label);
  }

  // Curves
  const line = d3.line().defined(d => d[1] > 0).x(d => x(d[0])).y(d => y(d[1]));
  const tt = tooltip(container);

  for (const entry of data) {
    if (!TAIL_THETAS.includes(entry.theta)) continue;
    const pts = entry.tail
      .filter(d => d.score >= SCORE_MIN && d.survival > 0)
      .map(d => [d.score, d.survival]);
    if (pts.length === 0) continue;

    const color = thetaColor(entry.theta);
    g.append('path')
      .datum(pts)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', entry.theta === 0 ? 2.5 : 1.8)
      .attr('opacity', entry.theta === 0 ? 1 : 0.7);

    // Label at the end
    const last = pts[pts.length - 1];
    if (last[1] > 1e-15) {
      g.append('text')
        .attr('x', x(last[0]) + 3).attr('y', y(last[1]) + 4)
        .attr('fill', color)
        .style('font-size', '9px')
        .text(`θ=${formatTheta(entry.theta)}`);
    }
  }

  // Axes
  drawAxis(g.append('g').attr('transform', `translate(0,${height})`), x, 'bottom', 'Score');
  drawLogAxis(g, y, yTicks, 'P(score \u2265 x)');
}

// ── Chart B: Games needed to observe ─────────────────────────────────

export async function initTailGamesNeeded() {
  const data = await DataLoader.tailExact();
  const container = document.getElementById('chart-tail-games-needed');
  if (!container) return;

  const chart = createChart('chart-tail-games-needed-svg', { aspectRatio: 0.55 });
  if (!chart) return;
  const { g, width, height } = chart;

  const GAMES_XMIN = 248.4;
  const x = d3.scaleLinear().domain([GAMES_XMIN, 374]).range([0, width]);
  const y = d3.scaleLog().domain([1, 1e15]).range([height, 0]).clamp(true);

  // Grid
  const yTicks = [1, 1e3, 1e6, 1e9, 1e12, 1e15];
  g.append('g').selectAll('line').data(yTicks).join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

  // Reference lines
  const refs = [
    { n: 2000, label: '1 lifetime (2,000 games)' },
    { n: 1e6, label: '1 million games' },
    { n: 1e9, label: '1 billion games' },
    { n: 1e12, label: '1 trillion games' },
    { n: 1e15, label: '1 quadrillion games' },
  ];
  for (const r of refs) {
    g.append('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', y(r.n)).attr('y2', y(r.n))
      .attr('stroke', 'var(--accent)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '5,3')
      .attr('opacity', 0.5);
    g.append('text')
      .attr('x', 4).attr('y', y(r.n) - 5)
      .attr('fill', 'var(--accent)')
      .style('font-size', '9px')
      .text(r.label);
  }

  // Curves + build lookup for tooltip
  const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));
  const curvePts = new Map();

  for (const entry of data) {
    if (!TAIL_THETAS.includes(entry.theta)) continue;
    const pts = entry.tail
      .filter(d => d.score >= GAMES_XMIN && d.survival > 0)
      .map(d => [d.score, 1 / d.survival]);
    if (pts.length === 0) continue;
    curvePts.set(entry.theta, pts);

    const color = thetaColor(entry.theta);
    g.append('path')
      .datum(pts)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', entry.theta === 0 ? 2.5 : 1.8)
      .attr('opacity', entry.theta === 0 ? 1 : 0.7);
  }

  // Legend at top-right (inside margin)
  const legendG = g.append('g').attr('transform', 'translate(0, -14)');
  TAIL_THETAS.forEach((theta, i) => {
    const lx = width - (TAIL_THETAS.length - i) * 72;
    legendG.append('rect').attr('x', lx).attr('y', 0)
      .attr('width', 10).attr('height', 10).attr('rx', 2)
      .attr('fill', thetaColor(theta));
    legendG.append('text').attr('x', lx + 14).attr('y', 9)
      .attr('fill', getTextColor()).style('font-size', '9px')
      .text(`θ=${formatTheta(theta)}`);
  });

  // Hover tooltip
  const tt = tooltip(container);
  g.append('rect')
    .attr('width', width).attr('height', height)
    .attr('fill', 'none')
    .attr('pointer-events', 'all')
    .on('mousemove', (event) => {
      const [mx] = d3.pointer(event);
      const score = Math.round(x.invert(mx));
      if (score < Math.ceil(GAMES_XMIN) || score > 374) { tt.hide(); return; }
      let rows = '';
      for (const theta of TAIL_THETAS) {
        const pts = curvePts.get(theta);
        if (!pts) continue;
        const pt = pts.find(p => p[0] === score);
        if (!pt) continue;
        const n = pt[1];
        const nFmt = n >= 1e15 ? '>10<sup style="line-height:0;font-size:0.75em;vertical-align:0.4em">15</sup>' : fmtSci(n);
        rows += `<tr>
          <td style="color:${thetaColor(theta)};font-weight:600;padding:0 6px 0 0;white-space:nowrap;border:none">θ=${formatTheta(theta)}</td>
          <td style="padding:0;white-space:nowrap;border:none">${nFmt}</td>
        </tr>`;
      }
      tt.show(`<div class="tt-label">Score \u2265 ${score}</div><table style="border-collapse:collapse;border-spacing:0;margin-top:2px;line-height:1.3">${rows}</table>`, event);
    })
    .on('mouseleave', () => tt.hide());

  // Axes
  drawAxis(g.append('g').attr('transform', `translate(0,${height})`), x, 'bottom', 'Score');
  drawLogAxis(g, y, yTicks, 'Games needed');
}

// ── Chart C: Tail PMF bars ───────────────────────────────────────────

export async function initTailBars() {
  const data = await DataLoader.tailExact();
  const container = document.getElementById('chart-tail-bars');
  if (!container) return;

  const chart = createChart('chart-tail-bars-svg', { aspectRatio: 0.55 });
  if (!chart) return;
  const { g, width, height } = chart;

  const thresholds = [320, 330, 340, 350, 360, 370, 374];
  const thetas = TAIL_THETAS;

  // For each threshold, compute P(score >= threshold) per theta
  const groups = thresholds.map(t => {
    return {
      score: t,
      values: thetas.map(theta => {
        const entry = data.find(d => Math.abs(d.theta - theta) < 0.001);
        if (!entry) return 0;
        return entry.tail
          .filter(d => d.score >= t)
          .reduce((sum, d) => sum + d.prob, 0);
      }),
    };
  });

  const x0 = d3.scaleBand().domain(thresholds).range([0, width]).paddingInner(0.25);
  const x1 = d3.scaleBand().domain(d3.range(thetas.length)).range([0, x0.bandwidth()]).padding(0.05);
  const y = d3.scaleLog().domain([1e-15, 0.1]).range([height, 0]).clamp(true);

  // Grid
  const yTicks = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14];
  g.append('g').selectAll('line').data(yTicks).join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,3');

  // Bars
  const tt = tooltip(container);
  for (const group of groups) {
    for (let i = 0; i < thetas.length; i++) {
      const val = group.values[i];
      if (val <= 0) continue;
      const barX = x0(group.score) + x1(i);
      const barH = height - y(val);
      g.append('rect')
        .attr('x', barX)
        .attr('y', y(val))
        .attr('width', x1.bandwidth())
        .attr('height', Math.max(barH, 1))
        .attr('fill', thetaColor(thetas[i]))
        .attr('opacity', 0.8)
        .attr('rx', 1)
        .on('mousemove', (event) => {
          tt.show(
            `<div class="tt-label">\u2265 ${group.score}, \u03b8=${formatTheta(thetas[i])}</div>
             <div>P = <span class="tt-value">${fmtSci(val)}</span></div>
             <div>~1 in ${fmtSci(1/val)} games</div>`,
            event);
        })
        .on('mouseleave', () => tt.hide());
    }
  }

  // Legend
  const legendG = g.append('g').attr('transform', `translate(0, -12)`);
  for (let i = 0; i < thetas.length; i++) {
    const lx = i * 80;
    legendG.append('rect').attr('x', lx).attr('y', 0)
      .attr('width', 10).attr('height', 10).attr('rx', 2)
      .attr('fill', thetaColor(thetas[i]));
    legendG.append('text').attr('x', lx + 14).attr('y', 9)
      .attr('fill', getTextColor()).style('font-size', '9px')
      .text(`θ=${formatTheta(thetas[i])}`);
  }

  // Axes
  const xAxis = d3.axisBottom(x0).tickFormat(d => `\u2265${d}`);
  const xG = g.append('g').attr('transform', `translate(0,${height})`).call(xAxis);
  xG.selectAll('line').attr('stroke', getGridColor());
  xG.selectAll('path').attr('stroke', getGridColor());
  xG.selectAll('text').attr('fill', getMutedColor()).style('font-size', '10px')
    .style('font-family', "'Newsreader', Georgia, serif");
  xG.append('text').attr('x', width / 2).attr('y', 35)
    .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
    .style('font-size', '12px').text('Score threshold');

  drawLogAxis(g, y, yTicks, 'Probability');
}

// ── Chart D: θ-score heatmap ─────────────────────────────────────────

export async function initTailHeatmap() {
  const data = await DataLoader.tailExact();
  const container = document.getElementById('chart-tail-heatmap');
  if (!container) return;

  const chart = createChart('chart-tail-heatmap-svg', { aspectRatio: 0.55 });
  if (!chart) return;
  const { g, width, height } = chart;

  // Use all available thetas, sorted
  const entries = data
    .filter(d => TAIL_THETAS.includes(d.theta) || d.theta === -0.1 || d.theta === -0.3)
    .sort((a, b) => a.theta - b.theta);
  const thetas = entries.map(d => d.theta);
  const scoreRange = d3.range(SCORE_MIN, 375);

  const x = d3.scaleBand().domain(thetas.map(String)).range([0, width]).padding(0);
  const y = d3.scaleBand().domain(scoreRange).range([0, height]).padding(0);

  // Build heatmap data
  const color = d3.scaleSequential(d3.interpolateViridis)
    .domain([-15, -1]); // log10 scale

  const tt = tooltip(container);

  for (const entry of entries) {
    const thetaStr = String(entry.theta);
    for (const d of entry.tail) {
      if (d.score < SCORE_MIN || d.prob <= 0) continue;
      const logP = Math.log10(d.prob);
      g.append('rect')
        .attr('x', x(thetaStr))
        .attr('y', y(d.score))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', color(logP))
        .on('mousemove', (event) => {
          tt.show(
            `<div class="tt-label">θ=${formatTheta(entry.theta)}, score=${d.score}</div>
             <div>P = <span class="tt-value">${fmtSci(d.prob)}</span></div>`,
            event);
        })
        .on('mouseleave', () => tt.hide());
    }
  }

  // Axes
  const xAxis = d3.axisBottom(x).tickFormat(d => formatTheta(+d).trim());
  const xG = g.append('g').attr('transform', `translate(0,${height})`).call(xAxis);
  xG.selectAll('line').attr('stroke', getGridColor());
  xG.selectAll('path').attr('stroke', getGridColor());
  xG.selectAll('text').attr('fill', getMutedColor()).style('font-size', '9px')
    .style('font-family', "'Newsreader', Georgia, serif");
  xG.append('text').attr('x', width / 2).attr('y', 35)
    .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
    .style('font-size', '12px').text('θ');

  // Y axis: show every 10th score
  const yTicks = scoreRange.filter(s => s % 10 === 0);
  const yAxis = d3.axisLeft(y).tickValues(yTicks);
  const yG = g.append('g').call(yAxis);
  yG.selectAll('line').attr('stroke', getGridColor());
  yG.selectAll('path').attr('stroke', getGridColor());
  yG.selectAll('text').attr('fill', getMutedColor()).style('font-size', '10px')
    .style('font-family', "'Newsreader', Georgia, serif");
  yG.append('text').attr('transform', 'rotate(-90)')
    .attr('x', -height / 2).attr('y', -42)
    .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
    .style('font-size', '12px').text('Score');

  // Color legend
  const legendW = 120, legendH = 10;
  const legendG = g.append('g').attr('transform', `translate(${width - legendW - 10}, -18)`);
  const legendScale = d3.scaleLinear().domain([-15, -1]).range([0, legendW]);
  const legendSteps = d3.range(-15, -0.5, 0.5);
  for (const v of legendSteps) {
    legendG.append('rect')
      .attr('x', legendScale(v))
      .attr('y', 0)
      .attr('width', legendW / legendSteps.length + 1)
      .attr('height', legendH)
      .attr('fill', color(v));
  }
  legendG.append('text').attr('x', 0).attr('y', -2)
    .attr('fill', getMutedColor()).style('font-size', '8px').text('10\u207B\u00B9\u2075');
  legendG.append('text').attr('x', legendW).attr('y', -2)
    .attr('text-anchor', 'end')
    .attr('fill', getMutedColor()).style('font-size', '8px').text('10\u207B\u00B9');
}
