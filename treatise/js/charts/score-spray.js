/**
 * Score Spray — animated beeswarm of 10K simulated Yatzy games.
 *
 * Canvas layer (bottom): 10K dots with physics animation
 * SVG overlay (top): axes, density curve, annotations
 */

import { DataLoader } from '../data-loader.js';
import { getTextColor, getMutedColor, getGridColor, thetaColor } from '../yatzy-viz.js';
const POP_LABELS = ['No bonus, no Yatzy', 'No bonus, Yatzy', 'Bonus, no Yatzy', 'Bonus + Yatzy'];

// Physics
const GRAVITY = 0.3;
const DOT_RADIUS = 1.1;
const BIN_WIDTH = 4;      // score bin width for stacking

export async function initScoreSpray() {
  const [sprayData, meta, densityExact] = await Promise.all([
    DataLoader.scoreSpray(),
    DataLoader.scoreSprayMeta(),
    DataLoader.densityExact(),
  ]);

  // Exact PMF from density evolution (zero variance, mean=248.44)
  const perfectKDE = densityExact.pmf;  // [[score, probability], ...]
  const refTheta0 = { mean: densityExact.mean, ...densityExact.percentiles };

  const container = document.getElementById('chart-score-spray');
  if (!container) return;

  const wrap = document.getElementById('chart-score-spray-wrap');
  if (!wrap) return;

  const DOT_COUNT = 2000;
  const allGames = sprayData.games;
  // Sample DOT_COUNT games randomly
  const shuffledAll = [...allGames];
  for (let i = shuffledAll.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffledAll[i], shuffledAll[j]] = [shuffledAll[j], shuffledAll[i]];
  }
  const games = shuffledAll.slice(0, DOT_COUNT);
  const n = games.length;

  // Controls
  const replayBtn = document.getElementById('spray-replay-btn');
  const popup = document.getElementById('spray-scorecard-popup');

  // Population filter state
  const popVisible = [true, true, true, true];

  // Layout
  const margin = { top: 20, right: 20, bottom: 45, left: 55 };

  function getSize() {
    const w = wrap.clientWidth || 700;
    const h = Math.round(w * 0.5);
    return {
      totalW: w, totalH: h,
      width: w - margin.left - margin.right,
      height: h - margin.top - margin.bottom,
    };
  }

  let { totalW, totalH, width, height } = getSize();

  // Scales (d3)
  // Compute population colors from coolwarm palette based on mean scores
  const popMeans = meta.populations.map(p => p.mean);
  const [loMean, hiMean] = [Math.min(...popMeans), Math.max(...popMeans)];
  const POP_COLORS = popMeans.map(m => {
    const t = -0.3 + (m - loMean) / (hiMean - loMean) * 0.6;  // map to [-0.3, 0.3]
    return thetaColor(t);
  });

  const xScale = d3.scaleLinear().domain([0, 380]).range([0, width]);
  const yMax = height;

  // Prepare dots
  let dots = [];
  let settled = [];          // count of settled dots per bin
  let animFrame = null;
  let spawnIndex = 0;
  let milestones = { mMean: false, mDone: false };

  // Spatial index for hit detection
  const CELL_SIZE = 10;
  let spatialGrid = new Map();

  // --- Canvas setup ---
  function setupCanvas() {
    wrap.querySelectorAll('canvas').forEach(c => c.remove());
    wrap.querySelectorAll('svg').forEach(s => s.remove());

    const canvas = document.createElement('canvas');
    canvas.width = totalW * devicePixelRatio;
    canvas.height = totalH * devicePixelRatio;
    canvas.style.width = totalW + 'px';
    canvas.style.height = totalH + 'px';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '2';
    wrap.style.height = totalH + 'px';
    wrap.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.scale(devicePixelRatio, devicePixelRatio);

    // SVG overlay (axes, annotations, live KDE — behind dots)
    const svg = d3.select(wrap)
      .append('svg')
      .attr('width', totalW)
      .attr('height', totalH)
      .style('position', 'absolute')
      .style('top', '0')
      .style('left', '0')
      .style('z-index', '1')
      .style('pointer-events', 'none');

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Top SVG overlay (perfect KDE — above dots)
    const svgTop = d3.select(wrap)
      .append('svg')
      .attr('width', totalW)
      .attr('height', totalH)
      .style('position', 'absolute')
      .style('top', '0')
      .style('left', '0')
      .style('z-index', '3')
      .style('pointer-events', 'none');

    const gTop = svgTop.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    return { canvas, ctx, svg, g, svgTop, gTop };
  }

  let { canvas, ctx, svg, g, svgTop, gTop } = setupCanvas();

  // --- Draw SVG elements ---
  function drawAxes() {
    g.selectAll('.spray-axis').remove();

    // X axis
    const xAxis = d3.axisBottom(xScale).ticks(8);
    const xG = g.append('g')
      .attr('class', 'spray-axis')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis);

    xG.selectAll('line').attr('stroke', getGridColor());
    xG.selectAll('path').attr('stroke', getGridColor());
    xG.selectAll('text')
      .attr('fill', getMutedColor())
      .style('font-size', '11px')
      .style('font-family', "'Newsreader', Georgia, serif");

    xG.append('text')
      .attr('x', width / 2)
      .attr('y', 35)
      .attr('fill', getMutedColor())
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Total Score');

    // Grid lines
    g.append('g')
      .attr('class', 'spray-axis')
      .selectAll('line')
      .data(xScale.ticks(8))
      .join('line')
      .attr('x1', d => xScale(d)).attr('x2', d => xScale(d))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getGridColor())
      .attr('stroke-dasharray', '2,4')
      .attr('opacity', 0.5);

    drawPerfectKDE();
  }

  function drawAnnotations() {
    g.selectAll('.spray-annot').remove();

    const settledCount = dots.filter(d => d.settled).length;

    // Mean line (after 5% of dots)
    if (settledCount >= n * 0.05) {
      const mx = xScale(refTheta0.mean);
      g.append('line')
        .attr('class', 'spray-annot')
        .attr('x1', mx).attr('x2', mx)
        .attr('y1', 0).attr('y2', height)
        .attr('stroke', '#F37021')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3');

      g.append('text')
        .attr('class', 'spray-annot')
        .attr('x', mx + 5)
        .attr('y', 14)
        .attr('fill', '#F37021')
        .style('font-size', '11px')
        .style('font-family', "'Newsreader', Georgia, serif")
        .text(`mean = ${refTheta0.mean.toFixed(1)}`);
    }

    // Live mean line (after 5% of dots)
    if (settledCount >= n * 0.05) {
      const settledScores = [];
      for (let i = 0; i < spawnIndex; i++) {
        if (dots[i].settled) settledScores.push(dots[i].game.total);
      }
      const liveMean = settledScores.reduce((a, b) => a + b, 0) / settledScores.length;
      const lmx = xScale(liveMean);
      g.append('line')
        .attr('class', 'spray-annot')
        .attr('x1', lmx).attr('x2', lmx)
        .attr('y1', 0).attr('y2', height)
        .attr('stroke', getMutedColor())
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');

      g.append('text')
        .attr('class', 'spray-annot')
        .attr('x', lmx - 5)
        .attr('y', 28)
        .attr('fill', getMutedColor())
        .attr('text-anchor', 'end')
        .style('font-size', '11px')
        .style('font-family', "'Newsreader', Georgia, serif")
        .text(`sample = ${liveMean.toFixed(1)}`);
    }

    // Live KDE curve (always)
    if (settledCount > 0) {
      const settledScores = [];
      for (let i = 0; i < spawnIndex; i++) {
        if (dots[i].settled) settledScores.push(dots[i].game.total);
      }
      drawLiveKDE(settledScores);
    }

    // Stats label (after all dots)
    if (settledCount >= n) {
      const depth = document.body.dataset.depth || '1';
      if (depth >= '3') {
        g.append('text')
          .attr('class', 'spray-annot')
          .attr('x', width - 5)
          .attr('y', 14)
          .attr('fill', getMutedColor())
          .attr('text-anchor', 'end')
          .style('font-size', '10px')
          .style('font-family', "'Newsreader', Georgia, serif")
          .text(`N=${meta.n.toLocaleString()}, \u03C3=${meta.std.toFixed(1)}`);
      }
    }
  }

  const perfectYMax = d3.max(perfectKDE, d => d[1]);

  function drawPerfectKDE() {
    gTop.selectAll('.spray-kde-perfect').remove();

    const kdeScale = d3.scaleLinear().domain([0, perfectYMax]).range([height, height * 0.35]);

    const line = d3.line()
      .x(d => xScale(d[0]))
      .y(d => kdeScale(d[1]))
      .curve(d3.curveBasis);

    gTop.append('path')
      .attr('class', 'spray-kde-perfect')
      .datum(perfectKDE)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', getTextColor())
      .attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '6,3')
      .attr('opacity', 0.7)
      .attr('pointer-events', 'none');
  }

  // Live Gaussian KDE from settled dot scores
  function computeLiveKDE(scores) {
    if (scores.length < 10) return [];
    const sn = scores.length;
    const mean = scores.reduce((a, b) => a + b, 0) / sn;
    const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / sn);
    const h = 0.7 * 1.06 * std * Math.pow(sn, -0.2);  // Silverman bandwidth × 0.7
    const points = [];
    for (let x = 0; x <= 380; x += 2) {
      let density = 0;
      for (const s of scores) {
        const u = (x - s) / h;
        density += Math.exp(-0.5 * u * u);
      }
      density /= (sn * h * Math.sqrt(2 * Math.PI));
      points.push([x, density]);
    }
    return points;
  }

  function drawLiveKDE(scores) {
    g.selectAll('.spray-kde').remove();
    const livePoints = computeLiveKDE(scores);
    if (livePoints.length === 0) return;

    // Scale to match perfect curve height
    const kdeScale = d3.scaleLinear().domain([0, perfectYMax]).range([height, height * 0.35]);

    const line = d3.line()
      .x(d => xScale(d[0]))
      .y(d => kdeScale(d[1]))
      .curve(d3.curveBasis);

    const area = d3.area()
      .x(d => xScale(d[0]))
      .y0(height)
      .y1(d => kdeScale(d[1]))
      .curve(d3.curveBasis);

    g.append('path')
      .attr('class', 'spray-kde')
      .datum(livePoints)
      .attr('d', area)
      .attr('fill', 'rgba(243, 112, 33, 0.06)')
      .attr('pointer-events', 'none');

    g.append('path')
      .attr('class', 'spray-kde')
      .datum(livePoints)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', '#F37021')
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.6)
      .attr('pointer-events', 'none');
  }

  // --- Dot management ---
  function binKey(score) {
    return Math.round(score / BIN_WIDTH);
  }

  function resetDots() {
    // Shuffle games for varied animation
    const shuffled = games.map((g, i) => ({ ...g, idx: i }));
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    dots = shuffled.map(game => {
      const px = xScale(game.total) + margin.left;
      const jitter = (Math.random() - 0.5) * BIN_WIDTH * 0.15;
      return {
        x: px + jitter,
        y: -Math.random() * totalH * 0.3,  // start above viewport
        vy: 0,
        targetY: 0,
        settled: false,
        game,
        pop: game.pop,
        visible: popVisible[game.pop],
      };
    });

    settled = {};
    spawnIndex = 0;
    milestones = { mMean: false, mDone: false };
    spatialGrid.clear();
    if (popup) popup.classList.add('hidden');
  }

  function computeTargetY(dot) {
    const bk = binKey(dot.game.total);
    if (!settled[bk]) settled[bk] = 0;
    settled[bk]++;
    const stack = settled[bk];
    return Math.max(margin.top, totalH - margin.bottom - stack * (DOT_RADIUS * 2));
  }

  // --- Canvas rendering ---
  function renderCanvas() {
    ctx.clearRect(0, 0, totalW, totalH);

    for (let i = 0; i < spawnIndex; i++) {
      const dot = dots[i];
      if (!dot.visible) continue;

      ctx.beginPath();
      ctx.arc(dot.x, dot.y, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = POP_COLORS[dot.pop];
      ctx.globalAlpha = dot.settled ? 0.85 : 0.4;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // --- Spatial index ---
  function rebuildSpatialIndex() {
    spatialGrid.clear();
    for (let i = 0; i < spawnIndex; i++) {
      const dot = dots[i];
      if (!dot.settled || !dot.visible) continue;
      const cx = Math.floor(dot.x / CELL_SIZE);
      const cy = Math.floor(dot.y / CELL_SIZE);
      const key = `${cx},${cy}`;
      if (!spatialGrid.has(key)) spatialGrid.set(key, []);
      spatialGrid.get(key).push(i);
    }
  }

  function findDotAt(mx, my) {
    const cx = Math.floor(mx / CELL_SIZE);
    const cy = Math.floor(my / CELL_SIZE);
    let bestDist = 8;
    let bestIdx = -1;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const key = `${cx + dx},${cy + dy}`;
        const cell = spatialGrid.get(key);
        if (!cell) continue;
        for (const idx of cell) {
          const dot = dots[idx];
          const dist = Math.sqrt((dot.x - mx) ** 2 + (dot.y - my) ** 2);
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = idx;
          }
        }
      }
    }
    return bestIdx >= 0 ? dots[bestIdx] : null;
  }

  // --- Tooltip ---
  let tooltipEl = container.querySelector('.chart-tooltip');
  if (!tooltipEl) {
    tooltipEl = document.createElement('div');
    tooltipEl.className = 'chart-tooltip';
    container.appendChild(tooltipEl);
  }

  // --- Animation loop ---
  function animate() {
    const dotsPerFrame = 15;

    // Spawn new dots
    const spawnEnd = Math.min(spawnIndex + dotsPerFrame, n);
    for (let i = spawnIndex; i < spawnEnd; i++) {
      dots[i].targetY = computeTargetY(dots[i]);
      dots[i].vy = 0;
    }
    spawnIndex = spawnEnd;

    // Physics
    let anyMoving = false;
    for (let i = 0; i < spawnIndex; i++) {
      const dot = dots[i];
      if (dot.settled) continue;

      dot.vy += GRAVITY;
      dot.y += dot.vy;

      if (dot.y >= dot.targetY) {
        dot.y = dot.targetY;
        dot.settled = true;
        dot.vy = 0;
      } else {
        anyMoving = true;
      }
    }

    // Milestone checks (proportional to n)
    const settledCount = dots.filter(d => d.settled).length;
    if (settledCount >= n * 0.05 && !milestones.mMean) {
      milestones.mMean = true;
      drawAnnotations();
    }
    if (settledCount > 0 && !milestones.mDone && settledCount - (milestones.lastKde || 0) >= 100) {
      milestones.lastKde = settledCount;
      drawAnnotations();
      updateStats();
      updateLegend();
    }
    if (settledCount >= n && !milestones.mDone) {
      milestones.mDone = true;
      drawAnnotations();
      rebuildSpatialIndex();
      updateStats();
      updateLegend();
    }

    renderCanvas();

    if (spawnIndex < n || anyMoving) {
      animFrame = requestAnimationFrame(animate);
    } else {
      rebuildSpatialIndex();
      drawAnnotations();
    }
  }

  // --- Percentile helper ---
  function percentile(sorted, p) {
    if (sorted.length === 0) return '—';
    const idx = Math.min(Math.floor(sorted.length * p), sorted.length - 1);
    return sorted[idx];
  }

  // --- Reference stats from exact density evolution ---
  const refStats = {
    p1: refTheta0.p1,
    p10: refTheta0.p10,
    mean: refTheta0.mean,
    p90: refTheta0.p90,
    p99: refTheta0.p99,
  };

  // --- Stats box ---
  const statsEl = document.getElementById('spray-stats');

  function updateStats() {
    if (!statsEl) return;
    const settledScores = [];
    for (let i = 0; i < spawnIndex; i++) {
      if (dots[i].settled) settledScores.push(dots[i].game.total);
    }
    settledScores.sort((a, b) => a - b);
    const sn = settledScores.length;

    const fmt = v => typeof v === 'number' ? v.toFixed(1) : '—';
    const liveMean = sn > 0 ? settledScores.reduce((a, b) => a + b, 0) / sn : null;
    const live = sn > 0 ? {
      p1: percentile(settledScores, 0.01),
      p10: percentile(settledScores, 0.1),
      mean: liveMean,
      p90: percentile(settledScores, 0.9),
      p99: percentile(settledScores, 0.99),
    } : { p1: '—', p10: '—', mean: '—', p90: '—', p99: '—' };

    statsEl.innerHTML = `<table>
      <tr><th></th><th>P1</th><th>P10</th><th>Mean</th><th>P90</th><th>P99</th></tr>
      <tr><td class="spray-stats-label">Exact</td><td>${fmt(refStats.p1)}</td><td>${fmt(refStats.p10)}</td><td>${fmt(refStats.mean)}</td><td>${fmt(refStats.p90)}</td><td>${fmt(refStats.p99)}</td></tr>
      <tr class="spray-stats-live"><td class="spray-stats-label">${sn.toLocaleString()} games</td><td>${fmt(live.p1)}</td><td>${fmt(live.p10)}</td><td>${fmt(live.mean)}</td><td>${fmt(live.p90)}</td><td>${fmt(live.p99)}</td></tr>
    </table>`;
  }

  // --- Population legend ---
  const legendItems = [];

  function buildLegend() {
    const legendEl = document.getElementById('spray-legend');
    if (!legendEl) return;
    legendEl.innerHTML = '';
    legendItems.length = 0;

    POP_LABELS.forEach((label, i) => {
      const item = document.createElement('div');
      item.className = 'spray-legend-item' + (popVisible[i] ? '' : ' muted');
      item.dataset.pop = i;
      item.innerHTML = `<span class="spray-legend-swatch" style="background:${POP_COLORS[i]}"></span><span class="spray-legend-text">${label} (—)</span>`;
      item.addEventListener('click', () => {
        popVisible[i] = !popVisible[i];
        item.classList.toggle('muted');
        applyFilter();
      });
      legendEl.appendChild(item);
      legendItems.push(item);
    });

    const hint = document.createElement('div');
    hint.className = 'spray-hint';
    hint.textContent = 'Click any dot to see its scorecard';
    legendEl.appendChild(hint);
  }

  function updateLegend() {
    if (legendItems.length === 0) return;
    const popCounts = [0, 0, 0, 0];
    let total = 0;
    for (let i = 0; i < spawnIndex; i++) {
      if (dots[i].settled) {
        popCounts[dots[i].pop]++;
        total++;
      }
    }
    POP_LABELS.forEach((label, i) => {
      const pct = total > 0 ? (popCounts[i] / total * 100).toFixed(1) : '—';
      const textEl = legendItems[i].querySelector('.spray-legend-text');
      if (textEl) textEl.textContent = `${label} (${pct}%)`;
    });
  }

  function applyFilter() {
    settled = {};
    for (let i = 0; i < spawnIndex; i++) {
      const dot = dots[i];
      dot.visible = popVisible[dot.pop];
      if (dot.visible && dot.settled) {
        const bk = binKey(dot.game.total);
        if (!settled[bk]) settled[bk] = 0;
        settled[bk]++;
        dot.targetY = Math.max(margin.top, totalH - margin.bottom - settled[bk] * (DOT_RADIUS * 2));
        dot.y = dot.targetY;
      }
    }
    renderCanvas();
    rebuildSpatialIndex();
    drawAnnotations();
  }

  // --- Scorecard popup ---
  function showScorecard(dot, mx, my) {
    if (!popup) return;

    const game = dot.game;
    const catNames = meta.category_names;

    let html = `<button class="spray-popup-close" id="spray-close">&times;</button>`;
    html += `<div class="spray-popup-title">Game Score: ${game.total}</div>`;

    // Upper section
    let upperSum = 0;
    for (let i = 0; i < 6; i++) {
      const s = game.cats[i];
      upperSum += s;
      html += `<div class="spray-popup-row${s === 0 ? ' zero' : ''}">`
        + `<span>${catNames[i]}</span><span>${s}</span></div>`;
    }
    html += `<div class="spray-popup-divider"></div>`;
    html += `<div class="spray-popup-row"><span>Upper subtotal</span><span>${upperSum}/63</span></div>`;
    if (game.bonus) {
      html += `<div class="spray-popup-row spray-popup-bonus"><span>Bonus</span><span>+50</span></div>`;
    }
    html += `<div class="spray-popup-divider"></div>`;

    // Lower section
    for (let i = 6; i < 15; i++) {
      const s = game.cats[i];
      html += `<div class="spray-popup-row${s === 0 ? ' zero' : ''}">`
        + `<span>${catNames[i]}</span><span>${s}</span></div>`;
    }

    html += `<div class="spray-popup-divider"></div>`;
    html += `<div class="spray-popup-total">Total: ${game.total}</div>`;
    html += `<div class="spray-popup-badge">`
      + `<span class="spray-legend-swatch" style="background:${POP_COLORS[game.pop]}"></span>`
      + `${POP_LABELS[game.pop]}</div>`;

    popup.innerHTML = html;
    popup.classList.remove('hidden');

    // Position near click, clamped to container
    const rect = container.getBoundingClientRect();
    let left = mx + 15;
    let top = my - 50;
    if (left + 280 > rect.width) left = mx - 290;
    if (top < 0) top = 10;
    if (top + 400 > rect.height) top = rect.height - 410;
    popup.style.left = left + 'px';
    popup.style.top = top + 'px';

    document.getElementById('spray-close').addEventListener('click', () => {
      popup.classList.add('hidden');
    });
  }

  // --- Event handlers ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const dot = findDotAt(mx, my);

    if (dot) {
      canvas.style.cursor = 'pointer';
      tooltipEl.innerHTML = `<div class="tt-label">Score: ${dot.game.total}</div>`
        + `<div>${POP_LABELS[dot.pop]}</div>`;
      tooltipEl.classList.add('visible');
      const cRect = container.getBoundingClientRect();
      let tx = e.clientX - cRect.left + 12;
      let ty = e.clientY - cRect.top - 10;
      if (tx + 150 > cRect.width) tx -= 170;
      tooltipEl.style.left = tx + 'px';
      tooltipEl.style.top = ty + 'px';
    } else {
      canvas.style.cursor = 'default';
      tooltipEl.classList.remove('visible');
    }
  });

  canvas.addEventListener('mouseleave', () => {
    tooltipEl.classList.remove('visible');
    canvas.style.cursor = 'default';
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const dot = findDotAt(mx, my);
    if (dot) {
      showScorecard(dot, mx, my);
    } else if (popup) {
      popup.classList.add('hidden');
    }
  });

  // Replay button
  if (replayBtn) {
    replayBtn.addEventListener('click', () => {
      if (animFrame) cancelAnimationFrame(animFrame);
      resetDots();
      drawAxes();
      drawAnnotations();
      buildLegend();
      updateStats();
      updateLegend();
      animFrame = requestAnimationFrame(animate);
    });
  }

  // --- Init ---
  function start() {
    ({ totalW, totalH, width, height } = getSize());
    xScale.range([0, width]);

    // Rebuild canvas/SVG
    ({ canvas, ctx, svg, g, svgTop, gTop } = setupCanvas());

    // Rebind canvas events
    canvas.addEventListener('mousemove', canvasMouseMove);
    canvas.addEventListener('mouseleave', canvasMouseLeave);
    canvas.addEventListener('click', canvasClick);

    resetDots();
    drawAxes();
    buildLegend();
    updateStats();
    updateLegend();
    animFrame = requestAnimationFrame(animate);
  }

  // Named handlers for rebinding
  function canvasMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const dot = findDotAt(mx, my);

    if (dot) {
      canvas.style.cursor = 'pointer';
      tooltipEl.innerHTML = `<div class="tt-label">Score: ${dot.game.total}</div>`
        + `<div>${POP_LABELS[dot.pop]}</div>`;
      tooltipEl.classList.add('visible');
      const cRect = container.getBoundingClientRect();
      let tx = e.clientX - cRect.left + 12;
      let ty = e.clientY - cRect.top - 10;
      if (tx + 150 > cRect.width) tx -= 170;
      tooltipEl.style.left = tx + 'px';
      tooltipEl.style.top = ty + 'px';
    } else {
      canvas.style.cursor = 'default';
      tooltipEl.classList.remove('visible');
    }
  }

  function canvasMouseLeave() {
    tooltipEl.classList.remove('visible');
    canvas.style.cursor = 'default';
  }

  function canvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const dot = findDotAt(mx, my);
    if (dot) {
      showScorecard(dot, mx, my);
    } else if (popup) {
      popup.classList.add('hidden');
    }
  }

  start();
}
