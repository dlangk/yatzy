import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, normalPDF,
  getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initMixtureDecomposition() {
  const [{ populations }, density] = await Promise.all([
    DataLoader.mixture(),
    DataLoader.densityExact(),
  ]);
  const container = document.getElementById('chart-mixture');
  if (!container) return;

  const pmf = density.pmf;
  const stats = {
    mean: density.mean,
    median: density.percentiles.p50,
    p5: density.percentiles.p5,
    p95: density.percentiles.p95,
  };

  // Build smooth KDE from exact PMF
  const bandwidth = 3.5;
  const xMin = 0, xMax = 374, nPoints = 300;
  const step = (xMax - xMin) / nPoints;

  const trueDensity = [];
  for (let j = 0; j <= nPoints; j++) {
    const xv = xMin + j * step;
    let val = 0;
    for (const [score, prob] of pmf) {
      if (prob < 1e-15) continue;
      val += prob * normalPDF(xv, score, bandwidth);
    }
    trueDensity.push({ x: xv, y: val });
  }

  // 16 Gaussian mixture components (bonus x yatzy x ss x ls)
  // Color by binary contribution: base gray, shift by each event
  const eventColors = {
    bonus: [78, 121, 167],   // blue
    yatzy: [180, 4, 38],     // red
    ss: [44, 160, 44],       // green
    ls: [148, 103, 189],     // purple
  };

  function componentColor(pop) {
    let r = 180, g = 180, b = 180; // base gray for no events
    let count = 0;
    if (pop.bonus) { r = 0; g = 0; b = 0; count = 0; } else { r = 0; g = 0; b = 0; count = 0; }
    // Blend active event colors
    const active = [];
    if (pop.bonus) active.push(eventColors.bonus);
    if (pop.yatzy) active.push(eventColors.yatzy);
    if (pop.ss) active.push(eventColors.ss);
    if (pop.ls) active.push(eventColors.ls);
    if (active.length === 0) return 'rgb(160,160,160)';
    r = Math.round(active.reduce((s, c) => s + c[0], 0) / active.length);
    g = Math.round(active.reduce((s, c) => s + c[1], 0) / active.length);
    b = Math.round(active.reduce((s, c) => s + c[2], 0) / active.length);
    return `rgb(${r},${g},${b})`;
  }

  const components = populations.map(pop => {
    const pts = [];
    for (let j = 0; j <= nPoints; j++) {
      const xv = xMin + j * step;
      pts.push({ x: xv, y: pop.fraction * normalPDF(xv, pop.mean, pop.std) });
    }
    return { ...pop, points: pts, color: componentColor(pop), visible: true };
  });

  // Toggle state: null = both, true = yes, false = no
  const filters = { bonus: null, yatzy: null, ss: null, ls: null };

  function matchesFilter(comp) {
    for (const key of ['bonus', 'yatzy', 'ss', 'ls']) {
      if (filters[key] !== null && comp[key] !== filters[key]) return false;
    }
    return true;
  }

  // Compute y-domain from true density
  const yMax = d3.max(trueDensity, d => d.y) * 1.08;

  const chart = createChart('chart-mixture-svg', {
    aspectRatio: 0.5,
    marginLeft: 15,
    marginRight: 15,
    marginBottom: 35,
    marginTop: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  // Move caption below SVG
  const caption = container.querySelector('.chart-caption');
  if (caption) container.appendChild(caption);

  const x = d3.scaleLinear().domain([xMin, xMax]).range([0, width]);
  const y = d3.scaleLinear().domain([0, yMax]).range([height, 0]);

  const lineFn = d3.line().x(d => x(d.x)).y(d => y(d.y)).curve(d3.curveBasis);
  const areaFn = d3.area().x(d => x(d.x)).y0(height).y1(d => y(d.y)).curve(d3.curveBasis);

  // Grid lines
  g.append('g').selectAll('line')
    .data(y.ticks(4))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // Component areas (drawn in order, toggled via opacity)
  const compGroups = components.map((comp, i) => {
    const cg = g.append('g');
    cg.append('path')
      .datum(comp.points)
      .attr('class', 'comp-area')
      .attr('d', areaFn)
      .attr('fill', comp.color)
      .attr('opacity', 0.2);
    cg.append('path')
      .datum(comp.points)
      .attr('class', 'comp-line')
      .attr('d', lineFn)
      .attr('fill', 'none')
      .attr('stroke', comp.color)
      .attr('stroke-width', 1.2);
    return cg;
  });

  // Combined sum of visible components
  function computeSum() {
    const pts = [];
    for (let j = 0; j <= nPoints; j++) {
      let total = 0;
      components.forEach(c => { if (c.visible) total += c.points[j].y; });
      pts.push({ x: xMin + j * step, y: total });
    }
    return pts;
  }

  const sumArea = g.append('path')
    .datum(computeSum())
    .attr('d', areaFn)
    .attr('fill', getMutedColor())
    .attr('opacity', 0.06);

  const sumLine = g.append('path')
    .datum(computeSum())
    .attr('d', lineFn)
    .attr('fill', 'none')
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 2);

  // True density (exact PMF, always visible, on top)
  g.append('path')
    .datum(trueDensity)
    .attr('d', lineFn)
    .attr('fill', 'none')
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '6,3');

  // Reference lines: p5, mean, p95
  const refLines = [
    { val: stats.p5, label: 'P5', anchor: 'start' },
    { val: stats.mean, label: 'Mean', anchor: 'middle' },
    { val: stats.p95, label: 'P95', anchor: 'end' },
  ];

  const refG = g.append('g');
  refLines.forEach(({ val, label, anchor }) => {
    refG.append('line')
      .attr('x1', x(val)).attr('x2', x(val))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', getMutedColor())
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3');
    refG.append('text')
      .attr('x', x(val))
      .attr('y', -5)
      .attr('text-anchor', anchor)
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text(`${label}: ${label === 'Mean' ? val.toFixed(1) : val}`);
  });

  // X axis
  const xAxisG = g.append('g').attr('transform', `translate(0,${height})`);
  const xAxis = d3.axisBottom(x).ticks(8).tickSize(0);
  xAxisG.call(xAxis);
  xAxisG.select('.domain').attr('stroke', getGridColor());
  xAxisG.selectAll('text').attr('fill', getMutedColor()).style('font-size', '10px');

  // Tooltip
  const tt = tooltip(container);
  g.append('rect')
    .attr('width', width).attr('height', height)
    .attr('fill', 'none')
    .attr('pointer-events', 'all')
    .on('mousemove', (event) => {
      const [mx] = d3.pointer(event);
      const xVal = x.invert(mx);
      const idx = Math.round((xVal - xMin) / step);
      if (idx < 0 || idx > nPoints) return;
      const score = (xMin + idx * step).toFixed(0);
      let html = `<div class="tt-label">Score: ${score}</div>`;
      let visTotal = 0;
      components.forEach(c => {
        if (!c.visible) return;
        visTotal += c.points[idx].y;
      });
      html += `<div>Selected density: <span class="tt-value">${(visTotal * 1000).toFixed(2)}&permil;</span></div>`;
      html += `<div>True density: ${(trueDensity[idx].y * 1000).toFixed(2)}&permil;</div>`;
      tt.show(html, event);
    })
    .on('mouseleave', () => tt.hide());

  // Toggle controls: 4 independent groups, 2 buttons each (toggle + Both)
  const controls = document.createElement('div');
  controls.className = 'chart-controls';
  controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:0.2rem 0.15rem;align-items:center;';
  const cap = container.querySelector('.chart-caption');
  if (cap) container.insertBefore(controls, cap);
  else container.appendChild(controls);

  const toggleDefs = [
    { key: 'bonus', label: 'Bonus' },
    { key: 'yatzy', label: 'Yatzy' },
    { key: 'ss', label: 'S.Straight' },
    { key: 'ls', label: 'L.Straight' },
  ];

  const btnStyle = 'font-size:0.7rem;padding:0.2rem 0.45rem;';

  toggleDefs.forEach(({ key, label }, idx) => {
    // Separator between row 1 (bonus+yatzy) and row 2 (straights)
    if (idx === 2) {
      const br = document.createElement('div');
      br.style.cssText = 'width:100%;height:0;';
      controls.appendChild(br);
    } else if (idx === 1 || idx === 3) {
      const sep = document.createElement('span');
      sep.style.cssText = 'width:1px;height:18px;background:var(--border);margin:0 0.15rem;';
      controls.appendChild(sep);
    }

    // Toggle button: cycles Hit ↔ Miss
    const toggle = document.createElement('button');
    toggle.className = 'chart-btn active';
    toggle.style.cssText = btnStyle;
    toggle.textContent = `${label}: Both`;
    toggle.dataset.state = 'both'; // both | hit | miss

    // Both button
    const bothBtn = document.createElement('button');
    bothBtn.className = 'chart-btn';
    bothBtn.style.cssText = btnStyle;
    bothBtn.textContent = 'Hit';

    const missBtn = document.createElement('button');
    missBtn.className = 'chart-btn';
    missBtn.style.cssText = btnStyle;
    missBtn.textContent = 'Miss';

    function activate(val) {
      filters[key] = val;
      toggle.classList.toggle('active', val === null);
      bothBtn.classList.toggle('active', val === true);
      missBtn.classList.toggle('active', val === false);
      updateVisibility();
    }

    toggle.addEventListener('click', () => activate(null));
    bothBtn.addEventListener('click', () => activate(true));
    missBtn.addEventListener('click', () => activate(false));

    controls.appendChild(missBtn);
    controls.appendChild(toggle);
    controls.appendChild(bothBtn);
  });

  function updateVisibility() {
    components.forEach((comp, i) => {
      comp.visible = matchesFilter(comp);
      compGroups[i].select('.comp-area')
        .transition().duration(300)
        .attr('opacity', comp.visible ? 0.2 : 0);
      compGroups[i].select('.comp-line')
        .transition().duration(300)
        .attr('opacity', comp.visible ? 1 : 0);
    });
    const sumPts = computeSum();
    sumArea.datum(sumPts).transition().duration(300).attr('d', areaFn);
    sumLine.datum(sumPts).transition().duration(300).attr('d', lineFn);
  }
}
