import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

/**
 * Compute upper-section reachability via forward DP.
 * R[n][mask] = can the 6-bit subset `mask` of {1..6} sum to exactly n?
 * Then aggregate by popcount for the heatmap display.
 */
function computeReachability() {
  const maxUpper = 105; // theoretical max: 6×6×5 = 180, but cap display at 63
  const nMasks = 64; // 2^6

  // R[n][mask] = reachable boolean
  // Use flat arrays for speed
  const R = new Uint8Array((maxUpper + 1) * nMasks);
  const idx = (n, m) => n * nMasks + m;

  // Base: empty set, score 0
  R[idx(0, 0)] = 1;

  // Forward pass: for each die face d (1..6), extend all reachable (n, mask)
  for (let d = 0; d < 6; d++) {
    const bit = 1 << d;
    const faceVal = d + 1; // 1..6
    // Process in reverse to avoid counting a face twice
    // But since each face appears exactly once in the mask, we just need
    // to iterate masks that don't yet include this bit
    for (let n = maxUpper; n >= 0; n--) {
      for (let m = 0; m < nMasks; m++) {
        if (R[idx(n, m)] && !(m & bit)) {
          // Score this face: multiply by 1..5 (number of dice showing this face)
          for (let count = 1; count <= 5; count++) {
            const newN = n + faceVal * count;
            if (newN <= maxUpper) {
              R[idx(newN, m | bit)] = 1;
            }
          }
        }
      }
    }
  }

  // Also: a face in the mask can score 0 (assigned to a lower category that doesn't count)
  // Actually, upper score only counts if assigned to the matching upper category.
  // A face being "scored" means its upper category is used. If used, score is sum of matching dice (1-5 of that face).
  // So each face in mask contributes faceVal*k for k in {1..5}, not 0.
  // But wait - scoring a category means you USED it, and the score is the actual dice.
  // Minimum contribution is faceVal*1, max is faceVal*5.

  // Aggregate by popcount: for each (popcount, upper_score), what fraction of masks are reachable?
  const result = []; // { pop, upper, fraction, reachable, total }
  for (let pop = 0; pop <= 6; pop++) {
    for (let upper = 0; upper <= 63; upper++) {
      let reachable = 0;
      let total = 0;
      for (let m = 0; m < nMasks; m++) {
        if (popcount6(m) !== pop) continue;
        total++;
        // Check if this mask can reach this upper score
        // Cap: anything >= 63 maps to 63
        let found = false;
        if (upper < 63) {
          found = R[idx(upper, m)] === 1;
        } else {
          // upper = 63 means "63 or more"
          for (let n = 63; n <= maxUpper; n++) {
            if (R[idx(n, m)]) { found = true; break; }
          }
        }
        if (found) reachable++;
      }
      if (total > 0) {
        result.push({ pop, upper, fraction: reachable / total, reachable, total });
      }
    }
  }

  return result;
}

function popcount6(m) {
  let c = 0;
  for (let i = 0; i < 6; i++) if (m & (1 << i)) c++;
  return c;
}

export async function initReachabilityGrid() {
  const container = document.getElementById('chart-reachability-grid');
  if (!container) return;

  const data = computeReachability();

  const chart = createChart('chart-reachability-grid', {
    aspectRatio: 0.5,
    marginLeft: 70,
    marginBottom: 50,
    marginTop: 30,
    marginRight: 20,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const pops = [0, 1, 2, 3, 4, 5, 6];
  const uppers = d3.range(0, 64);

  const cellW = width / uppers.length;
  const cellH = height / pops.length;

  // Color scale: 0 = gray, 1 = full orange
  const colorScale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(t => {
      if (t === 0) return getGridColor();
      return d3.interpolateOranges(0.2 + t * 0.7);
    });

  const tt = tooltip(container);

  // Draw cells
  data.forEach(d => {
    const x = d.upper * cellW;
    const y = d.pop * cellH;

    g.append('rect')
      .attr('x', x)
      .attr('y', y)
      .attr('width', cellW - 0.5)
      .attr('height', cellH - 0.5)
      .attr('fill', d.fraction === 0 ? getGridColor() : colorScale(d.fraction))
      .attr('opacity', d.fraction === 0 ? 0.3 : 0.85)
      .attr('rx', 1)
      .on('mousemove', (event) => {
        tt.show(
          `<div class="tt-label">Upper cats: ${d.pop}, Score: ${d.upper}${d.upper === 63 ? '+' : ''}</div>
           <div>Reachable: <span class="tt-value">${d.reachable}/${d.total}</span> (${(d.fraction * 100).toFixed(0)}%)</div>`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());
  });

  // Bonus threshold line at x=63
  const bonusX = 63 * cellW;
  g.append('line')
    .attr('x1', bonusX).attr('x2', bonusX)
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', COLORS.accent)
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3');

  g.append('text')
    .attr('x', bonusX - 4)
    .attr('y', -8)
    .attr('text-anchor', 'end')
    .attr('fill', COLORS.accent)
    .style('font-size', '9px')
    .style('font-weight', '600')
    .text('Bonus threshold (63)');

  // Axes
  const xScale = d3.scaleLinear().domain([0, 63]).range([0, width]);
  drawAxis(
    g.append('g').attr('transform', `translate(0,${height})`),
    xScale, 'bottom', 'Upper Section Score'
  );

  const yScale = d3.scaleBand().domain(pops).range([0, height]);
  const yAxisG = g.append('g').call(d3.axisLeft(yScale));
  yAxisG.selectAll('line').attr('stroke', getGridColor());
  yAxisG.selectAll('path').attr('stroke', getGridColor());
  yAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");
  yAxisG.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -50)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text('Upper Categories Scored');

  // Title annotation
  g.append('text')
    .attr('x', width / 2)
    .attr('y', -12)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '12px')
    .style('font-weight', '600')
    .text('Gray cells = unreachable (32% of state space pruned)');
}
