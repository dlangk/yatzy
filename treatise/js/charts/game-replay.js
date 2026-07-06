import {
  tooltip, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor,
} from '../yatzy-viz.js';

// Curated game: "The Surge"
// P1 leads throughout. P1 scores Yatzy on turn 10, opening a 60-point gap.
// P2 adapts with maximum risk, scores their own Yatzy on turn 11, and wins by 11.
// Score arrays: index 1-15 = cumulative after each turn.
const P1 = [15, 33, 48, 63, 80, 98, 112, 125, 138, 155, 205, 215, 225, 232, 240];
const P2 = [12, 24, 36, 51, 63, 81, 95, 114, 127, 144, 168, 218, 233, 243, 251];
// θ selected by P2 for each turn (index 0 = turn 1, ..., 14 = turn 15).
// Computed as gap / V_total, clamped to [-0.05, +0.07]. Turns 1-2 forced to 0.
const THETA = [
  0, 0, 0.010, 0.012, 0.015, 0.021, 0.022, 0.026,
  0.027, 0.041, 0.070, 0.070, 0.044, -0.017, -0.050,
];
const V_TOTAL = [2960, 2700, 2440, 2200, 1920, 1660, 1400, 1140, 900, 680, 480, 300, 160, 60, 20];
const P1_FINAL = 290;
const P2_FINAL = 301;

export async function initGameReplay() {
  const container = document.getElementById('chart-game-replay');
  if (!container) return;
  const target = document.getElementById('chart-game-replay-svg');
  if (!target) return;

  const tt = tooltip(container);
  const W = target.clientWidth || 635;
  const H = Math.round(W * 0.52);
  const ml = 48, mr = 58, mt = 10, mb = 40, gap = 6;
  const w = W - ml - mr;
  const sH = Math.round((H - mt - mb - gap) * 0.62);
  const tH = H - mt - mb - gap - sH;

  const svg = d3.select(target).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W).attr('height', H)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const x = d3.scaleLinear().domain([1, 15]).range([0, w]);

  // ── Score panel ──
  const sg = svg.append('g').attr('transform', `translate(${ml},${mt})`);
  const yS = d3.scaleLinear().domain([0, 270]).range([sH, 0]);

  // horizontal grid
  yS.ticks(5).forEach(v => {
    sg.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', yS(v)).attr('y2', yS(v))
      .attr('stroke', getGridColor()).attr('stroke-dasharray', '2,4');
  });

  // Y axis
  sg.append('g').call(d3.axisLeft(yS).ticks(5))
    .call(g => g.select('.domain').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick line').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick text').attr('fill', getMutedColor()).style('font-size', '11px'));
  sg.append('text')
    .attr('transform', `translate(-34,${sH / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '11px')
    .text('Score');

  // P1 line (muted)
  sg.append('path').datum(P1)
    .attr('d', d3.line().x((_, i) => x(i + 1)).y(d => yS(d)))
    .attr('fill', 'none').attr('stroke', getMutedColor())
    .attr('stroke-width', 1.5).attr('opacity', 0.5);

  // P2 line segments colored by θ
  for (let i = 1; i < 15; i++) {
    sg.append('line')
      .attr('x1', x(i)).attr('y1', yS(P2[i - 1]))
      .attr('x2', x(i + 1)).attr('y2', yS(P2[i]))
      .attr('stroke', thetaColor(THETA[i])).attr('stroke-width', 2.5);
  }

  // dots
  P1.forEach((s, i) => sg.append('circle')
    .attr('cx', x(i + 1)).attr('cy', yS(s)).attr('r', 2.5)
    .attr('fill', getMutedColor()).attr('opacity', 0.5));
  P2.forEach((s, i) => sg.append('circle')
    .attr('cx', x(i + 1)).attr('cy', yS(s)).attr('r', 3)
    .attr('fill', thetaColor(THETA[i])));

  // Yatzy annotations
  sg.append('text').attr('x', x(10)).attr('y', yS(205) - 10)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor())
    .style('font-size', '10px').style('font-weight', '600').text('Yatzy!');
  sg.append('text').attr('x', x(11)).attr('y', yS(218) - 10)
    .attr('text-anchor', 'middle').attr('fill', thetaColor(0.07))
    .style('font-size', '10px').style('font-weight', '600').text('Yatzy!');

  // final scores
  sg.append('text').attr('x', x(15) + 6).attr('y', yS(P1[14]))
    .attr('dominant-baseline', 'middle').attr('fill', getMutedColor())
    .style('font-size', '10px').text(P1_FINAL);
  sg.append('text').attr('x', x(15) + 6).attr('y', yS(P2[14]))
    .attr('dominant-baseline', 'middle').attr('fill', getTextColor())
    .style('font-size', '10px').style('font-weight', '600').text(`${P2_FINAL} \u2605`);

  // legend (bottom of score panel)
  const ly = sH - 8;
  sg.append('line').attr('x1', w - 96).attr('x2', w - 80).attr('y1', ly).attr('y2', ly)
    .attr('stroke', getMutedColor()).attr('stroke-width', 1.5).attr('opacity', 0.5);
  sg.append('text').attr('x', w - 76).attr('y', ly).attr('dominant-baseline', 'middle')
    .attr('fill', getMutedColor()).style('font-size', '10px').text('Player 1');
  sg.append('line').attr('x1', w - 96).attr('x2', w - 80).attr('y1', ly - 16).attr('y2', ly - 16)
    .attr('stroke', thetaColor(0.03)).attr('stroke-width', 2.5);
  sg.append('text').attr('x', w - 76).attr('y', ly - 16).attr('dominant-baseline', 'middle')
    .attr('fill', getTextColor()).style('font-size', '10px').text('Player 2');

  // ── Divider line ──
  svg.append('line')
    .attr('x1', ml).attr('x2', ml + w)
    .attr('y1', mt + sH + gap / 2).attr('y2', mt + sH + gap / 2)
    .attr('stroke', getGridColor()).attr('stroke-width', 0.5);

  // ── Theta panel ──
  const tg = svg.append('g').attr('transform', `translate(${ml},${mt + sH + gap})`);
  const yT = d3.scaleLinear().domain([-0.075, 0.085]).range([tH, 0]);

  // Remaining variance area (background, drawn first so bars sit on top)
  const yV = d3.scaleLinear().domain([0, 3000]).range([yT(0), 0]);
  const varArea = d3.area()
    .x((_, i) => x(i + 1))
    .y0(yT(0))
    .y1((d) => yV(d))
    .curve(d3.curveMonotoneX);
  tg.append('path').datum(V_TOTAL)
    .attr('d', varArea)
    .attr('fill', getMutedColor()).attr('opacity', 0.06);
  tg.append('path').datum(V_TOTAL)
    .attr('d', d3.line().x((_, i) => x(i + 1)).y(d => yV(d)).curve(d3.curveMonotoneX))
    .attr('fill', 'none').attr('stroke', getMutedColor())
    .attr('stroke-width', 0.75).attr('opacity', 0.25);

  // Variance label
  tg.append('text')
    .attr('x', x(2.5)).attr('y', yV(2600) - 4)
    .attr('fill', getMutedColor()).style('font-size', '9px').attr('opacity', 0.5)
    .text('Remaining variance');

  // zero line
  tg.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', yT(0)).attr('y2', yT(0))
    .attr('stroke', getGridColor());

  // clamp lines + left-side labels
  [0.07, -0.05].forEach(v => {
    tg.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', yT(v)).attr('y2', yT(v))
      .attr('stroke', getMutedColor()).attr('stroke-dasharray', '3,3').attr('stroke-width', 0.5);
    tg.append('text').attr('x', -4).attr('y', yT(v))
      .attr('dominant-baseline', 'middle').attr('text-anchor', 'end')
      .attr('fill', getMutedColor())
      .style('font-size', '9px').text(v > 0 ? '+0.07' : '\u22120.05');
  });

  // θ bars
  const bw = w / 15 * 0.35;
  THETA.forEach((th, i) => {
    const y0 = yT(0), y1 = yT(th);
    tg.append('rect')
      .attr('x', x(i + 1) - bw / 2)
      .attr('y', Math.min(y0, y1))
      .attr('width', bw)
      .attr('height', Math.max(1, Math.abs(y1 - y0)))
      .attr('fill', thetaColor(th)).attr('rx', 1);
  });

  // "Risk" label on left side
  tg.append('text')
    .attr('transform', `translate(-34,${tH / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '11px')
    .text('Risk');

  // X axis
  tg.append('g').attr('transform', `translate(0,${tH})`)
    .call(d3.axisBottom(x).tickValues(d3.range(1, 16)).tickFormat(d => d))
    .call(g => g.select('.domain').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick line').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick text').attr('fill', getMutedColor()).style('font-size', '10px'));
  tg.append('text').attr('x', w / 2).attr('y', tH + 24)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '11px')
    .text('Turn');

  // ── Hover interaction ──
  const hl = svg.append('line')
    .attr('y1', mt).attr('y2', mt + sH + gap + tH)
    .attr('stroke', getTextColor()).attr('stroke-width', 0.5)
    .attr('opacity', 0).attr('pointer-events', 'none');

  svg.append('rect')
    .attr('x', ml).attr('y', mt).attr('width', w).attr('height', sH + gap + tH)
    .attr('fill', 'transparent')
    .on('mousemove', function (event) {
      const mx = d3.pointer(event, this)[0];
      const turn = Math.round(x.invert(mx - ml));
      if (turn < 1 || turn > 15) { hl.attr('opacity', 0); tt.hide(); return; }
      hl.attr('x1', ml + x(turn)).attr('x2', ml + x(turn)).attr('opacity', 0.3);
      const p1s = turn === 1 ? P1[0] : P1[turn - 1] - P1[turn - 2];
      const p2s = turn === 1 ? P2[0] : P2[turn - 1] - P2[turn - 2];
      const th = THETA[turn - 1];
      const sc = P1[turn - 1] - P2[turn - 1];
      const v = V_TOTAL[turn - 1];
      tt.show(`<div class="tt-label">Turn ${turn}</div>
        <table style="border-collapse:collapse;margin-top:4px;line-height:1.4">
        <tr><td style="padding:0 8px 0 0;border:none;color:${getMutedColor()}">P1 scored</td>
            <td style="padding:0;border:none;text-align:right">${p1s}${turn === 10 ? ' (Yatzy)' : ''}</td></tr>
        <tr><td style="padding:0 8px 0 0;border:none">P2 scored</td>
            <td style="padding:0;border:none;text-align:right">${p2s}${turn === 11 ? ' (Yatzy)' : ''}</td></tr>
        <tr><td style="padding:0 8px 0 0;border:none">Gap</td>
            <td style="padding:0;border:none;text-align:right">${sc > 0 ? '+' : ''}${sc}</td></tr>
        <tr><td style="padding:0 8px 0 0;border:none">Variance</td>
            <td style="padding:0;border:none;text-align:right">${v.toLocaleString()}</td></tr>
        <tr><td style="padding:0 8px 0 0;border:none">θ</td>
            <td style="padding:0;border:none;text-align:right;color:${thetaColor(th)}">${formatTheta(th)}</td></tr>
        </table>`, event);
    })
    .on('mouseleave', () => { hl.attr('opacity', 0); tt.hide(); });
}
