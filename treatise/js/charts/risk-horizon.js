import {
  tooltip, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor,
} from '../yatzy-viz.js';

// Approximate combined remaining variance (both players) at each turn.
// Decreases as categories are scored and uncertainty resolves.
const V_TOTAL = [
  2960, 2700, 2440, 2200, 1920, 1660, 1400, 1140, 900, 680, 480, 300, 160, 60, 20,
];

export async function initRiskHorizon() {
  const container = document.getElementById('chart-risk-horizon');
  if (!container) return;
  const target = document.getElementById('chart-risk-horizon-svg');
  if (!target) return;

  const slider = container.querySelector('.chart-slider');
  const sliderLabel = container.querySelector('.slider-value');
  const tt = tooltip(container);

  const W = target.clientWidth || 635;
  const H = Math.round(W * 0.42);
  const ml = 48, mr = 20, mt = 10, mb = 36;
  const w = W - ml - mr, h = H - mt - mb;

  const svg = d3.select(target).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W).attr('height', H)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${ml},${mt})`);
  const x = d3.scaleLinear().domain([1, 15]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.08, 0.12]).range([h, 0]);

  // zero line
  g.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', y(0)).attr('y2', y(0)).attr('stroke', getGridColor());

  // degeneration zones (faint shading)
  g.append('rect').attr('x', 0).attr('y', y(0.12)).attr('width', w)
    .attr('height', y(0.07) - y(0.12))
    .attr('fill', thetaColor(0.20)).attr('opacity', 0.06);
  g.append('rect').attr('x', 0).attr('y', y(-0.05)).attr('width', w)
    .attr('height', y(-0.08) - y(-0.05))
    .attr('fill', thetaColor(-0.20)).attr('opacity', 0.06);

  // zone label
  g.append('text').attr('x', w - 4).attr('y', y(0.095))
    .attr('text-anchor', 'end').attr('fill', getMutedColor())
    .style('font-size', '9px').attr('opacity', 0.5).text('CARA degeneration');

  // clamp lines
  [0.07, -0.05].forEach(v => {
    g.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', getMutedColor()).attr('stroke-dasharray', '4,3').attr('stroke-width', 0.5);
  });

  // axes
  g.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickValues(d3.range(1, 16)).tickFormat(d => d))
    .call(g => g.select('.domain').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick line').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick text').attr('fill', getMutedColor()).style('font-size', '10px'));
  g.append('text').attr('x', w / 2).attr('y', h + 32)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '11px')
    .text('Turn');

  g.append('g')
    .call(d3.axisLeft(y).tickValues([-0.05, 0, 0.03, 0.05, 0.07, 0.10]).tickFormat(d3.format('+.02f')))
    .call(g => g.select('.domain').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick line').attr('stroke', getGridColor()))
    .call(g => g.selectAll('.tick text').attr('fill', getMutedColor()).style('font-size', '10px'));
  g.append('text')
    .attr('transform', `translate(-34,${h / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').attr('fill', getMutedColor()).style('font-size', '11px')
    .text('θ');

  const dyn = g.append('g');

  function render(deficit) {
    dyn.selectAll('*').remove();
    if (deficit === 0) {
      dyn.append('text').attr('x', w / 2).attr('y', y(0) - 14)
        .attr('text-anchor', 'middle').attr('fill', getMutedColor())
        .style('font-size', '11px').style('font-style', 'italic')
        .text('Scores even: no risk adjustment needed');
      return;
    }

    const sign = deficit > 0 ? 1 : -1;
    const line = d3.line().x(d => x(d[0])).y(d => y(d[1]));

    // Variance-scaled: raw (unclamped, faded)
    const raw = [];
    for (let i = 0; i < 15; i++) {
      const th = i <= 1 ? 0 : deficit / V_TOTAL[i];
      raw.push([i + 1, th]);
    }
    const clipped = raw.map(d => [d[0], Math.max(-0.08, Math.min(0.12, d[1]))]);
    dyn.append('path').datum(clipped).attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', thetaColor(sign * 0.08)).attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3').attr('opacity', 0.30);

    // Variance-scaled: clamped (solid, colored segments)
    const clamped = [];
    for (let i = 0; i < 15; i++) {
      const th = i <= 1 ? 0 : Math.max(-0.05, Math.min(0.07, deficit / V_TOTAL[i]));
      clamped.push([i + 1, th]);
    }
    for (let i = 1; i < clamped.length; i++) {
      dyn.append('line')
        .attr('x1', x(clamped[i - 1][0])).attr('y1', y(clamped[i - 1][1]))
        .attr('x2', x(clamped[i][0])).attr('y2', y(clamped[i][1]))
        .attr('stroke', thetaColor(clamped[i][1])).attr('stroke-width', 2.5);
    }
    clamped.forEach(d => {
      if (d[0] >= 3) {
        dyn.append('circle').attr('cx', x(d[0])).attr('cy', y(d[1]))
          .attr('r', 3).attr('fill', thetaColor(d[1]));
      }
    });

    // Linear policy (dashed gray)
    const linTh = Math.max(-0.03, Math.min(0.05, deficit / 50));
    const linData = [];
    for (let i = 0; i < 15; i++) linData.push([i + 1, i <= 1 ? 0 : linTh]);
    dyn.append('path').datum(linData).attr('d', line)
      .attr('fill', 'none').attr('stroke', getMutedColor())
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '6,4').attr('opacity', 0.7);

    // Legend
    const ls = 12;
    const lyCen = (y(0.12) + y(0.07)) / 2;
    const lx = 14, ly0 = lyCen - ls;
    dyn.append('line').attr('x1', lx).attr('x2', lx + 18).attr('y1', ly0).attr('y2', ly0)
      .attr('stroke', thetaColor(sign * 0.04)).attr('stroke-width', 2.5);
    dyn.append('text').attr('x', lx + 22).attr('y', ly0).attr('dominant-baseline', 'middle')
      .attr('fill', getTextColor()).style('font-size', '10px').text('Variance-scaled');
    dyn.append('line').attr('x1', lx).attr('x2', lx + 18).attr('y1', ly0 + ls).attr('y2', ly0 + ls)
      .attr('stroke', getMutedColor()).attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '6,4').attr('opacity', 0.7);
    dyn.append('text').attr('x', lx + 22).attr('y', ly0 + ls).attr('dominant-baseline', 'middle')
      .attr('fill', getMutedColor()).style('font-size', '10px').text('Linear');
    dyn.append('line').attr('x1', lx).attr('x2', lx + 18).attr('y1', ly0 + 2 * ls).attr('y2', ly0 + 2 * ls)
      .attr('stroke', thetaColor(sign * 0.08)).attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '3,3').attr('opacity', 0.3);
    dyn.append('text').attr('x', lx + 22).attr('y', ly0 + 2 * ls).attr('dominant-baseline', 'middle')
      .attr('fill', getMutedColor()).style('font-size', '10px').text('Unclamped θ*');
  }

  render(20);

  if (slider) {
    slider.addEventListener('input', function () {
      const d = +this.value;
      if (sliderLabel) {
        sliderLabel.textContent = d > 0 ? `+${d}` : d === 0 ? '\u00a00' : `${d}`;
      }
      render(d);
    });
  }
}
