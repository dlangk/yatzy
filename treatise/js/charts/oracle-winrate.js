import {
  createChart, tooltip, drawAxis, thetaColor, formatTheta,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

// Per-band data: optimal θ and win rates vs EV-optimal opponent.
// EV rates interpolated from θ=−0.005 and θ=+0.005 rows of winrate_conditional.csv.
// Oracle rates: best fixed θ for each band from the same dataset.
const BANDS = [
  { label: '< 200',   evRate: 0.9552, oracleRate: 0.9598, bestTheta: -0.030, weight: 87894 },
  { label: '200-220', evRate: 0.8523, oracleRate: 0.8591, bestTheta: -0.030, weight: 112525 },
  { label: '220-240', evRate: 0.6902, oracleRate: 0.6927, bestTheta: -0.015, weight: 206655 },
  { label: '240-250', evRate: 0.5364, oracleRate: 0.5371, bestTheta: -0.005, weight: 101813 },
  { label: '250-260', evRate: 0.4305, oracleRate: 0.4320, bestTheta: +0.010, weight: 109461 },
  { label: '260-280', evRate: 0.2980, oracleRate: 0.3036, bestTheta: +0.020, weight: 159071 },
  { label: '280+',    evRate: 0.1087, oracleRate: 0.1213, bestTheta: +0.040, weight: 222581 },
];

export async function initOracleWinrate() {
  const container = document.getElementById('chart-oracle-winrate');
  if (!container) return;

  const totalWeight = BANDS.reduce((s, b) => s + b.weight, 0);
  const oracleOverall = BANDS.reduce((s, b) => s + b.oracleRate * b.weight, 0) / totalWeight;
  const evOverall = BANDS.reduce((s, b) => s + b.evRate * b.weight, 0) / totalWeight;
  const gainPP = (oracleOverall - evOverall) * 100;

  const marginTop = 36, marginRight = 24, marginBottom = 72, marginLeft = 56;
  const containerWidth = container.clientWidth || 635;
  const totalWidth = containerWidth;
  const totalHeight = Math.round(totalWidth * 0.52);
  const width  = totalWidth  - marginLeft - marginRight;
  const height = totalHeight - marginTop  - marginBottom;

  container.querySelectorAll('svg').forEach(s => s.remove());

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`)
    .attr('width', totalWidth)
    .attr('height', totalHeight)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const g = svg.append('g').attr('transform', `translate(${marginLeft},${marginTop})`);

  // Scales
  const bandW = width / BANDS.length;
  const barGap = 0.18;
  const barW = bandW * (0.5 - barGap / 2);

  const x = d3.scaleBand()
    .domain(BANDS.map(b => b.label))
    .range([0, width])
    .paddingInner(0.12)
    .paddingOuter(0.06);

  const y = d3.scaleLinear()
    .domain([0, 1.05])
    .range([height, 0]);

  // Grid lines
  g.append('g').selectAll('line')
    .data(y.ticks(6))
    .join('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', d => y(d)).attr('y2', d => y(d))
    .attr('stroke', getGridColor())
    .attr('stroke-dasharray', '2,3');

  // 50% reference
  g.append('line')
    .attr('x1', 0).attr('x2', width)
    .attr('y1', y(0.5)).attr('y2', y(0.5))
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '6,3');

  g.append('text')
    .attr('x', 4).attr('y', y(0.5) - 5)
    .attr('fill', getMutedColor())
    .style('font-size', '10px')
    .text('50%');

  const bw = x.bandwidth();

  // EV bars (muted)
  g.selectAll('.bar-ev')
    .data(BANDS)
    .join('rect')
    .attr('class', 'bar-ev')
    .attr('x', d => x(d.label) + bw * 0.03)
    .attr('y', d => y(d.evRate))
    .attr('width', bw * 0.46)
    .attr('height', d => height - y(d.evRate))
    .attr('fill', getMutedColor())
    .attr('opacity', 0.35)
    .attr('rx', 1);

  // Oracle bars (colored by best theta)
  g.selectAll('.bar-oracle')
    .data(BANDS)
    .join('rect')
    .attr('class', 'bar-oracle')
    .attr('x', d => x(d.label) + bw * 0.51)
    .attr('y', d => y(d.oracleRate))
    .attr('width', bw * 0.46)
    .attr('height', d => height - y(d.oracleRate))
    .attr('fill', d => thetaColor(d.bestTheta))
    .attr('rx', 1);

  // θ labels below oracle bar (inside bottom of band label area)
  g.selectAll('.lbl-theta')
    .data(BANDS)
    .join('text')
    .attr('class', 'lbl-theta')
    .attr('x', d => x(d.label) + bw * 0.51 + bw * 0.23)
    .attr('y', height + 28)
    .attr('text-anchor', 'middle')
    .attr('fill', d => thetaColor(d.bestTheta))
    .style('font-size', '10px')
    .style('font-weight', '600')
    .text(d => {
      const s = d.bestTheta > 0 ? '+' : (d.bestTheta < 0 ? '' : '\u00a0');
      return `\u03b8${s}${d.bestTheta.toFixed(3)}`;
    });

  // Legend
  const legX = width - 160, legY = 8;
  g.append('rect').attr('x', legX).attr('y', legY).attr('width', 12).attr('height', 12)
    .attr('fill', getMutedColor()).attr('opacity', 0.35).attr('rx', 1);
  g.append('text').attr('x', legX + 16).attr('y', legY + 10)
    .attr('fill', getTextColor()).style('font-size', '11px').text('EV-optimal (\u03b8 = 0)');

  g.append('rect').attr('x', legX).attr('y', legY + 18).attr('width', 12).attr('height', 12)
    .attr('fill', COLORS.accent).attr('rx', 1);
  g.append('text').attr('x', legX + 16).attr('y', legY + 28)
    .attr('fill', getTextColor()).style('font-size', '11px').text('Oracle adaptive');

  // Oracle gain annotation
  g.append('text')
    .attr('x', width / 2)
    .attr('y', -18)
    .attr('text-anchor', 'middle')
    .attr('fill', getTextColor())
    .style('font-size', '11px')
    .html(`Oracle upper bound: EV ${(evOverall * 100).toFixed(2)}%  \u2192  ${(oracleOverall * 100).toFixed(2)}%  (+${gainPP.toFixed(3)} pp)`);

  // Axes
  const xAxis = d3.axisBottom(x);
  const xAxisG = g.append('g')
    .attr('transform', `translate(0,${height})`)
    .call(xAxis);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");
  xAxisG.append('text')
    .attr('x', width / 2).attr('y', 56)
    .attr('fill', getMutedColor()).attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text("Opponent's final score band");

  drawAxis(g, y, 'left', 'Your win rate', { tickFormat: d3.format('.0%') });

  // Tooltip
  const tt = tooltip(container);

  // Invisible hit areas per band
  BANDS.forEach(band => {
    g.append('rect')
      .attr('x', x(band.label))
      .attr('y', 0)
      .attr('width', x.bandwidth())
      .attr('height', height)
      .attr('fill', 'transparent')
      .attr('cursor', 'default')
      .on('mousemove', (event) => {
        const gain = (band.oracleRate - band.evRate) * 100;
        const theta_s = (band.bestTheta > 0 ? '+' : '') + band.bestTheta.toFixed(3);
        tt.show(
          `<div class="tt-label">Opponent scored ${band.label}</div>
           <table style="border-collapse:collapse;margin-top:4px;line-height:1.4">
             <tr><td style="color:${getMutedColor()};padding:0 8px 0 0;white-space:nowrap;border:none">EV-optimal</td>
                 <td style="padding:0;white-space:nowrap;border:none;font-weight:600">${(band.evRate * 100).toFixed(1)}%</td></tr>
             <tr><td style="color:${thetaColor(band.bestTheta)};padding:0 8px 0 0;white-space:nowrap;border:none">Oracle (\u03b8=${theta_s})</td>
                 <td style="padding:0;white-space:nowrap;border:none;font-weight:600">${(band.oracleRate * 100).toFixed(1)}%</td></tr>
             <tr><td style="padding:0 8px 0 0;white-space:nowrap;border:none">Gain</td>
                 <td style="padding:0;white-space:nowrap;border:none;color:${COLORS.accent};font-weight:600">+${gain.toFixed(2)} pp</td></tr>
           </table>`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());
  });
}
