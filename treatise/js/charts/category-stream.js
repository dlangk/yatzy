import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

const CATEGORY_NAMES = [
  'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
  'One Pair', 'Two Pairs', 'Three of a Kind', 'Four of a Kind',
  'Small Straight', 'Large Straight', 'Full House', 'Chance', 'Yatzy',
];

// Upper in cool tones, lower in warm tones
const CATEGORY_COLORS = [
  '#4e79a7', '#59a14f', '#76b7b2', '#6baed6', '#9ecae1', '#3b4cc0',
  '#e15759', '#f28e2b', '#edc948', '#b07aa1',
  '#ff9da7', '#d4a373', '#d95f02', '#636363', '#F37021',
];

export async function initCategoryStream() {
  const raw = await DataLoader.categorySankey();
  const container = document.getElementById('chart-category-stream');
  if (!container) return;

  // Find all unique categories and turns
  const categories = [...new Set(raw.map(d => d.category))];
  const turns = [...new Set(raw.map(d => d.turn))].sort((a, b) => a - b);

  // Pivot to turn × category matrix
  const dataByTurn = turns.map(t => {
    const row = { turn: t };
    categories.forEach(cat => { row[cat] = 0; });
    raw.filter(d => d.turn === t).forEach(d => { row[d.category] = d.mass; });
    return row;
  });

  // Sort categories: use CATEGORY_NAMES order
  const orderedCats = CATEGORY_NAMES.filter(c => categories.includes(c));

  const chart = createChart('chart-category-stream', {
    aspectRatio: 0.65,
    marginTop: 15,
    marginRight: 20,
    marginBottom: 90,
    marginLeft: 45,
  });
  if (!chart) return;
  const { svg, g, width, height } = chart;

  const x = d3.scaleLinear()
    .domain(d3.extent(turns))
    .range([0, width]);

  const stack = d3.stack()
    .keys(orderedCats)
    .offset(d3.stackOffsetWiggle)
    .order(d3.stackOrderInsideOut);

  const series = stack(dataByTurn);

  const yExtent = [
    d3.min(series, s => d3.min(s, d => d[0])),
    d3.max(series, s => d3.max(s, d => d[1])),
  ];
  const y = d3.scaleLinear().domain(yExtent).range([height, 0]);

  // Color lookup
  const catColorMap = new Map();
  orderedCats.forEach((cat, i) => {
    catColorMap.set(cat, CATEGORY_COLORS[CATEGORY_NAMES.indexOf(cat)] || CATEGORY_COLORS[i % CATEGORY_COLORS.length]);
  });

  const area = d3.area()
    .x(d => x(d.data.turn))
    .y0(d => y(d[0]))
    .y1(d => y(d[1]))
    .curve(d3.curveBasis);

  // Draw streams
  const paths = g.selectAll('.stream-layer')
    .data(series)
    .join('path')
    .attr('class', 'stream-layer')
    .attr('d', area)
    .attr('fill', d => catColorMap.get(d.key))
    .attr('opacity', 0.85)
    .attr('stroke', 'none');

  // Interaction: dim others on hover
  const tt = tooltip(container);

  const scrubber = g.append('line')
    .attr('y1', 0).attr('y2', height)
    .attr('stroke', getMutedColor())
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '3,3')
    .attr('opacity', 0);

  paths
    .on('mouseenter', function (event, d) {
      paths.attr('opacity', 0.15);
      d3.select(this).attr('opacity', 1);
      scrubber.attr('opacity', 0.6);
    })
    .on('mousemove', function (event, d) {
      const [mx] = d3.pointer(event);
      const turnVal = x.invert(mx);
      const turnIdx = Math.round(turnVal);
      const clamped = Math.max(turns[0], Math.min(turns[turns.length - 1], turnIdx));
      const row = dataByTurn.find(r => r.turn === clamped);
      const mass = row ? row[d.key] : 0;

      scrubber.attr('x1', mx).attr('x2', mx);

      tt.show(
        `<div class="tt-label">${d.key}</div>
         <div>Turn ${clamped}: <span class="tt-value">${(mass * 100).toFixed(1)}%</span></div>`,
        event
      );
    })
    .on('mouseleave', () => {
      paths.attr('opacity', 0.85);
      scrubber.attr('opacity', 0);
      tt.hide();
    });

  // X axis
  const xAxisG = g.append('g').attr('transform', `translate(0,${height})`);
  const xAxis = d3.axisBottom(x).ticks(15).tickFormat(d3.format('d'));
  xAxisG.call(xAxis);
  xAxisG.selectAll('line').attr('stroke', getGridColor());
  xAxisG.selectAll('path').attr('stroke', getGridColor());
  xAxisG.selectAll('text')
    .attr('fill', getMutedColor())
    .style('font-size', '11px')
    .style('font-family', "'Newsreader', Georgia, serif");

  xAxisG.append('text')
    .attr('x', width / 2)
    .attr('y', 35)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .text('Turn');

  // Legend — compact rows below chart
  const legendG = g.append('g')
    .attr('transform', `translate(0, ${height + 45})`);

  const swatchSize = 10;
  const colWidth = width / 5;

  orderedCats.forEach((cat, i) => {
    const col = i % 5;
    const row = Math.floor(i / 5);
    const lx = col * colWidth;
    const ly = row * 14;

    legendG.append('rect')
      .attr('x', lx).attr('y', ly)
      .attr('width', swatchSize).attr('height', swatchSize)
      .attr('fill', catColorMap.get(cat))
      .attr('rx', 2);

    legendG.append('text')
      .attr('x', lx + swatchSize + 4)
      .attr('y', ly + swatchSize - 1)
      .attr('fill', getMutedColor())
      .style('font-size', '9px')
      .text(cat);
  });
}
