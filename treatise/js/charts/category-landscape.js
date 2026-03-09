import { DataLoader } from '../data-loader.js';
import {
  createChart,
  getGridColor,
  getMutedColor,
  tooltip
} from '../yatzy-viz.js';

const PROPERTIES = {
  mean_fill_turn:        { label: 'Mean fill turn',        fmt: '.1f',  domain: [1, 15]       },
  score_pct_ceiling:     { label: 'Score % of ceiling',    fmt: '.0%',  domain: [0, 1]        },
  variance_contribution: { label: 'Variance contribution', fmt: '.1f',  domain: null          },
  mean_score:            { label: 'Mean score',            fmt: '.1f',  domain: [0, 50]       },
  ceiling:               { label: 'Ceiling',               fmt: 'd',    domain: [0, 50]       },
  zero_rate:             { label: 'Zero rate',             fmt: '.1%',  domain: [0, 1]        },
  hit_rate:              { label: 'Hit rate',              fmt: '.1%',  domain: [0, 1]        },
  score_std:             { label: 'Score std dev',         fmt: '.2f',  domain: null          },
  score_skewness:        { label: 'Score skewness',        fmt: '.2f',  domain: null          },
  fill_turn_std:         { label: 'Fill turn std dev',     fmt: '.2f',  domain: [0, 15]       },
  fill_turn_entropy:     { label: 'Fill turn entropy',     fmt: '.2f',  domain: [0, Math.log2(15)] },
  bonus_dependency:      { label: 'Bonus dependency',      fmt: '.2f',  domain: null          },
  opportunity_cost:      { label: 'Opportunity cost',      fmt: '.2f',  domain: null          },
};

const INSIGHTS = [
  {
    label: 'The Yatzy Outlier',
    x: 'score_std', y: 'variance_contribution', size: 'opportunity_cost',
    text: 'As the name of the game suggests, Yatzy is an outlier. Its variance contribution (583) is 4x the next highest. Its score variance and opportunity cost are also outliers. It is the single biggest source of outcome randomness, even under perfect play.',
  },
  {
    label: 'Risk vs Reward',
    x: 'zero_rate', y: 'opportunity_cost', size: 'variance_contribution',
    text: 'Categories that fail most often also cost the most when they do. Yatzy, Straights, and Four of a Kind are all "dangerous", i.e. they have high zero rates and high opportunity cost.',
  },
  {
    label: 'Bonus Dependency',
    x: 'bonus_dependency', y: 'score_pct_ceiling', size: 'mean_score',
    text: 'We define this as mean(category_score | got_bonus) - mean(category_score | no_bonus). Four of a Kind and Sixes both have higher scores in bonus games, while Yatzy and Ones have negative dependency.',
  },
  {
    label: 'Timing Flexibility',
    x: 'mean_fill_turn', y: 'fill_turn_entropy', size: 'score_std',
    text: 'There is no fixed order. Every category can be filled on almost any turn. The solver adapts to what the dice give it rather than following a rigid script like "fill Sixes on turn 3".',
  },
];

export async function initCategoryLandscape() {
  const data = await DataLoader.categoryLandscape();
  const container = document.getElementById('chart-category-landscape');
  if (!container) return;

  // State (matches first insight: The Yatzy outlier)
  let xProp = 'score_std';
  let yProp = 'variance_contribution';
  let sizeProp = 'opportunity_cost';
  let activeInsight = 0;

  // Move caption to end of container (below SVG)
  const caption = container.querySelector('.chart-caption');
  container.appendChild(caption);

  // Insight buttons (inserted first, so they appear above dropdowns)
  const insightRow = document.createElement('div');
  insightRow.className = 'chart-controls';
  container.prepend(insightRow);

  // Insight text (after buttons, before dropdowns)
  const insightText = document.createElement('p');
  insightText.className = 'insight-text';
  insightRow.after(insightText);

  // Dropdown controls (after insight text)
  const controlsDiv = document.createElement('div');
  controlsDiv.className = 'chart-controls';
  insightText.after(controlsDiv);

  const selects = {};

  function makeSelect(labelText, key, initial, onChange) {
    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.alignItems = 'center';
    wrapper.style.gap = '0.35rem';

    const lbl = document.createElement('label');
    lbl.textContent = labelText;
    wrapper.appendChild(lbl);

    const sel = document.createElement('select');
    sel.className = 'chart-select';
    for (const [k, cfg] of Object.entries(PROPERTIES)) {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = cfg.label;
      if (k === initial) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      onChange(sel.value);
      clearActiveInsight();
    });
    wrapper.appendChild(sel);
    controlsDiv.appendChild(wrapper);
    selects[key] = sel;
    return sel;
  }

  makeSelect('X axis', 'x', xProp, v => { xProp = v; update(); });
  makeSelect('Y axis', 'y', yProp, v => { yProp = v; update(); });
  makeSelect('Size', 'size', sizeProp, v => { sizeProp = v; update(); });

  const insightBtns = INSIGHTS.map((ins, i) => {
    const btn = document.createElement('button');
    btn.className = 'chart-btn';
    btn.textContent = ins.label;
    btn.addEventListener('click', () => applyInsight(i));
    insightRow.appendChild(btn);
    return btn;
  });

  function applyInsight(i) {
    activeInsight = i;
    const ins = INSIGHTS[i];
    xProp = ins.x;
    yProp = ins.y;
    sizeProp = ins.size;
    selects.x.value = ins.x;
    selects.y.value = ins.y;
    selects.size.value = ins.size;
    insightBtns.forEach((b, j) => b.classList.toggle('active', j === i));
    insightText.textContent = ins.text;
    insightText.style.display = '';
    update();
  }

  function clearActiveInsight() {
    activeInsight = -1;
    insightBtns.forEach(b => b.classList.remove('active'));
    insightText.style.display = 'none';
  }

  // Set "Overview" button active + text visible (but don't call update() yet — chart not built)
  insightBtns[0].classList.add('active');
  insightText.textContent = INSIGHTS[0].text;

  // Lock insight panel height to the tallest text so switching never causes layout shift
  let maxH = 0;
  for (const ins of INSIGHTS) {
    insightText.textContent = ins.text;
    maxH = Math.max(maxH, insightText.offsetHeight);
  }
  insightText.style.minHeight = maxH + 'px';
  insightText.textContent = INSIGHTS[0].text;

  // Chart — extra right margin for legend outside data area
  const chart = createChart('chart-category-landscape', {
    aspectRatio: 0.65,
    marginTop: 20,
    marginRight: 150,
    marginBottom: 50,
    marginLeft: 60,
  });
  if (!chart) return;
  const { g, width, height } = chart;

  const upperColor = '#4e79a7';
  const lowerColor = '#e15759';

  // Scales (will be updated)
  const x = d3.scaleLinear().range([0, width]);
  const y = d3.scaleLinear().range([height, 0]);
  const r = d3.scaleSqrt().range([4, 32]);

  // Grid lines groups
  const hGridG = g.append('g').attr('class', 'hgrid-group');
  const vGridG = g.append('g').attr('class', 'vgrid-group');

  // Axis groups
  const xAxisG = g.append('g')
    .attr('class', 'axis x-axis')
    .attr('transform', `translate(0,${height})`);
  const yAxisG = g.append('g')
    .attr('class', 'axis y-axis');

  // Axis labels
  const xLabel = xAxisG.append('text')
    .attr('x', width / 2)
    .attr('y', 35)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px');

  const yLabel = yAxisG.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -45)
    .attr('fill', getMutedColor())
    .attr('text-anchor', 'middle')
    .style('font-size', '12px');

  // Bubbles
  const tt = tooltip(container);

  const bubbleG = g.append('g');
  const bubbles = bubbleG.selectAll('circle.bubble')
    .data(data)
    .join('circle')
    .attr('class', 'bubble')
    .attr('fill', d => d.section === 'upper' ? upperColor : lowerColor)
    .attr('opacity', 0.6)
    .attr('stroke', d => d.section === 'upper' ? upperColor : lowerColor)
    .attr('stroke-width', 1.5);

  // Labels
  const labelG = g.append('g');
  const labels = labelG.selectAll('text.cat-label')
    .data(data)
    .join('text')
    .attr('class', 'cat-label')
    .attr('text-anchor', 'middle')
    .attr('fill', getMutedColor())
    .style('font-size', d => d.name === 'Yatzy' ? '10px' : '9px')
    .style('font-weight', d => d.name === 'Yatzy' ? '600' : '400')
    .text(d => d.name);

  // Interaction
  bubbles
    .on('mousemove', (event, d) => {
      const xFmt = d3.format(PROPERTIES[xProp].fmt);
      const yFmt = d3.format(PROPERTIES[yProp].fmt);
      const sFmt = d3.format(PROPERTIES[sizeProp].fmt);
      tt.show(
        `<div class="tt-label">${d.name}</div>
         <div>${PROPERTIES[xProp].label}: <span class="tt-value">${xFmt(d[xProp])}</span></div>
         <div>${PROPERTIES[yProp].label}: <span class="tt-value">${yFmt(d[yProp])}</span></div>
         <div>${PROPERTIES[sizeProp].label}: <span class="tt-value">${sFmt(d[sizeProp])}</span></div>`,
        event
      );
    })
    .on('mouseleave', () => tt.hide());

  // Legend — placed in right margin, outside the data area
  const legendG = g.append('g')
    .attr('transform', `translate(${width + 16}, 10)`);

  [{ label: 'Upper section', color: upperColor }, { label: 'Lower section', color: lowerColor }].forEach((item, i) => {
    legendG.append('circle')
      .attr('cx', 6).attr('cy', i * 18)
      .attr('r', 5)
      .attr('fill', item.color)
      .attr('opacity', 0.6);
    legendG.append('text')
      .attr('x', 16).attr('y', i * 18 + 4)
      .attr('fill', getMutedColor())
      .style('font-size', '10px')
      .text(item.label);
  });

  // Size legend — below section legend, also in right margin
  const sizeLegG = g.append('g')
    .attr('transform', `translate(${width + 16}, 50)`);

  const sizeLegLabel = sizeLegG.append('text')
    .attr('x', 0).attr('y', 0)
    .attr('fill', getMutedColor())
    .style('font-size', '9px');

  // Compute axis domain, reserving pixel space so no bubble+label overflows.
  // Uses each point's actual bubble radius (from current sizeProp) rather
  // than a worst-case constant, so the domain stays tight when it can.
  function computeDomain(prop, pixelExtent) {
    const fixed = PROPERTIES[prop].domain;
    let lo = fixed ? fixed[0] : d3.min(data, d => d[prop]);
    let hi = fixed ? fixed[1] : d3.max(data, d => d[prop]);
    const dataRange = hi - lo || 1;

    // pixels-per-unit for a naive domain
    const pxPerUnit = pixelExtent / dataRange;
    const labelPx = 14;

    // Find worst-case overflow at each edge
    let loPad = 0, hiPad = 0;
    for (const d of data) {
      const bubbleR = r(Math.abs(d[sizeProp]));
      const clearPx = bubbleR + labelPx;
      const clearData = clearPx / pxPerUnit;

      // How far beyond lo / hi does this point's bubble reach?
      const overLo = (lo + clearData) - d[prop];
      const overHi = d[prop] - (hi - clearData);
      if (overLo > loPad) loPad = overLo;
      if (overHi > hiPad) hiPad = overHi;
    }

    lo -= Math.max(loPad, 0);
    hi += Math.max(hiPad, 0);
    return [lo, hi];
  }

  function update() {
    const t = d3.transition().duration(400);

    // Size scale first (needed to compute bubble radii for domain padding)
    const maxSize = d3.max(data, d => Math.abs(d[sizeProp])) || 1;
    r.domain([0, maxSize]);

    // Update axis domains with pixel-aware padding
    x.domain(computeDomain(xProp, width));
    y.domain(computeDomain(yProp, height));

    // Axes
    const xAxis = d3.axisBottom(x).ticks(8).tickFormat(d3.format(PROPERTIES[xProp].fmt));
    const yAxis = d3.axisLeft(y).ticks(6).tickFormat(d3.format(PROPERTIES[yProp].fmt));

    xAxisG.transition(t).call(xAxis);
    yAxisG.transition(t).call(yAxis);

    // Style axes after transition
    xAxisG.selectAll('line').attr('stroke', getGridColor());
    xAxisG.selectAll('path').attr('stroke', getGridColor());
    xAxisG.selectAll('text')
      .attr('fill', getMutedColor())
      .style('font-size', '11px')
      .style('font-family', "'Newsreader', Georgia, serif");

    yAxisG.selectAll('line').attr('stroke', getGridColor());
    yAxisG.selectAll('path').attr('stroke', getGridColor());
    yAxisG.selectAll('text')
      .attr('fill', getMutedColor())
      .style('font-size', '11px')
      .style('font-family', "'Newsreader', Georgia, serif");

    // Labels
    xLabel.text(PROPERTIES[xProp].label);
    yLabel.text(PROPERTIES[yProp].label);
    sizeLegLabel.text('Bubble size = ' + PROPERTIES[sizeProp].label.toLowerCase());

    // Grid
    const yTicks = y.ticks(5);
    hGridG.selectAll('line')
      .data(yTicks, d => d)
      .join(
        enter => enter.append('line')
          .attr('x1', 0).attr('x2', width)
          .attr('stroke', getGridColor())
          .attr('stroke-dasharray', '2,3')
          .attr('y1', d => y(d)).attr('y2', d => y(d))
          .attr('opacity', 0)
          .call(e => e.transition(t).attr('opacity', 1)),
        update => update.transition(t)
          .attr('y1', d => y(d)).attr('y2', d => y(d)),
        exit => exit.transition(t).attr('opacity', 0).remove()
      );

    const xTicks = x.ticks(8);
    vGridG.selectAll('line')
      .data(xTicks, d => d)
      .join(
        enter => enter.append('line')
          .attr('y1', 0).attr('y2', height)
          .attr('stroke', getGridColor())
          .attr('stroke-dasharray', '2,3')
          .attr('x1', d => x(d)).attr('x2', d => x(d))
          .attr('opacity', 0)
          .call(e => e.transition(t).attr('opacity', 1)),
        update => update.transition(t)
          .attr('x1', d => x(d)).attr('x2', d => x(d)),
        exit => exit.transition(t).attr('opacity', 0).remove()
      );

    // Bubbles
    bubbles.transition(t)
      .attr('cx', d => x(d[xProp]))
      .attr('cy', d => y(d[yProp]))
      .attr('r', d => r(Math.abs(d[sizeProp])));

    // Labels
    labels.transition(t)
      .attr('x', d => x(d[xProp]))
      .attr('y', d => y(d[yProp]) - r(Math.abs(d[sizeProp])) - 4);
  }

  // Initial render (Overview insight is already active from above)
  update();
}
