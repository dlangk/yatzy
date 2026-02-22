import { DataLoader } from '../data-loader.js';
import {
  createChart, tooltip, drawAxis,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

export async function initBackwardWave() {
  const container = document.getElementById('chart-backward-wave');
  if (!container) return;

  const data = await DataLoader.backwardWave();
  const layers = data.layers; // 16 layers, num_scored: 15 → 0

  const playBtn = document.getElementById('wave-play-btn');
  const resetBtn = document.getElementById('wave-reset-btn');
  const scrub = document.getElementById('wave-scrub');
  const label = document.getElementById('wave-label');

  let activeLayer = 15; // Start at terminal (layer 15)
  let animTimer = null;

  function render() {
    const chart = createChart('chart-backward-wave-svg', {
      aspectRatio: 0.55,
      marginLeft: 65,
      marginBottom: 50,
      marginTop: 25,
      marginRight: 30,
    });
    if (!chart) return;
    const { g, width, height } = chart;

    // Map layers: x = upper_score (0-63), y = num_scored (15 at top, 0 at bottom)
    const xScale = d3.scaleLinear().domain([0, 63]).range([0, width]);
    const yScale = d3.scaleLinear().domain([15, 0]).range([0, height]);

    // EV color scale (shared across all layers)
    const evMin = 0;
    const evMax = 260;
    const colorScale = d3.scaleSequential()
      .domain([evMin, evMax])
      .interpolator(d3.interpolateWarm);

    // Draw all dots, dimmed or bright depending on activeLayer
    layers.forEach((layer) => {
      const layerIdx = layer.num_scored;
      const isSolved = layerIdx >= activeLayer;
      const isActive = layerIdx === activeLayer;
      const cy = yScale(layerIdx);

      layer.states.forEach((s) => {
        const cx = xScale(s.upper);
        const r = isActive ? 4.5 : (isSolved ? 3 : 2);
        const opacity = isSolved ? (isActive ? 1 : 0.6) : 0.12;

        const dot = g.append('circle')
          .attr('cx', cx)
          .attr('cy', cy)
          .attr('r', r)
          .attr('fill', isSolved ? colorScale(s.ev) : getGridColor())
          .attr('opacity', opacity)
          .attr('class', 'wave-dot')
          .attr('data-layer', layerIdx)
          .attr('data-upper', s.upper)
          .attr('data-ev', s.ev);

        if (isActive) {
          dot.attr('stroke', 'gold')
            .attr('stroke-width', 1.5)
            .classed('highlight', true);
        }
      });
    });

    // Golden payoff annotation at layer 0
    if (activeLayer <= 0) {
      const startState = layers.find(l => l.num_scored === 0);
      if (startState && startState.states.length > 0) {
        const s0 = startState.states[0];
        g.append('text')
          .attr('x', xScale(s0.upper) + 8)
          .attr('y', yScale(0) - 8)
          .attr('fill', 'gold')
          .attr('font-weight', '700')
          .style('font-size', '13px')
          .text(`EV = ${s0.ev.toFixed(1)}`);
      }
    }

    // Wave-front line
    if (activeLayer >= 0 && activeLayer <= 15) {
      const wy = yScale(activeLayer);
      g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', wy)
        .attr('y2', wy)
        .attr('stroke', 'gold')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3')
        .attr('opacity', 0.5);
    }

    // Axes
    drawAxis(
      g.append('g').attr('transform', `translate(0,${height})`),
      xScale, 'bottom', 'Upper Section Score'
    );

    const yAxis = d3.axisLeft(yScale).ticks(8).tickFormat(d => d);
    const yAxisG = g.append('g').call(yAxis);
    yAxisG.selectAll('line').attr('stroke', getGridColor());
    yAxisG.selectAll('path').attr('stroke', getGridColor());
    yAxisG.selectAll('text')
      .attr('fill', getMutedColor())
      .style('font-size', '11px')
      .style('font-family', "'Newsreader', Georgia, serif");
    yAxisG.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('fill', getMutedColor())
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text('Categories Scored');

    // Tooltip
    const tt = tooltip(container);
    g.selectAll('.wave-dot')
      .on('mousemove', function (event) {
        const layerIdx = +this.getAttribute('data-layer');
        const upper = +this.getAttribute('data-upper');
        const ev = +this.getAttribute('data-ev');
        tt.show(
          `<div class="tt-label">Layer ${layerIdx}, upper = ${upper}</div>
           <div>EV: <span class="tt-value">${ev.toFixed(1)}</span></div>`,
          event
        );
      })
      .on('mouseleave', () => tt.hide());

    // Update label
    if (activeLayer > 0) {
      label.textContent = `Layer ${activeLayer} \u2014 ${activeLayer} categories scored`;
    } else if (activeLayer === 0) {
      label.textContent = 'Layer 0 \u2014 game start (EV = 248.4)';
    } else {
      label.textContent = 'Layer 15 \u2014 terminal states';
    }
  }

  // Controls
  playBtn.addEventListener('click', () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
      playBtn.innerHTML = '&#9654; Play';
      return;
    }
    activeLayer = 15;
    scrub.value = 15;
    playBtn.innerHTML = '&#9646;&#9646; Pause';
    render();

    animTimer = setInterval(() => {
      if (activeLayer > 0) {
        activeLayer--;
        scrub.value = activeLayer;
        render();
      } else {
        clearInterval(animTimer);
        animTimer = null;
        playBtn.innerHTML = '&#9654; Play';
      }
    }, 600);
  });

  resetBtn.addEventListener('click', () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
    activeLayer = 15;
    scrub.value = 15;
    render();
  });

  scrub.addEventListener('input', () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
    activeLayer = +scrub.value;
    render();
  });

  render();
}
