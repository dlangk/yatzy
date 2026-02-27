import {
  createChart,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

function binomial(n, k) {
  if (k > n || k < 0) return 0;
  if (k === 0 || k === n) return 1;
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return Math.round(result);
}

export async function initTopologicalCascade() {
  const container = document.getElementById('chart-topological-cascade');
  if (!container) return;

  // C(15, k) widgets per layer, processing 15 → 0
  const layers = [];
  for (let k = 15; k >= 0; k--) {
    layers.push({ scored: k, widgets: binomial(15, k), label: `|C|=${k}` });
  }

  const playBtn = document.getElementById('cascade-play-btn');
  const resetBtn = document.getElementById('cascade-reset-btn');
  const scrub = document.getElementById('cascade-scrub');
  const label = document.getElementById('cascade-label');

  let activeStep = -1; // -1 = nothing solved, 0 = first layer (k=15) solved, etc.
  let animTimer = null;

  function render() {
    const chart = createChart('chart-topological-cascade-svg', {
      aspectRatio: 0.6,
      marginLeft: 60,
      marginBottom: 30,
      marginTop: 15,
      marginRight: 60,
    });
    if (!chart) return;
    const { g, width, height } = chart;

    const maxLog = Math.log10(binomial(15, 7)); // C(15,7)=6435 is the max
    const barMaxW = width * 0.85;
    const rowH = height / layers.length;

    // Arrow marker
    g.append('defs').append('marker')
      .attr('id', 'arrow-cascade')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', getMutedColor());

    layers.forEach((layer, i) => {
      const bw = layer.widgets === 1 ? 20 : (Math.log10(layer.widgets) / maxLog) * barMaxW;
      const bx = (width - bw) / 2;
      const by = i * rowH + 2;
      const bh = rowH - 4;

      const isSolved = i <= activeStep;
      const isActive = i === activeStep;

      let fill;
      if (isActive) fill = 'rgba(255, 200, 50, 0.5)';
      else if (isSolved) fill = 'rgba(42, 160, 42, 0.35)';
      else fill = getGridColor();

      let stroke;
      if (isActive) stroke = 'gold';
      else if (isSolved) stroke = COLORS.optimal;
      else stroke = 'transparent';

      g.append('rect')
        .attr('x', bx)
        .attr('y', by)
        .attr('width', bw)
        .attr('height', bh)
        .attr('rx', 3)
        .attr('fill', fill)
        .attr('opacity', isSolved ? 0.9 : 0.4)
        .attr('stroke', stroke)
        .attr('stroke-width', isActive ? 2 : 1);

      // Layer label (left)
      g.append('text')
        .attr('x', -8)
        .attr('y', by + bh / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', isSolved ? getTextColor() : getMutedColor())
        .style('font-size', '10px')
        .style('font-weight', isActive ? '700' : '400')
        .text(layer.label);

      // Widget count (right)
      g.append('text')
        .attr('x', width + 8)
        .attr('y', by + bh / 2)
        .attr('dy', '0.35em')
        .attr('fill', isSolved ? getTextColor() : getMutedColor())
        .style('font-size', '10px')
        .style('font-variant-numeric', 'tabular-nums')
        .text(layer.widgets.toLocaleString());

      // Dependency arrow (pointing up — each layer depends on the one above it being solved first)
      if (i > 0) {
        const prevBw = layers[i - 1].widgets === 1 ? 20 : (Math.log10(layers[i - 1].widgets) / maxLog) * barMaxW;
        const arrowX = width / 2 + bw / 2 + 12;
        g.append('line')
          .attr('x1', arrowX)
          .attr('y1', by)
          .attr('x2', arrowX)
          .attr('y2', by - 4)
          .attr('stroke', getMutedColor())
          .attr('stroke-width', 0.7)
          .attr('opacity', 0.4)
          .attr('marker-end', 'url(#arrow-cascade)');
      }
    });

    // Wavefront line
    if (activeStep >= 0 && activeStep < layers.length) {
      const wy = activeStep * rowH + rowH;
      g.append('line')
        .attr('x1', 0).attr('x2', width)
        .attr('y1', wy).attr('y2', wy)
        .attr('stroke', 'gold')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3')
        .attr('opacity', 0.7);
    }

    // Update label
    if (activeStep < 0) {
      label.textContent = 'All layers pending';
    } else if (activeStep < layers.length) {
      const layer = layers[activeStep];
      label.textContent = `Solving layer ${layer.scored} — ${layer.widgets.toLocaleString()} widgets`;
    } else {
      label.textContent = 'All 16 layers solved — E[start] = 248.4';
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
    activeStep = -1;
    scrub.value = -1;
    playBtn.innerHTML = '&#9646;&#9646; Pause';
    render();

    animTimer = setInterval(() => {
      if (activeStep < layers.length - 1) {
        activeStep++;
        scrub.value = activeStep;
        render();
      } else {
        clearInterval(animTimer);
        animTimer = null;
        playBtn.innerHTML = '&#9654; Play';
      }
    }, 400);
  });

  resetBtn.addEventListener('click', () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
    activeStep = -1;
    scrub.value = -1;
    render();
  });

  scrub.addEventListener('input', () => {
    if (animTimer) {
      clearInterval(animTimer);
      animTimer = null;
      playBtn.innerHTML = '&#9654; Play';
    }
    activeStep = +scrub.value;
    render();
  });

  render();
}
