import {
  createChart, tooltip,
  getTextColor, getMutedColor, getGridColor, COLORS,
} from '../yatzy-viz.js';

// Hardcoded illustrative data for one widget (turn 1, empty scorecard, specific dice)
// This is pedagogical — shows the structure, not real computed values.
const GROUPS = [
  {
    id: 6,
    label: 'Group 6',
    type: 'decision',
    title: 'Category Assignment',
    desc: 'max over available categories',
    equation: 'E₆(r) = max_c [ s(r,c) + V(m′, C∪{c}) ]',
    nodes: [
      { label: '⚂⚂⚃⚄⚅ → Fours (8)', ev: 249.1, best: false },
      { label: '⚂⚂⚃⚄⚅ → Chance (19)', ev: 243.7, best: false },
      { label: '⚂⚂⚃⚄⚅ → Sm. Straight (15)', ev: 241.2, best: false },
      { label: '⚂⚂⚃⚄⚅ → Threes (6)', ev: 252.8, best: true },
      { label: '⚂⚂⚃⚄⚅ → One Pair (8)', ev: 240.1, best: false },
    ],
  },
  {
    id: 5,
    label: 'Group 5',
    type: 'chance',
    title: 'Final Roll Outcomes',
    desc: 'E[·] over dice outcomes after second reroll',
    equation: 'E₅(k) = Σ P(r|k) · E₆(r)',
    nodes: [
      { label: 'keep {3,3} → E[·]', ev: 248.3, best: true },
      { label: 'keep {3,4,5} → E[·]', ev: 245.1, best: false },
      { label: 'keep {4,5} → E[·]', ev: 244.7, best: false },
    ],
  },
  {
    id: 4,
    label: 'Group 4',
    type: 'decision',
    title: 'Second Keep Choice',
    desc: 'max over keep-multisets',
    equation: 'E₄(r) = max_k E₅(k)',
    nodes: [
      { label: '⚂⚃⚃⚄⚅ → keep best', ev: 249.6, best: true },
      { label: '⚁⚂⚃⚄⚅ → keep best', ev: 247.2, best: false },
      { label: '⚂⚂⚂⚄⚅ → keep best', ev: 251.3, best: false },
      { label: '⚃⚃⚃⚃⚅ → keep best', ev: 253.8, best: false },
    ],
  },
  {
    id: 3,
    label: 'Group 3',
    type: 'chance',
    title: 'Second Roll Outcomes',
    desc: 'E[·] over dice outcomes after first reroll',
    equation: 'E₃(k) = Σ P(r|k) · E₄(r)',
    nodes: [
      { label: 'keep {3,4,5} → E[·]', ev: 249.7, best: true },
      { label: 'keep {3,3} → E[·]', ev: 247.9, best: false },
      { label: 'keep {5} → E[·]', ev: 246.2, best: false },
    ],
  },
  {
    id: 2,
    label: 'Group 2',
    type: 'decision',
    title: 'First Keep Choice',
    desc: 'max over keep-multisets',
    equation: 'E₂(r) = max_k E₃(k)',
    nodes: [
      { label: '⚂⚂⚃⚄⚅ → keep best', ev: 249.7, best: true },
    ],
  },
  {
    id: 1,
    label: 'Group 1',
    type: 'chance',
    title: 'Entry (Initial Roll)',
    desc: 'E[·] over all 252 initial outcomes',
    equation: 'V(m,C) = Σ P(r) · E₂(r)',
    nodes: [
      { label: 'V(0, ∅) = E[·]', ev: 248.4, best: true },
    ],
  },
];

export async function initWidgetExplorer() {
  const container = document.getElementById('chart-widget-explorer');
  if (!container) return;

  let activeGroup = -1; // -1 = all dim, 0..5 = highlighting GROUPS[idx]
  const totalGroups = GROUPS.length;

  const stepBtn = document.getElementById('widget-step-btn');
  const resetBtn = document.getElementById('widget-reset-btn');
  const stepLabel = document.getElementById('widget-step-label');

  function render() {
    const chart = createChart('chart-widget-explorer-svg', {
      aspectRatio: 0.55,
      marginLeft: 10,
      marginRight: 10,
      marginTop: 15,
      marginBottom: 10,
    });
    if (!chart) return;
    const { g, width, height } = chart;

    const isDark = document.documentElement.classList.contains('dark');
    const colWidth = width / totalGroups;
    const maxNodes = Math.max(...GROUPS.map(gr => gr.nodes.length));
    const nodeH = 28;
    const nodeGap = 6;

    // Draw groups right-to-left (Group 6 on the right, Group 1 on the left)
    GROUPS.forEach((group, gi) => {
      const colX = (totalGroups - 1 - gi) * colWidth;
      const isActive = gi === activeGroup;
      const isSolved = gi > activeGroup && activeGroup >= 0;
      const opacity = activeGroup < 0 ? 0.4 : (isActive ? 1.0 : (isSolved ? 0.8 : 0.25));

      // Column background
      const gg = g.append('g')
        .attr('transform', `translate(${colX}, 0)`)
        .attr('opacity', opacity);

      // Header
      const headerColor = group.type === 'decision' ? COLORS.accent : COLORS.riskAverse;
      gg.append('rect')
        .attr('x', 4)
        .attr('y', 0)
        .attr('width', colWidth - 8)
        .attr('height', 22)
        .attr('rx', 4)
        .attr('fill', headerColor)
        .attr('opacity', isActive ? 0.25 : 0.1);

      gg.append('text')
        .attr('x', colWidth / 2)
        .attr('y', 15)
        .attr('text-anchor', 'middle')
        .attr('fill', isActive ? headerColor : getMutedColor())
        .style('font-size', '10px')
        .style('font-weight', '600')
        .text(group.label);

      // Type indicator
      gg.append('text')
        .attr('x', colWidth / 2)
        .attr('y', 34)
        .attr('text-anchor', 'middle')
        .attr('fill', getMutedColor())
        .style('font-size', '8px')
        .text(group.type === 'decision' ? '(max)' : '(Σ P·x)');

      // Nodes
      const nodesStartY = 44;
      group.nodes.forEach((node, ni) => {
        const ny = nodesStartY + ni * (nodeH + nodeGap);
        const nodeColor = node.best && (isActive || isSolved)
          ? headerColor
          : getGridColor();
        const textColor = node.best && (isActive || isSolved)
          ? getTextColor()
          : getMutedColor();

        gg.append('rect')
          .attr('x', 6)
          .attr('y', ny)
          .attr('width', colWidth - 12)
          .attr('height', nodeH)
          .attr('rx', 4)
          .attr('fill', nodeColor)
          .attr('opacity', node.best ? 0.3 : 0.15)
          .attr('stroke', node.best && isActive ? headerColor : 'none')
          .attr('stroke-width', 1.5);

        // Truncate label to fit
        const maxChars = Math.floor((colWidth - 20) / 5.5);
        const displayLabel = node.label.length > maxChars
          ? node.label.substring(0, maxChars - 1) + '…'
          : node.label;

        gg.append('text')
          .attr('x', 12)
          .attr('y', ny + 12)
          .attr('fill', textColor)
          .style('font-size', '8px')
          .text(displayLabel);

        gg.append('text')
          .attr('x', colWidth - 16)
          .attr('y', ny + 23)
          .attr('text-anchor', 'end')
          .attr('fill', node.best && (isActive || isSolved) ? headerColor : textColor)
          .style('font-size', '9px')
          .style('font-weight', node.best ? '700' : '400')
          .text(node.ev.toFixed(1));
      });

      // Equation (show for active group)
      if (isActive) {
        const eqY = height - 10;
        gg.append('text')
          .attr('x', colWidth / 2)
          .attr('y', eqY)
          .attr('text-anchor', 'middle')
          .attr('fill', headerColor)
          .style('font-size', '9px')
          .style('font-style', 'italic')
          .text(group.equation);
      }
    });

    // Draw edges between groups (connecting best nodes)
    if (activeGroup >= 0) {
      for (let gi = 0; gi < totalGroups - 1; gi++) {
        if (gi > activeGroup) continue;
        const srcGroup = GROUPS[gi];
        const dstGroup = GROUPS[gi + 1];
        const srcCol = (totalGroups - 1 - gi) * colWidth;
        const dstCol = (totalGroups - 2 - gi) * colWidth;
        const srcBestIdx = srcGroup.nodes.findIndex(n => n.best);
        const dstBestIdx = dstGroup.nodes.findIndex(n => n.best);
        if (srcBestIdx < 0 || dstBestIdx < 0) continue;

        const nodesStartY = 44;
        const srcY = nodesStartY + srcBestIdx * (nodeH + nodeGap) + nodeH / 2;
        const dstY = nodesStartY + dstBestIdx * (nodeH + nodeGap) + nodeH / 2;
        const srcX = srcCol + 6;
        const dstX = dstCol + colWidth - 6;

        const isSolved = gi >= activeGroup;
        g.append('path')
          .attr('d', `M${srcX},${srcY} C${srcX - 15},${srcY} ${dstX + 15},${dstY} ${dstX},${dstY}`)
          .attr('fill', 'none')
          .attr('stroke', isSolved ? COLORS.accent : getGridColor())
          .attr('stroke-width', isSolved ? 2 : 1)
          .attr('stroke-dasharray', isSolved ? 'none' : '3,3')
          .attr('opacity', isSolved ? 0.6 : 0.3);
      }
    }

    // Update label
    if (activeGroup < 0) {
      stepLabel.textContent = 'Click "Step Backward" to begin';
    } else {
      const group = GROUPS[activeGroup];
      stepLabel.textContent = `Solving ${group.label}: ${group.title}`;
    }
  }

  stepBtn.addEventListener('click', () => {
    if (activeGroup < totalGroups - 1) {
      activeGroup++;
    }
    render();
  });

  resetBtn.addEventListener('click', () => {
    activeGroup = -1;
    render();
  });

  render();
}
