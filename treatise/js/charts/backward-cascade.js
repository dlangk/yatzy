export async function initBackwardCascade() {
  const container = document.getElementById('chart-backward-cascade');
  if (!container) return;

  function scorecard(filled, empty) {
    const squares = [];
    for (let i = 0; i < filled; i++) squares.push('<div class="cascade-sq filled"></div>');
    for (let i = 0; i < empty; i++) squares.push('<div class="cascade-sq empty"></div>');
    return `<div class="cascade-scorecard">${squares.join('')}</div>`;
  }

  function card(layer, subtitle, filled, desc, footnote, extra = '') {
    const cls = layer === 1 ? 'cascade-card cascade-card-final' : 'cascade-card';
    return `
      <div class="${cls}">
        <div class="cascade-card-header">
          <span class="cascade-layer-badge">Layer ${layer}</span>
          <span class="cascade-card-subtitle">${subtitle}</span>
        </div>
        ${scorecard(filled, 15 - filled)}
        <p class="cascade-desc">${desc}</p>
        ${extra}
        <p class="cascade-footnote">${footnote}</p>
      </div>`;
  }

  function arrow() {
    return `
      <div class="cascade-arrow">
        <div class="cascade-arrow-line"></div>
        <span class="cascade-arrow-label">already solved</span>
        <div class="cascade-arrow-line"></div>
      </div>`;
  }

  const bonusFork = `
    <div class="cascade-bonus-fork">
      <div class="cascade-bonus-box">
        <span class="cascade-bonus-label">upper &lt; 63</span>
        <span class="cascade-bonus-value">bonus: 0</span>
      </div>
      <div class="cascade-bonus-box cascade-bonus-earned">
        <span class="cascade-bonus-label">upper &ge; 63</span>
        <span class="cascade-bonus-value">bonus: +50 &#9733;</span>
      </div>
    </div>`;

  container.innerHTML = `<div class="backward-cascade">
    ${card(16, 'Game over', 15,
      'All 15 categories scored. Nothing to decide.',
      '1 state-check per state. No dice, no decisions.',
      bonusFork)}
    ${arrow()}
    ${card(15, 'One category left', 14,
      'One category remains. You must score it. Roll, keep, reroll, keep, reroll, score. Then look up the final answer from Layer 16.',
      '15 widgets &middot; 1 scoring choice each')}
    ${arrow()}
    ${card(14, 'Two categories left', 13,
      'Two categories remain. Try scoring each one. Each choice leads to a Layer 15 state, already solved. Pick whichever gives the higher expected score.',
      '105 widgets &middot; 2 scoring choices each')}
    <div class="cascade-ellipsis">
      <div class="cascade-arrow-line"></div>
      <span class="cascade-ellipsis-dots">&middot; &middot; &middot;</span>
      <div class="cascade-arrow-line"></div>
    </div>
    ${card(1, 'Game start', 0,
      'All 15 open. 1 widget. <strong>E[start] = 248.4</strong>',
      'The answer to the entire game.')}
  </div>`;
}
