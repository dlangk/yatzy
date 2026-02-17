/**
 * Player Card: simulation-backed profile analytics shown after quiz completion.
 *
 * Sections:
 *   A. Headline stats (your mean, optimal mean, gap)
 *   B. Score range box-plot (p5 to p95 with optimal reference)
 *   C. Counterfactual coaching bars (where you're losing points)
 *   D. Behavioral fingerprint comparison table
 */
import { subscribe, getState } from '../store.js';
import {
  isGridLoaded, lookupGrid, computeCounterfactuals,
  getOptimalStats, loadPlayerCardGrid,
} from '../player-card-data.js';

export function initPlayerCard(container) {
  const el = document.createElement('div');
  el.className = 'player-card';
  container.appendChild(el);

  function render(state) {
    const allAnswered = state.scenarios.length > 0 && state.answers.length >= state.scenarios.length;
    if (!allAnswered || !state.profile || !isGridLoaded()) {
      el.style.display = 'none';
      el.innerHTML = '';
      return;
    }

    const p = state.profile;
    const yours = lookupGrid(p.theta, p.beta, p.gamma, p.d);
    const optStats = getOptimalStats();

    if (!yours || !optStats) {
      el.style.display = 'none';
      return;
    }

    el.style.display = 'block';

    const gap = optStats.mean - yours.mean;
    const counterfactuals = computeCounterfactuals(p.theta, p.beta, p.gamma, p.d);
    const maxDelta = Math.max(...counterfactuals.map(c => c.delta), 1);

    el.innerHTML = `
      <h3 class="pc-title">Simulated Performance</h3>
      ${renderHeadline(yours, optStats, gap)}
      ${renderBoxplot(yours, optStats)}
      ${renderCounterfactuals(counterfactuals, maxDelta)}
      ${renderFingerprint(yours, optStats)}
      <p class="pc-note">Based on 10K Monte Carlo simulated games per parameter combination.</p>
    `;
  }

  function renderHeadline(yours, optimal, gap) {
    return `
      <div class="pc-headline">
        <div class="pc-stat-box">
          <span class="pc-stat-label">Your Score</span>
          <span class="pc-stat-value">${Math.round(yours.mean)}</span>
        </div>
        <div class="pc-stat-box">
          <span class="pc-stat-label">Optimal</span>
          <span class="pc-stat-value">${Math.round(optimal.mean)}</span>
        </div>
        <div class="pc-stat-box">
          <span class="pc-stat-label">Gap</span>
          <span class="pc-stat-value pc-stat-gap">${Math.round(gap)} pts</span>
        </div>
      </div>
    `;
  }

  function renderBoxplot(yours, optimal) {
    const lo = Math.min(yours.p5, optimal.p5) - 20;
    const hi = Math.max(yours.p95, optimal.p95) + 20;
    const range = hi - lo;
    const pct = v => ((v - lo) / range * 100).toFixed(1);

    return `
      <div class="pc-boxplot">
        <div class="pc-section-label">Score Distribution</div>
        <div class="pc-boxplot-track">
          <div class="pc-boxplot-whisker" style="left:${pct(yours.p5)}%;width:${(pct(yours.p95) - pct(yours.p5)).toFixed(1)}%;"></div>
          <div class="pc-boxplot-box" style="left:${pct(yours.p25)}%;width:${(pct(yours.p75) - pct(yours.p25)).toFixed(1)}%;"></div>
          <div class="pc-boxplot-median" style="left:${pct(yours.p50)}%;"></div>
          <div class="pc-boxplot-optimal" style="left:${pct(optimal.p50)}%;" title="Optimal median: ${optimal.p50}"></div>
        </div>
        <div class="pc-boxplot-axis">
          <span>p5: ${yours.p5}</span>
          <span>p50: ${yours.p50}</span>
          <span>p95: ${yours.p95}</span>
        </div>
        <div class="pc-boxplot-legend">
          <span class="pc-legend-yours">&#9644; Your range</span>
          <span class="pc-legend-opt">&#9670; Optimal median</span>
        </div>
      </div>
    `;
  }

  function renderCounterfactuals(cfs, maxDelta) {
    if (cfs.length === 0) return '';

    const rows = cfs.map(cf => {
      const pct = maxDelta > 0 ? (Math.max(cf.delta, 0) / maxDelta * 100) : 0;
      const sign = cf.delta >= 0 ? '+' : '';
      return `
        <div class="pc-cf-row">
          <span class="pc-cf-label">${cf.label}</span>
          <div class="pc-cf-bar-track">
            <div class="pc-cf-bar" style="width:${pct.toFixed(1)}%;"></div>
          </div>
          <span class="pc-cf-value">${sign}${cf.delta.toFixed(1)}</span>
        </div>
      `;
    }).join('');

    return `
      <div class="pc-counterfactuals">
        <div class="pc-section-label">Where You're Losing Points</div>
        ${rows}
        <div class="pc-cf-note">Individual costs overlap &mdash; combined effect is less than the sum.</div>
      </div>
    `;
  }

  function renderFingerprint(yours, optimal) {
    const rows = [
      ['Median Score', yours.p50, optimal.p50],
      ['Best Games (p95)', yours.p95, optimal.p95],
      ['Worst Games (p5)', yours.p5, optimal.p5],
    ];

    const tableRows = rows.map(([label, you, opt]) => `
      <tr>
        <td class="pc-fp-label">${label}</td>
        <td class="pc-fp-you">${you}</td>
        <td class="pc-fp-opt">${opt}</td>
      </tr>
    `).join('');

    return `
      <div class="pc-fingerprint">
        <div class="pc-section-label">Performance Comparison</div>
        <table class="pc-fp-table">
          <thead>
            <tr><th></th><th>You</th><th>Optimal</th></tr>
          </thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    `;
  }

  render(getState());
  subscribe((state) => render(state));

  // Re-render when grid data finishes loading
  loadPlayerCardGrid().then(() => render(getState()));
}
