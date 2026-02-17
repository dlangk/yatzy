/**
 * Parameter chart: live 4-parameter visualization with confidence intervals.
 * Each parameter row is clickable — opens a concept drawer with explanation.
 */
import { subscribe, getState } from '../store.js';

const PARAMS = [
  {
    key: 'theta', ci: 'ci_theta', label: 'Risk (θ)', min: -0.1, max: 0.1,
    fmt: v => v.toFixed(3),
    minLabel: '−0.1', maxLabel: '+0.1', center: 0,
    ticks: [{ val: 0, label: '0' }],
  },
  {
    key: 'beta', ci: 'ci_beta', label: 'Precision (β)', min: 0.1, max: 20,
    fmt: v => v.toFixed(1),
    minLabel: '0', maxLabel: '20',
  },
  {
    key: 'gamma', ci: 'ci_gamma', label: 'Horizon (γ)', min: 0.1, max: 1.0,
    fmt: v => v.toFixed(2),
    minLabel: '0', maxLabel: '1',
  },
  {
    key: 'd', ci: null, label: 'Depth (d)', min: 5, max: 999,
    fmt: v => v === 999 ? '∞' : String(v),
    minLabel: '5', maxLabel: '∞',
    steps: [
      { val: 5, label: '5' }, { val: 8, label: '8' }, { val: 10, label: '10' },
      { val: 15, label: '15' }, { val: 20, label: '20' }, { val: 999, label: '∞' },
    ],
  },
];

const CONCEPT_HTML = {
  theta: `
    <h2>Risk Attitude (θ)</h2>
    <p><strong>Range:</strong> −0.1 to +0.1</p>
    <p>The risk parameter θ controls how the player values <em>variance</em> in outcomes,
       beyond just the expected score. It comes from
       <strong>exponential utility</strong>: instead of maximising E[score], we maximise
       E[e<sup>θ·score</sup>].</p>
    <h3>Interpretation</h3>
    <ul>
      <li><strong>θ = 0</strong> — risk-neutral. Maximises expected score.</li>
      <li><strong>θ &lt; 0</strong> — risk-averse. Prefers consistent scores, avoids gambles.</li>
      <li><strong>θ &gt; 0</strong> — risk-seeking. Chases high scores, accepts more bad games.</li>
    </ul>
    <h3>Why this range?</h3>
    <p>For Yatzy (scores ~250), |θ| &gt; 0.1 makes the utility function saturate —
       every strategy looks the same. The sweet spot for meaningful differences is
       |θ| &lt; 0.05.</p>
    <h3>Example</h3>
    <p>A risk-averse player (θ ≈ −0.03) will keep a safe pair of fives rather than
       rerolling for a straight. A risk-seeker (θ ≈ +0.05) takes that gamble.</p>
  `,
  beta: `
    <h2>Decision Precision (β)</h2>
    <p><strong>Range:</strong> 0.1 to 20</p>
    <p>β is the <strong>inverse temperature</strong> of a softmax choice model:
       P(action) ∝ e<sup>β·Q(action)</sup>. It measures how reliably you pick the
       highest-value option.</p>
    <h3>Interpretation</h3>
    <ul>
      <li><strong>β → 0</strong> — random play. All actions equally likely regardless of value.</li>
      <li><strong>β ≈ 2–5</strong> — casual human play. Usually picks good options with occasional mistakes.</li>
      <li><strong>β ≈ 5–10</strong> — strong play. Rarely deviates from the best action.</li>
      <li><strong>β &gt; 10</strong> — near-optimal. Almost always picks the highest-Q action.</li>
    </ul>
    <h3>What it captures</h3>
    <p>β absorbs everything that makes a player "noisy": fatigue, distraction,
       miscounting, or simply not knowing the value of each option. Two players
       with identical θ, γ, d but different β will agree on <em>which</em> option is
       best but differ in how often they actually pick it.</p>
  `,
  gamma: `
    <h2>Planning Horizon (γ)</h2>
    <p><strong>Range:</strong> 0.1 to 1.0</p>
    <p>γ is a <strong>discount factor</strong> that controls how far ahead the player
       looks when scoring a category. Future turns are weighted by γ<sup>t</sup>.</p>
    <h3>Interpretation</h3>
    <ul>
      <li><strong>γ = 1.0</strong> — fully forward-looking. Weighs all remaining turns equally (optimal play).</li>
      <li><strong>γ ≈ 0.8–0.9</strong> — moderate foresight. Considers a few turns ahead.</li>
      <li><strong>γ &lt; 0.5</strong> — myopic / greedy. Strongly favours immediate score over future potential.</li>
    </ul>
    <h3>Example</h3>
    <p>You roll [1,2,3,4,5]. A greedy player (γ ≈ 0.3) scores Small Straight now for 15 points.
       A forward-looking player (γ ≈ 0.95) sees that keeping 2-3-4-5 and rerolling the 1
       gives a shot at Large Straight (20 points) while still falling back on Small Straight.</p>
    <h3>Why it matters</h3>
    <p>Category decisions in Yatzy are inherently sequential — scoring a category now removes
       it for the rest of the game. Players with low γ systematically sacrifice long-term
       value for short-term gains, typically losing 10–30 points per game.</p>
  `,
  d: `
    <h2>Strategic Depth (d)</h2>
    <p><strong>Values:</strong> 5, 8, 10, 15, 20, ∞ (optimal)</p>
    <p>d models the player's <strong>resolution</strong> for evaluating game states.
       Higher d means the player can distinguish fine differences in position value.
       It's calibrated from decision tree models trained on the exact solver.</p>
    <h3>Calibration</h3>
    <table>
      <tr><th>d</th><th>Mean Score</th><th>EV Loss</th><th>Level</th></tr>
      <tr><td>5</td><td>157</td><td>91 pts</td><td>Novice</td></tr>
      <tr><td>8</td><td>192</td><td>56 pts</td><td>Developing</td></tr>
      <tr><td>10</td><td>216</td><td>32 pts</td><td>Intermediate</td></tr>
      <tr><td>15</td><td>239</td><td>9 pts</td><td>Advanced</td></tr>
      <tr><td>20</td><td>245</td><td>3 pts</td><td>Expert</td></tr>
      <tr><td>∞</td><td>248</td><td>0 pts</td><td>Optimal</td></tr>
    </table>
    <h3>How it works</h3>
    <p>The solver knows the exact value of every game state. A depth-d player sees a
       <em>noisy</em> version: V<sub>d</sub>(s) = V(s) + σ<sub>d</sub>·noise(s), where
       σ<sub>d</sub> decreases with d. This means low-d players confuse similar-valued
       states, while high-d players can tell them apart.</p>
    <h3>Why discrete?</h3>
    <p>d is discrete because it's calibrated from actual decision trees of known depth.
       Each value maps to a concrete "skill level" with measured mean score.
       It does not have a confidence interval — the estimator picks the d value
       that best explains your choices.</p>
  `,
};

function normalizePct(key, v, min, max) {
  if (key === 'd') {
    const dMap = { 5: 10, 8: 25, 10: 40, 15: 60, 20: 80, 999: 100 };
    return dMap[v] || 50;
  }
  return ((v - min) / (max - min)) * 100;
}

function clampPct(v) {
  return Math.max(0, Math.min(100, v));
}

// Drawer state
let drawerEl = null;
let drawerTitleEl = null;
let drawerBodyEl = null;
let backdropEl = null;
let currentDrawerKey = null;

function ensureDrawer() {
  if (drawerEl) return;

  backdropEl = document.createElement('div');
  backdropEl.className = 'concept-drawer-backdrop';
  document.body.appendChild(backdropEl);

  drawerEl = document.createElement('aside');
  drawerEl.className = 'concept-drawer';
  drawerEl.innerHTML = `
    <div class="concept-drawer-header">
      <span class="concept-drawer-title"></span>
      <button class="concept-drawer-close" aria-label="Close">&times;</button>
    </div>
    <div class="concept-drawer-body"></div>
  `;
  document.body.appendChild(drawerEl);

  drawerTitleEl = drawerEl.querySelector('.concept-drawer-title');
  drawerBodyEl = drawerEl.querySelector('.concept-drawer-body');

  drawerEl.querySelector('.concept-drawer-close').addEventListener('click', closeDrawer);
  backdropEl.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
  });
}

function closeDrawer() {
  drawerEl.classList.remove('open');
  backdropEl.classList.remove('open');
  document.querySelectorAll('.profile-param-label.active').forEach(el => el.classList.remove('active'));
  currentDrawerKey = null;
}

function openDrawer(key, triggerEl) {
  ensureDrawer();

  if (currentDrawerKey === key) {
    closeDrawer();
    return;
  }

  document.querySelectorAll('.profile-param-label.active').forEach(el => el.classList.remove('active'));
  triggerEl.classList.add('active');
  currentDrawerKey = key;

  const html = CONCEPT_HTML[key] || '<p>No description available.</p>';
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  const h2 = tmp.querySelector('h2');
  if (h2) {
    drawerTitleEl.textContent = h2.textContent;
    h2.remove();
  }
  drawerBodyEl.innerHTML = tmp.innerHTML;

  drawerEl.classList.add('open');
  backdropEl.classList.add('open');
  drawerBodyEl.scrollTop = 0;
}

export function initParameterChart(container) {
  const el = document.createElement('div');
  el.className = 'profile-params';
  container.appendChild(el);

  function buildTicks(p) {
    // Discrete steps
    if (p.steps) {
      return p.steps.map(s => {
        const pct = clampPct(normalizePct(p.key, s.val, p.min, p.max));
        return `<span class="profile-param-tick" style="left:${pct}%"><span class="profile-param-tick-line"></span><span class="profile-param-tick-label">${s.label}</span></span>`;
      }).join('');
    }
    // Named ticks (e.g. center zero)
    if (p.ticks) {
      return p.ticks.map(t => {
        const pct = clampPct(normalizePct(p.key, t.val, p.min, p.max));
        return `<span class="profile-param-tick" style="left:${pct}%"><span class="profile-param-tick-line"></span><span class="profile-param-tick-label">${t.label}</span></span>`;
      }).join('');
    }
    return '';
  }

  el.innerHTML = `
    <h3 class="profile-params-title">Your Profile</h3>
    <div class="profile-params-grid">
      ${PARAMS.map(p => `
        <div class="profile-param-row">
          <span class="profile-param-label" data-param="${p.key}">${p.label}</span>
          <div class="profile-param-bar-wrap">
            <div class="profile-param-bar-track">
              ${p.center != null ? `<div class="profile-param-center" style="left:${clampPct(normalizePct(p.key, p.center, p.min, p.max))}%"></div>` : ''}
              <div class="profile-param-ci" id="ci-${p.key}"></div>
              <div class="profile-param-marker" id="bar-${p.key}"></div>
              ${buildTicks(p)}
            </div>
            ${!p.steps ? `<div class="profile-param-axis">
              <span class="profile-param-axis-min">${p.minLabel || ''}</span>
              <span class="profile-param-axis-max">${p.maxLabel || ''}</span>
            </div>` : '<div class="profile-param-axis-spacer"></div>'}
          </div>
          <span class="profile-param-value" id="val-${p.key}">--</span>
        </div>
      `).join('')}
    </div>
  `;

  // Click handler for parameter labels
  el.addEventListener('click', (e) => {
    const label = e.target.closest('.profile-param-label');
    if (!label) return;
    openDrawer(label.dataset.param, label);
  });

  function render(state) {
    const profile = state.profile;
    const visible = profile && (state.phase === 'answering' || state.phase === 'complete');

    el.style.opacity = visible ? '1' : '0.3';

    if (!profile) {
      for (const p of PARAMS) {
        const marker = el.querySelector(`#bar-${p.key}`);
        const ciEl = el.querySelector(`#ci-${p.key}`);
        const val = el.querySelector(`#val-${p.key}`);
        if (marker) marker.style.display = 'none';
        if (ciEl) ciEl.style.display = 'none';
        if (val) val.textContent = '--';
      }
      return;
    }

    for (const p of PARAMS) {
      const marker = el.querySelector(`#bar-${p.key}`);
      const ciEl = el.querySelector(`#ci-${p.key}`);
      const val = el.querySelector(`#val-${p.key}`);
      if (!marker || !val) continue;

      const v = profile[p.key];
      val.textContent = p.fmt(v);

      // Point estimate marker
      const pct = clampPct(normalizePct(p.key, v, p.min, p.max));
      marker.style.left = `${pct}%`;
      marker.style.display = 'block';

      // Confidence interval band — clamp to parameter domain
      if (ciEl && p.ci && profile[p.ci]) {
        const ci = profile[p.ci];
        const lo = Array.isArray(ci) ? Math.max(p.min, ci[0]) : v;
        const hi = Array.isArray(ci) ? Math.min(p.max, ci[1]) : v;

        const loPct = clampPct(normalizePct(p.key, lo, p.min, p.max));
        const hiPct = clampPct(normalizePct(p.key, hi, p.min, p.max));

        ciEl.style.left = `${loPct}%`;
        ciEl.style.width = `${Math.max(1, hiPct - loPct)}%`;
        ciEl.style.display = 'block';
      } else if (ciEl) {
        ciEl.style.display = 'none';
      }
    }
  }

  render(getState());
  subscribe((state) => render(state));
}
