import { DataLoader } from '../data-loader.js';
import { getTextColor, getMutedColor, COLORS } from '../yatzy-viz.js';
import { ScrollDriver } from '../utils/scroll-driver.js';
import { animateCounter } from '../utils/animated-counter.js';

export async function initStateSpaceCounter() {
  const container = document.getElementById('chart-state-space-counter');
  if (!container) return;

  const [stepsData, reachData] = await Promise.all([
    DataLoader.stateCounterSteps(),
    DataLoader.reachability(),
  ]);

  const stickyEl = document.getElementById('counter-sticky');
  const displayEl = document.getElementById('counter-display');
  const stepsContainer = document.getElementById('counter-steps');
  const stepEls = stepsContainer.querySelectorAll('.step');

  // Build the persistent display
  displayEl.innerHTML = `
    <div class="counter-big" id="counter-number">0</div>
    <div class="counter-label" id="counter-sublabel"></div>
    <div class="counter-bar-wrap" id="counter-bar-wrap" style="display:none">
      <div class="counter-bar-track">
        <div class="counter-bar-fill" id="counter-bar-fill"></div>
      </div>
      <div class="counter-bar-labels">
        <span id="counter-bar-left"></span>
        <span id="counter-bar-right"></span>
      </div>
    </div>
  `;

  const numberEl = document.getElementById('counter-number');
  const sublabelEl = document.getElementById('counter-sublabel');
  const barWrap = document.getElementById('counter-bar-wrap');
  const barFill = document.getElementById('counter-bar-fill');
  const barLeft = document.getElementById('counter-bar-left');
  const barRight = document.getElementById('counter-bar-right');

  let prevNumber = 0;

  const depth = () => document.body.getAttribute('data-depth') || '1';

  function updateDisplay(stepIdx) {
    const step = stepsData.steps[stepIdx];
    barWrap.style.display = 'none';

    if (stepIdx === 0) {
      // 2^15 = 32,768 bitmasks
      const target = 32768;
      animateCounter(numberEl, prevNumber, target);
      prevNumber = target;
      sublabelEl.textContent = 'possible scorecards';
      if (depth() >= '2') {
        sublabelEl.textContent = '2\u00B9\u2075 = 32,768 subsets of 15 categories';
      }
    } else if (stepIdx === 1) {
      // × 64 upper scores = 2,097,152
      const target = 2097152;
      animateCounter(numberEl, prevNumber, target);
      prevNumber = target;
      sublabelEl.textContent = 'state slots';
      if (depth() >= '2') {
        sublabelEl.textContent = '32,768 \u00D7 64 upper scores = 2,097,152';
      }
      if (depth() >= '3') {
        sublabelEl.textContent = 'S = (C, m) where C \u2286 {1..15}, m \u2208 [0, 63]';
      }
    } else if (stepIdx === 2) {
      // Pruned to ~1,430,000
      const target = 1430000;
      animateCounter(numberEl, prevNumber, target);
      prevNumber = target;
      sublabelEl.textContent = 'reachable states';
      // Show bar
      barWrap.style.display = 'block';
      const pct = ((1 - 1430000 / 2097152) * 100).toFixed(1);
      barFill.style.width = `${100 - parseFloat(pct)}%`;
      barFill.style.background = COLORS.accent;
      barLeft.textContent = `${pct}% pruned`;
      barRight.textContent = '1,430,000 reachable';
      if (depth() >= '3') {
        sublabelEl.textContent = 'state_index(up, scored) = scored * 128 + up';
      }
    } else if (stepIdx === 3) {
      // Widget: 252 × 462 per state
      const target = 1681;
      animateCounter(numberEl, prevNumber, target);
      prevNumber = target;
      sublabelEl.textContent = 'decisions per state';
      barWrap.style.display = 'block';
      barFill.style.width = '100%';
      barFill.style.background = COLORS.riskAverse;
      barLeft.textContent = '252 outcomes';
      barRight.textContent = '462 keep-multisets';
      if (depth() >= '3') {
        sublabelEl.textContent = 'KeepTable: f32 transition probabilities';
      }
    } else if (stepIdx === 4) {
      // Final: 8 MB
      numberEl.textContent = '8 MB';
      prevNumber = 8;
      sublabelEl.textContent = 'one complete strategy table';
      barWrap.style.display = 'none';
      if (depth() >= '2') {
        sublabelEl.textContent = '16 MB with STATE_STRIDE=128 padding';
      }
    }
  }

  // Setup scroll steps
  const steps = Array.from(stepEls).map((el, idx) => ({
    el,
    enter() {
      el.classList.add('active');
      updateDisplay(idx);
    },
    leave() {
      el.classList.remove('active');
    },
  }));

  const driver = new ScrollDriver(container, steps, null);
  driver.start();

  // Activate first step by default
  if (steps.length > 0) {
    steps[0].enter();
  }
}
