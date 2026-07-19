/**
 * The headline metric for Delta Yatzy: the running total of expected points
 * given up versus optimal play. Delta Yatzy is a skill game where the goal is
 * to keep this delta at zero, not to maximize score.
 *
 * The per-decision deltas live on the trajectory and are always computed (hints
 * only gate their display elsewhere), so this box shows regardless of hints and
 * updates on every decision.
 */
import { getState, subscribe } from '../store.ts';
import type { TrajectoryPoint } from '../types.ts';
import { pointDelta } from '../types.ts';
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';
import { LABELS } from './RiskSlider.ts';

function cumulativeDelta(trajectory: TrajectoryPoint[], level: number): number {
  let sum = 0;
  for (const p of trajectory) {
    const d = pointDelta(p, level);
    if (d != null) sum += d;
  }
  return sum;
}

export function initDeltaBox(container: HTMLElement): void {
  container.className = 'delta-box';

  const label = document.createElement('div');
  label.className = 'delta-box-label';
  label.textContent = 'Delta vs optimal';
  attachTooltip(label, TIP.deltaBox);

  const value = document.createElement('div');
  value.className = 'delta-box-value';

  const sub = document.createElement('div');
  sub.className = 'delta-box-sub';

  container.appendChild(label);
  container.appendChild(value);
  container.appendChild(sub);

  function render(): void {
    const state = getState();
    const d = cumulativeDelta(state.trajectory, state.prefs.riskLevel);
    const perfect = d > -0.005;
    value.textContent = perfect ? '0.00' : d.toFixed(2);
    container.classList.toggle('perfect', perfect);
    const neutral = state.prefs.riskLevel === 2;
    if (neutral) {
      sub.textContent = perfect
        ? 'perfect play so far — keep the delta at zero'
        : 'expected points behind perfect play';
    } else {
      const label = LABELS[state.prefs.riskLevel].toLowerCase();
      sub.textContent = perfect
        ? `optimal for ${label} play so far`
        : `expected points behind ${label}-optimal play`;
    }
  }

  render();
  subscribe((state, prev) => {
    if (state.trajectory === prev.trajectory && state.prefs.riskLevel === prev.prefs.riskLevel) return;
    render();
  });
}
