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
import { attachTooltip } from './Tooltip.ts';
import { TIP } from '../tooltips.ts';

function cumulativeDelta(trajectory: TrajectoryPoint[]): number {
  let sum = 0;
  for (const p of trajectory) if (p.delta != null) sum += p.delta;
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
    const d = cumulativeDelta(getState().trajectory);
    const perfect = d > -0.005;
    value.textContent = perfect ? '0.00' : d.toFixed(2);
    container.classList.toggle('perfect', perfect);
    sub.textContent = perfect
      ? 'perfect play so far — keep the delta at zero'
      : 'expected points behind perfect play';
  }

  render();
  subscribe((state, prev) => {
    if (state.trajectory === prev.trajectory) return;
    render();
  });
}
