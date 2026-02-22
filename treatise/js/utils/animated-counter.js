/**
 * Tween a number display with easeOutCubic.
 */
export function animateCounter(el, from, to, duration = 800, format = null) {
  const fmt = format || ((n) => n.toLocaleString());
  const start = performance.now();
  const diff = to - from;

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function tick(now) {
    const elapsed = now - start;
    const t = Math.min(elapsed / duration, 1);
    const value = from + diff * easeOutCubic(t);
    el.textContent = fmt(Math.round(value));
    if (t < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}
