/**
 * Shared score-probability math: tail probabilities from an exact score PMF.
 * Pure functions, no DOM, no imports. Used by the treatise risk-theta chart and
 * the standalone Probabilities tab so their probability numbers cannot drift.
 * @module score-prob
 */

/**
 * Integrate a density curve above or below a score threshold (trapezoidal rule).
 * `points` is a sorted array of { x: score, y: density }. The curves in
 * kde_curves.json integrate to ~1, so this returns a probability.
 * @param {Array<{x:number,y:number}>} points
 * @param {number} threshold
 * @param {boolean} above true for P(score >= threshold), false for P(score <= threshold)
 * @returns {number}
 */
export function integrateTail(points, threshold, above) {
  let sum = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const x0 = points[i].x, x1 = points[i + 1].x;
    const y0 = points[i].y, y1 = points[i + 1].y;
    if (above) {
      if (x1 <= threshold) continue;
      const lo = Math.max(x0, threshold);
      const frac = (x1 === x0) ? 1 : (x1 - lo) / (x1 - x0);
      const yLo = y0 + (y1 - y0) * (1 - frac);
      sum += (yLo + y1) / 2 * (x1 - lo);
    } else {
      if (x0 >= threshold) continue;
      const hi = Math.min(x1, threshold);
      const frac = (x1 === x0) ? 1 : (hi - x0) / (x1 - x0);
      const yHi = y0 + (y1 - y0) * frac;
      sum += (y0 + yHi) / 2 * (hi - x0);
    }
  }
  return sum;
}

/**
 * Probability of meeting or exceeding a target score.
 * @param {Array<{x:number,y:number}>} points
 * @param {number} target
 * @returns {number}
 */
export function pAtLeast(points, target) {
  return integrateTail(points, target, true);
}
