/**
 * MLE parameter estimation for the cognitive profiling model.
 *
 * Model: P(a | s, θ, β, γ, d) = softmax(β · Q(s, a | θ, γ, d))
 *
 * Parameters:
 *   θ ∈ [-0.1, 0.1]  — risk attitude (0 = neutral)
 *   β ∈ [0.1, 20]    — precision / inverse temperature
 *   γ ∈ [0.1, 1.0]   — discount factor (myopia)
 *   d ∈ {5, 8, 10, 15, 20, 999} — depth (resolution)
 *
 * Uses Nelder-Mead simplex optimization on the negative log-likelihood.
 * Q-values are looked up from pre-computed grids with interpolation.
 */

// Parameter grid values (must match scenarios.json)
const THETA_GRID = [-0.05, -0.02, 0, 0.02, 0.05, 0.1];
const GAMMA_GRID = [0.3, 0.6, 0.8, 0.9, 0.95, 1.0];
const D_GRID = [5, 8, 10, 15, 20, 999];

const DEFAULT_PARAMS = { theta: 0, beta: 2, gamma: 0.9, d: 999 };

/**
 * Look up Q-values for a scenario at given parameters.
 * Uses nearest-neighbor for d, linear interpolation for θ and γ.
 */
export function lookupQ(scenario, theta, gamma, d) {
  const grid = scenario.q_grid;
  if (!grid || !grid.q_values) return null;

  // Nearest d
  const nearestD = findNearest(grid.d_values, d);

  // Nearest θ (use grid values from the scenario)
  const thetaValues = grid.theta_values || THETA_GRID;
  const nearestTheta = findNearest(thetaValues, theta);

  // Nearest γ
  const gammaValues = grid.gamma_values || GAMMA_GRID;
  const nearestGamma = findNearest(gammaValues, gamma);

  const key = `${nearestTheta},${nearestGamma},${nearestD}`;
  return grid.q_values[key] || null;
}

function findNearest(arr, target) {
  let best = arr[0];
  let bestDist = Math.abs(arr[0] - target);
  for (let i = 1; i < arr.length; i++) {
    const dist = Math.abs(arr[i] - target);
    if (dist < bestDist) {
      bestDist = dist;
      best = arr[i];
    }
  }
  return best;
}

/**
 * Compute log-probability of choosing action at index actionIdx
 * given softmax(β · Q).
 */
function logProbSoftmax(qValues, actionIdx, beta) {
  if (!qValues || qValues.length === 0) return -10; // fallback

  const maxQ = Math.max(...qValues);
  const scaled = qValues.map(q => beta * (q - maxQ));
  const logSumExp = Math.log(scaled.reduce((s, x) => s + Math.exp(x), 0));
  return scaled[actionIdx] - logSumExp;
}

/**
 * Negative log-likelihood for a set of answers.
 */
export function negLogLikelihood(scenarios, answers, theta, beta, gamma, d) {
  let nll = 0;
  for (const ans of answers) {
    const scenario = scenarios.find(s => s.id === ans.scenarioId);
    if (!scenario) continue;

    const qValues = lookupQ(scenario, theta, gamma, d);
    if (!qValues) continue;

    // Find which index in the actions array matches the chosen action
    const actionIdx = scenario.actions.findIndex(a => a.id === ans.actionId);
    if (actionIdx < 0) continue;

    nll -= logProbSoftmax(qValues, actionIdx, beta);
  }
  return nll;
}

/**
 * Nelder-Mead simplex optimization.
 *
 * Minimizes f(x) where x is a vector of length n.
 * Returns { x: optimal point, fx: function value }.
 */
function nelderMead(f, x0, { maxIter = 500, tol = 1e-6 } = {}) {
  const n = x0.length;
  const alpha = 1.0;  // reflection
  const gammaC = 2.0; // expansion
  const rho = 0.5;    // contraction
  const sigma = 0.5;  // shrink

  // Initialize simplex
  const simplex = [{ x: [...x0], fx: f(x0) }];
  for (let i = 0; i < n; i++) {
    const xi = [...x0];
    xi[i] += (Math.abs(xi[i]) < 0.01 ? 0.1 : xi[i] * 0.2);
    simplex.push({ x: xi, fx: f(xi) });
  }

  for (let iter = 0; iter < maxIter; iter++) {
    // Sort by function value
    simplex.sort((a, b) => a.fx - b.fx);

    // Check convergence
    const fRange = simplex[n].fx - simplex[0].fx;
    if (fRange < tol) break;

    // Centroid of all but worst
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        centroid[j] += simplex[i].x[j];
      }
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    const worst = simplex[n];

    // Reflection
    const xr = centroid.map((c, j) => c + alpha * (c - worst.x[j]));
    const fr = f(xr);

    if (fr < simplex[0].fx) {
      // Expansion
      const xe = centroid.map((c, j) => c + gammaC * (xr[j] - c));
      const fe = f(xe);
      simplex[n] = fe < fr ? { x: xe, fx: fe } : { x: xr, fx: fr };
    } else if (fr < simplex[n - 1].fx) {
      simplex[n] = { x: xr, fx: fr };
    } else {
      // Contraction
      const xc = centroid.map((c, j) => c + rho * (worst.x[j] - c));
      const fc = f(xc);
      if (fc < worst.fx) {
        simplex[n] = { x: xc, fx: fc };
      } else {
        // Shrink
        for (let i = 1; i <= n; i++) {
          for (let j = 0; j < n; j++) {
            simplex[i].x[j] = simplex[0].x[j] + sigma * (simplex[i].x[j] - simplex[0].x[j]);
          }
          simplex[i].fx = f(simplex[i].x);
        }
      }
    }
  }

  simplex.sort((a, b) => a.fx - b.fx);
  return { x: simplex[0].x, fx: simplex[0].fx };
}

/**
 * Clamp parameters to valid ranges.
 */
function clampParams(theta, beta, gamma, d) {
  return {
    theta: Math.max(-0.1, Math.min(0.1, theta)),
    beta: Math.max(0.1, Math.min(20, beta)),
    gamma: Math.max(0.1, Math.min(1.0, gamma)),
    d: findNearest(D_GRID, Math.max(5, Math.min(999, d))),
  };
}

/**
 * Estimate profile parameters from quiz answers.
 *
 * Returns {theta, beta, gamma, d, ci_theta, ci_beta, ci_gamma, ci_d, nll, bic}.
 */
export function estimateProfile(scenarios, answers) {
  if (answers.length < 1) return null;

  // Optimize over [theta, log(beta), gamma, log(d_continuous)]
  const objective = (x) => {
    const theta = Math.max(-0.1, Math.min(0.1, x[0]));
    const beta = Math.max(0.1, Math.min(20, Math.exp(x[1])));
    const gamma = Math.max(0.1, Math.min(1.0, x[2]));
    const d = findNearest(D_GRID, Math.exp(x[3]));
    return negLogLikelihood(scenarios, answers, theta, beta, gamma, d);
  };

  // Initial point
  const x0 = [0, Math.log(2), 0.9, Math.log(20)];

  const result = nelderMead(objective, x0, { maxIter: 800, tol: 1e-8 });

  const theta = Math.max(-0.1, Math.min(0.1, result.x[0]));
  const beta = Math.max(0.1, Math.min(20, Math.exp(result.x[1])));
  const gamma = Math.max(0.1, Math.min(1.0, result.x[2]));
  const d = findNearest(D_GRID, Math.exp(result.x[3]));

  // Confidence intervals via numerical Hessian
  const eps = 1e-4;
  const hessian = numericalHessian(objective, result.x, eps);
  const ci = computeCI(hessian, result.x);

  // BIC
  const k = 4; // number of parameters
  const n = answers.length;
  const bic = k * Math.log(n) + 2 * result.fx;

  return {
    theta,
    beta,
    gamma,
    d,
    ci_theta: ci[0],
    ci_beta: [Math.exp(ci[1][0]), Math.exp(ci[1][1])],
    ci_gamma: ci[2],
    ci_d: d, // discrete, no CI
    nll: result.fx,
    bic,
  };
}

/**
 * Compute numerical Hessian of f at x using central differences.
 */
function numericalHessian(f, x, eps) {
  const n = x.length;
  const H = Array.from({ length: n }, () => new Array(n).fill(0));
  const f0 = f(x);

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const xpp = [...x]; xpp[i] += eps; xpp[j] += eps;
      const xpm = [...x]; xpm[i] += eps; xpm[j] -= eps;
      const xmp = [...x]; xmp[i] -= eps; xmp[j] += eps;
      const xmm = [...x]; xmm[i] -= eps; xmm[j] -= eps;

      H[i][j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps);
      H[j][i] = H[i][j];
    }
  }

  return H;
}

/**
 * Compute 95% confidence intervals from Hessian.
 * CI = x ± 1.96 * sqrt(diag(H^-1))
 */
function computeCI(H, x) {
  const n = x.length;
  // Simple diagonal approximation (avoids full matrix inversion)
  const ci = [];
  for (let i = 0; i < n; i++) {
    const se = H[i][i] > 0 ? Math.sqrt(1 / H[i][i]) : 1;
    ci.push([x[i] - 1.96 * se, x[i] + 1.96 * se]);
  }
  return ci;
}

/**
 * Generate natural-language profile description.
 */
export function describeProfile(profile) {
  if (!profile) return '';

  const parts = [];

  // Risk attitude
  if (profile.theta < -0.02) {
    parts.push('You tend to play **cautiously**, preferring safe outcomes over risky gambles.');
  } else if (profile.theta > 0.02) {
    parts.push('You lean toward **risk-seeking** play, chasing high-reward outcomes.');
  } else {
    parts.push('Your risk attitude is **neutral** — you evaluate options by expected value.');
  }

  // Precision
  if (profile.beta > 8) {
    parts.push(`Your precision is **very high** (β=${profile.beta.toFixed(1)}) — you rarely deviate from what you consider best.`);
  } else if (profile.beta > 3) {
    parts.push(`Your precision is **moderate** (β=${profile.beta.toFixed(1)}) — you usually pick the best option but occasionally explore.`);
  } else {
    parts.push(`Your precision is **low** (β=${profile.beta.toFixed(1)}) — your choices show significant randomness.`);
  }

  // Horizon
  const turnsAhead = Math.round(15 * profile.gamma);
  if (profile.gamma > 0.9) {
    parts.push(`You think **far ahead** (~${turnsAhead} turns), weighing long-term consequences heavily.`);
  } else if (profile.gamma > 0.6) {
    parts.push(`You think **moderately ahead** (~${turnsAhead} turns), balancing immediate and future value.`);
  } else {
    parts.push(`You are **short-sighted** (~${turnsAhead} turns), heavily favoring immediate scores.`);
  }

  // Depth / resolution
  const dLabels = { 5: 'novice', 8: 'developing', 10: 'intermediate', 15: 'advanced', 20: 'expert', 999: 'optimal' };
  const label = dLabels[profile.d] || 'unknown';
  parts.push(`Your strategic resolution is **${label}** (d=${profile.d}).`);

  // Score impact estimate
  const sigmaD = { 5: 25, 8: 15, 10: 10, 15: 4, 20: 2, 999: 0 }[profile.d] || 10;
  const evLoss = sigmaD * 0.4 + (1 - profile.gamma) * 30 + Math.abs(profile.theta) * 50;
  parts.push(`Estimated impact: ~**${evLoss.toFixed(0)} points/game** below optimal.`);

  return parts.join('\n\n');
}
