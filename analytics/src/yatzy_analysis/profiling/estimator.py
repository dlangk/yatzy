"""Reference MLE estimator for profiling (Python/scipy).

Mirror of the JS estimator for validation purposes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

D_GRID = [5, 8, 10, 15, 20, 999]


@dataclass
class ProfileEstimate:
    """Estimated cognitive profile."""

    theta: float
    beta: float
    gamma: float
    d: int
    nll: float
    bic: float
    ci_theta: tuple[float, float] | None = None
    ci_beta: tuple[float, float] | None = None
    ci_gamma: tuple[float, float] | None = None


def _find_nearest(arr: list, target: float) -> float:
    return min(arr, key=lambda x: abs(x - target))


def _lookup_q(scenario: dict, theta: float, gamma: float, d: int) -> np.ndarray | None:
    grid = scenario.get("q_grid")
    if not grid:
        return None

    q_values = grid.get("q_values", {})
    theta_vals = grid.get("theta_values", [])
    gamma_vals = grid.get("gamma_values", [])
    d_vals = grid.get("d_values", [])

    nearest_theta = _find_nearest(theta_vals, theta) if theta_vals else 0
    nearest_gamma = _find_nearest(gamma_vals, gamma) if gamma_vals else 1.0
    nearest_d = _find_nearest(d_vals, d) if d_vals else 999

    key = f"{nearest_theta},{nearest_gamma},{nearest_d}"
    vals = q_values.get(key)
    if vals is None:
        return None
    return np.array(vals, dtype=np.float64)


def neg_log_likelihood(
    scenarios: list[dict],
    answers: list[dict],
    theta: float,
    beta: float,
    gamma: float,
    d: int,
) -> float:
    """Compute negative log-likelihood for a set of answers."""
    nll = 0.0
    for ans in answers:
        scenario = next((s for s in scenarios if s["id"] == ans["scenarioId"]), None)
        if scenario is None:
            continue

        q = _lookup_q(scenario, theta, gamma, d)
        if q is None or len(q) == 0:
            continue

        actions = scenario["actions"]
        action_idx = next(
            (i for i, a in enumerate(actions) if a["id"] == ans["actionId"]), None
        )
        if action_idx is None or action_idx >= len(q):
            continue

        # Softmax log-probability
        q_shifted = beta * (q - q.max())
        log_sum_exp = np.log(np.sum(np.exp(q_shifted)))
        nll -= q_shifted[action_idx] - log_sum_exp

    return nll


def estimate_profile(
    scenarios: list[dict],
    answers: list[dict],
) -> ProfileEstimate:
    """Estimate profile parameters using L-BFGS-B optimization."""
    # Optimize over [theta, log(beta), gamma, d_idx]
    # We try all d values and pick the best

    best_result = None
    best_nll = float("inf")
    best_d = D_GRID[-1]

    for d in D_GRID:

        def objective(x: np.ndarray) -> float:
            theta = np.clip(x[0], -0.1, 0.1)
            beta = np.clip(np.exp(x[1]), 0.1, 20.0)
            gamma = np.clip(x[2], 0.1, 1.0)
            return neg_log_likelihood(scenarios, answers, theta, beta, gamma, d)

        x0 = np.array([0.0, np.log(2.0), 0.9])
        bounds = [(-0.1, 0.1), (np.log(0.1), np.log(20.0)), (0.1, 1.0)]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
            best_d = d

    if best_result is None:
        # Fallback
        return ProfileEstimate(
            theta=0.0, beta=2.0, gamma=0.9, d=999, nll=0.0, bic=0.0
        )

    theta = np.clip(best_result.x[0], -0.1, 0.1)
    beta = np.clip(np.exp(best_result.x[1]), 0.1, 20.0)
    gamma = np.clip(best_result.x[2], 0.1, 1.0)

    # BIC
    k = 4
    n = len(answers)
    bic = k * np.log(n) + 2 * best_nll

    # Confidence intervals via numerical Hessian
    ci_theta = None
    ci_beta = None
    ci_gamma = None
    try:
        from scipy.optimize import approx_fprime

        eps = 1e-5

        def _obj(x):
            return neg_log_likelihood(
                scenarios,
                answers,
                np.clip(x[0], -0.1, 0.1),
                np.clip(np.exp(x[1]), 0.1, 20.0),
                np.clip(x[2], 0.1, 1.0),
                best_d,
            )

        # Numerical Hessian diagonal
        x_opt = best_result.x
        hess_diag = np.zeros(3)
        f0 = _obj(x_opt)
        for i in range(3):
            xp = x_opt.copy()
            xm = x_opt.copy()
            xp[i] += eps
            xm[i] -= eps
            hess_diag[i] = (_obj(xp) - 2 * f0 + _obj(xm)) / (eps**2)

        for i in range(3):
            if hess_diag[i] > 0:
                se = np.sqrt(1.0 / hess_diag[i])
                ci = (x_opt[i] - 1.96 * se, x_opt[i] + 1.96 * se)
                if i == 0:
                    ci_theta = ci
                elif i == 1:
                    ci_beta = (np.exp(ci[0]), np.exp(ci[1]))
                elif i == 2:
                    ci_gamma = ci
    except Exception:
        pass

    return ProfileEstimate(
        theta=float(theta),
        beta=float(beta),
        gamma=float(gamma),
        d=best_d,
        nll=float(best_nll),
        bic=float(bic),
        ci_theta=ci_theta,
        ci_beta=ci_beta,
        ci_gamma=ci_gamma,
    )
