"""Convergence validation for the θ estimator.

Simulates synthetic humans with known (θ_true, β_true), runs the Bayesian
estimator, and checks how quickly the posterior converges to the ground truth.

This validates that:
1. The estimator recovers θ_true within the credible interval for β >= 2
2. Posterior width shrinks monotonically for fixed-θ synthetic humans
3. Posterior width plateaus for adaptive synthetic humans (detecting inconsistency)

Uses vectorized index-based APIs for speed (~0.3s per trial).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .theta_estimator import ThetaEstimator, _scenario_values


def _sample_action_vectorized(
    estimator: ThetaEstimator,
    scenario_idx: int,
    theta_true: float,
    beta_true: float,
    rng: np.random.Generator,
) -> int:
    """Sample an action from a synthetic human using precomputed values.

    Uses the estimator's precomputed value matrix instead of dict lookups.
    Returns the category_id of the chosen action.
    """
    # Find the theta index closest to theta_true
    ti = int(np.argmin(np.abs(estimator.theta_grid - theta_true)))

    values = estimator._values[scenario_idx]  # (n_theta, n_actions)
    action_vals = values[ti, :]  # (n_actions,)
    action_ids = estimator._action_ids[scenario_idx]

    if len(action_ids) == 0:
        return 0

    # Softmax with β
    logits = beta_true * action_vals
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    chosen_col = rng.choice(len(action_ids), p=probs)
    return int(action_ids[chosen_col])


def _sample_action_adaptive(
    estimator: ThetaEstimator,
    scenario_idx: int,
    scenario: dict[str, Any],
    theta_early: float,
    theta_late: float,
    beta_true: float,
    rng: np.random.Generator,
) -> int:
    """Simulate an adaptive human: theta_early for turns 1-7, theta_late for 8-15."""
    turn = scenario.get("turn", 8)
    theta = theta_early if turn <= 7 else theta_late
    return _sample_action_vectorized(estimator, scenario_idx, theta, beta_true, rng)


def run_convergence_trial(
    estimator_template: ThetaEstimator,
    scenarios: list[dict[str, Any]],
    theta_true: float,
    beta_true: float,
    max_questions: int = 50,
    adaptive: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run one trial: present scenarios adaptively, track posterior convergence.

    Args:
        estimator_template: A ThetaEstimator with precomputed values (reused for init speed).
        scenarios: Pool of pivotal scenarios.
        theta_true: Ground-truth θ (ignored if adaptive is set).
        beta_true: Ground-truth rationality.
        max_questions: Maximum number of questions.
        adaptive: If set, (theta_early, theta_late) for adaptive human.
        rng: Random number generator.

    Returns dict with:
        - ci_widths: list of 95% CI widths after each question
        - theta_means: list of posterior mean θ after each question
        - final_mean: final θ estimate
        - final_ci: (low, high)
        - recovered: whether ground truth is within final CI
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fresh estimator reusing precomputed scenario data
    estimator = ThetaEstimator.__new__(ThetaEstimator)
    estimator.theta_grid = estimator_template.theta_grid
    estimator.beta_grid = estimator_template.beta_grid
    estimator.scenarios = estimator_template.scenarios
    estimator.history = []
    estimator.log_posterior = np.zeros_like(estimator_template.log_posterior)
    estimator._sid_to_idx = estimator_template._sid_to_idx
    estimator._action_ids = estimator_template._action_ids
    estimator._values = estimator_template._values
    estimator._action_id_to_col = estimator_template._action_id_to_col
    estimator._optimal_col = estimator_template._optimal_col

    # Build pool as indices
    pool_indices = list(range(len(scenarios)))

    ci_widths: list[float] = []
    theta_means: list[float] = []

    n_questions = min(max_questions, len(pool_indices))
    for _ in range(n_questions):
        if not pool_indices:
            break

        # Select next best scenario (by index)
        si = estimator.select_next_idx(pool_indices)
        pool_indices = [i for i in pool_indices if i != si]

        scenario = scenarios[si]
        scenario_id = scenario["id"]

        # Synthetic human responds
        if adaptive is not None:
            action = _sample_action_adaptive(
                estimator, si, scenario,
                adaptive[0], adaptive[1], beta_true, rng,
            )
        else:
            action = _sample_action_vectorized(
                estimator, si, theta_true, beta_true, rng,
            )

        estimator.update_by_idx(si, action, scenario_id)

        # Track convergence
        mean, ci_low, ci_high = estimator.theta_estimate()
        ci_widths.append(ci_high - ci_low)
        theta_means.append(mean)

    final_mean, final_low, final_high = estimator.theta_estimate()
    if adaptive is not None:
        avg_theta = (adaptive[0] + adaptive[1]) / 2
        recovered = final_low <= avg_theta <= final_high
    else:
        recovered = final_low <= theta_true <= final_high

    return {
        "ci_widths": ci_widths,
        "theta_means": theta_means,
        "final_mean": final_mean,
        "final_ci": (final_low, final_high),
        "recovered": recovered,
    }


def run_validation(
    scenarios: list[dict[str, Any]],
    n_trials: int = 100,
    max_questions: int = 30,
) -> dict[str, Any]:
    """Run full validation suite across multiple (θ_true, β_true) combinations.

    Returns a dict with results for each parameter combination.
    """
    # Precompute scenario data once (shared across all trials)
    print("  Precomputing scenario value matrices...", flush=True)
    template = ThetaEstimator(scenarios)

    test_thetas = [0.0, 0.03, 0.05, 0.10, 0.15]
    test_betas = [0.5, 2.0, 5.0, 50.0]

    results: dict[str, Any] = {}

    for theta_true in test_thetas:
        for beta_true in test_betas:
            key = f"theta={theta_true:.2f}_beta={beta_true:.1f}"
            print(f"  Validating {key} ({n_trials} trials)...", end="", flush=True)

            trial_results = []
            for trial in range(n_trials):
                seed = trial * 1000 + int(theta_true * 100) + int(beta_true * 10)
                rng = np.random.default_rng(seed)
                result = run_convergence_trial(
                    template, scenarios, theta_true, beta_true, max_questions,
                    rng=rng,
                )
                trial_results.append(result)

            # Aggregate
            recovery_rate = np.mean([r["recovered"] for r in trial_results])
            mean_final_width = np.mean(
                [r["ci_widths"][-1] if r["ci_widths"] else 0.2 for r in trial_results]
            )
            mean_final_estimate = np.mean([r["final_mean"] for r in trial_results])
            bias = mean_final_estimate - theta_true

            widths_at: dict[int, float | None] = {}
            for n_q in [5, 8, 10, 15, 20, 25, 30]:
                if n_q <= max_questions:
                    ws = [
                        r["ci_widths"][n_q - 1]
                        for r in trial_results
                        if len(r["ci_widths"]) >= n_q
                    ]
                    widths_at[n_q] = float(np.mean(ws)) if ws else None

            results[key] = {
                "theta_true": theta_true,
                "beta_true": beta_true,
                "recovery_rate": float(recovery_rate),
                "mean_final_width": float(mean_final_width),
                "mean_final_estimate": float(mean_final_estimate),
                "bias": float(bias),
                "widths_at": widths_at,
            }
            print(f" recovery={recovery_rate:.0%}, width={mean_final_width:.3f}")

    # Adaptive humans
    adaptive_configs = [
        (0.01, 0.12, 5.0, "conservative->risk-seeking"),
        (0.10, 0.02, 5.0, "risk-seeking->conservative"),
    ]

    for theta_early, theta_late, beta_true, label in adaptive_configs:
        key = f"adaptive_{label}"
        print(f"  Validating {key} ({n_trials} trials)...", end="", flush=True)

        trial_results = []
        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 2000 + int(theta_early * 100))
            result = run_convergence_trial(
                template, scenarios, 0.0, beta_true, max_questions,
                adaptive=(theta_early, theta_late),
                rng=rng,
            )
            trial_results.append(result)

        mean_final_width = np.mean(
            [r["ci_widths"][-1] if r["ci_widths"] else 0.2 for r in trial_results]
        )
        mean_est = float(np.mean([r["final_mean"] for r in trial_results]))

        results[key] = {
            "theta_early": theta_early,
            "theta_late": theta_late,
            "beta_true": beta_true,
            "label": label,
            "mean_final_width": float(mean_final_width),
            "mean_final_estimate": mean_est,
        }
        print(f" width={mean_final_width:.3f}, mean_θ={mean_est:.3f}")

    return results


def print_validation_results(results: dict[str, Any]) -> None:
    """Print formatted validation results."""
    print()
    print("=" * 75)
    print("  Convergence Validation Results")
    print("=" * 75)
    print()

    # Fixed-θ results
    print("  Fixed-θ synthetic humans:")
    print(
        f"  {'θ_true':>6s} {'β_true':>6s} | {'recovery':>8s} | "
        f"{'bias':>6s} | {'CI@5':>5s} {'CI@10':>5s} {'CI@15':>5s} {'CI@20':>5s}"
    )
    print("  " + "-" * 68)

    for key, r in results.items():
        if key.startswith("adaptive"):
            continue
        wid = r["widths_at"]

        def _w(n: int) -> str:
            v = wid.get(n)
            return f"{v:.3f}" if v is not None else "  -  "

        print(
            f"  {r['theta_true']:>6.2f} {r['beta_true']:>6.1f} | "
            f"{r['recovery_rate']:>7.0%} | "
            f"{r['bias']:>+6.3f} | "
            f"{_w(5)} {_w(10)} {_w(15)} {_w(20)}"
        )

    # Adaptive results
    adaptive_results = {k: v for k, v in results.items() if k.startswith("adaptive")}
    if adaptive_results:
        print()
        print("  Adaptive synthetic humans (θ varies by game phase):")
        print(f"  {'label':>30s} | {'mean_θ':>6s} | {'CI_width':>8s}")
        print("  " + "-" * 52)
        for _key, r in adaptive_results.items():
            print(
                f"  {r['label']:>30s} | "
                f"{r['mean_final_estimate']:>6.3f} | "
                f"{r['mean_final_width']:>8.3f}"
            )
        print()
        print("  (Adaptive humans should show wider final CI than fixed-θ humans)")

    # Summary
    print()
    fixed = {k: v for k, v in results.items() if not k.startswith("adaptive")}
    good_beta = {k: v for k, v in fixed.items() if v["beta_true"] >= 2.0}
    if good_beta:
        avg_recovery = np.mean([v["recovery_rate"] for v in good_beta.values()])
        print(f"  Average recovery rate (β >= 2): {avg_recovery:.0%}")
        widths_15 = [
            v["widths_at"].get(15, 0.2) for v in good_beta.values()
            if v["widths_at"].get(15) is not None
        ]
        if widths_15:
            avg_width_15 = np.mean(widths_15)
            print(f"  Average CI width after 15 questions (β >= 2): +/-{avg_width_15 / 2:.3f}")
