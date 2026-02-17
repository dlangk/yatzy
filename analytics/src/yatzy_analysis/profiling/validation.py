"""Parameter recovery validation for cognitive profiling.

Tests that the MLE estimator can recover known parameters from synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import estimate_profile
from .synthetic import ARCHETYPES, SyntheticPlayer, generate_synthetic_dataset


@dataclass
class RecoveryResult:
    """Result of a single recovery test."""

    player: SyntheticPlayer
    seed: int
    estimated_theta: float
    estimated_beta: float
    estimated_gamma: float
    estimated_d: int
    theta_error: float
    beta_error: float
    gamma_error: float
    d_correct: bool
    nll: float


@dataclass
class RecoverySummary:
    """Aggregate recovery statistics for one archetype."""

    name: str
    true_theta: float
    true_beta: float
    true_gamma: float
    true_d: int
    n_trials: int
    theta_bias: float
    theta_rmse: float
    beta_bias: float
    beta_rmse: float
    gamma_bias: float
    gamma_rmse: float
    d_accuracy: float


def run_recovery_test(
    scenarios: list[dict],
    player: SyntheticPlayer,
    n_trials: int = 100,
    base_seed: int = 0,
) -> list[RecoveryResult]:
    """Run n_trials recovery tests for a single player archetype."""
    results = []
    for i in range(n_trials):
        seed = base_seed + i
        answers = generate_synthetic_dataset(scenarios, player, seed=seed)
        est = estimate_profile(scenarios, answers)

        results.append(
            RecoveryResult(
                player=player,
                seed=seed,
                estimated_theta=est.theta,
                estimated_beta=est.beta,
                estimated_gamma=est.gamma,
                estimated_d=est.d,
                theta_error=est.theta - player.theta,
                beta_error=est.beta - player.beta,
                gamma_error=est.gamma - player.gamma,
                d_correct=est.d == player.d,
                nll=est.nll,
            )
        )

    return results


def summarize_recovery(
    player: SyntheticPlayer, results: list[RecoveryResult]
) -> RecoverySummary:
    """Compute aggregate statistics from recovery test results."""
    theta_errors = [r.theta_error for r in results]
    beta_errors = [r.beta_error for r in results]
    gamma_errors = [r.gamma_error for r in results]
    d_correct = [r.d_correct for r in results]

    return RecoverySummary(
        name=player.name,
        true_theta=player.theta,
        true_beta=player.beta,
        true_gamma=player.gamma,
        true_d=player.d,
        n_trials=len(results),
        theta_bias=float(np.mean(theta_errors)),
        theta_rmse=float(np.sqrt(np.mean(np.square(theta_errors)))),
        beta_bias=float(np.mean(beta_errors)),
        beta_rmse=float(np.sqrt(np.mean(np.square(beta_errors)))),
        gamma_bias=float(np.mean(gamma_errors)),
        gamma_rmse=float(np.sqrt(np.mean(np.square(gamma_errors)))),
        d_accuracy=float(np.mean(d_correct)),
    )


def run_full_validation(
    scenarios: list[dict],
    n_trials: int = 100,
    base_seed: int = 0,
) -> list[RecoverySummary]:
    """Run recovery tests for all standard archetypes."""
    summaries = []
    for i, player in enumerate(ARCHETYPES):
        print(f"  Testing {player.name}... ", end="", flush=True)
        results = run_recovery_test(
            scenarios, player, n_trials=n_trials, base_seed=base_seed + i * 1000
        )
        summary = summarize_recovery(player, results)
        summaries.append(summary)
        print(
            f"theta RMSE={summary.theta_rmse:.4f}, "
            f"beta RMSE={summary.beta_rmse:.3f}, "
            f"gamma RMSE={summary.gamma_rmse:.4f}, "
            f"d acc={summary.d_accuracy:.1%}"
        )
    return summaries


def print_validation_report(summaries: list[RecoverySummary]) -> None:
    """Print a formatted validation report."""
    print("\n=== Parameter Recovery Validation ===\n")
    header = (
        f"{'Name':<20} {'θ RMSE':>8} {'θ bias':>8} "
        f"{'β RMSE':>8} {'β bias':>8} "
        f"{'γ RMSE':>8} {'γ bias':>8} "
        f"{'d acc':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.name:<20} {s.theta_rmse:>8.4f} {s.theta_bias:>+8.4f} "
            f"{s.beta_rmse:>8.3f} {s.beta_bias:>+8.3f} "
            f"{s.gamma_rmse:>8.4f} {s.gamma_bias:>+8.4f} "
            f"{s.d_accuracy:>7.1%}"
        )

    # Overall
    print()
    mean_theta_rmse = np.mean([s.theta_rmse for s in summaries])
    mean_beta_rmse = np.mean([s.beta_rmse for s in summaries])
    mean_gamma_rmse = np.mean([s.gamma_rmse for s in summaries])
    mean_d_acc = np.mean([s.d_accuracy for s in summaries])
    med_beta_rmse = float(np.median([s.beta_rmse for s in summaries]))
    med_gamma_rmse = float(np.median([s.gamma_rmse for s in summaries]))
    print(
        f"{'MEAN':<20} {mean_theta_rmse:>8.4f} {'':>8} "
        f"{mean_beta_rmse:>8.3f} {'':>8} "
        f"{mean_gamma_rmse:>8.4f} {'':>8} "
        f"{mean_d_acc:>7.1%}"
    )
    print(
        f"{'MEDIAN':<20} {'':>8} {'':>8} "
        f"{med_beta_rmse:>8.3f} {'':>8} "
        f"{med_gamma_rmse:>8.4f}"
    )

    # Thresholds calibrated for 30-question quiz with 4 confounded parameters.
    # Use median for β and γ: high-β players are inherently unidentifiable
    # (near-deterministic → flat NLL surface) and inflate the mean.
    print("\n=== Threshold Check ===")
    print(f"  θ RMSE < 0.2:        {'PASS' if mean_theta_rmse < 0.2 else 'FAIL'} ({mean_theta_rmse:.4f})")
    print(f"  β RMSE median < 2.0: {'PASS' if med_beta_rmse < 2.0 else 'FAIL'} ({med_beta_rmse:.3f})")
    print(f"  γ RMSE median < 0.2: {'PASS' if med_gamma_rmse < 0.2 else 'FAIL'} ({med_gamma_rmse:.4f})")
    print(f"  d acc >= 60%:        {'PASS' if mean_d_acc >= 0.6 else 'FAIL'} ({mean_d_acc:.1%})")
