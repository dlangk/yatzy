"""CLI for RL training and evaluation.

Commands:
    yatzy-rl train-a    Train Approach A (theta-switching)
    yatzy-rl train-c    Train Approach C (hybrid value correction)
    yatzy-rl train-b    Train Approach B (direct action with IQN)
    yatzy-rl evaluate   Evaluate a trained model
    yatzy-rl baseline   Evaluate table baselines
    yatzy-rl compare    Compare all approaches
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np


@click.group()
def cli() -> None:
    """RL agents for Yatzy: can we beat the optimal solver on tail percentiles?"""
    pass


@cli.command("train-a")
@click.option("--base-path", default="backend", help="Path to backend directory")
@click.option("--output", default="rl/checkpoints/approach_a", help="Output directory")
@click.option("--lambdas", default="0,0.5,1,2,5", help="Comma-separated lambda values")
@click.option("--episodes", default=409_600, type=int, help="Training episodes per lambda")
@click.option("--batch-size", default=4096, type=int)
@click.option("--device", default="cpu")
@click.option("--thetas", default="0.0,0.02,0.05,0.08,0.12", help="Theta values for tables")
def train_a(
    base_path: str,
    output: str,
    lambdas: str,
    episodes: int,
    batch_size: int,
    device: str,
    thetas: str,
) -> None:
    """Train Approach A: learned theta-switching with REINFORCE."""
    from .approach_a.train import TrainConfig, train

    theta_list = [float(t) for t in thetas.split(",")]
    lambda_list = [float(l) for l in lambdas.split(",")]

    for lam in lambda_list:
        print(f"\n{'='*60}")
        print(f"Training lambda={lam}")
        print(f"{'='*60}")
        config = TrainConfig(
            n_episodes=episodes,
            batch_size=batch_size,
            lam=lam,
            thetas=theta_list,
        )
        train(base_path, config, output, device=device)


@cli.command("train-c")
@click.option("--base-path", default="backend", help="Path to backend directory")
@click.option("--output", default="rl/checkpoints/approach_c", help="Output directory")
@click.option("--episodes", default=100_000, type=int)
@click.option("--batch-size", default=2048, type=int)
@click.option("--device", default="cpu")
def train_c(
    base_path: str,
    output: str,
    episodes: int,
    batch_size: int,
    device: str,
) -> None:
    """Train Approach C: hybrid value correction with actor-critic."""
    from .approach_c.train import TrainConfig, train

    config = TrainConfig(
        n_episodes=episodes,
        batch_size=batch_size,
        device=device,
    )
    train(base_path, config, output)


@cli.command("train-b")
@click.option("--base-path", default="backend", help="Path to backend directory")
@click.option("--output", default="rl/checkpoints/approach_b", help="Output directory")
@click.option("--bc-episodes", default=100_000, type=int, help="Behavioral cloning episodes")
@click.option("--rl-episodes", default=200_000, type=int, help="RL fine-tuning episodes")
@click.option("--device", default="cpu")
@click.option("--alpha", default=0.75, type=float, help="CVaR level")
@click.option("--bc-only", is_flag=True, help="Only run behavioral cloning (skip RL)")
def train_b(
    base_path: str,
    output: str,
    bc_episodes: int,
    rl_episodes: int,
    device: str,
    alpha: float,
    bc_only: bool,
) -> None:
    """Train Approach B: direct action with IQN."""
    from .approach_b.train import TrainConfig, train

    config = TrainConfig(
        bc_episodes=bc_episodes,
        rl_episodes=0 if bc_only else rl_episodes,
        alpha=alpha,
        device=device,
    )
    train(base_path, config, output)


@cli.command("evaluate")
@click.option("--approach", required=True, type=click.Choice(["a", "b", "c"]))
@click.option("--checkpoint", required=True, help="Path to checkpoint .pt file")
@click.option("--base-path", default="backend", help="Path to backend directory")
@click.option("--output", default="analytics/results/bin_files/rl", help="Output directory")
@click.option("--games", default=1_000_000, type=int)
@click.option("--device", default="cpu")
@click.option("--thetas", default="0.0,0.02,0.05,0.08,0.12", help="Theta values (Approach A)")
def evaluate(
    approach: str,
    checkpoint: str,
    base_path: str,
    output: str,
    games: int,
    device: str,
    thetas: str,
) -> None:
    """Evaluate a trained RL model."""
    import torch

    from .bridge import RustBridge
    from .evaluate import compute_stats, evaluate_approach_a, save_results
    from .tables import load_state_values, state_file_path

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    if approach == "a":
        from .approach_a.policy import ThetaPolicy

        theta_list = [float(t) for t in thetas.split(",")]
        bridge = RustBridge(base_path, theta_list)
        base_sv = load_state_values(state_file_path(base_path, 0.0))

        config = ckpt.get("config", {})
        policy = ThetaPolicy(n_actions=config.get("n_actions", len(theta_list)))
        policy.load_state_dict(ckpt["policy_state_dict"])

        scores = evaluate_approach_a(policy, bridge, base_sv, games, device=device)
        label = f"rl_a_lambda_{config.get('lam', 0):.1f}"
    else:
        raise NotImplementedError(f"Approach {approach} evaluation not yet implemented with bridge")

    save_results(scores, output, label)
    stats = compute_stats(scores)
    print(f"\nResults: mean={stats['mean']:.1f}, p95={stats['p95']:.0f}, "
          f"p5={stats['p5']:.0f}, std={stats['std']:.1f}")


@cli.command("baseline")
@click.option("--base-path", default="backend", help="Path to backend directory")
@click.option("--output", default="analytics/results/bin_files/rl", help="Output directory")
@click.option("--games", default=1_000_000, type=int)
@click.option("--theta", default=0.0, type=float, help="Theta value for baseline")
def baseline(
    base_path: str,
    output: str,
    games: int,
    theta: float,
) -> None:
    """Evaluate a fixed-theta table baseline."""
    from .bridge import RustBridge
    from .evaluate import compute_stats, evaluate_table_baseline, save_results

    bridge = RustBridge(base_path, [theta])
    scores = evaluate_table_baseline(bridge, 0, games)
    label = f"baseline_theta_{theta:.3f}"
    save_results(scores, output, label)

    stats = compute_stats(scores)
    print(f"\nBaseline theta={theta}: mean={stats['mean']:.1f}, p95={stats['p95']:.0f}")


@cli.command("compare")
@click.option("--results-dir", default="analytics/results/bin_files/rl", help="Results directory")
def compare(results_dir: str) -> None:
    """Compare all evaluated approaches."""
    from .evaluate import compute_stats

    results_path = Path(results_dir)
    if not results_path.exists():
        click.echo(f"No results found at {results_path}")
        return

    print(f"\n{'Label':<30} {'Mean':>7} {'Std':>6} {'p5':>5} {'p50':>5} "
          f"{'p90':>5} {'p95':>5} {'p99':>5}")
    print("-" * 90)

    for subdir in sorted(results_path.iterdir()):
        if not subdir.is_dir():
            continue
        scores_file = subdir / "scores.bin"
        if not scores_file.exists():
            continue

        scores = np.fromfile(scores_file, dtype=np.int32)
        stats = compute_stats(scores)
        print(
            f"{subdir.name:<30} {stats['mean']:7.1f} {stats['std']:6.1f} "
            f"{stats['p5']:5.0f} {stats['p50']:5.0f} {stats['p90']:5.0f} "
            f"{stats['p95']:5.0f} {stats['p99']:5.0f}"
        )


if __name__ == "__main__":
    cli()
