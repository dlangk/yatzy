"""Approach A: REINFORCE training with quantile-tilted reward.

Trains the ThetaPolicy to select optimal theta tables per turn,
maximizing a shaped reward that encourages high upper-tail scores.

Uses Rust FFI bridge for ~1000x faster game simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from ..bridge import RustBridge
from ..scoring import CATEGORY_COUNT, is_category_scored, state_index
from .policy import ThetaPolicy


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    n_episodes: int = 50_000
    batch_size: int = 4096
    lr: float = 3e-4
    gamma: float = 1.0  # No discounting (episodic)
    entropy_coef: float = 0.01
    lam: float = 1.0  # Quantile-tilt strength
    baseline_ema: float = 0.999
    seed: int = 42
    log_interval: int = 2
    save_interval: int = 10_000
    thetas: list[float] | None = None

    def __post_init__(self) -> None:
        if self.thetas is None:
            self.thetas = [0.0, 0.02, 0.05, 0.08, 0.12]


def compute_shaped_reward(scores: np.ndarray, lam: float, running_p95: float) -> np.ndarray:
    """Quantile-tilted reward: score + lambda * max(0, score - running_p95).

    Encourages the agent to push scores into the upper tail.
    """
    bonus = np.maximum(0.0, scores - running_p95)
    return scores + lam * bonus


def compute_base_obs_batch(
    upper_scores: np.ndarray,
    scored_cats: np.ndarray,
    total_scores: np.ndarray,
    state_values: np.ndarray,
) -> np.ndarray:
    """Vectorized base observation for N games. Returns (N, 10) float32."""
    n = len(upper_scores)
    obs = np.zeros((n, 10), dtype=np.float32)

    # cats_left per game
    cats_scored = np.zeros(n, dtype=np.int32)
    for bit in range(CATEGORY_COUNT):
        cats_scored += ((scored_cats >> bit) & 1).astype(np.int32)
    cats_left = CATEGORY_COUNT - cats_scored

    upper_cats_left = np.zeros(n, dtype=np.int32)
    for c in range(6):
        upper_cats_left += (1 - ((scored_cats >> c) & 1)).astype(np.int32)

    obs[:, 0] = upper_scores / 63.0
    obs[:, 1] = cats_left / 15.0
    obs[:, 2] = upper_cats_left / 6.0
    obs[:, 3] = (upper_scores >= 63).astype(np.float32)

    # Bonus reachable: max remaining upper = sum of 5*(c+1) for unscored upper cats
    max_remaining_upper = np.zeros(n, dtype=np.float32)
    for c in range(6):
        unscored = (1 - ((scored_cats >> c) & 1)).astype(np.float32)
        max_remaining_upper += 5.0 * (c + 1) * unscored
    deficit = np.maximum(0, 63 - upper_scores).astype(np.float32)
    obs[:, 4] = (max_remaining_upper >= deficit).astype(np.float32)

    # Bonus health
    expected_upper = max_remaining_upper * 0.6
    safe_expected = np.maximum(expected_upper, 1.0)
    obs[:, 5] = np.clip(1.0 - deficit / safe_expected, 0.0, 1.0)
    no_remaining = max_remaining_upper == 0
    obs[no_remaining, 5] = (upper_scores[no_remaining] >= 63).astype(np.float32)

    # High-variance categories left
    high_var_cats = [10, 11, 12, 14]
    has_hv = np.zeros(n, dtype=np.float32)
    for c in high_var_cats:
        has_hv = np.maximum(has_hv, (1 - ((scored_cats >> c) & 1)).astype(np.float32))
    obs[:, 6] = has_hv

    obs[:, 7] = total_scores / 374.0

    # EV remaining
    si = scored_cats.astype(np.int64) * 64 + upper_scores.astype(np.int64)
    si = np.clip(si, 0, len(state_values) - 1)
    obs[:, 8] = state_values[si] / 200.0

    # Score z-score
    turns_played = (15 - cats_left).astype(np.float32)
    expected_score = 248.0 * turns_played / 15.0
    sigma = np.maximum(40.0 * np.sqrt(turns_played / 15.0), 1.0)
    obs[:, 9] = np.where(turns_played > 0, (total_scores - expected_score) / sigma, 0.0)

    return obs


def collect_batch_fast(
    bridge: RustBridge,
    base_sv: np.ndarray,
    policy: ThetaPolicy,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a batch of episodes using Rust bridge.

    Runs all batch_size games simultaneously, stepping through 15 turns.

    Returns:
        all_obs: (batch_size * 15, 10) float32 — observations at each step
        all_actions: (batch_size * 15,) int64 — actions taken
        episode_scores: (batch_size,) int32 — final game scores
    """
    n = batch_size
    upper_scores = np.zeros(n, dtype=np.int32)
    scored_cats = np.zeros(n, dtype=np.int32)
    total_scores = np.zeros(n, dtype=np.float32)

    all_obs_list = []
    all_actions_list = []

    for turn in range(CATEGORY_COUNT):
        # Compute observations
        obs = compute_base_obs_batch(upper_scores, scored_cats, total_scores.astype(np.int32), base_sv)
        all_obs_list.append(obs)

        # Policy selects theta indices
        obs_t = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            actions, _ = policy.get_action(obs_t)
        theta_indices = actions.cpu().numpy().astype(np.int32)
        all_actions_list.append(theta_indices)

        # Rust bridge simulates the turn for all games
        seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
        cats, scrs, new_ups = bridge.batch_turn(
            theta_indices, upper_scores, scored_cats, seeds, turn
        )

        # Update game state
        upper_scores = new_ups
        scored_cats = scored_cats | (1 << cats)
        total_scores += scrs.astype(np.float32)

    # Add bonus
    bonus = np.where(upper_scores >= 63, 50, 0).astype(np.float32)
    total_scores += bonus

    all_obs = np.concatenate(all_obs_list, axis=0)  # (N*15, 10)
    all_actions = np.concatenate(all_actions_list, axis=0)  # (N*15,)
    episode_scores = total_scores.astype(np.int32)

    return all_obs, all_actions, episode_scores


def train(
    base_path: str | Path,
    config: TrainConfig,
    output_dir: str | Path,
    device: str = "cpu",
) -> ThetaPolicy:
    """Train Approach A policy using Rust bridge for fast simulation.

    Args:
        base_path: Path to backend directory (contains data/*.bin)
        config: Training hyperparameters
        output_dir: Where to save checkpoints
        device: "cpu", "mps", or "cuda"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)

    # Initialize Rust bridge
    assert config.thetas is not None
    bridge = RustBridge(str(base_path), config.thetas)

    # Load theta=0 state values for observation features
    from ..tables import load_state_values, state_file_path
    base_sv = load_state_values(state_file_path(base_path, 0.0))

    # Create policy
    policy = ThetaPolicy(n_actions=len(config.thetas)).to(torch_device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed)

    # Running statistics
    running_p95 = 300.0
    baseline = 248.0

    print(f"Training Approach A: thetas={config.thetas}, lambda={config.lam}")
    print(f"  batch_size={config.batch_size}, lr={config.lr}, device={device}")
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    best_p95 = 0.0
    n_batches = config.n_episodes // config.batch_size

    for batch_num in range(n_batches):
        policy.train()

        # Collect batch using Rust bridge
        all_obs, all_actions, episode_scores = collect_batch_fast(
            bridge, base_sv, policy, config.batch_size, torch_device, rng
        )

        scores_arr = episode_scores.astype(np.float64)

        # Update running statistics
        batch_mean = float(scores_arr.mean())
        batch_p95 = float(np.percentile(scores_arr, 95))
        running_p95 = config.baseline_ema * running_p95 + (1 - config.baseline_ema) * batch_p95

        # Compute shaped rewards
        shaped = compute_shaped_reward(scores_arr, config.lam, running_p95)
        baseline = config.baseline_ema * baseline + (1 - config.baseline_ema) * float(shaped.mean())

        # Build tensors for policy update
        obs_tensor = torch.from_numpy(all_obs).float().to(torch_device)
        actions_tensor = torch.from_numpy(all_actions).long().to(torch_device)

        # Expand episode returns to per-step (15 steps per episode)
        advantages = shaped - baseline  # (batch_size,)
        returns_per_step = np.repeat(advantages, 15)  # (batch_size * 15,)
        returns_tensor = torch.from_numpy(returns_per_step).float().to(torch_device)

        # Policy gradient
        log_probs, entropy = policy.evaluate_actions(obs_tensor, actions_tensor)
        policy_loss = -(log_probs * returns_tensor).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + config.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        # Logging
        if batch_num % config.log_interval == 0:
            p5 = float(np.percentile(scores_arr, 5))
            p50 = float(np.percentile(scores_arr, 50))
            print(
                f"  batch {batch_num:5d} | mean={batch_mean:.1f} p5={p5:.0f} "
                f"p50={p50:.0f} p95={batch_p95:.0f} | "
                f"loss={loss.item():.4f} ent={(-entropy_loss).item():.3f}"
            )

        # Save checkpoint
        if batch_p95 > best_p95:
            best_p95 = batch_p95
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "config": {
                        "thetas": config.thetas,
                        "lam": config.lam,
                        "n_actions": len(config.thetas),
                    },
                    "stats": {
                        "mean": batch_mean,
                        "p95": batch_p95,
                        "batch": batch_num,
                    },
                },
                output_dir / f"best_lambda_{config.lam:.1f}.pt",
            )

    # Final save
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": {
                "thetas": config.thetas,
                "lam": config.lam,
                "n_actions": len(config.thetas),
            },
        },
        output_dir / f"final_lambda_{config.lam:.1f}.pt",
    )

    print(f"  Training complete. Best p95={best_p95:.0f}")
    return policy
