"""Approach C: Turn-level value correction with PPO.

The actor learns a per-turn correction that selects a continuous risk level
conditioned on accumulated score, bonus health, and game phase.

Unlike Approach A (discrete θ-switching), Approach C uses a continuous
action space and PPO for stable updates, providing better gradient signal.

Key differences from Approach A:
- Continuous action (θ value) instead of discrete index
- PPO with clipped objective instead of REINFORCE
- GAE for variance reduction
- Critic baseline for advantage estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..bridge import RustBridge
from ..scoring import CATEGORY_COUNT
from ..approach_a.train import compute_base_obs_batch


@dataclass
class TrainConfig:
    """Training hyperparameters for Approach C."""
    n_episodes: int = 1_024_000
    batch_size: int = 4096
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    n_epochs: int = 4  # PPO epochs per batch
    seed: int = 42
    log_interval: int = 5
    thetas: list[float] | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.thetas is None:
            # Dense grid of available θ values
            self.thetas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
                           0.06, 0.07, 0.08, 0.09, 0.10, 0.12]


class ContinuousActor(nn.Module):
    """Actor: maps state to a distribution over θ-table indices.

    Uses a larger network than Approach A and outputs a softmax
    distribution over all available θ tables. This is effectively
    a richer version of θ-switching with more options and better training.

    Architecture: 10 -> 128 -> 128 -> K
    """

    def __init__(self, obs_dim: int = 10, n_actions: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class ValueCritic(nn.Module):
    """Critic: estimates V(s) = E[total_score | state].

    Architecture: 10 -> 128 -> 128 -> 1
    """

    def __init__(self, obs_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def collect_batch_ppo(
    bridge: RustBridge,
    base_sv: np.ndarray,
    actor: ContinuousActor,
    critic: ValueCritic,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Collect batch with PPO-style rollouts using Rust bridge.

    Returns arrays for all N*15 steps plus episode scores.
    """
    n = batch_size
    upper_scores = np.zeros(n, dtype=np.int32)
    scored_cats = np.zeros(n, dtype=np.int32)
    total_scores = np.zeros(n, dtype=np.float32)

    # Storage: (turns, batch)
    all_obs = np.zeros((CATEGORY_COUNT, n, 10), dtype=np.float32)
    all_actions = np.zeros((CATEGORY_COUNT, n), dtype=np.int32)
    all_log_probs = np.zeros((CATEGORY_COUNT, n), dtype=np.float32)
    all_values = np.zeros((CATEGORY_COUNT, n), dtype=np.float32)
    all_turn_scores = np.zeros((CATEGORY_COUNT, n), dtype=np.float32)

    actor.eval()
    critic.eval()

    for turn in range(CATEGORY_COUNT):
        obs = compute_base_obs_batch(
            upper_scores, scored_cats, total_scores.astype(np.int32), base_sv
        )
        all_obs[turn] = obs

        obs_t = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            actions, log_probs = actor.get_action(obs_t)
            values = critic(obs_t)

        theta_indices = actions.cpu().numpy().astype(np.int32)
        all_actions[turn] = theta_indices
        all_log_probs[turn] = log_probs.cpu().numpy()
        all_values[turn] = values.cpu().numpy()

        seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
        cats, scrs, new_ups = bridge.batch_turn(
            theta_indices, upper_scores, scored_cats, seeds, turn
        )

        all_turn_scores[turn] = scrs.astype(np.float32)
        upper_scores = new_ups
        scored_cats = scored_cats | (1 << cats)
        total_scores += scrs.astype(np.float32)

    # Add bonus
    bonus = np.where(upper_scores >= 63, 50, 0).astype(np.float32)
    total_scores += bonus

    return {
        "obs": all_obs,  # (15, N, 10)
        "actions": all_actions,  # (15, N)
        "log_probs": all_log_probs,  # (15, N)
        "values": all_values,  # (15, N)
        "turn_scores": all_turn_scores,  # (15, N)
        "episode_scores": total_scores.astype(np.int32),  # (N,)
        "bonus": bonus,  # (N,)
    }


def compute_advantages(
    episode_scores: np.ndarray,
    values: np.ndarray,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns.

    Args:
        episode_scores: (N,) final game scores
        values: (15, N) critic value estimates per turn

    Returns:
        advantages: (15, N)
        returns: (15, N)
    """
    n_turns, n = values.shape
    advantages = np.zeros_like(values)
    returns = np.zeros_like(values)

    # Terminal reward at turn 14, zero intermediate
    rewards = np.zeros_like(values)
    rewards[-1] = episode_scores.astype(np.float32)

    # GAE backward pass
    last_gae = np.zeros(n, dtype=np.float32)
    for t in range(n_turns - 1, -1, -1):
        if t == n_turns - 1:
            next_value = np.zeros(n, dtype=np.float32)  # terminal
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def train(
    base_path: str | Path,
    config: TrainConfig,
    output_dir: str | Path,
) -> ContinuousActor:
    """Train Approach C with PPO.

    Uses Rust bridge for fast simulation, PPO for stable policy updates,
    and a critic baseline for variance reduction.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    assert config.thetas is not None
    bridge = RustBridge(str(base_path), config.thetas)

    from ..tables import load_state_values, state_file_path
    base_sv = load_state_values(state_file_path(base_path, 0.0))

    n_actions = len(config.thetas)
    actor = ContinuousActor(n_actions=n_actions).to(device)
    critic = ValueCritic().to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=config.lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=config.lr_critic)
    rng = np.random.default_rng(config.seed)

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Training Approach C (PPO): thetas={config.thetas}")
    print(f"  batch_size={config.batch_size}, device={config.device}")
    print(f"  Actor params: {actor_params:,}, Critic params: {critic_params:,}")
    print(f"  PPO epochs: {config.n_epochs}, clip_eps: {config.clip_eps}")

    best_p95 = 0.0
    n_batches = config.n_episodes // config.batch_size

    for batch_num in range(n_batches):
        # Collect rollouts
        data = collect_batch_ppo(
            bridge, base_sv, actor, critic,
            config.batch_size, device, rng
        )

        scores = data["episode_scores"]
        batch_mean = float(scores.mean())
        batch_p95 = float(np.percentile(scores, 95))

        # Compute advantages with GAE
        advantages, returns = compute_advantages(
            scores, data["values"]
        )

        # Flatten: (15, N) -> (15*N,)
        flat_obs = torch.from_numpy(
            data["obs"].reshape(-1, 10)
        ).float().to(device)
        flat_actions = torch.from_numpy(
            data["actions"].reshape(-1)
        ).long().to(device)
        flat_old_log_probs = torch.from_numpy(
            data["log_probs"].reshape(-1)
        ).float().to(device)
        flat_advantages = torch.from_numpy(
            advantages.reshape(-1).astype(np.float32)
        ).float().to(device)
        flat_returns = torch.from_numpy(
            returns.reshape(-1).astype(np.float32)
        ).float().to(device)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # PPO epochs
        actor.train()
        critic.train()
        total_steps = len(flat_obs)
        indices = np.arange(total_steps)

        for epoch in range(config.n_epochs):
            rng.shuffle(indices)

            # Mini-batch SGD
            mb_size = min(4096 * 15, total_steps)  # one full batch
            for start in range(0, total_steps, mb_size):
                end = min(start + mb_size, total_steps)
                mb_idx = indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_lp = flat_old_log_probs[mb_idx]
                mb_adv = flat_advantages[mb_idx]
                mb_ret = flat_returns[mb_idx]

                # Actor loss (PPO clipped)
                new_log_probs, entropy = actor.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - config.entropy_coef * entropy.mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_actor.step()

                # Critic loss (MSE on returns)
                values = critic(mb_obs)
                critic_loss = nn.functional.mse_loss(values, mb_ret)

                opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_critic.step()

        # Logging
        if batch_num % config.log_interval == 0:
            p5 = float(np.percentile(scores, 5))
            p50 = float(np.percentile(scores, 50))

            # Check action distribution
            action_dist = np.bincount(data["actions"].reshape(-1), minlength=n_actions)
            action_pct = action_dist / action_dist.sum() * 100
            top3 = np.argsort(action_pct)[-3:][::-1]
            top3_str = ", ".join(
                f"θ={config.thetas[i]:.2f}:{action_pct[i]:.0f}%"
                for i in top3
            )
            print(
                f"  batch {batch_num:5d} | mean={batch_mean:.1f} p5={p5:.0f} "
                f"p50={p50:.0f} p95={batch_p95:.0f} | "
                f"act_loss={actor_loss.item():.4f} crit_loss={critic_loss.item():.1f} | "
                f"top: {top3_str}"
            )

        # Save best
        if batch_p95 > best_p95:
            best_p95 = batch_p95
            torch.save(
                {
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "config": {
                        "thetas": config.thetas,
                        "n_actions": n_actions,
                    },
                    "stats": {
                        "mean": batch_mean,
                        "p95": batch_p95,
                        "batch": batch_num,
                    },
                },
                output_dir / "best_approach_c.pt",
            )

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": {
                "thetas": config.thetas,
                "n_actions": n_actions,
            },
        },
        output_dir / "final_approach_c.pt",
    )

    print(f"  Training complete. Best p95={best_p95:.0f}")
    return actor
