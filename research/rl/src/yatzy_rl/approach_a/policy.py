"""Approach A: Theta-switching policy network.

Small MLP that maps game state to a distribution over K theta values.
The agent decides which precomputed theta table to use for each turn.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThetaPolicy(nn.Module):
    """MLP policy: 10-dim state -> K theta choices.

    Architecture: 10 -> 64 -> 64 -> K (~5K params for K=4).
    """

    def __init__(self, n_actions: int = 4, obs_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.net(obs)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
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
        """Return (log_prob, entropy) for given (obs, action) pairs."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
