"""Approach C: Hybrid value correction network.

Learns a small delta(state, accumulated_score) that adjusts the theta=0 EV
when making decisions. The correction captures risk-sensitive residuals that
the fixed-theta solver cannot express (because it lacks accumulated score).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CorrectionNetwork(nn.Module):
    """Correction network: adjusts theta=0 EV with a learned delta.

    adjusted_value(action) = EV_theta0(action) + delta(state, total_score, action)

    Architecture: 18 -> 128 -> 128 -> 1 (~20K params).
    Input: full observation (base + dice features).
    Output: scalar correction delta.
    """

    def __init__(self, obs_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return correction delta for the given observation."""
        return self.net(obs).squeeze(-1)


class CorrectionCritic(nn.Module):
    """Distributional critic: estimates quantile values of return.

    Uses quantile regression to learn the full return distribution,
    enabling risk-sensitive evaluation.

    Architecture: 18 -> 128 -> 128 -> n_quantiles
    """

    def __init__(self, obs_dim: int = 18, n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_quantiles),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return quantile values for the given observation.

        Output shape: (batch, n_quantiles) representing sorted quantile estimates.
        """
        return self.net(obs)


class CorrectionActor(nn.Module):
    """Actor network for direct action selection with correction.

    For reroll phase: outputs preference over 32 masks.
    For score phase: outputs preference over 15 categories.
    The actor can be used standalone or combined with theta=0 EV baseline.
    """

    def __init__(self, obs_dim: int = 18, n_reroll_actions: int = 32, n_score_actions: int = 15):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.reroll_head = nn.Linear(128, n_reroll_actions)
        self.score_head = nn.Linear(128, n_score_actions)

    def forward(self, obs: torch.Tensor, phase: str = "reroll") -> torch.Tensor:
        """Return action logits for the given phase."""
        h = self.shared(obs)
        if phase == "reroll":
            return self.reroll_head(h)
        return self.score_head(h)

    def get_action(
        self,
        obs: torch.Tensor,
        phase: str,
        valid_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob).

        If valid_mask provided, invalid actions get -inf logits.
        """
        logits = self.forward(obs, phase)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action)
