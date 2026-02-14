"""Approach B: Implicit Quantile Network (IQN) for direct action selection.

Learns the full return distribution Z(s,a) using implicit quantile regression.
Action selection uses CVaR: average over the bottom-alpha quantile samples
for risk-sensitive decisions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CosineQuantileEmbedding(nn.Module):
    """Cosine basis embedding for quantile levels tau.

    Maps tau in [0,1] to a d-dimensional embedding using:
        phi(tau) = ReLU(W @ [cos(pi * i * tau) for i in 0..n_cos] + b)
    """

    def __init__(self, n_cos: int = 64, embed_dim: int = 128):
        super().__init__()
        self.n_cos = n_cos
        self.embed_dim = embed_dim
        self.linear = nn.Linear(n_cos, embed_dim)
        # Register pi * i as buffer
        self.register_buffer(
            "pi_i",
            torch.arange(0, n_cos, dtype=torch.float32) * np.pi,
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Embed quantile levels.

        Args:
            tau: (..., n_tau) quantile levels in [0, 1]

        Returns:
            (..., n_tau, embed_dim) embeddings
        """
        # tau: (..., n_tau) -> (..., n_tau, 1) -> (..., n_tau, n_cos)
        cos_input = tau.unsqueeze(-1) * self.pi_i
        cos_features = torch.cos(cos_input)
        return torch.relu(self.linear(cos_features))


class IQN(nn.Module):
    """Implicit Quantile Network for Yatzy.

    Architecture:
        State encoder: obs_dim -> 128 (shared)
        Quantile embedding: tau -> 128 (cosine basis)
        Combined: 128 * 128 (element-wise) -> 128 -> n_actions

    For reroll: n_actions = 32 (masks)
    For scoring: n_actions = 15 (categories)
    """

    def __init__(
        self,
        obs_dim: int = 18,
        n_reroll_actions: int = 32,
        n_score_actions: int = 15,
        hidden_dim: int = 128,
        n_cos: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Quantile embedding
        self.quantile_embed = CosineQuantileEmbedding(n_cos, hidden_dim)

        # Action value heads
        self.reroll_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_reroll_actions),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_score_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        tau: torch.Tensor,
        phase: str = "reroll",
    ) -> torch.Tensor:
        """Compute quantile values for given states and quantile levels.

        Args:
            obs: (batch, obs_dim) state observations
            tau: (batch, n_tau) quantile levels
            phase: "reroll" or "score"

        Returns:
            (batch, n_tau, n_actions) quantile action values
        """
        batch_size = obs.shape[0]
        n_tau = tau.shape[1]

        # Encode state: (batch, hidden)
        state_feat = self.state_encoder(obs)

        # Encode quantiles: (batch, n_tau, hidden)
        tau_feat = self.quantile_embed(tau)

        # Combine: element-wise product
        # state_feat: (batch, 1, hidden) * tau_feat: (batch, n_tau, hidden)
        combined = state_feat.unsqueeze(1) * tau_feat

        # Reshape for action head: (batch * n_tau, hidden)
        combined_flat = combined.view(batch_size * n_tau, self.hidden_dim)

        if phase == "reroll":
            q_values = self.reroll_head(combined_flat)
        else:
            q_values = self.score_head(combined_flat)

        # Reshape: (batch, n_tau, n_actions)
        return q_values.view(batch_size, n_tau, -1)

    def get_action(
        self,
        obs: torch.Tensor,
        phase: str,
        n_tau: int = 32,
        alpha: float = 1.0,
        valid_mask: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Select action using CVaR criterion.

        Args:
            obs: (batch, obs_dim)
            phase: "reroll" or "score"
            n_tau: number of quantile samples
            alpha: CVaR level (1.0 = mean, 0.1 = bottom 10%)
            valid_mask: (batch, n_actions) boolean mask
            deterministic: if True, take argmax; else sample proportionally

        Returns:
            (batch,) selected actions
        """
        batch_size = obs.shape[0]
        tau = torch.rand(batch_size, n_tau, device=obs.device)

        if alpha < 1.0:
            # CVaR: only sample from [0, alpha] range
            tau = tau * alpha

        with torch.no_grad():
            q_values = self.forward(obs, tau, phase)  # (batch, n_tau, n_actions)
            # Mean over quantile samples: (batch, n_actions)
            mean_q = q_values.mean(dim=1)

            if valid_mask is not None:
                mean_q = mean_q.masked_fill(~valid_mask, float("-inf"))

            if deterministic:
                return mean_q.argmax(dim=-1)
            else:
                # Softmax sampling with temperature
                probs = torch.softmax(mean_q * 10.0, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)
