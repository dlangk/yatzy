"""Synthetic player simulation for profiling validation.

Generates quiz answers from players with known (θ, β, γ, d) parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SyntheticPlayer:
    """A player with known cognitive parameters."""

    theta: float
    beta: float
    gamma: float
    d: int
    name: str = ""

    def answer_scenario(
        self, scenario: dict, rng: np.random.Generator | None = None
    ) -> dict:
        """Choose an action for a scenario using softmax(β · Q)."""
        if rng is None:
            rng = np.random.default_rng()

        q_values = self._lookup_q(scenario)
        if q_values is None or len(q_values) == 0:
            # Fallback: random choice
            actions = scenario["actions"]
            idx = rng.integers(len(actions))
            return {
                "scenarioId": scenario["id"],
                "actionId": actions[idx]["id"],
                "responseTimeMs": int(rng.exponential(2000)),
            }

        # Softmax
        q = np.array(q_values, dtype=np.float64)
        q_shifted = self.beta * (q - q.max())
        probs = np.exp(q_shifted)
        probs /= probs.sum()

        idx = rng.choice(len(probs), p=probs)
        actions = scenario["actions"]
        action_id = actions[idx]["id"] if idx < len(actions) else actions[0]["id"]

        return {
            "scenarioId": scenario["id"],
            "actionId": action_id,
            "responseTimeMs": int(rng.exponential(2000)),
        }

    def _lookup_q(self, scenario: dict) -> list[float] | None:
        """Look up Q-values from the scenario's pre-computed grid."""
        grid = scenario.get("q_grid")
        if not grid:
            return None

        q_values = grid.get("q_values", {})
        if not q_values:
            return None

        # Find nearest grid point
        theta_vals = grid.get("theta_values", [])
        gamma_vals = grid.get("gamma_values", [])
        d_vals = grid.get("d_values", [])

        nearest_theta = _find_nearest(theta_vals, self.theta) if theta_vals else 0
        nearest_gamma = _find_nearest(gamma_vals, self.gamma) if gamma_vals else 1.0
        nearest_d = _find_nearest(d_vals, self.d) if d_vals else 999

        # Find key by matching nearest values (handles float formatting differences)
        best_key = None
        best_dist = float("inf")
        for key in q_values:
            parts = key.split(",")
            if len(parts) != 3:
                continue
            try:
                kt, kg, kd = float(parts[0]), float(parts[1]), int(float(parts[2]))
            except ValueError:
                continue
            dist = abs(kt - nearest_theta) + abs(kg - nearest_gamma) + (0 if kd == int(nearest_d) else 1e6)
            if dist < best_dist:
                best_dist = dist
                best_key = key

        if best_key is None or best_dist > 0.1:
            return None
        return q_values[best_key]


def _find_nearest(arr: list, target: float) -> float:
    """Find the nearest value in a sorted list."""
    if not arr:
        return target
    return min(arr, key=lambda x: abs(x - target))


# Standard archetypes for validation.
# d values must match D_GRID in estimator.py: {8, 20, 999}
# β values kept in identifiable range (0.5-5): β>5 approaches deterministic,
# making it indistinguishable from any higher value with limited observations.
ARCHETYPES: list[SyntheticPlayer] = [
    SyntheticPlayer(theta=0.0, beta=4.0, gamma=0.95, d=999, name="expert_neutral"),
    SyntheticPlayer(theta=-0.05, beta=2.5, gamma=0.9, d=20, name="cautious_strong"),
    SyntheticPlayer(theta=0.05, beta=2.0, gamma=0.8, d=20, name="risky_decent"),
    SyntheticPlayer(theta=0.0, beta=0.8, gamma=0.5, d=8, name="sloppy_myopic"),
    SyntheticPlayer(theta=-0.03, beta=5.0, gamma=0.9, d=999, name="precise_cautious"),
    SyntheticPlayer(theta=0.03, beta=1.5, gamma=0.7, d=8, name="gambler_mid"),
]


def generate_synthetic_dataset(
    scenarios: list[dict],
    player: SyntheticPlayer,
    seed: int = 42,
) -> list[dict]:
    """Generate a full set of quiz answers for a synthetic player."""
    rng = np.random.default_rng(seed)
    return [player.answer_scenario(s, rng) for s in scenarios]


def load_scenarios(path: str | Path) -> list[dict]:
    """Load scenarios from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["scenarios"]
