"""Evaluation pipeline: run N games with a trained policy and output results.

Produces analytics-compatible output (scores array, summary stats, percentiles).
Uses Rust FFI bridge for fast simulation (~50K games/s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .approach_a.train import compute_base_obs_batch
from .bridge import RustBridge
from .scoring import CATEGORY_COUNT


def evaluate_approach_a(
    policy: torch.nn.Module,
    bridge: RustBridge,
    base_sv: np.ndarray,
    n_games: int = 1_000_000,
    seed: int = 12345,
    device: str = "cpu",
    batch_size: int = 50_000,
) -> np.ndarray:
    """Evaluate Approach A policy over N games using Rust bridge.

    Processes games in batches for memory efficiency.
    """
    torch_device = torch.device(device)
    all_scores = []
    rng = np.random.default_rng(seed)
    policy.eval()

    for start in range(0, n_games, batch_size):
        n = min(batch_size, n_games - start)
        upper_scores = np.zeros(n, dtype=np.int32)
        scored_cats = np.zeros(n, dtype=np.int32)
        total_scores = np.zeros(n, dtype=np.float32)

        for turn in range(CATEGORY_COUNT):
            obs = compute_base_obs_batch(
                upper_scores, scored_cats, total_scores.astype(np.int32), base_sv
            )
            obs_t = torch.from_numpy(obs).float().to(torch_device)
            with torch.no_grad():
                actions, _ = policy.get_action(obs_t, deterministic=True)
            theta_indices = actions.cpu().numpy().astype(np.int32)

            seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
            cats, scrs, new_ups = bridge.batch_turn(
                theta_indices, upper_scores, scored_cats, seeds, turn
            )
            upper_scores = new_ups
            scored_cats = scored_cats | (1 << cats)
            total_scores += scrs.astype(np.float32)

        bonus = np.where(upper_scores >= 63, 50, 0).astype(np.float32)
        total_scores += bonus
        all_scores.append(total_scores.astype(np.int32))

    return np.concatenate(all_scores)


def evaluate_table_baseline(
    bridge: RustBridge,
    theta_index: int,
    n_games: int = 1_000_000,
    seed: int = 12345,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Evaluate a fixed-theta table baseline using Rust bridge."""
    all_scores = []
    for start in range(0, n_games, batch_size):
        n = min(batch_size, n_games - start)
        seeds = np.arange(start + seed, start + seed + n, dtype=np.uint64)
        scores = bridge.batch_game(theta_index, seeds)
        all_scores.append(scores)
    return np.concatenate(all_scores)


def compute_stats(scores: np.ndarray) -> dict:
    """Compute summary statistics from score array."""
    return {
        "n_games": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": int(scores.min()),
        "max": int(scores.max()),
        "median": float(np.median(scores)),
        "p5": float(np.percentile(scores, 5)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def save_results(
    scores: np.ndarray,
    output_dir: str | Path,
    label: str,
) -> None:
    """Save scores and stats to output directory.

    Creates:
        {output_dir}/{label}/scores.bin  — raw int32 scores
        {output_dir}/{label}/stats.json  — summary statistics
    """
    out = Path(output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    # Save raw scores
    scores.astype(np.int32).tofile(out / "scores.bin")

    # Save stats
    stats = compute_stats(scores)
    stats["label"] = label
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved {label}: mean={stats['mean']:.1f} p95={stats['p95']:.0f} ({len(scores)} games)")
