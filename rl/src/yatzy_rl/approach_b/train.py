"""Approach B: IQN training with behavioral cloning warm-start.

Two-phase training using the Rust bridge for fast batch simulation:
1. Behavioral cloning: imitate theta=0 table decisions (reroll masks + categories)
2. Distributional RL fine-tuning with IQN loss and CVaR action selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..bridge import RustBridge
from ..scoring import CATEGORY_COUNT, is_category_scored
from ..approach_a.train import compute_base_obs_batch
from .policy import IQN


@dataclass
class TrainConfig:
    """Training hyperparameters for Approach B."""
    # Behavioral cloning
    bc_episodes: int = 100_000
    bc_batch_size: int = 512
    bc_lr: float = 1e-3
    bc_epochs: int = 10
    # RL fine-tuning
    rl_episodes: int = 200_000
    rl_batch_size: int = 2048
    rl_lr: float = 3e-4
    n_tau: int = 32
    n_tau_prime: int = 32
    kappa: float = 1.0  # Huber threshold
    alpha: float = 0.75  # CVaR level for action selection
    gamma: float = 1.0
    target_update_freq: int = 200
    buffer_size: int = 200_000
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    seed: int = 42
    log_interval: int = 1000
    device: str = "cpu"


def compute_full_obs_batch(
    upper_scores: np.ndarray,
    scored_cats: np.ndarray,
    total_scores: np.ndarray,
    base_sv: np.ndarray,
    dice: np.ndarray,
    rerolls_remaining: int,
) -> np.ndarray:
    """Compute full 18-dim observation for N games.

    Combines base_obs (10) with dice features (8).
    """
    n = len(upper_scores)
    base = compute_base_obs_batch(upper_scores, scored_cats, total_scores, base_sv)

    # Dice features: face counts / 5, rerolls_remaining / 2, dice_set_index / 252
    dice_obs = np.zeros((n, 8), dtype=np.float32)
    for face in range(1, 7):
        dice_obs[:, face - 1] = np.sum(dice == face, axis=1) / 5.0
    dice_obs[:, 6] = rerolls_remaining / 2.0
    # Dice set index: approximate with a hash (exact index not needed for features)
    # Use sum of sorted dice as simple feature instead
    dice_obs[:, 7] = dice.sum(axis=1) / 30.0  # max sum = 30

    return np.concatenate([base, dice_obs], axis=1)


def collect_bc_data(
    bridge: RustBridge,
    base_sv: np.ndarray,
    n_episodes: int,
    batch_size: int = 10_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Collect behavioral cloning data by running expert games through the bridge.

    Returns dict with:
        reroll_obs: (M, 18) observations at reroll decisions
        reroll_actions: (M,) expert reroll masks
        score_obs: (K, 18) observations at score decisions
        score_actions: (K,) expert category choices
    """
    rng = np.random.default_rng(seed)
    reroll_obs_list: list[np.ndarray] = []
    reroll_act_list: list[np.ndarray] = []
    score_obs_list: list[np.ndarray] = []
    score_act_list: list[np.ndarray] = []

    remaining = n_episodes
    collected = 0
    while remaining > 0:
        n = min(remaining, batch_size)
        remaining -= n
        collected += n
        if collected % 50_000 == 0 or remaining == 0:
            print(f"    Collected {collected:,}/{n_episodes:,} episodes...", flush=True)

        ups = np.zeros(n, dtype=np.int32)
        scs = np.zeros(n, dtype=np.int32)
        totals = np.zeros(n, dtype=np.int32)

        for turn in range(CATEGORY_COUNT):
            # Roll initial dice
            seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
            dice = bridge.batch_roll(seeds)

            # Decision 1: first reroll
            obs1 = compute_full_obs_batch(ups, scs, totals, base_sv, dice, 2)
            masks1 = bridge.batch_expert_reroll(dice, ups, scs, 2)
            reroll_obs_list.append(obs1)
            reroll_act_list.append(masks1)

            # Apply first reroll
            seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
            dice = bridge.batch_apply_reroll(dice, masks1, seeds)

            # Decision 2: second reroll
            obs2 = compute_full_obs_batch(ups, scs, totals, base_sv, dice, 1)
            masks2 = bridge.batch_expert_reroll(dice, ups, scs, 1)
            reroll_obs_list.append(obs2)
            reroll_act_list.append(masks2)

            # Apply second reroll
            seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
            dice = bridge.batch_apply_reroll(dice, masks2, seeds)

            # Decision 3: category
            obs3 = compute_full_obs_batch(ups, scs, totals, base_sv, dice, 0)
            cats = bridge.batch_expert_category(dice, ups, scs, turn)
            score_obs_list.append(obs3)
            score_act_list.append(cats)

            # Update state
            scores, new_ups = bridge.batch_score_category(dice, cats, ups, scs)
            ups = new_ups
            scs = scs | (1 << cats)
            totals += scores

    return {
        "reroll_obs": np.concatenate(reroll_obs_list),
        "reroll_actions": np.concatenate(reroll_act_list),
        "score_obs": np.concatenate(score_obs_list),
        "score_actions": np.concatenate(score_act_list),
    }


def behavioral_cloning(
    model: IQN,
    bc_data: dict[str, np.ndarray],
    config: TrainConfig,
    device: torch.device,
) -> None:
    """Phase 1: Train IQN to imitate expert decisions."""
    print("  Phase 1: Behavioral cloning warm-start...")
    optimizer = optim.Adam(model.parameters(), lr=config.bc_lr)

    r_obs = bc_data["reroll_obs"]
    r_act = bc_data["reroll_actions"]
    s_obs = bc_data["score_obs"]
    s_act = bc_data["score_actions"]
    print(f"    Reroll steps: {len(r_obs):,}, Score steps: {len(s_obs):,}")

    for epoch in range(config.bc_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Train reroll head
        perm = np.random.permutation(len(r_obs))
        for start in range(0, len(r_obs), config.bc_batch_size):
            idx = perm[start : start + config.bc_batch_size]
            obs_t = torch.from_numpy(r_obs[idx]).float().to(device)
            act_t = torch.from_numpy(r_act[idx]).long().to(device)

            tau = torch.rand(len(idx), config.n_tau, device=device)
            q_values = model(obs_t, tau, "reroll").mean(dim=1)
            loss = F.cross_entropy(q_values, act_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Train score head
        perm = np.random.permutation(len(s_obs))
        for start in range(0, len(s_obs), config.bc_batch_size):
            idx = perm[start : start + config.bc_batch_size]
            obs_t = torch.from_numpy(s_obs[idx]).float().to(device)
            act_t = torch.from_numpy(s_act[idx]).long().to(device)

            tau = torch.rand(len(idx), config.n_tau, device=device)
            q_values = model(obs_t, tau, "score").mean(dim=1)
            loss = F.cross_entropy(q_values, act_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Measure accuracy
        model.eval()
        with torch.no_grad():
            sample_r = min(10000, len(r_obs))
            idx_r = np.random.choice(len(r_obs), sample_r, replace=False)
            obs_r = torch.from_numpy(r_obs[idx_r]).float().to(device)
            tau_r = torch.rand(sample_r, config.n_tau, device=device)
            pred_r = model(obs_r, tau_r, "reroll").mean(dim=1).argmax(dim=1).cpu().numpy()
            acc_r = (pred_r == r_act[idx_r]).mean()

            sample_s = min(10000, len(s_obs))
            idx_s = np.random.choice(len(s_obs), sample_s, replace=False)
            obs_s = torch.from_numpy(s_obs[idx_s]).float().to(device)
            tau_s = torch.rand(sample_s, config.n_tau, device=device)
            pred_s = model(obs_s, tau_s, "score").mean(dim=1).argmax(dim=1).cpu().numpy()
            acc_s = (pred_s == s_act[idx_s]).mean()

        print(f"    Epoch {epoch + 1}/{config.bc_epochs}: loss={avg_loss:.4f} "
              f"reroll_acc={acc_r:.3f} score_acc={acc_s:.3f}")


class ReplayBuffer:
    """Ring buffer for transition storage."""

    def __init__(self, capacity: int, obs_dim: int = 18):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.phases = np.zeros(capacity, dtype=np.int32)  # current: 0=reroll, 1=score
        self.next_phases = np.zeros(capacity, dtype=np.int32)  # next: 0=reroll, 1=score
        self.size = 0
        self.pos = 0

    def push_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        phases: np.ndarray,
        next_phases: np.ndarray,
    ) -> None:
        n = len(obs)
        for i in range(n):
            self.obs[self.pos] = obs[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.next_obs[self.pos] = next_obs[i]
            self.dones[self.pos] = dones[i]
            self.phases[self.pos] = phases[i]
            self.next_phases[self.pos] = next_phases[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "phases": self.phases[idx],
            "next_phases": self.next_phases[idx],
        }


def get_valid_mask_batch(scored_cats: np.ndarray) -> np.ndarray:
    """Compute valid category mask for batch. Returns (N, 15) bool."""
    n = len(scored_cats)
    valid = np.zeros((n, CATEGORY_COUNT), dtype=bool)
    for c in range(CATEGORY_COUNT):
        valid[:, c] = (scored_cats & (1 << c)) == 0
    return valid


def rl_finetune(
    bridge: RustBridge,
    base_sv: np.ndarray,
    model: IQN,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Phase 2: Distributional RL fine-tuning with IQN loss.

    Runs N parallel games, collecting transitions and training online.
    """
    print("  Phase 2: Distributional RL fine-tuning...")
    optimizer = optim.Adam(model.parameters(), lr=config.rl_lr)

    target = IQN().to(device)
    target.load_state_dict(model.state_dict())

    buffer = ReplayBuffer(config.buffer_size)
    rng = np.random.default_rng(config.seed + 1000)

    # Track per-game state for N parallel games
    par_n = 4096  # games running in parallel
    ups = np.zeros(par_n, dtype=np.int32)
    scs = np.zeros(par_n, dtype=np.int32)
    totals = np.zeros(par_n, dtype=np.int32)
    turns = np.zeros(par_n, dtype=np.int32)
    # phase: 0=roll, 1=reroll1_done, 2=reroll2_done
    phases = np.zeros(par_n, dtype=np.int32)
    dice = np.zeros((par_n, 5), dtype=np.int32)
    # For tracking episode stats
    completed_scores: list[int] = []
    total_steps = 0
    best_p95 = 0.0

    # Initial roll for all games
    seeds = rng.integers(0, 2**63, size=par_n, dtype=np.uint64)
    dice = bridge.batch_roll(seeds)
    phases[:] = 0  # at first reroll decision

    def get_epsilon() -> float:
        frac = min(1.0, total_steps / config.epsilon_decay_steps)
        return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

    while len(completed_scores) < config.rl_episodes:
        epsilon = get_epsilon()
        model.eval()

        # All games are at a reroll or score decision point
        # Phase 0: need first reroll decision
        # Phase 1: need second reroll decision
        # Phase 2: need score decision

        # Games at reroll phase (0 or 1)
        reroll_mask_0 = phases < 2
        score_mask = phases == 2
        n_reroll = int(reroll_mask_0.sum())
        n_score = int(score_mask.sum())

        # Process reroll decisions
        if n_reroll > 0:
            idx_r = np.where(reroll_mask_0)[0]
            rerolls_rem = np.where(phases[idx_r] == 0, 2, 1)
            obs_r = compute_full_obs_batch(
                ups[idx_r], scs[idx_r], totals[idx_r], base_sv,
                dice[idx_r], int(rerolls_rem[0])  # all same rerolls_rem in this batch
            )
            # Actually rerolls_rem may differ per game... handle separately
            # Split by rerolls remaining
            for rr in [2, 1]:
                sub_mask = rerolls_rem == rr
                if not sub_mask.any():
                    continue
                sub_idx = idx_r[sub_mask]
                n_sub = len(sub_idx)

                obs_sub = compute_full_obs_batch(
                    ups[sub_idx], scs[sub_idx], totals[sub_idx], base_sv,
                    dice[sub_idx], rr,
                )
                obs_t = torch.from_numpy(obs_sub).float().to(device)

                # Epsilon-greedy
                with torch.no_grad():
                    tau = torch.rand(n_sub, config.n_tau, device=device)
                    if config.alpha < 1.0:
                        tau = tau * config.alpha
                    q_vals = model(obs_t, tau, "reroll").mean(dim=1)  # (n_sub, 32)
                    greedy_actions = q_vals.argmax(dim=1).cpu().numpy()

                random_mask = rng.random(n_sub) < epsilon
                random_actions = rng.integers(0, 32, size=n_sub)
                actions = np.where(random_mask, random_actions, greedy_actions)

                # Apply reroll
                seeds_r = rng.integers(0, 2**63, size=n_sub, dtype=np.uint64)
                new_dice = bridge.batch_apply_reroll(dice[sub_idx], actions, seeds_r)

                # Compute next obs
                next_rr = rr - 1
                if next_rr > 0:
                    next_obs = compute_full_obs_batch(
                        ups[sub_idx], scs[sub_idx], totals[sub_idx], base_sv,
                        new_dice, next_rr,
                    )
                    next_phase = np.zeros(n_sub, dtype=np.int32)  # reroll
                else:
                    next_obs = compute_full_obs_batch(
                        ups[sub_idx], scs[sub_idx], totals[sub_idx], base_sv,
                        new_dice, 0,
                    )
                    next_phase = np.ones(n_sub, dtype=np.int32)  # score

                # Store transitions (reward=0, not done for reroll)
                next_rr_val = rr - 1
                np_phase = np.zeros(n_sub, dtype=np.int32) if next_rr_val > 0 else np.ones(n_sub, dtype=np.int32)
                buffer.push_batch(
                    obs_sub, actions,
                    np.zeros(n_sub, dtype=np.float32),
                    next_obs,
                    np.zeros(n_sub, dtype=np.float32),
                    np.zeros(n_sub, dtype=np.int32),  # current = reroll
                    np_phase,  # next = reroll or score
                )

                dice[sub_idx] = new_dice
                phases[sub_idx] += 1
                total_steps += n_sub

        # Process score decisions
        if n_score > 0:
            idx_s = np.where(score_mask)[0]
            n_s = len(idx_s)

            obs_s = compute_full_obs_batch(
                ups[idx_s], scs[idx_s], totals[idx_s], base_sv,
                dice[idx_s], 0,
            )
            obs_t = torch.from_numpy(obs_s).float().to(device)

            # Valid categories mask
            valid = get_valid_mask_batch(scs[idx_s])
            valid_t = torch.from_numpy(valid).bool().to(device)

            with torch.no_grad():
                tau = torch.rand(n_s, config.n_tau, device=device)
                if config.alpha < 1.0:
                    tau = tau * config.alpha
                q_vals = model(obs_t, tau, "score").mean(dim=1)
                q_vals.masked_fill_(~valid_t, float("-inf"))
                greedy_actions = q_vals.argmax(dim=1).cpu().numpy()

            # Epsilon-greedy with valid mask
            random_mask = rng.random(n_s) < epsilon
            random_actions = np.zeros(n_s, dtype=np.int32)
            for j in range(n_s):
                valid_cats = np.where(valid[j])[0]
                random_actions[j] = rng.choice(valid_cats)
            actions = np.where(random_mask, random_actions, greedy_actions)

            # Score
            scores, new_ups = bridge.batch_score_category(
                dice[idx_s], actions, ups[idx_s], scs[idx_s],
            )

            new_scs = scs[idx_s] | (1 << actions)
            new_totals = totals[idx_s] + scores
            new_turns = turns[idx_s] + 1

            # Check for done
            done_mask = new_turns >= CATEGORY_COUNT
            bonus = np.where(done_mask & (new_ups >= 63), 50, 0).astype(np.int32)
            final_scores = new_totals + bonus

            # Compute rewards: only at terminal
            rewards = np.zeros(n_s, dtype=np.float32)
            rewards[done_mask] = final_scores[done_mask].astype(np.float32)

            # Next obs for non-done games
            # For done games, next_obs doesn't matter (won't be used)
            next_dice = np.zeros_like(dice[idx_s])
            not_done = ~done_mask
            if not_done.any():
                seeds_n = rng.integers(0, 2**63, size=int(not_done.sum()), dtype=np.uint64)
                next_dice[not_done] = bridge.batch_roll(seeds_n)

            next_obs = compute_full_obs_batch(
                new_ups, new_scs, new_totals, base_sv,
                next_dice, 2,
            )

            buffer.push_batch(
                obs_s, actions, rewards, next_obs,
                done_mask.astype(np.float32),
                np.ones(n_s, dtype=np.int32),  # current = score
                np.zeros(n_s, dtype=np.int32),  # next = reroll (new turn)
            )

            # Update state
            ups[idx_s] = new_ups
            scs[idx_s] = new_scs
            totals[idx_s] = new_totals
            turns[idx_s] = new_turns

            # Reset done games
            done_idx = idx_s[done_mask]
            if len(done_idx) > 0:
                done_final = final_scores[done_mask]
                for s in done_final:
                    completed_scores.append(int(s))

                ups[done_idx] = 0
                scs[done_idx] = 0
                totals[done_idx] = 0
                turns[done_idx] = 0
                seeds_new = rng.integers(0, 2**63, size=len(done_idx), dtype=np.uint64)
                dice[done_idx] = bridge.batch_roll(seeds_new)
                phases[done_idx] = 0
            # Non-done: advance to reroll phase with new dice
            not_done_idx = idx_s[not_done]
            if len(not_done_idx) > 0:
                dice[not_done_idx] = next_dice[not_done]
                phases[not_done_idx] = 0

            total_steps += n_s

        # Train step
        if buffer.size >= config.rl_batch_size and total_steps % 8 == 0:
            model.train()
            batch = buffer.sample(min(config.rl_batch_size, buffer.size))

            obs_t = torch.from_numpy(batch["obs"]).float().to(device)
            act_t = torch.from_numpy(batch["actions"]).long().to(device)
            rew_t = torch.from_numpy(batch["rewards"]).float().to(device)
            next_t = torch.from_numpy(batch["next_obs"]).float().to(device)
            done_t = torch.from_numpy(batch["dones"]).float().to(device)
            phase_t = batch["phases"]
            next_phase_t = batch["next_phases"]

            for phase_val, phase_name in [(0, "reroll"), (1, "score")]:
                pm = phase_t == phase_val
                if pm.sum() < 4:
                    continue

                p_obs = obs_t[pm]
                p_act = act_t[pm]
                p_rew = rew_t[pm]
                p_next = next_t[pm]
                p_done = done_t[pm]
                p_next_phase = next_phase_t[pm]
                n_p = p_obs.shape[0]

                # Current quantile values
                tau = torch.rand(n_p, config.n_tau, device=device)
                q_vals = model(p_obs, tau, phase_name)
                q_a = q_vals.gather(
                    2, p_act.unsqueeze(1).unsqueeze(2).expand(-1, config.n_tau, -1)
                ).squeeze(2)

                # Target: handle mixed next phases
                with torch.no_grad():
                    tau_prime = torch.rand(n_p, config.n_tau_prime, device=device)

                    # Split by next phase to use correct head
                    target_vals = torch.zeros(n_p, config.n_tau_prime, device=device)
                    for np_val, np_name in [(0, "reroll"), (1, "score")]:
                        np_mask = p_next_phase == np_val
                        if np_mask.sum() == 0:
                            continue
                        np_idx = np.where(np_mask)[0]
                        np_idx_t = torch.from_numpy(np_idx).long().to(device)
                        t_q = target(p_next[np_idx_t], tau_prime[np_idx_t], np_name)
                        t_mean = t_q.mean(dim=1)
                        t_best = t_mean.max(dim=-1)[0].unsqueeze(1).expand(-1, config.n_tau_prime)
                        target_vals[np_idx_t] = (
                            p_rew[np_idx_t].unsqueeze(1)
                            + (1 - p_done[np_idx_t].unsqueeze(1)) * t_best
                        )
                    # For done transitions, target = reward only
                    done_mask_b = p_done > 0.5
                    if done_mask_b.any():
                        target_vals[done_mask_b] = p_rew[done_mask_b].unsqueeze(1)

                # Quantile Huber loss
                delta = target_vals.unsqueeze(1) - q_a.unsqueeze(2)
                huber = torch.where(
                    delta.abs() <= config.kappa,
                    0.5 * delta.pow(2),
                    config.kappa * (delta.abs() - 0.5 * config.kappa),
                )
                tau_expanded = tau.unsqueeze(2)
                weight = torch.abs(tau_expanded - (delta < 0).float())
                loss = (weight * huber).sum(dim=2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Update target network
            if total_steps % config.target_update_freq == 0:
                target.load_state_dict(model.state_dict())

        # Logging
        if len(completed_scores) > 0 and len(completed_scores) % config.log_interval < par_n:
            recent = np.array(completed_scores[-5000:])
            mean = float(recent.mean())
            p5 = float(np.percentile(recent, 5))
            p50 = float(np.percentile(recent, 50))
            p95 = float(np.percentile(recent, 95))
            eps = get_epsilon()
            print(
                f"    ep {len(completed_scores):7d} | mean={mean:.1f} p5={p5:.0f} "
                f"p50={p50:.0f} p95={p95:.0f} | eps={eps:.3f} buf={buffer.size}"
            )

            if p95 > best_p95 and len(completed_scores) > 5000:
                best_p95 = p95
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stats": {"mean": mean, "p95": p95, "p5": p5},
                        "episodes": len(completed_scores),
                    },
                    output_dir / "best_approach_b.pt",
                )


def train(
    base_path: str | Path,
    config: TrainConfig,
    output_dir: str | Path,
) -> IQN:
    """Train Approach B: behavioral cloning + distributional RL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    bridge = RustBridge(str(base_path), [0.0])

    from ..tables import load_state_values, state_file_path
    base_sv = load_state_values(state_file_path(base_path, 0.0))

    model = IQN().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training Approach B: IQN with behavioral cloning warm-start")
    print(f"  Model params: {n_params:,}")
    print(f"  CVaR alpha={config.alpha}, device={config.device}")

    # Phase 1: Collect BC data and train
    print(f"  Collecting BC data from {config.bc_episodes:,} expert games...")
    bc_data = collect_bc_data(bridge, base_sv, config.bc_episodes, seed=config.seed)
    behavioral_cloning(model, bc_data, config, device)

    # Evaluate after BC
    _evaluate_quick(bridge, base_sv, model, config, device, "post-BC")

    # Phase 2: RL fine-tuning
    rl_finetune(bridge, base_sv, model, config, device, output_dir)

    # Final save
    torch.save(
        {"model_state_dict": model.state_dict()},
        output_dir / "final_approach_b.pt",
    )

    _evaluate_quick(bridge, base_sv, model, config, device, "final")
    print("  Training complete.")
    return model


def _evaluate_quick(
    bridge: RustBridge,
    base_sv: np.ndarray,
    model: IQN,
    config: TrainConfig,
    device: torch.device,
    label: str,
    n_games: int = 50_000,
) -> None:
    """Quick evaluation: run n_games with model policy."""
    model.eval()
    rng = np.random.default_rng(12345)
    n = n_games

    ups = np.zeros(n, dtype=np.int32)
    scs = np.zeros(n, dtype=np.int32)
    totals = np.zeros(n, dtype=np.int32)

    for turn in range(CATEGORY_COUNT):
        seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
        dice = bridge.batch_roll(seeds)

        for rerolls_rem in [2, 1]:
            obs = compute_full_obs_batch(ups, scs, totals, base_sv, dice, rerolls_rem)
            obs_t = torch.from_numpy(obs).float().to(device)

            with torch.no_grad():
                actions = model.get_action(
                    obs_t, "reroll", config.n_tau, config.alpha,
                    deterministic=True,
                ).cpu().numpy()

            seeds = rng.integers(0, 2**63, size=n, dtype=np.uint64)
            dice = bridge.batch_apply_reroll(dice, actions, seeds)

        # Category
        obs = compute_full_obs_batch(ups, scs, totals, base_sv, dice, 0)
        obs_t = torch.from_numpy(obs).float().to(device)
        valid = get_valid_mask_batch(scs)
        valid_t = torch.from_numpy(valid).bool().to(device)

        with torch.no_grad():
            cats = model.get_action(
                obs_t, "score", config.n_tau, config.alpha,
                valid_mask=valid_t, deterministic=True,
            ).cpu().numpy().astype(np.int32)

        scores, new_ups = bridge.batch_score_category(dice, cats, ups, scs)
        ups = new_ups
        scs = scs | (1 << cats)
        totals += scores

    bonus = np.where(ups >= 63, 50, 0)
    totals += bonus

    print(f"  [{label}] {n_games:,} games: mean={totals.mean():.1f} "
          f"p5={np.percentile(totals, 5):.0f} p50={np.percentile(totals, 50):.0f} "
          f"p95={np.percentile(totals, 95):.0f} p99={np.percentile(totals, 99):.0f}")
