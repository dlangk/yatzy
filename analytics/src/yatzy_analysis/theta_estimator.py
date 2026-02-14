"""Bayesian θ estimator from human category decisions.

Uses a softmax choice model on a grid of (θ, β) to infer a human's risk
preference from their responses to curated pivotal scenarios.

Mathematical framework:
    P(action a | state s, θ, β) = exp(β · V_θ(a, s)) / Σ_{a'} exp(β · V_θ(a', s))

where V_θ(c, s) = score(c) + sv_θ[successor_state] is read from the scenario JSON.

The posterior P(θ, β | D) is maintained on a discrete grid (no MCMC needed).

All hot-path computations are fully vectorized with numpy broadcasting —
no Python loops over (θ, β, action) in the critical path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .scorecard import scorecard_from_scenario


# ── Theta/Beta grids ──────────────────────────────────────────────────────────

THETA_GRID = np.array([
    -3.00, -2.00, -1.50, -1.00, -0.75, -0.50, -0.30,
    -0.200, -0.150, -0.100, -0.070, -0.050, -0.040, -0.030, -0.020, -0.015, -0.010, -0.005,
    0.000,
    0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.070, 0.100, 0.150, 0.200,
    0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
])  # 37 values: progressive spacing, dense near 0, sparse at tails
BETA_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0])  # 6 rationality levels

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]


# ── Scenario loading ─────────────────────────────────────────────────────────

def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load pivotal scenarios from JSON file."""
    with open(path) as f:
        return json.load(f)


def _scenario_values(scenario: dict, theta_str: str) -> dict[int, float]:
    """Extract {category_id: V_θ} for a given theta string from a scenario.

    Used by simulate_synthetic_human (not in the hot path).
    """
    result = {}
    for cat in scenario["available"]:
        cat_id = cat["id"]
        if theta_str in cat["values"]:
            result[cat_id] = cat["values"][theta_str]
    return result


# ── Core estimator ────────────────────────────────────────────────────────────

class ThetaEstimator:
    """Bayesian grid estimator for (θ, β) from category decisions.

    All value lookups are precomputed at init into numpy arrays.
    Likelihood/selection use only numpy broadcasting — no Python loops
    over (θ, β, actions).
    """

    def __init__(self, scenarios: list[dict[str, Any]]) -> None:
        self.theta_grid = THETA_GRID
        self.beta_grid = BETA_GRID
        self.scenarios = scenarios
        self.history: list[tuple[int, int]] = []  # (scenario_id, action_category_id)

        n_theta = len(self.theta_grid)
        n_beta = len(self.beta_grid)
        self.log_posterior = np.zeros((n_theta, n_beta))  # log-uniform = 0

        # ── Precompute scenario value matrices ────────────────────────────
        self._sid_to_idx: dict[int, int] = {}
        self._action_ids: list[np.ndarray] = []       # per-scenario: (n_actions,)
        self._values: list[np.ndarray] = []            # per-scenario: (n_theta, n_actions)
        self._action_id_to_col: list[dict[int, int]] = []
        # Per-scenario, per-theta: which column is the argmax action
        self._optimal_col: list[np.ndarray] = []       # per-scenario: (n_theta,)

        for i, scenario in enumerate(scenarios):
            self._sid_to_idx[scenario["id"]] = i

            cats = scenario["available"]
            action_ids = np.array([c["id"] for c in cats], dtype=np.int32)
            n_actions = len(action_ids)

            values = np.zeros((n_theta, n_actions), dtype=np.float64)
            for ti, theta in enumerate(self.theta_grid):
                theta_str = f"{theta:.3f}"
                for ai, cat in enumerate(cats):
                    if theta_str in cat["values"]:
                        values[ti, ai] = cat["values"][theta_str]

            id_to_col = {int(aid): ai for ai, aid in enumerate(action_ids)}

            # For θ<0, decision nodes minimize; for θ≥0, maximize
            opt_col = np.where(
                self.theta_grid < 0,
                np.argmin(values, axis=1),
                np.argmax(values, axis=1),
            )

            self._action_ids.append(action_ids)
            self._values.append(values)
            self._action_id_to_col.append(id_to_col)
            self._optimal_col.append(opt_col)  # (n_theta,)

    # ── Posterior access ──────────────────────────────────────────────────

    @property
    def posterior(self) -> np.ndarray:
        """Normalized posterior P(θ, β | D), shape (n_theta, n_beta)."""
        lp = self.log_posterior - self.log_posterior.max()
        p = np.exp(lp)
        return p / p.sum()

    @property
    def theta_marginal(self) -> np.ndarray:
        """Marginal P(θ | D), shape (n_theta,)."""
        return self.posterior.sum(axis=1)

    @property
    def beta_marginal(self) -> np.ndarray:
        """Marginal P(β | D), shape (n_beta,)."""
        return self.posterior.sum(axis=0)

    # ── Vectorized likelihood ─────────────────────────────────────────────

    def _all_action_probs(self, scenario_idx: int) -> np.ndarray:
        """P(a | θ, β) for all actions, shape (n_theta, n_beta, n_actions).

        Single numpy broadcast — no Python loops.
        For θ<0, values are negated so that the softmax still assigns highest
        probability to the optimal (minimum-V) action.
        """
        values = self._values[scenario_idx]  # (n_theta, n_actions)
        # For θ<0 the optimal action minimizes V, so negate to make softmax work
        sign = np.where(self.theta_grid < 0, -1.0, 1.0)[:, np.newaxis]  # (n_theta, 1)
        oriented = values * sign  # (n_theta, n_actions)
        # logits[ti, bi, ai] = beta[bi] * oriented[ti, ai]
        logits = self.beta_grid[np.newaxis, :, np.newaxis] * oriented[:, np.newaxis, :]
        logits -= logits.max(axis=2, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=2, keepdims=True)

    def likelihood_by_idx(self, scenario_idx: int, action_id: int) -> np.ndarray:
        """P(action_id | scenario, θ, β), shape (n_theta, n_beta). Vectorized."""
        id_to_col = self._action_id_to_col[scenario_idx]
        if action_id not in id_to_col:
            n_actions = len(self._action_ids[scenario_idx])
            return np.full(
                (len(self.theta_grid), len(self.beta_grid)),
                1.0 / max(n_actions, 1),
            )
        col = id_to_col[action_id]
        return self._all_action_probs(scenario_idx)[:, :, col]

    def likelihood(self, scenario: dict[str, Any], action_id: int) -> np.ndarray:
        """P(action_id | scenario, θ, β), shape (n_theta, n_beta)."""
        idx = self._sid_to_idx[scenario["id"]]
        return self.likelihood_by_idx(idx, action_id)

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, scenario: dict[str, Any], action_id: int) -> None:
        """Update posterior with a new observation."""
        lik = self.likelihood(scenario, action_id)
        self.log_posterior += np.log(np.clip(lik, 1e-30, None))
        self.history.append((scenario["id"], action_id))

    def update_by_idx(self, scenario_idx: int, action_id: int, scenario_id: int) -> None:
        """Update posterior by scenario index (avoids dict lookup)."""
        lik = self.likelihood_by_idx(scenario_idx, action_id)
        self.log_posterior += np.log(np.clip(lik, 1e-30, None))
        self.history.append((scenario_id, action_id))

    # ── Estimates ─────────────────────────────────────────────────────────

    def theta_estimate(self) -> tuple[float, float, float]:
        """Return (mean θ, 95% CI low, 95% CI high) from marginal posterior."""
        marginal = self.theta_marginal
        mean = float(np.average(self.theta_grid, weights=marginal))

        cumsum = np.cumsum(marginal)
        low_idx = np.searchsorted(cumsum, 0.025)
        high_idx = np.searchsorted(cumsum, 0.975)
        ci_low = float(self.theta_grid[max(0, low_idx)])
        ci_high = float(self.theta_grid[min(len(self.theta_grid) - 1, high_idx)])

        return mean, ci_low, ci_high

    def beta_estimate(self) -> float:
        """Return posterior mean of β."""
        return float(np.average(self.beta_grid, weights=self.beta_marginal))

    def posterior_entropy(self) -> float:
        """Shannon entropy of the joint posterior (in nats)."""
        p = self.posterior.ravel()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    # ── Vectorized scenario selection ─────────────────────────────────────

    def select_next(self, pool: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the next scenario maximizing expected information gain.

        Fully vectorized: computes all action likelihoods via broadcasting,
        then batch-computes hypothetical posterior entropies.
        """
        if not pool:
            raise ValueError("Empty scenario pool")

        h_before = self.posterior_entropy()
        posterior = self.posterior  # (n_theta, n_beta)
        best_scenario = pool[0]
        best_eig = -float("inf")

        for scenario in pool:
            si = self._sid_to_idx[scenario["id"]]
            n_actions = len(self._action_ids[si])
            if n_actions <= 1:
                continue

            # All action probs: (n_theta, n_beta, n_actions)
            all_probs = self._all_action_probs(si)

            # P(action_a) = Σ_{θ,β} P(a|θ,β) · P(θ,β) → (n_actions,)
            p_actions = np.einsum("ijk,ij->k", all_probs, posterior)

            # Log-likelihood for all actions: (n_theta, n_beta, n_actions)
            log_lik = np.log(np.clip(all_probs, 1e-30, None))

            # Hypothetical log-posteriors: (n_actions, n_theta, n_beta)
            # log_post_new[a] = log_posterior + log_lik[:, :, a]
            log_post_all = (
                self.log_posterior[np.newaxis, :, :]
                + log_lik.transpose(2, 0, 1)
            )  # (n_actions, n_theta, n_beta)

            # Normalize each → (n_actions, n_theta * n_beta)
            lp_flat = log_post_all.reshape(n_actions, -1)
            lp_flat = lp_flat - lp_flat.max(axis=1, keepdims=True)
            p_flat = np.exp(lp_flat)
            p_flat /= p_flat.sum(axis=1, keepdims=True)

            # Entropy of each hypothetical posterior: (n_actions,)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_p = np.where(p_flat > 0, np.log(p_flat), 0.0)
            entropies = -np.sum(p_flat * log_p, axis=1)

            # Expected entropy after observing the response
            # Weight by P(action) but skip negligible actions
            mask = p_actions > 1e-15
            expected_entropy = float(np.dot(p_actions[mask], entropies[mask]))

            eig = h_before - expected_entropy
            if eig > best_eig:
                best_eig = eig
                best_scenario = scenario

        return best_scenario

    def select_next_idx(self, pool_indices: list[int]) -> int:
        """select_next by scenario indices (faster for validation loops)."""
        h_before = self.posterior_entropy()
        posterior = self.posterior

        best_idx = pool_indices[0]
        best_eig = -float("inf")

        for si in pool_indices:
            n_actions = len(self._action_ids[si])
            if n_actions <= 1:
                continue

            all_probs = self._all_action_probs(si)
            p_actions = np.einsum("ijk,ij->k", all_probs, posterior)
            log_lik = np.log(np.clip(all_probs, 1e-30, None))

            log_post_all = (
                self.log_posterior[np.newaxis, :, :]
                + log_lik.transpose(2, 0, 1)
            )
            lp_flat = log_post_all.reshape(n_actions, -1)
            lp_flat = lp_flat - lp_flat.max(axis=1, keepdims=True)
            p_flat = np.exp(lp_flat)
            p_flat /= p_flat.sum(axis=1, keepdims=True)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_p = np.where(p_flat > 0, np.log(p_flat), 0.0)
            entropies = -np.sum(p_flat * log_p, axis=1)

            mask = p_actions > 1e-15
            expected_entropy = float(np.dot(p_actions[mask], entropies[mask]))

            eig = h_before - expected_entropy
            if eig > best_eig:
                best_eig = eig
                best_idx = si

        return best_idx

    # ── Consistency checks ────────────────────────────────────────────────

    def consistency(self) -> tuple[str, float]:
        """Check how well the best-fit θ explains the observed data.

        Returns (rating, match_fraction):
            - "HIGH": >80% of answers match argmax under best θ
            - "MODERATE": 60-80%
            - "LOW": <60%
        """
        if not self.history:
            return "HIGH", 1.0

        best_ti = int(np.argmax(self.theta_marginal))

        matches = 0
        for scenario_id, action_id in self.history:
            si = self._sid_to_idx.get(scenario_id)
            if si is None:
                continue
            optimal_col = self._optimal_col[si][best_ti]
            optimal_cat = int(self._action_ids[si][optimal_col])
            if optimal_cat == action_id:
                matches += 1

        fraction = matches / len(self.history) if self.history else 1.0
        if fraction >= 0.8:
            rating = "HIGH"
        elif fraction >= 0.6:
            rating = "MODERATE"
        else:
            rating = "LOW"

        return rating, fraction

    def agreement_rates(self) -> dict[str, int]:
        """For each θ in the grid, count how many answers agree with that θ's optimal."""
        rates = {}
        for ti, theta in enumerate(self.theta_grid):
            matches = 0
            for scenario_id, action_id in self.history:
                si = self._sid_to_idx.get(scenario_id)
                if si is None:
                    continue
                optimal_col = self._optimal_col[si][ti]
                optimal_cat = int(self._action_ids[si][optimal_col])
                if optimal_cat == action_id:
                    matches += 1
            rates[f"{theta:.2f}"] = matches
        return rates

    def per_phase_theta(self) -> dict[str, float]:
        """Estimate θ separately for early-game (turns 1-5), mid (6-10), late (11-15)."""
        phases: dict[str, list[tuple[dict, int]]] = {
            "early (1-5)": [],
            "mid (6-10)": [],
            "late (11-15)": [],
        }

        for scenario_id, action_id in self.history:
            scenario = next((s for s in self.scenarios if s["id"] == scenario_id), None)
            if scenario is None:
                continue
            turn = scenario.get("turn", 8)
            if turn <= 5:
                phases["early (1-5)"].append((scenario, action_id))
            elif turn <= 10:
                phases["mid (6-10)"].append((scenario, action_id))
            else:
                phases["late (11-15)"].append((scenario, action_id))

        result = {}
        for phase_name, obs in phases.items():
            if not obs:
                continue
            temp = ThetaEstimator(self.scenarios)
            for scenario, action_id in obs:
                temp.update(scenario, action_id)
            mean, _, _ = temp.theta_estimate()
            result[phase_name] = mean

        return result


# ── Interactive questionnaire ─────────────────────────────────────────────────

def run_questionnaire(
    scenarios_path: str | Path,
    batch_size: int = 15,
    save_path: str | Path | None = None,
) -> None:
    """Run the interactive θ estimation questionnaire.

    Answers are auto-saved as (scenario_id, category_id) pairs to save_path
    (default: same directory as scenarios, named answers.json).
    """
    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        print("No scenarios loaded. Generate them first with pivotal-scenarios.")
        return

    if save_path is None:
        save_path = Path(scenarios_path).parent / "answers.json"

    estimator = ThetaEstimator(scenarios)
    pool = list(scenarios)  # mutable copy for removal

    print()
    print("=" * 60)
    print("  Yatzy Risk Profile Questionnaire")
    print("=" * 60)
    print()
    print(f"  {len(pool)} scenarios available")
    print(f"  Starting with {batch_size} questions")
    print(f"  Answers will be saved to {save_path}")
    print()

    question_num = 0

    while pool:
        # Run a batch
        batch_target = min(batch_size, len(pool))
        for _ in range(batch_target):
            question_num += 1

            # Select best next question
            scenario = estimator.select_next(pool)
            pool = [s for s in pool if s["id"] != scenario["id"]]

            # Display scenario with full scorecard
            available = scenario["available"]
            available_ids = [c["id"] for c in available]

            print(f"Question {question_num}:")
            print(scorecard_from_scenario(scenario, available_ids))
            print()

            # Show point values for available categories
            for i, cat in enumerate(available):
                score = cat["score"]
                print(f"  [{i + 1}] {cat['name']:<20s} -> {score:>2d} pts")
            print()

            # Get user input
            while True:
                try:
                    raw = input(f"  Your choice [1-{len(available)}] (q to quit): ").strip()
                    if raw.lower() == "q":
                        if estimator.history:
                            save_answers(estimator.history, save_path)
                        _print_results(estimator)
                        return
                    choice = int(raw) - 1
                    if 0 <= choice < len(available):
                        break
                    print(f"  Please enter 1-{len(available)}")
                except (ValueError, EOFError):
                    print(f"  Please enter 1-{len(available)} or 'q' to quit")

            action_id = available[choice]["id"]
            estimator.update(scenario, action_id)

            # Brief inline feedback
            mean, ci_low, ci_high = estimator.theta_estimate()
            print(f"  -> θ estimate: {mean:.2f} [{ci_low:.2f}, {ci_high:.2f}]")
            print()
            print("-" * 60)
            print()

        # Show batch results
        _print_results(estimator)

        if not pool:
            print("  No more scenarios available.")
            break

        # Ask to continue
        print()
        try:
            more = input(f"  Continue with {min(5, len(pool))} more questions? [y/n]: ").strip()
        except EOFError:
            break
        if more.lower() != "y":
            break

        batch_size = 5  # subsequent batches are 5 questions

    # Auto-save answers in robust format
    if estimator.history:
        save_answers(estimator.history, save_path)

    print()
    print("=" * 60)
    print("  Final Summary")
    print("=" * 60)
    _print_results(estimator, final=True)


def _print_results(estimator: ThetaEstimator, final: bool = False) -> None:
    """Print estimation results."""
    n = len(estimator.history)
    mean, ci_low, ci_high = estimator.theta_estimate()
    beta = estimator.beta_estimate()
    rating, fraction = estimator.consistency()

    label = "Final results" if final else f"Results after {n} questions"
    print(f"  === {label} ===")
    print(f"    Estimated theta: {mean:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    print(f"    Rationality (beta): ~{beta:.1f}")
    print(f"    Consistency: {rating} ({fraction * 100:.0f}% match)")

    # Agreement rates for a few key thetas
    rates = estimator.agreement_rates()
    for t_str in ["0.00", "0.05", "0.10"]:
        if t_str in rates:
            print(f"    Answers agreeing with θ={t_str} (EV): {rates[t_str]}/{n}")

    # Profile description
    if mean < 0.02:
        profile = "Risk-neutral. You play close to the EV-optimal strategy."
    elif mean < 0.05:
        profile = "Mildly risk-seeking. You slightly favor high-ceiling categories."
    elif mean < 0.10:
        profile = "Moderately risk-seeking. You chase upside at some mean cost."
    else:
        profile = "Strongly risk-seeking. You aggressively pursue high scores."
    print(f"    Profile: {profile}")

    # Per-phase breakdown if consistency is low
    if rating == "LOW":
        print()
        print("    Warning: Inconsistent risk profile detected.")
        phase_thetas = estimator.per_phase_theta()
        for phase, theta in phase_thetas.items():
            label = "conservative" if theta < 0.03 else "moderate" if theta < 0.08 else "risk-seeking"
            print(f"    {phase} theta: {theta:.2f} ({label})")
        print("    This suggests you adapt your strategy based on game state.")

    # Per-choice breakdown: show each answer and which θ range agrees
    if final and estimator.history:
        print()
        print("  === Per-choice breakdown ===")
        print(f"    {'#':>3}  {'Turn':>4}  {'Dice':<16} {'Your choice':<20} {'θ range'}")
        print(f"    {'─' * 3}  {'─' * 4}  {'─' * 16} {'─' * 20} {'─' * 20}")

        for qi, (scenario_id, action_id) in enumerate(estimator.history, 1):
            scenario = next(
                (s for s in estimator.scenarios if s["id"] == scenario_id), None,
            )
            if scenario is None:
                continue

            si = estimator._sid_to_idx[scenario_id]
            id_to_col = estimator._action_id_to_col[si]
            action_ids = estimator._action_ids[si]
            optimal_col = estimator._optimal_col[si]  # (n_theta,)

            # Find which θ values agree with this choice
            if action_id in id_to_col:
                chosen_col = id_to_col[action_id]
                agreeing = [
                    estimator.theta_grid[ti]
                    for ti in range(len(estimator.theta_grid))
                    if optimal_col[ti] == chosen_col
                ]
            else:
                agreeing = []

            cat_name = CATEGORY_NAMES[action_id] if action_id < 15 else f"cat {action_id}"
            dice_str = str(scenario.get("dice", []))
            turn = scenario.get("turn", "?")

            if not agreeing:
                theta_str = "none"
            elif len(agreeing) == len(estimator.theta_grid):
                theta_str = "all (unanimous)"
            else:
                lo = min(agreeing)
                hi = max(agreeing)
                theta_str = f"{lo:.2f}–{hi:.2f}"

            print(f"    {qi:>3}  {turn:>4}  {dice_str:<16} {cat_name:<20} {theta_str}")


def save_answers(
    history: list[tuple[int, int]],
    path: str | Path,
) -> None:
    """Save answers as (scenario_id, category_id) pairs.

    Format: list of {"scenario_id": int, "category_id": int} dicts.
    This is robust to grid changes — scenario IDs are stable.
    """
    data = [
        {"scenario_id": sid, "category_id": cid}
        for sid, cid in history
    ]
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
    print(f"  Saved {len(data)} answers to {path}")


def replay_answers(
    scenarios_path: str | Path,
    answers_path: str | Path,
) -> None:
    """Replay saved answers and print final results.

    Supports two formats:
    - New: list of {"scenario_id": int, "category_id": int} dicts (robust)
    - Legacy: list of integers (1-based choice indices, fragile)
    """
    scenarios = load_scenarios(scenarios_path)
    if not scenarios:
        print("No scenarios loaded.")
        return

    raw = json.loads(Path(answers_path).read_text())

    # Detect format
    if raw and isinstance(raw[0], dict):
        _replay_robust(scenarios, raw)
    else:
        _replay_legacy(scenarios, raw)


def _replay_robust(
    scenarios: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> None:
    """Replay answers stored as (scenario_id, category_id) pairs."""
    estimator = ThetaEstimator(scenarios)
    scenarios_by_id = {s["id"]: s for s in scenarios}

    for i, entry in enumerate(answers, 1):
        sid = entry["scenario_id"]
        cid = entry["category_id"]

        scenario = scenarios_by_id.get(sid)
        if scenario is None:
            print(f"  Q{i}: scenario {sid} not found (scenarios may have changed)")
            continue

        cat_name = CATEGORY_NAMES[cid] if cid < 15 else f"cat {cid}"
        estimator.update(scenario, cid)

        mean, ci_low, ci_high = estimator.theta_estimate()
        print(f"  Q{i:>2}: {cat_name:<20s} -> θ = {mean:.3f} [{ci_low:.2f}, {ci_high:.2f}]")

    print()
    _print_results(estimator, final=True)


def _replay_legacy(
    scenarios: list[dict[str, Any]],
    answers: list[int],
) -> None:
    """Replay legacy answers (1-based choice indices). Fragile — breaks if scenarios change."""
    print("  WARNING: Legacy answer format (position indices). May be invalid if scenarios changed.")
    print()
    estimator = ThetaEstimator(scenarios)
    pool = list(scenarios)

    for i, choice_idx in enumerate(answers, 1):
        scenario = estimator.select_next(pool)
        pool = [s for s in pool if s["id"] != scenario["id"]]

        available = scenario["available"]
        if choice_idx < 1 or choice_idx > len(available):
            print(f"  Q{i}: invalid choice {choice_idx} (1-{len(available)} available)")
            continue

        action_id = available[choice_idx - 1]["id"]
        estimator.update(scenario, action_id)

        mean, ci_low, ci_high = estimator.theta_estimate()
        print(f"  Q{i:>2}: chose [{choice_idx}] {CATEGORY_NAMES[action_id]:<20s} "
              f"-> θ = {mean:.3f} [{ci_low:.2f}, {ci_high:.2f}]")

    print()
    _print_results(estimator, final=True)
