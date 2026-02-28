#!/usr/bin/env python3
"""Generate 4 D3-ready JSON files for the resource rationality experiments.

Usage:
    research/rosetta/.venv/bin/python research/rosetta/generate_rosetta_experiments.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add analytics to path so we can import skill_ladder
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analytics" / "src"))

from yatzy_analysis.skill_ladder import (
    CATEGORY_NAMES,
    FEATURE_NAMES,
    load_regret_file,
)

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "profiler" / "data"


# ── Shared data loading ─────────────────────────────────────────────────

def load_shared_data():
    """Load and subsample regret data + skill ladder rules."""
    print("Loading regret data...")
    ds = load_regret_file(BASE / "outputs" / "rosetta" / "regret_category.bin", "category")
    print(f"  {ds.features.shape[0]:,} records, {ds.features.shape[1]} features, {ds.num_actions} actions")

    rng = np.random.default_rng(42)
    n_sample = min(300_000, ds.features.shape[0])
    idx = rng.choice(ds.features.shape[0], n_sample, replace=False)
    idx.sort()

    features = ds.features[idx]
    q_values = ds.q_values[idx]
    best_action = ds.best_action[idx]
    regret = ds.regret[idx]

    print(f"  Subsampled to {n_sample:,} records")

    with open(BASE / "outputs" / "rosetta" / "skill_ladder.json") as f:
        ladder = json.load(f)

    rules = ladder["category_rules"]
    meta = ladder["meta"]
    default_action = meta["default_category"]

    return features, q_values, best_action, regret, rules, default_action, meta


def eval_rule_mask(rule, features):
    """Evaluate a rule's conditions against feature matrix, return boolean mask."""
    mask = np.ones(features.shape[0], dtype=bool)
    for cond in rule["conditions"]:
        col_idx = FEATURE_NAMES.index(cond["feature"])
        col = features[:, col_idx]
        threshold = cond["threshold"]
        op = cond["op"]
        if op == "==":
            mask &= col == threshold
        elif op == ">=":
            mask &= col >= threshold
        elif op == "<=":
            mask &= col <= threshold
        elif op == ">":
            mask &= col > threshold
        elif op == "<":
            mask &= col < threshold
        else:
            raise ValueError(f"Unknown op: {op}")
    return mask


def mean_finite_regret(regret_arr, actions, n):
    """Mean regret for assigned actions, treating -inf (unavailable) as max regret."""
    per_record = regret_arr[np.arange(n), actions]
    # For unavailable categories (-inf regret), use max finite regret in that record
    neginf = np.isneginf(per_record)
    if neginf.any():
        # Replace -inf with max available regret (worst available choice)
        max_finite = np.where(
            np.isfinite(regret_arr),
            regret_arr,
            -np.inf,
        ).max(axis=1)
        per_record = np.where(neginf, max_finite, per_record)
    return float(per_record.mean())


# ── Experiment A: Cognitive Power Law ────────────────────────────────────

def experiment_a(features, q_values, regret, best_action, rules, default_action):
    """Show diminishing returns as rules are added.

    Uses scaling_law.json for calibrated game-level EVs at key rule counts,
    and regret data for per-rule coverage and marginal EV analysis.
    """
    print("\n=== Experiment A: Cognitive Power Law ===")
    n = features.shape[0]
    assigned = np.full(n, default_action, dtype=np.int32)
    covered = np.zeros(n, dtype=bool)

    # Load calibrated game-level EVs from scaling_law.json
    with open(BASE / "outputs" / "rosetta" / "scaling_law.json") as f:
        scaling = json.load(f)
    with open(BASE / "outputs" / "rosetta" / "eval_results.json") as f:
        eval_data = json.load(f)

    oracle_ev = eval_data["oracle_ev"]  # 248.44

    # Build lookup: n_rules → game EV
    scaling_ev = {s["n_rules"]: s["ev"] for s in scaling["sweep"]}
    default_ev = scaling_ev[0]  # 196.65

    # Compute per-rule regret reduction for marginal analysis
    prev_regret = mean_finite_regret(regret, assigned, n)

    results = []
    for k, rule in enumerate(rules):
        mask = eval_rule_mask(rule, features)
        newly_covered = mask & ~covered
        n_new = newly_covered.sum()

        covered |= newly_covered
        assigned[newly_covered] = rule["action"]

        # Per-decision mean regret (used for marginal EV)
        curr_regret = mean_finite_regret(regret, assigned, n)
        marginal_regret_reduction = prev_regret - curr_regret

        # Interpolate game-level EV from scaling_law checkpoints
        rule_count = k + 1
        if rule_count in scaling_ev:
            cumulative_ev = scaling_ev[rule_count]
        else:
            # Linear interpolation between nearest checkpoints
            keys = sorted(scaling_ev.keys())
            lo = max(x for x in keys if x <= rule_count)
            hi = min(x for x in keys if x >= rule_count)
            if lo == hi:
                cumulative_ev = scaling_ev[lo]
            else:
                t = (rule_count - lo) / (hi - lo)
                cumulative_ev = scaling_ev[lo] + t * (scaling_ev[hi] - scaling_ev[lo])

        action_name = CATEGORY_NAMES[rule["action"]] if isinstance(rule["action"], int) else str(rule["action"])
        cond_str = " AND ".join(
            f"{c['feature']} {c['op']} {c['threshold']}" for c in rule["conditions"]
        )

        results.append({
            "rank": rule_count,
            "name": f"{action_name} ({cond_str})",
            "action": action_name,
            "coverage_pct": round(n_new / n, 6),
            "cumulative_coverage": round(covered.sum() / n, 6),
            "marginal_ev": round(marginal_regret_reduction, 4),
            "cumulative_ev": round(cumulative_ev, 2),
        })
        prev_regret = curr_regret

        if (k + 1) % 25 == 0:
            print(f"  Rule {k+1}: coverage={covered.sum()/n:.1%}, EV={cumulative_ev:.2f}")

    output = {
        "rules": results,
        "oracle_ev": oracle_ev,
        "default_ev": round(default_ev, 2),
        "default_action": CATEGORY_NAMES[default_action],
        "n_records": n,
    }

    path = OUT_DIR / "exp_a_power_law.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {path} ({path.stat().st_size / 1024:.1f} KB)")

    return assigned, covered


# ── Experiment B: Anatomy of Mistakes ────────────────────────────────────

def experiment_b(features, q_values, best_action, assigned):
    """Hexbin heatmap of where the 100-rule policy leaks EV."""
    print("\n=== Experiment B: Anatomy of Mistakes ===")
    n = features.shape[0]
    disagree = assigned != best_action
    n_disagree = disagree.sum()
    print(f"  Disagreements: {n_disagree:,} / {n:,} ({n_disagree/n:.1%})")

    # Compute actual regret as Q(best) - Q(assigned)
    best_q = np.max(q_values, axis=1)
    assigned_q = q_values[np.arange(n), assigned]
    actual_regret = np.where(np.isfinite(assigned_q), best_q - assigned_q, best_q)
    dis_regret = actual_regret[disagree]
    dis_features = features[disagree]

    # Group by unique feature vectors
    dis_features_int = dis_features.astype(np.int32)
    _, inverse, counts = np.unique(dis_features_int, axis=0, return_inverse=True, return_counts=True)

    # Per-group mean regret
    n_groups = counts.shape[0]
    group_regret_sum = np.zeros(n_groups, dtype=np.float64)
    np.add.at(group_regret_sum, inverse, dis_regret)
    group_mean_regret = group_regret_sum / counts

    # Filter to groups with positive regret
    pos = group_mean_regret > 0
    freq = counts[pos].astype(float)
    reg = group_mean_regret[pos]

    print(f"  Unique disagreement states: {n_groups:,}, with positive regret: {pos.sum():,}")

    # Hexbin: bin on log10(freq) x log10(regret)
    log_freq = np.log10(np.clip(freq, 1, None))
    log_reg = np.log10(np.clip(reg, 1e-3, None))

    bin_radius = 0.15
    x_range = [0, float(np.ceil(log_freq.max() + 0.5))]
    y_range = [float(np.floor(log_reg.min() - 0.5)), float(np.ceil(log_reg.max() + 0.5))]

    # Create hex grid
    hex_w = bin_radius * 2
    hex_h = bin_radius * np.sqrt(3)

    hexbins = {}
    for lf, lr in zip(log_freq, log_reg):
        col = int(np.floor(lf / hex_w))
        row_shift = 0.5 * hex_w if col % 2 else 0
        row = int(np.floor((lr - row_shift) / hex_h))
        key = (col, row)
        hexbins[key] = hexbins.get(key, 0) + 1

    hex_data = []
    for (col, row), density in hexbins.items():
        cx = (col + 0.5) * hex_w
        row_shift = 0.5 * hex_h if col % 2 else 0
        cy = (row + 0.5) * hex_h + row_shift
        hex_data.append({
            "freq_bin": round(cx, 3),
            "regret_bin": round(cy, 3),
            "density": density,
        })

    hex_data.sort(key=lambda x: -x["density"])
    print(f"  Hex bins: {len(hex_data)}")

    # Extract 5 diverse example states with high regret
    top_regret_idx = np.argsort(dis_regret)[-50:][::-1]
    disagree_indices = np.where(disagree)[0]
    examples = []
    seen_turns = set()
    for i in top_regret_idx:
        global_idx = disagree_indices[i]
        oracle_act = int(best_action[global_idx])
        rule_act = int(assigned[global_idx])
        turn = int(features[global_idx, FEATURE_NAMES.index("turn")])
        key = (oracle_act, rule_act, turn)
        if key in seen_turns:
            continue
        seen_turns.add(key)
        examples.append({
            "freq": int(counts[inverse[i]]),
            "regret": round(float(dis_regret[i]), 2),
            "oracle": CATEGORY_NAMES[oracle_act],
            "rule_chose": CATEGORY_NAMES[rule_act],
            "turn": int(features[global_idx, FEATURE_NAMES.index("turn")]),
            "upper_score": int(features[global_idx, FEATURE_NAMES.index("upper_score")]),
            "categories_left": int(features[global_idx, FEATURE_NAMES.index("categories_left")]),
        })
        if len(examples) >= 5:
            break

    output = {
        "hexbins": hex_data,
        "examples": examples,
        "total_disagreements": int(n_disagree),
        "total_records": n,
        "hex_params": {
            "x_range": x_range,
            "y_range": y_range,
            "bin_radius": bin_radius,
        },
    }

    path = OUT_DIR / "exp_b_mistakes.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {path} ({path.stat().st_size / 1024:.1f} KB)")


# ── Experiment C: Cartography of Unhandled ───────────────────────────────

def experiment_c(features, q_values, best_action, assigned, covered, rules):
    """UMAP visualization of handled-core vs unhandled states."""
    print("\n=== Experiment C: Cartography of Unhandled ===")

    n = features.shape[0]

    # Core: covered by rules 1-20; Unhandled: not covered by any of 100 rules
    core_mask = np.zeros(n, dtype=bool)
    for rule in rules[:20]:
        mask = eval_rule_mask(rule, features)
        core_mask |= mask

    unhandled_mask = ~covered

    n_core_total = core_mask.sum()
    n_unhandled_total = unhandled_mask.sum()
    print(f"  Core (rules 1-20): {n_core_total:,}, Unhandled: {n_unhandled_total:,}")

    rng = np.random.default_rng(42)

    core_idx = np.where(core_mask)[0]
    unhandled_idx = np.where(unhandled_mask)[0]

    n_core_sample = min(5000, len(core_idx))
    n_unhandled_sample = min(5000, len(unhandled_idx))

    core_sample = rng.choice(core_idx, n_core_sample, replace=False)
    unhandled_sample = rng.choice(unhandled_idx, n_unhandled_sample, replace=False)

    combined_idx = np.concatenate([core_sample, unhandled_sample])
    combined_features = features[combined_idx]

    # Standardize
    mean = combined_features.mean(axis=0)
    std = combined_features.std(axis=0)
    std[std == 0] = 1.0
    standardized = (combined_features - mean) / std

    # UMAP
    print("  Running UMAP...")
    import umap
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(standardized)
    print(f"  UMAP done: {embedding.shape}")

    # Build rule_id lookup for core points
    rule_ids = np.full(n, -1, dtype=np.int32)
    rule_covered = np.zeros(n, dtype=bool)
    for k, rule in enumerate(rules[:20]):
        mask = eval_rule_mask(rule, features)
        newly = mask & ~rule_covered
        rule_ids[newly] = k + 1
        rule_covered |= newly

    # Compute per-point regret
    best_q = np.max(q_values, axis=1)
    assigned_q = q_values[np.arange(n), assigned]
    all_regret = np.where(np.isfinite(assigned_q), best_q - assigned_q, best_q)

    points = []
    for i, global_i in enumerate(combined_idx):
        is_core = i < n_core_sample
        oracle_act = int(best_action[global_i])
        points.append({
            "x": round(float(embedding[i, 0]), 3),
            "y": round(float(embedding[i, 1]), 3),
            "status": "Core" if is_core else "Unhandled",
            "rule_id": int(rule_ids[global_i]) if is_core and rule_ids[global_i] >= 0 else None,
            "oracle": CATEGORY_NAMES[oracle_act],
            "regret": round(float(all_regret[global_i]), 2),
        })

    output = {
        "points": points,
        "n_core": n_core_sample,
        "n_unhandled": n_unhandled_sample,
        "umap_params": {"n_neighbors": 30, "min_dist": 0.1, "metric": "cosine"},
    }

    path = OUT_DIR / "exp_c_unhandled_umap.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {path} ({path.stat().st_size / 1024:.1f} KB)")


# ── Experiment D: Resource-Rational Pareto Frontier ──────────────────────

def experiment_d():
    """Unified params-vs-EV plot across all policy families."""
    print("\n=== Experiment D: Resource-Rational Pareto Frontier ===")

    points = []

    # Surrogates
    with open(BASE / "outputs" / "surrogate" / "game_eval_results.json") as f:
        surrogates = json.load(f)

    for s in surrogates:
        name = s["name"]
        if name == "heuristic":
            family = "Baseline"
        elif name.startswith("dt_"):
            family = "Decision Tree"
        elif name.startswith("mlp_"):
            family = "MLP"
        else:
            family = "Other"
        points.append({
            "name": name,
            "params": s["total_params"],
            "ev": round(s["mean"], 2),
            "family": family,
        })

    # Symbolic rules from scaling_law.json
    with open(BASE / "outputs" / "rosetta" / "scaling_law.json") as f:
        scaling = json.load(f)

    for entry in scaling["sweep"]:
        n_rules = entry["n_rules"]
        params = n_rules * 3
        points.append({
            "name": f"{n_rules} rules",
            "params": params,
            "ev": round(entry["ev"], 2),
            "family": "Symbolic",
        })

    # Oracle (exact DP)
    with open(BASE / "outputs" / "rosetta" / "eval_results.json") as f:
        eval_data = json.load(f)
    points.append({
        "name": "Oracle (DP)",
        "params": 1_430_000,
        "ev": round(eval_data["oracle_ev"], 2),
        "family": "Exact",
    })

    # Sort by params for Pareto computation
    points.sort(key=lambda p: (p["params"], -p["ev"]))

    # Compute Pareto frontier
    max_ev = -float("inf")
    for p in points:
        if p["ev"] > max_ev:
            p["pareto"] = True
            max_ev = p["ev"]
        else:
            p["pareto"] = False

    families = sorted(set(p["family"] for p in points))
    n_pareto = sum(1 for p in points if p["pareto"])

    output = {
        "points": points,
        "families": families,
    }

    path = OUT_DIR / "exp_d_pareto.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {path} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(points)} points, {n_pareto} on Pareto frontier, families: {families}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    features, q_values, best_action, regret, rules, default_action, meta = load_shared_data()

    assigned, covered = experiment_a(features, q_values, regret, best_action, rules, default_action)
    experiment_b(features, q_values, best_action, assigned)
    experiment_d()
    experiment_c(features, q_values, best_action, assigned, covered, rules)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in ["exp_a_power_law", "exp_b_mistakes", "exp_c_unhandled_umap", "exp_d_pareto"]:
        path = OUT_DIR / f"{name}.json"
        size_kb = path.stat().st_size / 1024
        with open(path) as f:
            data = json.load(f)
        if name == "exp_a_power_law":
            desc = f"{len(data['rules'])} rules, EV: {data['default_ev']:.1f} → {data['rules'][-1]['cumulative_ev']:.1f}"
        elif name == "exp_b_mistakes":
            desc = f"{data['total_disagreements']:,} disagreements, {len(data['hexbins'])} hex bins"
        elif name == "exp_c_unhandled_umap":
            desc = f"{data['n_core']}+{data['n_unhandled']} points"
        elif name == "exp_d_pareto":
            n_p = sum(1 for p in data["points"] if p["pareto"])
            desc = f"{len(data['points'])} models, {n_p} Pareto-optimal"
        print(f"  {name:30s}  {size_kb:7.1f} KB  {desc}")


if __name__ == "__main__":
    main()
