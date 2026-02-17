"""Surrogate model training: decision trees and MLPs for Yatzy policy compression.

Trains classifiers on per-decision training data exported by yatzy-export-training-data,
evaluates via gap-weighted EV loss, and produces a Pareto frontier of model size vs accuracy.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

# ── Binary data loading ──────────────────────────────────────────────────

MAGIC = 0x59545244  # "YTRD"
VERSION = 1
HEADER_SIZE = 32


def load_training_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load binary file -> (features, labels, gaps) numpy arrays.

    Binary format:
      Header (32 bytes): magic u32, version u32, num_records u64,
                          num_features u32, num_actions u32, reserved u64
      Records: [f32 * num_features, u16, f32] per record

    Uses numpy structured dtype for zero-copy vectorized loading.
    """
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)

    magic, version, num_records, num_features, num_actions = struct.unpack_from(
        "<IIqII", header, 0
    )
    assert magic == MAGIC, f"Bad magic: {magic:#x}"
    assert version == VERSION, f"Bad version: {version}"

    # Build a structured dtype matching the record layout (packed, no padding)
    dt = np.dtype([
        ("features", np.float32, (num_features,)),
        ("label", np.uint16),
        ("gap", np.float32),
    ])

    # Memory-map the body for zero-copy read
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_SIZE)
    # View as structured array — works because dtype packing matches file layout
    records = np.frombuffer(raw, dtype=dt, count=num_records)

    features = np.array(records["features"], dtype=np.float32)
    labels = np.array(records["label"], dtype=np.int64)
    gaps = np.array(records["gap"], dtype=np.float32)

    return features, labels, gaps


# ── Evaluation metrics ───────────────────────────────────────────────────


def ev_loss_per_game(
    y_true: np.ndarray, y_pred: np.ndarray, gaps: np.ndarray, n_games: int
) -> float:
    """EV loss = sum(gap[i] * I(wrong[i])) / n_games."""
    wrong = (y_true != y_pred).astype(np.float64)
    return float((gaps * wrong).sum() / n_games)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def accuracy_by_turn(
    y_true: np.ndarray, y_pred: np.ndarray, n_decisions_per_game: int = 15
) -> np.ndarray:
    """Per-turn accuracy (assumes records are ordered by game, then turn)."""
    n = len(y_true)
    n_games = n // n_decisions_per_game
    correct = (y_true == y_pred).reshape(n_games, n_decisions_per_game)
    return correct.mean(axis=0)


# ── Model result container ───────────────────────────────────────────────


@dataclass
class ModelResult:
    name: str
    model_type: str  # "dt" or "mlp"
    decision_type: str  # "category", "reroll1", "reroll2"
    n_params: int
    accuracy: float
    ev_loss: float
    train_accuracy: float = 0.0
    train_ev_loss: float = 0.0
    accuracy_per_turn: np.ndarray = field(default_factory=lambda: np.array([]))
    extra: dict = field(default_factory=dict)


# ── Decision tree training ───────────────────────────────────────────────


def count_dt_params(tree) -> int:
    """Parameter count for a decision tree: 2 per internal node + 1 per leaf."""
    n_nodes = tree.tree_.node_count
    n_leaves = tree.tree_.n_leaves
    n_internal = n_nodes - n_leaves
    return n_internal * 2 + n_leaves


def train_decision_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gaps_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gaps_test: np.ndarray,
    n_train_games: int,
    n_test_games: int,
    decision_type: str,
    models_dir: Path | None = None,
) -> list[ModelResult]:
    """Train decision trees at various depths."""
    from sklearn.tree import DecisionTreeClassifier

    depths = [1, 2, 3, 5, 8, 10, 15, 20, 30, None]
    results = []

    for depth in depths:
        name = f"dt_d{depth}" if depth else "dt_full"
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)

        pred_test = clf.predict(X_test)
        pred_train = clf.predict(X_train)

        if models_dir is not None:
            joblib.dump(clf, models_dir / f"{decision_type}_{name}.pkl")

        results.append(
            ModelResult(
                name=name,
                model_type="dt",
                decision_type=decision_type,
                n_params=count_dt_params(clf),
                accuracy=accuracy(y_test, pred_test),
                ev_loss=ev_loss_per_game(y_test, pred_test, gaps_test, n_test_games),
                train_accuracy=accuracy(y_train, pred_train),
                train_ev_loss=ev_loss_per_game(y_train, pred_train, gaps_train, n_train_games),
                accuracy_per_turn=accuracy_by_turn(y_test, pred_test),
                extra={"depth": depth},
            )
        )
    return results


# ── Random Forest for feature importance ─────────────────────────────────


def compute_feature_importance(
    X_train: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    """Fit a Random Forest on a subsample and return feature importances."""
    from sklearn.ensemble import RandomForestClassifier

    # Subsample to 100K for speed (RF on 2.4M is very slow)
    n = len(X_train)
    if n > 100_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 100_000, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    rf = RandomForestClassifier(
        n_estimators=30, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_sub, y_sub)
    return rf.feature_importances_


# ── MLP training ─────────────────────────────────────────────────────────


def count_mlp_params(layer_sizes: list[int], n_features: int, n_classes: int) -> int:
    """Count total parameters in an MLP: sum of (in+1)*out for each layer."""
    total = 0
    prev = n_features
    for size in layer_sizes:
        total += (prev + 1) * size  # weights + bias
        prev = size
    total += (prev + 1) * n_classes  # output layer
    return total


def train_mlps(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gaps_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gaps_test: np.ndarray,
    n_train_games: int,
    n_test_games: int,
    decision_type: str,
    n_features: int,
    n_classes: int,
    models_dir: Path | None = None,
) -> list[ModelResult]:
    """Train MLPs with various architectures using sklearn (much faster than PyTorch for this scale)."""
    from sklearn.neural_network import MLPClassifier

    architectures: list[tuple[int, ...]] = [
        (8,),
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64),
    ]

    # Subsample training data for speed — MLPs converge with 300K records
    max_train = 300_000
    if len(X_train) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_train, replace=False)
        idx.sort()  # preserve ordering for game-based metrics
        X_tr_sub, y_tr_sub, gaps_tr_sub = X_train[idx], y_train[idx], gaps_train[idx]
        n_train_games_sub = max_train // 15
    else:
        X_tr_sub, y_tr_sub, gaps_tr_sub = X_train, y_train, gaps_train
        n_train_games_sub = n_train_games

    results = []

    for layers in architectures:
        name = f"mlp_{'_'.join(str(s) for s in layers)}"
        n_params = count_mlp_params(list(layers), n_features, n_classes)
        t_start = time.time()

        clf = MLPClassifier(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=4096,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        clf.fit(X_tr_sub, y_tr_sub)

        if models_dir is not None:
            joblib.dump(clf, models_dir / f"{decision_type}_{name}.pkl")

        pred_test = clf.predict(X_test)
        pred_train = clf.predict(X_tr_sub)

        elapsed = time.time() - t_start
        r = ModelResult(
            name=name,
            model_type="mlp",
            decision_type=decision_type,
            n_params=n_params,
            accuracy=accuracy(y_test, pred_test),
            ev_loss=ev_loss_per_game(y_test, pred_test, gaps_test, n_test_games),
            train_accuracy=accuracy(y_tr_sub, pred_train),
            train_ev_loss=ev_loss_per_game(
                y_tr_sub, pred_train, gaps_tr_sub, n_train_games_sub,
            ),
            accuracy_per_turn=accuracy_by_turn(y_test, pred_test),
            extra={"layers": list(layers), "epochs": clf.n_iter_},
        )
        results.append(r)
        print(
            f"    {name:>20s}: {n_params:>8,d} params, "
            f"acc={r.accuracy:.4f}, ev_loss={r.ev_loss:.3f}/game "
            f"({clf.n_iter_} iters, {elapsed:.1f}s)"
        )

    return results


# ── Pareto frontier ──────────────────────────────────────────────────────


def compute_pareto_frontier(results: list[ModelResult]) -> list[ModelResult]:
    """Extract Pareto-optimal models (minimize params AND ev_loss)."""
    sorted_results = sorted(results, key=lambda r: r.n_params)
    pareto: list[ModelResult] = []
    best_loss = float("inf")
    for r in sorted_results:
        if r.ev_loss < best_loss:
            pareto.append(r)
            best_loss = r.ev_loss
    return pareto


# ── Full training pipeline ───────────────────────────────────────────────

FEATURE_NAMES_CATEGORY = [
    "turn", "upper_score", "upper_cats_left", "bonus_secured", "bonus_deficit",
    "face1", "face2", "face3", "face4", "face5", "face6",
    "dice_sum", "max_face_count", "num_distinct",
    "cat0_avail", "cat1_avail", "cat2_avail", "cat3_avail", "cat4_avail",
    "cat5_avail", "cat6_avail", "cat7_avail", "cat8_avail", "cat9_avail",
    "cat10_avail", "cat11_avail", "cat12_avail", "cat13_avail", "cat14_avail",
]

FEATURE_NAMES_REROLL = FEATURE_NAMES_CATEGORY + ["rerolls_remaining"]


def run_training_pipeline(
    data_dir: Path,
    output_dir: Path,
    n_games: int = 200_000,
) -> dict[str, list[ModelResult]]:
    """Run full DT + MLP training for all 3 decision types."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[ModelResult]] = {}

    decision_configs = [
        ("category", "category_decisions.bin", 29, 15),
        ("reroll1", "reroll1_decisions.bin", 30, 32),
        ("reroll2", "reroll2_decisions.bin", 30, 32),
    ]

    for dtype, filename, n_features, n_classes in decision_configs:
        path = data_dir / filename
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Training {dtype} models ({n_features} features, {n_classes} classes)")
        print(f"{'='*60}")

        t0 = time.time()
        features, labels, gaps = load_training_data(path)
        print(f"  Loaded {len(features):,d} records in {time.time()-t0:.1f}s")
        print(f"  Mean gap: {gaps.mean():.3f}, zero-gap: {(gaps < 0.01).mean():.1%}")

        # Train/test split by game (80/20)
        n_records_per_game = 15
        n_total_games = len(features) // n_records_per_game
        n_train_games = int(n_total_games * 0.8)
        n_test_games = n_total_games - n_train_games
        train_end = n_train_games * n_records_per_game

        X_train, X_test = features[:train_end], features[train_end:]
        y_train, y_test = labels[:train_end], labels[train_end:]
        gaps_train, gaps_test = gaps[:train_end], gaps[train_end:]

        print(f"  Train: {len(X_train):,d} records ({n_train_games:,d} games)")
        print(f"  Test:  {len(X_test):,d} records ({n_test_games:,d} games)")

        results: list[ModelResult] = []

        # Decision trees
        print("\n  --- Decision Trees ---")
        dt_results = train_decision_trees(
            X_train, y_train, gaps_train,
            X_test, y_test, gaps_test,
            n_train_games, n_test_games, dtype,
            models_dir=models_dir,
        )
        for r in dt_results:
            print(
                f"    {r.name:>12s}: {r.n_params:>8,d} params, "
                f"acc={r.accuracy:.4f}, ev_loss={r.ev_loss:.3f}/game"
            )
        results.extend(dt_results)

        # Feature importance
        print("\n  Computing feature importance...")
        importances = compute_feature_importance(X_train, y_train)
        feat_names = FEATURE_NAMES_REROLL if n_features == 30 else FEATURE_NAMES_CATEGORY
        top_idx = np.argsort(importances)[::-1][:10]
        for idx in top_idx:
            print(f"    {feat_names[idx]:>20s}: {importances[idx]:.4f}")

        # Save feature importances
        np.savez(
            output_dir / f"feature_importance_{dtype}.npz",
            importances=importances,
            names=np.array(feat_names),
        )

        # MLPs
        print("\n  --- MLPs ---")
        mlp_results = train_mlps(
            X_train, y_train, gaps_train,
            X_test, y_test, gaps_test,
            n_train_games, n_test_games, dtype,
            n_features, n_classes,
            models_dir=models_dir,
        )
        results.extend(mlp_results)

        # Baselines
        random_ev_loss = ev_loss_per_game(
            y_test, np.random.RandomState(42).randint(0, n_classes, len(y_test)),
            gaps_test, n_test_games,
        )
        results.append(
            ModelResult(
                name="random",
                model_type="baseline",
                decision_type=dtype,
                n_params=0,
                accuracy=1.0 / n_classes,
                ev_loss=random_ev_loss,
            )
        )

        all_results[dtype] = results

        # Save per-type CSV
        _save_results_csv(results, output_dir / f"results_{dtype}.csv")

    # Combined Pareto frontier
    all_combined: list[ModelResult] = []
    for results in all_results.values():
        all_combined.extend(results)

    pareto = compute_pareto_frontier(
        [r for r in all_combined if r.model_type != "baseline"]
    )
    _save_pareto_csv(pareto, output_dir / "pareto_frontier.csv")
    _save_pareto_json(pareto, all_results, output_dir / "pareto_frontier.json")

    return all_results


def _save_results_csv(results: list[ModelResult], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "name", "model_type", "decision_type", "n_params",
            "accuracy", "ev_loss", "train_accuracy", "train_ev_loss",
        ])
        for r in results:
            w.writerow([
                r.name, r.model_type, r.decision_type, r.n_params,
                f"{r.accuracy:.6f}", f"{r.ev_loss:.6f}",
                f"{r.train_accuracy:.6f}", f"{r.train_ev_loss:.6f}",
            ])


def _save_pareto_csv(pareto: list[ModelResult], path: Path) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "model_type", "decision_type", "n_params", "accuracy", "ev_loss"])
        for r in pareto:
            w.writerow([
                r.name, r.model_type, r.decision_type, r.n_params,
                f"{r.accuracy:.6f}", f"{r.ev_loss:.6f}",
            ])


def _save_pareto_json(
    pareto: list[ModelResult],
    all_results: dict[str, list[ModelResult]],
    path: Path,
) -> None:
    summary = {
        "pareto_frontier": [
            {
                "name": r.name,
                "model_type": r.model_type,
                "decision_type": r.decision_type,
                "n_params": int(r.n_params),
                "accuracy": round(r.accuracy, 6),
                "ev_loss": round(r.ev_loss, 6),
            }
            for r in pareto
        ],
        "per_type_summary": {},
    }
    for dtype, results in all_results.items():
        best = min((r for r in results if r.model_type != "baseline"), key=lambda r: r.ev_loss)
        summary["per_type_summary"][dtype] = {
            "best_model": best.name,
            "best_ev_loss": round(best.ev_loss, 6),
            "best_accuracy": round(best.accuracy, 6),
            "best_n_params": int(best.n_params),
            "num_models_tested": len(results),
        }
    path.write_text(json.dumps(summary, indent=2))


# ── Diagnostic experiments ──────────────────────────────────────────────


def quantify_label_noise(
    features: np.ndarray, labels: np.ndarray, gaps: np.ndarray
) -> dict:
    """Group records by feature vector, check for conflicting labels.

    If zero conflicts, the feature set is lossless (no two identical feature
    vectors map to different optimal actions). If nonzero, bounds the
    irreducible EV loss floor.
    """
    # Hash each feature row to group duplicates efficiently
    # Use a dict mapping feature-tuple -> set of labels seen
    from collections import defaultdict

    label_sets: dict[bytes, set[int]] = defaultdict(set)
    gap_sums: dict[bytes, float] = defaultdict(float)
    counts: dict[bytes, int] = defaultdict(int)

    for i in range(len(features)):
        key = features[i].tobytes()
        label_sets[key].add(int(labels[i]))
        counts[key] += 1
        gap_sums[key] += float(gaps[i])

    n_unique = len(label_sets)
    conflicting = {k: v for k, v in label_sets.items() if len(v) > 1}
    n_conflicting = len(conflicting)

    # Estimate irreducible EV loss from conflicts
    total_conflict_gap = 0.0
    total_conflict_records = 0
    for key in conflicting:
        total_conflict_gap += gap_sums[key]
        total_conflict_records += counts[key]

    return {
        "n_records": len(features),
        "n_unique_vectors": n_unique,
        "n_conflicting": n_conflicting,
        "conflict_fraction": n_conflicting / n_unique if n_unique > 0 else 0,
        "total_conflict_records": total_conflict_records,
        "total_conflict_gap": total_conflict_gap,
    }


def analyze_error_distribution(
    X_test: np.ndarray,
    y_test: np.ndarray,
    gaps_test: np.ndarray,
    model,
    n_test_games: int,
    decision_type: str,
) -> dict:
    """Analyze where dt_full makes errors: gap distribution, turn distribution,
    bonus proximity clustering."""
    pred = model.predict(X_test)
    wrong = pred != y_test
    n_wrong = int(wrong.sum())
    n_total = len(y_test)

    if n_wrong == 0:
        return {
            "decision_type": decision_type,
            "n_total": n_total,
            "n_wrong": 0,
            "error_rate": 0.0,
        }

    error_gaps = gaps_test[wrong]
    error_features = X_test[wrong]

    # Gap distribution of errors
    gap_stats = {
        "mean": float(error_gaps.mean()),
        "median": float(np.median(error_gaps)),
        "p90": float(np.percentile(error_gaps, 90)),
        "p99": float(np.percentile(error_gaps, 99)),
        "max": float(error_gaps.max()),
        "near_zero_frac": float((error_gaps < 0.1).mean()),
    }

    # Turn distribution of errors
    n_features = X_test.shape[1]
    turn_idx = 0  # turn is always feature 0
    error_turns = (error_features[:, turn_idx] * 14).round().astype(int)
    all_turns = (X_test[:, turn_idx] * 14).round().astype(int)

    turn_error_rates = {}
    for t in range(15):
        total_at_t = int((all_turns == t).sum())
        errors_at_t = int((error_turns == t).sum())
        turn_error_rates[t] = {
            "total": total_at_t,
            "errors": errors_at_t,
            "rate": errors_at_t / total_at_t if total_at_t > 0 else 0,
        }

    # Bonus proximity: errors near bonus threshold (upper_score near 63)
    upper_score_idx = 1
    error_upper = error_features[:, upper_score_idx] * 63
    near_bonus = float(((error_upper > 40) & (error_upper < 63)).mean())

    return {
        "decision_type": decision_type,
        "n_total": n_total,
        "n_wrong": n_wrong,
        "error_rate": n_wrong / n_total,
        "ev_loss": float((gaps_test * wrong).sum() / n_test_games),
        "gap_stats": gap_stats,
        "turn_error_rates": turn_error_rates,
        "near_bonus_fraction": near_bonus,
    }


def run_data_scaling_experiment(
    features: np.ndarray,
    labels: np.ndarray,
    gaps: np.ndarray,
    decision_type: str,
    n_games_total: int,
    subsets: list[int] | None = None,
) -> list[dict]:
    """Train dt_full on increasing data subsets, measure EV loss.

    Returns list of {n_games, n_records, ev_loss, accuracy, n_params}.
    """
    from sklearn.tree import DecisionTreeClassifier

    if subsets is None:
        subsets = [25_000, 50_000, 100_000, 150_000, n_games_total]

    records_per_game = 15
    # Use last 20% of full dataset as fixed test set
    n_test_games = n_games_total // 5
    test_start = (n_games_total - n_test_games) * records_per_game
    X_test = features[test_start:]
    y_test = labels[test_start:]
    gaps_test = gaps[test_start:]

    results = []
    for n_games in subsets:
        if n_games > n_games_total - n_test_games:
            n_games = n_games_total - n_test_games
        n_train = n_games * records_per_game

        X_train = features[:n_train]
        y_train = labels[:n_train]

        clf = DecisionTreeClassifier(max_depth=None, random_state=42)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        wrong = pred != y_test
        el = float((gaps_test * wrong).sum() / n_test_games)
        acc = float((~wrong).mean())

        results.append({
            "n_games": n_games,
            "n_records": n_train,
            "ev_loss": el,
            "accuracy": acc,
            "n_params": count_dt_params(clf),
        })
        print(
            f"    {decision_type} @ {n_games:>7,d} games: "
            f"ev_loss={el:.4f}, acc={acc:.4f}, params={count_dt_params(clf):,d}"
        )

    return results


def run_feature_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gaps_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gaps_test: np.ndarray,
    n_train_games: int,
    n_test_games: int,
    decision_type: str,
    depth: int = 20,
) -> list[dict]:
    """Remove feature groups one at a time, measure delta EV loss vs full model."""
    from sklearn.tree import DecisionTreeClassifier

    from .feature_engineering import FEATURE_GROUPS

    n_features = X_train.shape[1]

    # Baseline: full feature set
    clf_full = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf_full.fit(X_train, y_train)
    pred_full = clf_full.predict(X_test)
    baseline_loss = ev_loss_per_game(y_test, pred_full, gaps_test, n_test_games)
    baseline_acc = accuracy(y_test, pred_full)

    results = [
        {
            "group": "all_features",
            "removed_indices": [],
            "ev_loss": baseline_loss,
            "accuracy": baseline_acc,
            "delta_ev_loss": 0.0,
            "delta_accuracy": 0.0,
            "n_params": count_dt_params(clf_full),
        }
    ]

    for group_name, indices in FEATURE_GROUPS.items():
        # Skip rerolls_rem for category decisions
        if group_name == "rerolls_rem" and n_features < 30:
            continue
        # Skip indices beyond feature count
        valid_indices = [i for i in indices if i < n_features]
        if not valid_indices:
            continue

        keep = [i for i in range(n_features) if i not in valid_indices]
        X_tr_sub = X_train[:, keep]
        X_te_sub = X_test[:, keep]

        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_tr_sub, y_train)
        pred = clf.predict(X_te_sub)

        el = ev_loss_per_game(y_test, pred, gaps_test, n_test_games)
        acc = accuracy(y_test, pred)

        results.append({
            "group": group_name,
            "removed_indices": valid_indices,
            "ev_loss": el,
            "accuracy": acc,
            "delta_ev_loss": el - baseline_loss,
            "delta_accuracy": acc - baseline_acc,
            "n_params": count_dt_params(clf),
        })
        print(
            f"    -{group_name:>16s}: ev_loss={el:.4f} "
            f"(Δ={el - baseline_loss:+.4f}), acc={acc:.4f}"
        )

    return results


def run_forward_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gaps_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    gaps_test: np.ndarray,
    n_train_games: int,
    n_test_games: int,
    decision_type: str,
    depth: int = 15,
    max_train_records: int = 300_000,
) -> list[dict]:
    """Greedy forward feature selection: start empty, add one feature per step.

    Subsamples training data to max_train_records for speed (forward selection
    trains O(n_features^2) models). Evaluates on full test set for accurate
    EV loss measurement.

    Returns list of {step, feature_idx, feature_name, ev_loss, accuracy}.
    """
    from sklearn.tree import DecisionTreeClassifier

    n_features = X_train.shape[1]
    feat_names = FEATURE_NAMES_REROLL[:n_features] if n_features == 30 else FEATURE_NAMES_CATEGORY

    # Subsample training data for speed
    if len(X_train) > max_train_records:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_train_records, replace=False)
        idx.sort()
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        X_tr = X_train
        y_tr = y_train

    selected: list[int] = []
    remaining = set(range(n_features))
    results = []

    for step in range(n_features):
        best_loss = float("inf")
        best_feat = -1

        for f in remaining:
            trial = selected + [f]
            clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
            clf.fit(X_tr[:, trial], y_tr)
            pred = clf.predict(X_test[:, trial])
            el = ev_loss_per_game(y_test, pred, gaps_test, n_test_games)
            if el < best_loss:
                best_loss = el
                best_feat = f

        if best_feat < 0:
            break

        selected.append(best_feat)
        remaining.discard(best_feat)

        # Compute accuracy for the step
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_tr[:, selected], y_tr)
        pred = clf.predict(X_test[:, selected])
        acc = accuracy(y_test, pred)

        results.append({
            "step": step + 1,
            "feature_idx": best_feat,
            "feature_name": feat_names[best_feat] if best_feat < len(feat_names) else f"feat_{best_feat}",
            "ev_loss": best_loss,
            "accuracy": acc,
            "n_features": len(selected),
        })
        print(
            f"    step {step + 1:>2d}: +{feat_names[best_feat]:>20s} "
            f"→ ev_loss={best_loss:.4f}, acc={acc:.4f}"
        )

        # Early stop if loss is very close to full model
        if best_loss < 0.01:
            break

    return results


def train_with_augmented_features(
    data_dir: Path,
    output_dir: Path,
    depth: int = 20,
) -> dict[str, list[ModelResult]]:
    """Train dt at given depth with base vs augmented features, compare."""
    from sklearn.tree import DecisionTreeClassifier

    from .feature_engineering import augment_features

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[ModelResult]] = {}

    decision_configs = [
        ("category", "category_decisions.bin", 29, 15),
        ("reroll1", "reroll1_decisions.bin", 30, 32),
        ("reroll2", "reroll2_decisions.bin", 30, 32),
    ]

    for dtype, filename, n_features, n_classes in decision_configs:
        path = data_dir / filename
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue

        print(f"\n  Augmented training: {dtype}")
        features, labels, gaps = load_training_data(path)

        # Train/test split
        records_per_game = 15
        n_total_games = len(features) // records_per_game
        n_train_games = int(n_total_games * 0.8)
        n_test_games = n_total_games - n_train_games
        train_end = n_train_games * records_per_game

        X_train, X_test = features[:train_end], features[train_end:]
        y_train, y_test = labels[:train_end], labels[train_end:]
        gaps_train, gaps_test = gaps[:train_end], gaps[train_end:]

        results: list[ModelResult] = []

        # Baseline: original features
        clf_base = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf_base.fit(X_train, y_train)
        pred_base = clf_base.predict(X_test)
        base_loss = ev_loss_per_game(y_test, pred_base, gaps_test, n_test_games)
        base_acc = accuracy(y_test, pred_base)
        results.append(ModelResult(
            name=f"dt_d{depth}_base",
            model_type="dt",
            decision_type=dtype,
            n_params=count_dt_params(clf_base),
            accuracy=base_acc,
            ev_loss=base_loss,
        ))
        print(f"    baseline:  ev_loss={base_loss:.4f}, acc={base_acc:.4f}")

        # Augmented features
        print("    Computing augmented features (this may take a minute)...")
        X_train_aug, aug_names = augment_features(X_train)
        X_test_aug, _ = augment_features(X_test)

        clf_aug = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf_aug.fit(X_train_aug, y_train)
        pred_aug = clf_aug.predict(X_test_aug)
        aug_loss = ev_loss_per_game(y_test, pred_aug, gaps_test, n_test_games)
        aug_acc = accuracy(y_test, pred_aug)
        results.append(ModelResult(
            name=f"dt_d{depth}_augmented",
            model_type="dt",
            decision_type=dtype,
            n_params=count_dt_params(clf_aug),
            accuracy=aug_acc,
            ev_loss=aug_loss,
        ))
        print(
            f"    augmented: ev_loss={aug_loss:.4f}, acc={aug_acc:.4f} "
            f"(Δ={aug_loss - base_loss:+.4f})"
        )

        all_results[dtype] = results

    return all_results
