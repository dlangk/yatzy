#!/usr/bin/env python3
"""Two-tower joint embeddings for Yatzy reroll decisions.

Trains a bilinear two-tower Q-network that jointly embeds game states and
keep-multiset actions into a shared 16D latent space. Visualizes with UMAP
to reveal emergent decision concepts.

Input:  outputs/rosetta/regret_reroll1.bin  (3M records, 121 floats each)
Output: outputs/rosetta/yatzy_joint_manifold.png  (1×3 panel, 30×10 inches)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_SIZE = 32
MAGIC = 0x52455052
N_FEATURES = 56
N_ACTIONS = 32
FLOATS_PER_RECORD = 121  # 56 features + 32 q-values + 1 best_mask + 32 regret
EMBED_DIM = 16
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
N_STATES_VIS = 25_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "outputs" / "rosetta" / "regret_reroll1.bin"
OUT_PATH = BASE / "outputs" / "rosetta" / "yatzy_joint_manifold.png"


# ---------------------------------------------------------------------------
# Step 1: 462 Keep-Multiset Vocabulary
# ---------------------------------------------------------------------------
def build_keep_vocabulary() -> np.ndarray:
    """Enumerate all [f1..f6] with sum <= 5, fi >= 0. Returns (462, 6)."""
    keeps = []
    for f1 in range(6):
        for f2 in range(6 - f1):
            for f3 in range(6 - f1 - f2):
                for f4 in range(6 - f1 - f2 - f3):
                    for f5 in range(6 - f1 - f2 - f3 - f4):
                        for f6 in range(6 - f1 - f2 - f3 - f4 - f5):
                            keeps.append([f1, f2, f3, f4, f5, f6])
    keeps = np.array(keeps, dtype=np.float32)
    assert keeps.shape == (462, 6), f"Expected 462 keeps, got {keeps.shape[0]}"
    return keeps


def build_keep_lookup(vocab: np.ndarray) -> dict[tuple[int, ...], int]:
    """Map (f1,..,f6) tuple → index in vocab."""
    return {tuple(int(v) for v in row): i for i, row in enumerate(vocab)}


# ---------------------------------------------------------------------------
# Step 2: Load data and build training triples
# ---------------------------------------------------------------------------
def load_binary(path: Path) -> np.ndarray:
    """Load regret binary → (N, 121) float32 array."""
    with open(path, "rb") as f:
        magic, version, n_records, n_feat, n_act, _ = struct.unpack("<IIQIIq", f.read(32))
        assert magic == MAGIC, f"Bad magic: {magic:#x}"
        assert n_feat == N_FEATURES and n_act == N_ACTIONS
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_records, FLOATS_PER_RECORD)
    print(f"Loaded {n_records:,} records from {path.name}")
    return data


def dice_from_face_counts(face_counts: np.ndarray) -> list[int]:
    """face_counts[0..6] (for faces 1-6) → sorted list of dice values."""
    dice = []
    for face_idx in range(6):
        count = int(round(face_counts[face_idx]))
        dice.extend([face_idx + 1] * count)
    return sorted(dice)


def kept_dice_counts(dice: list[int], mask: int) -> tuple[int, ...]:
    """Given 5 sorted dice and a 5-bit reroll mask, return 6D count of KEPT dice."""
    counts = [0] * 6
    for bit in range(5):
        if not (mask & (1 << bit)):  # bit=0 → keep
            counts[dice[bit] - 1] += 1
    return tuple(counts)


def build_triples(
    data: np.ndarray, keep_lookup: dict[tuple[int, ...], int], max_triples: int = 5_000_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (state_features, action_index, q_value) triples.

    Returns:
        features: (M, 56) float32
        action_ids: (M,) int64
        q_values: (M,) float32
    """
    feat_list = []
    act_list = []
    q_list = []
    n = len(data)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    count = 0
    for idx in indices:
        row = data[idx]
        features = row[:56]
        q_vals = row[56:88]

        face_counts = features[7:13]
        dice = dice_from_face_counts(face_counts)

        if len(dice) != 5:
            continue

        for mask in range(32):
            q = q_vals[mask]
            if not np.isfinite(q):
                continue
            kc = kept_dice_counts(dice, mask)
            aid = keep_lookup.get(kc)
            if aid is None:
                continue

            feat_list.append(features)
            act_list.append(aid)
            q_list.append(q)
            count += 1

            if count >= max_triples:
                break
        if count >= max_triples:
            break

    print(f"Built {count:,} training triples from {n:,} records")
    return (
        np.array(feat_list, dtype=np.float32),
        np.array(act_list, dtype=np.int64),
        np.array(q_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Step 3: Two-Tower Network
# ---------------------------------------------------------------------------
class TwoTowerQ(nn.Module):
    def __init__(self, n_features: int = 56, n_action_dim: int = 6, embed_dim: int = 16):
        super().__init__()
        self.state_tower = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.action_tower = nn.Sequential(
            nn.Linear(n_action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )
        self.state_bias = nn.Linear(n_features, 1)
        self.action_bias = nn.Linear(n_action_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z_s = self.state_tower(state)
        z_a = self.action_tower(action)
        dot = (z_s * z_a).sum(dim=-1, keepdim=True)
        b_s = self.state_bias(state)
        b_a = self.action_bias(action)
        return (dot + b_s + b_a).squeeze(-1)

    def embed_states(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_tower(state)

    def embed_actions(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_tower(action)


# ---------------------------------------------------------------------------
# Step 4: Semantic Labels for 462 Actions
# ---------------------------------------------------------------------------
def label_keep(counts: np.ndarray) -> str:
    """Classify a 6D kept-dice count vector into a semantic label."""
    s = int(counts.sum())
    mx = int(counts.max())
    sorted_desc = sorted(counts, reverse=True)

    if s == 5:
        return "Keep All"
    if s == 0:
        return "Reroll All"
    if mx == 5:
        return "Yatzy (5×)"
    if mx == 4:
        return "Four of a Kind"
    if mx == 3 and sorted_desc[1] == 2:
        return "Full House"
    if mx == 3:
        return "Triple"
    # Two pair: exactly two faces with count 2
    pairs = sum(1 for c in counts if c == 2)
    if pairs == 2:
        return "Two Pair"
    if pairs == 1 and s == 2:
        # Single pair — check face value
        face = int(np.argmax(counts))  # 0-indexed, face = index + 1
        if face >= 3:  # faces 4,5,6
            return "High Pair (4-6)"
        else:
            return "Low Pair (1-3)"
    # Straight draw: 4+ distinct faces, all counts <= 1
    distinct = int((counts > 0).sum())
    if distinct >= 4 and mx <= 1:
        return "Straight Draw"
    if 2 <= s <= 3 and mx <= 2:
        return "Partial"
    if s == 1:
        return "Single"
    return "Partial"


def label_all_keeps(vocab: np.ndarray) -> list[str]:
    return [label_keep(row) for row in vocab]


# ---------------------------------------------------------------------------
# Step 5 & 6: Train, Extract Embeddings, UMAP
# ---------------------------------------------------------------------------
def train_model(
    features: np.ndarray,
    action_ids: np.ndarray,
    q_values: np.ndarray,
    vocab: np.ndarray,
) -> TwoTowerQ:
    """Train two-tower model on (state, action, q) triples."""
    model = TwoTowerQ().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Convert action IDs to 6D count vectors
    action_vecs = vocab[action_ids]  # (M, 6)

    ds = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(action_vecs),
        torch.from_numpy(q_values),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches = 0
        for s_batch, a_batch, q_batch in loader:
            s_batch = s_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            q_batch = q_batch.to(DEVICE)

            q_pred = model(s_batch, a_batch)
            loss = criterion(q_pred, q_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={avg:.4f}")

    return model


def extract_embeddings(
    model: TwoTowerQ,
    vocab: np.ndarray,
    state_features: np.ndarray,
    n_states: int = N_STATES_VIS,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract action (462, 16) and state (n_states, 16) embeddings."""
    model.eval()
    with torch.no_grad():
        action_emb = model.embed_actions(
            torch.from_numpy(vocab).to(DEVICE)
        ).cpu().numpy()

        rng = np.random.default_rng(123)
        idx = rng.choice(len(state_features), size=min(n_states, len(state_features)), replace=False)
        sampled = state_features[idx]
        state_emb = model.embed_states(
            torch.from_numpy(sampled).to(DEVICE)
        ).cpu().numpy()

    return state_emb, action_emb


def run_umap(state_emb: np.ndarray, action_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint UMAP on concatenated state + action embeddings."""
    joint = np.vstack([state_emb, action_emb])
    print(f"Running UMAP on {joint.shape[0]:,} points ({state_emb.shape[0]:,} states + {action_emb.shape[0]} actions)...")
    reducer = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(joint)
    n_s = state_emb.shape[0]
    return coords[:n_s], coords[n_s:]


# ---------------------------------------------------------------------------
# Step 7: 1×3 Visualization
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "Keep All": "#1f77b4",
    "Reroll All": "#7f7f7f",
    "Yatzy (5×)": "#d62728",
    "Four of a Kind": "#ff7f0e",
    "Full House": "#2ca02c",
    "Triple": "#9467bd",
    "Two Pair": "#8c564b",
    "High Pair (4-6)": "#e377c2",
    "Low Pair (1-3)": "#bcbd22",
    "Straight Draw": "#17becf",
    "Partial": "#aec7e8",
    "Single": "#c7c7c7",
}


def plot_panels(
    state_coords: np.ndarray,
    action_coords: np.ndarray,
    labels: list[str],
    vocab: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel 1: Verb Clusters — actions colored by semantic label
    ax = axes[0]
    for lab in LABEL_COLORS:
        mask = np.array([lb == lab for lb in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(
            action_coords[mask, 0], action_coords[mask, 1],
            c=LABEL_COLORS[lab], s=40, alpha=0.8, label=lab, edgecolors="none",
        )
    ax.set_title("Verb Clusters", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False, markerscale=1.5,
    )

    # Panel 2: Grammar of Value — actions colored by sum(kept_dice)
    ax = axes[1]
    sums = vocab.sum(axis=1)
    sc = ax.scatter(
        action_coords[:, 0], action_coords[:, 1],
        c=sums, cmap="magma", s=40, alpha=0.8, edgecolors="none",
    )
    ax.set_title("Grammar of Value", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Dice Kept", fontsize=10)

    # Panel 3: Joint Orbit — states as gray dots + actions as colored stars
    ax = axes[2]
    ax.scatter(
        state_coords[:, 0], state_coords[:, 1],
        c="#cccccc", s=2, alpha=0.1, rasterized=True,
    )
    for lab in LABEL_COLORS:
        mask = np.array([lb == lab for lb in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(
            action_coords[mask, 0], action_coords[mask, 1],
            c=LABEL_COLORS[lab], s=150, marker="*", edgecolors="black", linewidths=0.5,
        )
    ax.set_title("Joint Orbit", fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    # Small legend for action types
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=LABEL_COLORS[lab],
               markeredgecolor="black", markersize=10, label=lab)
        for lab in LABEL_COLORS
    ]
    handles.insert(0, Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc",
                              markersize=6, label="States", alpha=0.5))
    ax.legend(
        handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Two-Tower Joint Embeddings — Yatzy Reroll Decisions")
    print("=" * 60)

    # Step 1: Build vocabulary
    vocab = build_keep_vocabulary()
    keep_lookup = build_keep_lookup(vocab)
    print(f"Keep vocabulary: {vocab.shape[0]} multisets")

    # Step 2: Load data and build triples
    data = load_binary(DATA_PATH)
    features, action_ids, q_values = build_triples(data, keep_lookup, max_triples=5_000_000)
    del data  # free ~1.4 GB

    # Step 3: Train
    print(f"\nTraining on {DEVICE}...")
    model = train_model(features, action_ids, q_values, vocab)

    # Step 4: Label actions
    labels = label_all_keeps(vocab)
    label_dist = {}
    for lab in labels:
        label_dist[lab] = label_dist.get(lab, 0) + 1
    print("\nAction label distribution:")
    for lab, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {lab:20s} {cnt:4d}")

    # Step 5: Extract embeddings
    state_emb, action_emb = extract_embeddings(model, vocab, features)
    print(f"State embeddings: {state_emb.shape}, Action embeddings: {action_emb.shape}")

    # Step 6: Joint UMAP
    state_coords, action_coords = run_umap(state_emb, action_emb)

    # Step 7: Visualize
    plot_panels(state_coords, action_coords, labels, vocab, OUT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
