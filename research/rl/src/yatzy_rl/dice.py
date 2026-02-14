"""Dice set enumeration, indexing, and transition probabilities.

Port of backend/src/dice_mechanics.rs and parts of phase0_tables.rs.
Enumerates all C(10,5)=252 sorted 5-dice multisets from {1..6} and builds
the keep-multiset transition table (462 unique keeps in sparse CSR format).
"""

from __future__ import annotations

import numpy as np
from math import factorial

NUM_DICE_SETS = 252
NUM_KEEP_MULTISETS = 462


def build_all_dice_sets() -> tuple[np.ndarray, np.ndarray]:
    """Enumerate all 252 sorted 5-dice multisets and build 5D reverse lookup.

    Returns:
        all_dice_sets: (252, 5) int32 array of sorted dice
        index_lookup: (6, 6, 6, 6, 6) int32 reverse lookup table
    """
    all_dice_sets = np.zeros((252, 5), dtype=np.int32)
    index_lookup = np.full((6, 6, 6, 6, 6), -1, dtype=np.int32)
    idx = 0
    for a in range(1, 7):
        for b in range(a, 7):
            for c in range(b, 7):
                for d in range(c, 7):
                    for e in range(d, 7):
                        all_dice_sets[idx] = [a, b, c, d, e]
                        index_lookup[a - 1, b - 1, c - 1, d - 1, e - 1] = idx
                        idx += 1
    assert idx == 252
    return all_dice_sets, index_lookup


def find_dice_set_index(index_lookup: np.ndarray, dice: np.ndarray) -> int:
    """Map sorted dice to index in R_{5,6} (0-251)."""
    return int(index_lookup[dice[0] - 1, dice[1] - 1, dice[2] - 1, dice[3] - 1, dice[4] - 1])


def compute_dice_set_probabilities(all_dice_sets: np.ndarray) -> np.ndarray:
    """Compute P(empty -> r) for all 252 dice sets. Multinomial formula."""
    probs = np.zeros(252, dtype=np.float64)
    total_outcomes = 6.0**5
    for i in range(252):
        fc = np.zeros(7, dtype=np.int32)
        for d in all_dice_sets[i]:
            fc[d] += 1
        numerator = float(factorial(5))
        denominator = 1.0
        for f in range(1, 7):
            if fc[f] > 1:
                denominator *= float(factorial(int(fc[f])))
        probs[i] = numerator / denominator / total_outcomes
    return probs


class KeepTable:
    """Sparse CSR keep-multiset transition table.

    Stores P(keep -> target_dice_set) for each of 462 unique keep-multisets.
    Also provides per-dice-set dedup mappings (mask -> keep index).
    """

    def __init__(
        self,
        vals: np.ndarray,
        cols: np.ndarray,
        row_start: np.ndarray,
        unique_count: np.ndarray,
        unique_keep_ids: np.ndarray,
        mask_to_keep: np.ndarray,
        keep_to_mask: np.ndarray,
    ):
        self.vals = vals
        self.cols = cols
        self.row_start = row_start
        self.unique_count = unique_count
        self.unique_keep_ids = unique_keep_ids
        self.mask_to_keep = mask_to_keep
        self.keep_to_mask = keep_to_mask

    def row_slice(self, ki: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (vals, cols) for keep-multiset ki."""
        start = self.row_start[ki]
        end = self.row_start[ki + 1]
        return self.vals[start:end], self.cols[start:end]


def build_keep_table(
    all_dice_sets: np.ndarray, index_lookup: np.ndarray
) -> KeepTable:
    """Build the keep-multiset transition table (port of phase0_tables.rs precompute_keep_table).

    Three sub-steps:
    3a: Enumerate all 462 keep-multisets as frequency vectors
    3b: Compute P(K->T) for each keep K and target T (sparse CSR)
    3c: For each (ds, mask), compute dedup mappings
    """
    factorials = np.array([factorial(i) for i in range(6)], dtype=np.int64)

    # 3a: Enumerate keeps
    keep_freq = np.zeros((NUM_KEEP_MULTISETS, 6), dtype=np.int32)
    keep_size = np.zeros(NUM_KEEP_MULTISETS, dtype=np.int32)
    keep_lookup = np.full(6**6, -1, dtype=np.int32)
    num_keeps = 0

    for f1 in range(6):
        for f2 in range(6 - f1):
            for f3 in range(6 - f1 - f2):
                for f4 in range(6 - f1 - f2 - f3):
                    for f5 in range(6 - f1 - f2 - f3 - f4):
                        for f6 in range(6 - f1 - f2 - f3 - f4 - f5):
                            keep_freq[num_keeps] = [f1, f2, f3, f4, f5, f6]
                            keep_size[num_keeps] = f1 + f2 + f3 + f4 + f5 + f6
                            key = ((((f1 * 6 + f2) * 6 + f3) * 6 + f4) * 6 + f5) * 6 + f6
                            keep_lookup[key] = num_keeps
                            num_keeps += 1

    assert num_keeps == NUM_KEEP_MULTISETS

    # 3b: Compute P(K->T) in CSR format
    pow6 = [1, 6, 36, 216, 1296, 7776]
    vals_list: list[float] = []
    cols_list: list[int] = []
    row_start = np.zeros(NUM_KEEP_MULTISETS + 1, dtype=np.int32)

    for ki in range(num_keeps):
        row_start[ki] = len(vals_list)
        n = 5 - keep_size[ki]  # dice rerolled

        if n == 0:
            # Keep all 5: deterministic
            dice = []
            for face in range(6):
                dice.extend([face + 1] * keep_freq[ki, face])
            if len(dice) == 5:
                ti = index_lookup[dice[0] - 1, dice[1] - 1, dice[2] - 1, dice[3] - 1, dice[4] - 1]
                vals_list.append(1.0)
                cols_list.append(int(ti))
            continue

        inv_pow6n = 1.0 / pow6[n]
        fact_n = factorials[n]

        for ti in range(252):
            # Target frequency vector
            tf = np.zeros(6, dtype=np.int32)
            for j in range(5):
                tf[all_dice_sets[ti, j] - 1] += 1

            # Check subset: keep_freq <= target_freq for all faces
            valid = True
            denom = 1
            for f in range(6):
                if keep_freq[ki, f] > tf[f]:
                    valid = False
                    break
                denom *= int(factorials[tf[f] - keep_freq[ki, f]])
            if not valid:
                continue

            vals_list.append(float(fact_n) / denom * inv_pow6n)
            cols_list.append(ti)

    row_start[num_keeps] = len(vals_list)

    vals = np.array(vals_list, dtype=np.float64)
    cols = np.array(cols_list, dtype=np.int32)

    # 3c: Per dice-set dedup
    unique_count = np.zeros(NUM_DICE_SETS, dtype=np.int32)
    unique_keep_ids = np.zeros((NUM_DICE_SETS, 31), dtype=np.int32)
    mask_to_keep = np.full(NUM_DICE_SETS * 32, -1, dtype=np.int32)
    keep_to_mask = np.zeros(NUM_DICE_SETS * 32, dtype=np.int32)

    for ds in range(252):
        dice = all_dice_sets[ds]
        seen: list[int] = []
        n_unique = 0

        for mask in range(1, 32):
            # Frequency vector of kept dice (bits NOT set)
            kf = np.zeros(6, dtype=np.int32)
            for i in range(5):
                if (mask & (1 << i)) == 0:
                    kf[dice[i] - 1] += 1
            key = ((((kf[0] * 6 + kf[1]) * 6 + kf[2]) * 6 + kf[3]) * 6 + kf[4]) * 6 + kf[5]
            kid = int(keep_lookup[key])
            mask_to_keep[ds * 32 + mask] = kid

            if kid not in seen:
                seen.append(kid)
                unique_keep_ids[ds, n_unique] = kid
                keep_to_mask[ds * 32 + n_unique] = mask
                n_unique += 1

        unique_count[ds] = n_unique

        # mask=0: keep all
        kf_all = np.zeros(6, dtype=np.int32)
        for i in range(5):
            kf_all[dice[i] - 1] += 1
        key = ((((kf_all[0] * 6 + kf_all[1]) * 6 + kf_all[2]) * 6 + kf_all[3]) * 6 + kf_all[4]) * 6 + kf_all[5]
        mask_to_keep[ds * 32] = int(keep_lookup[key])

    return KeepTable(
        vals=vals,
        cols=cols,
        row_start=row_start,
        unique_count=unique_count,
        unique_keep_ids=unique_keep_ids,
        mask_to_keep=mask_to_keep,
        keep_to_mask=keep_to_mask,
    )
