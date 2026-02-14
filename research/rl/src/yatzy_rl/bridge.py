"""Python ctypes wrapper for the Rust FFI bridge.

Loads libyatzy_rl_bridge.{dylib,so} and provides a Pythonic interface
for batch turn simulation at Rust speed (~500K games/s vs ~50 games/s in Python).
"""

from __future__ import annotations

import ctypes
import platform
import sys
from pathlib import Path

import numpy as np

# Library file extension
_EXT = {
    "Darwin": "dylib",
    "Linux": "so",
    "Windows": "dll",
}.get(platform.system(), "so")

# Default library search paths (relative to repo root)
_DEFAULT_PATHS = [
    Path(__file__).parents[3] / "bridge" / "target" / "release" / f"libyatzy_rl_bridge.{_EXT}",
    Path("rl/bridge/target/release") / f"libyatzy_rl_bridge.{_EXT}",
]


def _find_library() -> Path | None:
    """Find the bridge shared library."""
    for p in _DEFAULT_PATHS:
        if p.exists():
            return p
    return None


class RustBridge:
    """High-level wrapper around the Rust FFI bridge.

    Provides batch turn simulation and batch game simulation.
    """

    def __init__(self, base_path: str, thetas: list[float]):
        lib_path = _find_library()
        if lib_path is None:
            raise FileNotFoundError(
                f"Bridge library not found. Build it with:\n"
                f"  cd rl/bridge && cargo build --release"
            )

        self._lib = ctypes.CDLL(str(lib_path))
        self._setup_ffi()

        # Initialize context
        thetas_arr = np.array(thetas, dtype=np.float32)
        self._ctx = self._lib.rl_bridge_init(
            base_path.encode("utf-8"),
            thetas_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(thetas),
        )
        if not self._ctx:
            raise RuntimeError("Failed to initialize bridge context")

        self.thetas = thetas
        self.n_thetas = len(thetas)

    def _setup_ffi(self) -> None:
        """Set up ctypes function signatures."""
        lib = self._lib
        INT_P = ctypes.POINTER(ctypes.c_int)
        U64_P = ctypes.POINTER(ctypes.c_uint64)
        FLOAT_P = ctypes.POINTER(ctypes.c_float)

        lib.rl_bridge_init.argtypes = [ctypes.c_char_p, FLOAT_P, ctypes.c_int]
        lib.rl_bridge_init.restype = ctypes.c_void_p

        lib.rl_bridge_free.argtypes = [ctypes.c_void_p]
        lib.rl_bridge_free.restype = None

        lib.rl_bridge_batch_turn.argtypes = [
            ctypes.c_void_p, ctypes.c_int, INT_P, INT_P, INT_P, U64_P,
            ctypes.c_int, INT_P, INT_P, INT_P,
        ]
        lib.rl_bridge_batch_turn.restype = None

        lib.rl_bridge_batch_game.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, U64_P, INT_P,
        ]
        lib.rl_bridge_batch_game.restype = None

        lib.rl_bridge_n_thetas.argtypes = [ctypes.c_void_p]
        lib.rl_bridge_n_thetas.restype = ctypes.c_int

        lib.rl_bridge_theta_at.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rl_bridge_theta_at.restype = ctypes.c_float

        lib.rl_bridge_starting_ev.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.rl_bridge_starting_ev.restype = ctypes.c_float

        # Per-decision functions for Approach B
        lib.rl_bridge_batch_roll.argtypes = [ctypes.c_void_p, ctypes.c_int, U64_P, INT_P]
        lib.rl_bridge_batch_roll.restype = None

        lib.rl_bridge_batch_apply_reroll.argtypes = [
            ctypes.c_void_p, ctypes.c_int, INT_P, INT_P, U64_P, INT_P,
        ]
        lib.rl_bridge_batch_apply_reroll.restype = None

        lib.rl_bridge_batch_score_category.argtypes = [
            ctypes.c_void_p, ctypes.c_int, INT_P, INT_P, INT_P, INT_P, INT_P, INT_P,
        ]
        lib.rl_bridge_batch_score_category.restype = None

        lib.rl_bridge_batch_expert_reroll.argtypes = [
            ctypes.c_void_p, ctypes.c_int, INT_P, INT_P, INT_P,
            ctypes.c_int, ctypes.c_int, INT_P,
        ]
        lib.rl_bridge_batch_expert_reroll.restype = None

        lib.rl_bridge_batch_expert_category.argtypes = [
            ctypes.c_void_p, ctypes.c_int, INT_P, INT_P, INT_P,
            ctypes.c_int, ctypes.c_int, INT_P,
        ]
        lib.rl_bridge_batch_expert_category.restype = None

    def batch_turn(
        self,
        theta_indices: np.ndarray,
        upper_scores: np.ndarray,
        scored_cats: np.ndarray,
        seeds: np.ndarray,
        turn: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate one turn for N games in parallel.

        Args:
            theta_indices: (N,) int32 — which theta table each game uses
            upper_scores: (N,) int32 — current upper score per game
            scored_cats: (N,) int32 — current scored-categories bitmask per game
            seeds: (N,) uint64 — RNG seed per game
            turn: current turn number (0-14)

        Returns:
            categories: (N,) int32 — category scored
            scores: (N,) int32 — score awarded
            new_upper_scores: (N,) int32 — new upper score
        """
        n = len(theta_indices)
        assert len(upper_scores) == n and len(scored_cats) == n and len(seeds) == n

        theta_indices = np.ascontiguousarray(theta_indices, dtype=np.int32)
        upper_scores = np.ascontiguousarray(upper_scores, dtype=np.int32)
        scored_cats = np.ascontiguousarray(scored_cats, dtype=np.int32)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)

        out_cats = np.empty(n, dtype=np.int32)
        out_scores = np.empty(n, dtype=np.int32)
        out_new_ups = np.empty(n, dtype=np.int32)

        self._lib.rl_bridge_batch_turn(
            self._ctx,
            n,
            theta_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            upper_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            scored_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            turn,
            out_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_new_ups.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        return out_cats, out_scores, out_new_ups

    def batch_game(
        self,
        theta_index: int,
        seeds: np.ndarray,
    ) -> np.ndarray:
        """Simulate full games for N seeds with a fixed theta.

        Args:
            theta_index: which theta table to use
            seeds: (N,) uint64 — RNG seeds

        Returns:
            scores: (N,) int32 — final game scores including bonus
        """
        n = len(seeds)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        out_scores = np.empty(n, dtype=np.int32)

        self._lib.rl_bridge_batch_game(
            self._ctx,
            n,
            theta_index,
            seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )

        return out_scores

    def starting_ev(self, theta_index: int) -> float:
        """Get the starting EV for the given theta index."""
        return float(self._lib.rl_bridge_starting_ev(self._ctx, theta_index))

    # Per-decision methods for Approach B

    def batch_roll(self, seeds: np.ndarray) -> np.ndarray:
        """Roll initial dice for N games. Returns (N, 5) int32 sorted dice."""
        n = len(seeds)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        out_dice = np.empty(n * 5, dtype=np.int32)
        self._lib.rl_bridge_batch_roll(
            self._ctx, n,
            seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_dice.reshape(n, 5)

    def batch_apply_reroll(
        self, dice: np.ndarray, masks: np.ndarray, seeds: np.ndarray,
    ) -> np.ndarray:
        """Apply reroll masks. dice: (N,5), masks: (N,), seeds: (N,). Returns (N,5)."""
        n = len(masks)
        dice = np.ascontiguousarray(dice.reshape(-1), dtype=np.int32)
        masks = np.ascontiguousarray(masks, dtype=np.int32)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint64)
        out_dice = np.empty(n * 5, dtype=np.int32)
        self._lib.rl_bridge_batch_apply_reroll(
            self._ctx, n,
            dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            seeds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_dice.reshape(n, 5)

    def batch_score_category(
        self, dice: np.ndarray, categories: np.ndarray,
        upper_scores: np.ndarray, scored_cats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score categories. Returns (scores, new_upper_scores)."""
        n = len(categories)
        dice = np.ascontiguousarray(dice.reshape(-1), dtype=np.int32)
        categories = np.ascontiguousarray(categories, dtype=np.int32)
        upper_scores = np.ascontiguousarray(upper_scores, dtype=np.int32)
        scored_cats = np.ascontiguousarray(scored_cats, dtype=np.int32)
        out_scores = np.empty(n, dtype=np.int32)
        out_new_ups = np.empty(n, dtype=np.int32)
        self._lib.rl_bridge_batch_score_category(
            self._ctx, n,
            dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            categories.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            upper_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            scored_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_new_ups.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_scores, out_new_ups

    def batch_expert_reroll(
        self, dice: np.ndarray, upper_scores: np.ndarray,
        scored_cats: np.ndarray, rerolls_remaining: int, theta_index: int = 0,
    ) -> np.ndarray:
        """Get expert reroll masks. Returns (N,) int32 masks."""
        n = len(upper_scores)
        dice = np.ascontiguousarray(dice.reshape(-1), dtype=np.int32)
        upper_scores = np.ascontiguousarray(upper_scores, dtype=np.int32)
        scored_cats = np.ascontiguousarray(scored_cats, dtype=np.int32)
        out_masks = np.empty(n, dtype=np.int32)
        self._lib.rl_bridge_batch_expert_reroll(
            self._ctx, n,
            dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            upper_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            scored_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            rerolls_remaining, theta_index,
            out_masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_masks

    def batch_expert_category(
        self, dice: np.ndarray, upper_scores: np.ndarray,
        scored_cats: np.ndarray, turn: int, theta_index: int = 0,
    ) -> np.ndarray:
        """Get expert category choices. Returns (N,) int32 categories."""
        n = len(upper_scores)
        dice = np.ascontiguousarray(dice.reshape(-1), dtype=np.int32)
        upper_scores = np.ascontiguousarray(upper_scores, dtype=np.int32)
        scored_cats = np.ascontiguousarray(scored_cats, dtype=np.int32)
        out_cats = np.empty(n, dtype=np.int32)
        self._lib.rl_bridge_batch_expert_category(
            self._ctx, n,
            dice.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            upper_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            scored_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            turn, theta_index,
            out_cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_cats

    def __del__(self) -> None:
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.rl_bridge_free(self._ctx)
            self._ctx = None
