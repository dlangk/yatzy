"""Scenario card: composite scorecard + decision analysis image.

Generates a single-page PNG per scenario combining:
- Left: monospace game-state block (always includes full scorecard)
- Right top: signed-advantage chart showing decision crossover across theta
- Right bottom: per-theta decision table
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# All sorted 5d6 combinations (252 entries, index 0–251)
SORTED_DICE = list(combinations_with_replacement(range(1, 7), 5))

_DTYPE_STR_TO_INT = {"reroll1": 0, "reroll2": 1, "category": 2}
_DTYPE_INT_TO_STR = {v: k for k, v in _DTYPE_STR_TO_INT.items()}


def encode_scenario_id(scenario: dict) -> str:
    """Encode a scenario's game state into a deterministic 8-char hex ID.

    Bit layout of the u32:
        bits [30..16] = scored_categories  (15 bits)
        bits [15..10] = upper_score        (6 bits)
        bits  [9..2]  = dice_index         (8 bits)
        bits  [1..0]  = decision_type      (2 bits)
    """
    scored = scenario["scored_categories"]
    upper = scenario["upper_score"]
    dice_idx = SORTED_DICE.index(tuple(sorted(scenario["dice"])))
    dtype = _DTYPE_STR_TO_INT[scenario["decision_type"]]
    packed = (scored << 16) | (upper << 10) | (dice_idx << 2) | dtype
    return f"{packed:08X}"


def decode_scenario_id(sid: str) -> dict:
    """Decode an 8-char hex scenario ID back to game state fields."""
    packed = int(sid, 16)
    dtype = packed & 0x3
    dice_idx = (packed >> 2) & 0xFF
    upper = (packed >> 10) & 0x3F
    scored = (packed >> 16) & 0x7FFF
    return {
        "scored_categories": scored,
        "upper_score": upper,
        "dice": list(SORTED_DICE[dice_idx]),
        "decision_type": _DTYPE_INT_TO_STR[dtype],
    }

CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

# Unicode die faces: index 1-6
DIE_FACES = {1: "\u2680", 2: "\u2681", 3: "\u2682", 4: "\u2683", 5: "\u2684", 6: "\u2685"}

# Fixed figure dimensions (all cards identical)
CARD_WIDTH = 15
CARD_HEIGHT = 11


def compute_score(dice: list[int], cat: int) -> int:
    """Compute Scandinavian Yatzy score for a category given 5 dice."""
    counts = Counter(dice)
    total = sum(dice)
    sd = sorted(dice)

    if cat < 6:
        return counts.get(cat + 1, 0) * (cat + 1)
    if cat == 6:  # One Pair
        for f in range(6, 0, -1):
            if counts.get(f, 0) >= 2:
                return f * 2
        return 0
    if cat == 7:  # Two Pairs
        ps = [f for f in range(6, 0, -1) if counts.get(f, 0) >= 2]
        return (ps[0] * 2 + ps[1] * 2) if len(ps) >= 2 else 0
    if cat == 8:  # Three of a Kind
        for f in range(6, 0, -1):
            if counts.get(f, 0) >= 3:
                return f * 3
        return 0
    if cat == 9:  # Four of a Kind
        for f in range(6, 0, -1):
            if counts.get(f, 0) >= 4:
                return f * 4
        return 0
    if cat == 10:  # Small Straight
        return 15 if sd == [1, 2, 3, 4, 5] else 0
    if cat == 11:  # Large Straight
        return 20 if sd == [2, 3, 4, 5, 6] else 0
    if cat == 12:  # Full House
        vs = sorted(counts.values())
        return total if vs == [2, 3] else 0
    if cat == 13:  # Chance
        return total
    if cat == 14:  # Yatzy
        return 50 if len(counts) == 1 else 0
    return 0


def _is_scored(scored_mask: int, cat: int) -> bool:
    return (scored_mask & (1 << cat)) != 0


def _build_state_text(scenario: dict, scenario_id: str = "") -> str:
    """Build monospace text block with full scorecard, always shown."""
    dice = scenario["dice"]
    turn = scenario["turn"]
    upper_score = scenario["upper_score"]
    scored = scenario["scored_categories"]
    dtype = scenario["decision_type"]
    phase = scenario["game_phase"]
    theta0_id = scenario["theta_0_action_id"]
    flip_id = scenario.get("flip_action_id", -1)

    lines: list[str] = []

    # Scenario ID
    if scenario_id:
        lines.append(f"  Scenario {scenario_id}")
        lines.append("")

    # Dice
    faces = "  ".join(DIE_FACES.get(d, str(d)) for d in dice)
    nums = ", ".join(str(d) for d in dice)
    lines.append(f"  {faces}")
    lines.append(f"  [{nums}]")
    lines.append("")

    # Game context
    n_scored = bin(scored).count("1")
    lines.append(f"  Turn {turn + 1} of 15  ({phase} game)")
    lines.append(f"  Scored: {n_scored}/15 categories")
    if upper_score >= 63:
        lines.append(f"  Upper:  {upper_score}/63 (bonus!)")
    else:
        lines.append(f"  Upper:  {upper_score}/63")
    lines.append("")

    # Decision type header
    dtype_label = dtype.replace("reroll", "Reroll ").title()
    if dtype == "category":
        dtype_label = "Category Choice"
    lines.append(f"  Decision: {dtype_label}")
    lines.append("")

    # Always show the full scorecard
    _append_category_table(lines, dice, scored, theta0_id, flip_id, dtype)

    # Decision summary
    lines.append("")
    lines.append(f"  EV-optimal:    {scenario['theta_0_action']}")
    if scenario.get("has_flip"):
        ft = scenario["flip_theta"]
        lines.append(f"  Flips at:      theta = {ft}")
        lines.append(f"  Risk-seeking:  {scenario['flip_action']}")
    else:
        lines.append(f"  Flips at:      (none)")
        lines.append(f"  Risk-seeking:  (same)")
    lines.append(f"  Visit freq:    {scenario['visit_count']:,} / 100K")
    if "state_frequency" in scenario:
        lines.append(f"  Board freq:    {scenario['state_frequency']:,} / 100K decisions")

    return "\n".join(lines)


def _append_category_table(
    lines: list[str],
    dice: list[int],
    scored: int,
    theta0_id: int,
    flip_id: int,
    dtype: str,
) -> None:
    """Append a category score table to the text lines.

    For category decisions, marks the EV-optimal and flip choices.
    For reroll decisions, shows scores without markers.
    """
    show_markers = dtype == "category"

    lines.append("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    lines.append("  \u2502 Category         \u2502 Score \u2502      \u2502")
    lines.append("  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2524")

    for c in range(15):
        name = CATEGORY_NAMES[c]
        if _is_scored(scored, c):
            lines.append(f"  \u2502 {name:<16} \u2502   \u2713   \u2502      \u2502")
        else:
            scr = compute_score(dice, c)
            marker = ""
            if show_markers:
                if c == theta0_id:
                    marker = "\u25c0 EV"
                elif c == flip_id:
                    marker = "\u25c0 \u03b8+"
            lines.append(f"  \u2502 {name:<16} \u2502 {scr:>5} \u2502 {marker:<4} \u2502")

        # Separator after upper section
        if c == 5:
            lines.append("  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                         "\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2524")

    lines.append("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2518")


def _plot_advantage_chart(scenario: dict, ax: plt.Axes) -> None:
    """Plot signed advantage of the theta=0 action across theta values.

    Positive = theta=0 action is still best.
    Negative = alternative has taken over.
    Zero crossing = flip point.

    Excludes theta=0 (EV-domain, different scale from log-domain theta>0).
    """
    results = scenario["theta_results"]
    theta0_id = scenario["theta_0_action_id"]

    # Separate theta=0 from theta>0 (different value domains)
    results_pos = [r for r in results if r["theta"] > 0]
    if not results_pos:
        return

    thetas = [r["theta"] for r in results_pos]
    signed_gaps = []
    for r in results_pos:
        if r["action_id"] == theta0_id:
            signed_gaps.append(r["gap"])
        else:
            signed_gaps.append(-r["gap"])

    thetas_arr = np.array(thetas)
    gaps_arr = np.array(signed_gaps)

    # Filled regions
    ax.fill_between(
        thetas_arr, gaps_arr, 0,
        where=gaps_arr >= 0, interpolate=True,
        color="#2ca02c", alpha=0.15, label=None,
    )
    ax.fill_between(
        thetas_arr, gaps_arr, 0,
        where=gaps_arr < 0, interpolate=True,
        color="#d62728", alpha=0.15, label=None,
    )

    # Line + markers
    colors = ["#2ca02c" if g >= 0 else "#d62728" for g in signed_gaps]
    ax.plot(thetas_arr, gaps_arr, color="#333333", linewidth=1.8, zorder=4)
    for t, g, c in zip(thetas, signed_gaps, colors):
        ax.plot(t, g, "o", color=c, markersize=7, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

    # Zero line
    ax.axhline(y=0, color="#888888", linewidth=0.8, linestyle="-", zorder=2)

    # Flip vertical line (no arrow/annotation)
    if scenario.get("has_flip"):
        flip_theta = scenario["flip_theta"]
        ax.axvline(x=flip_theta, color="#d62728", linewidth=1.5,
                   linestyle="--", alpha=0.6, zorder=3)

    # Labels
    ax.set_xlabel("theta  (risk preference)", fontsize=11)
    ax.set_ylabel("Advantage of EV-optimal action", fontsize=11)

    # Action legend
    theta0_action = scenario["theta_0_action"]
    flip_action = scenario.get("flip_action", "")

    green_patch = mpatches.Patch(color="#2ca02c", alpha=0.5, label=f"{theta0_action} (EV-optimal)")
    red_patch = mpatches.Patch(color="#d62728", alpha=0.5, label=f"{flip_action} (risk-seeking)")
    ax.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=9,
              framealpha=0.9)

    # theta=0 note
    r0 = next((r for r in results if r["theta"] == 0.0), None)
    if r0:
        ax.text(
            0.02, 0.04,
            f"At theta=0: {r0['action']} wins by {r0['gap']:.2f} EV pts",
            transform=ax.transAxes, fontsize=8.5, color="#555555",
            style="italic",
        )

    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max(thetas) * 1.05)


def _plot_theta_table(scenario: dict, ax: plt.Axes) -> None:
    """Plot a small colored table showing per-theta decisions."""
    results = scenario["theta_results"]
    theta0_id = scenario["theta_0_action_id"]

    ax.axis("off")

    col_labels = ["theta", "Best action", "Runner-up", "Gap"]
    cell_text = []
    cell_colors = []

    for r in results:
        is_theta0 = r["action_id"] == theta0_id
        bg = "#e8f5e9" if is_theta0 else "#ffebee"

        theta_str = f"{r['theta']:.3f}" if r["theta"] != 0.0 else "0 (EV)"
        cell_text.append([
            theta_str,
            r["action"],
            r["runner_up"],
            f"{r['gap']:.4f}",
        ])
        cell_colors.append([bg] * 4)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colWidths=[0.12, 0.35, 0.35, 0.18],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e0e0e0")


def plot_scenario_card(
    scenario: dict,
    out_path: Path,
    scenario_id: str = "",
    dpi: int = 200,
) -> None:
    """Generate a composite scenario card image.

    Fixed layout — every card has identical dimensions.
    Left: monospace scorecard. Right top: advantage chart. Right bottom: theta table.
    """
    fig = plt.figure(figsize=(CARD_WIDTH, CARD_HEIGHT), facecolor="white")

    # Fixed grid: 2 rows x 2 cols
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.8, 1.4],
        height_ratios=[1.3, 1],
        hspace=0.25, wspace=0.05,
        left=0.02, right=0.98, top=0.88, bottom=0.04,
    )

    # Title
    dtype_label = scenario["decision_type"].replace("reroll", "Reroll ").title()
    if scenario["decision_type"] == "category":
        dtype_label = "Category Choice"
    turn = scenario["turn"] + 1
    dice_str = ", ".join(str(d) for d in scenario["dice"])

    id_prefix = f"[{scenario_id}]  " if scenario_id else ""
    fig.suptitle(
        f"{id_prefix}Decision Sensitivity: {dtype_label} at Turn {turn}    "
        f"Dice [{dice_str}]",
        fontsize=15, fontweight="bold", x=0.5, y=0.96,
    )

    # Subtitle
    theta0_action = scenario["theta_0_action"]
    flip_action = scenario.get("flip_action", "")
    flip_theta = scenario.get("flip_theta", 0)
    if scenario.get("has_flip"):
        subtitle = (
            f"EV-optimal: {theta0_action}   \u2192   "
            f"Risk-seeking (theta >= {flip_theta}): {flip_action}"
        )
    else:
        subtitle = f"EV-optimal: {theta0_action}   (no flip in theta 0-0.2)"
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=12, color="#555555")

    # Left: monospace scorecard (spans both rows, fixed position)
    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    text = _build_state_text(scenario, scenario_id=scenario_id)
    ax_text.text(
        0.03, 0.97, text,
        transform=ax_text.transAxes,
        fontfamily="monospace", fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="square,pad=0.6", facecolor="#f8f9fa",
                  edgecolor="#dee2e6", linewidth=1),
    )

    # Right top: advantage chart (fixed axes)
    ax_chart = fig.add_subplot(gs[0, 1])
    _plot_advantage_chart(scenario, ax_chart)

    # Right bottom: per-theta table (fixed axes)
    ax_table = fig.add_subplot(gs[1, 1])
    _plot_theta_table(scenario, ax_table)

    # Save with fixed size — no bbox_inches="tight" so dimensions are constant
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_example_cards(
    flips_path: Path,
    out_dir: Path,
    n: int = 5,
    dpi: int = 200,
) -> list[Path]:
    """Generate scenario cards for the N most interesting flips."""
    with open(flips_path) as f:
        flips = json.load(f)

    # Score scenarios: prefer small gap at theta=0 (tight decisions), high state frequency,
    # and category decisions (most intuitive)
    def interest_score(s: dict) -> float:
        gap = s.get("gap_at_theta0", 999)
        state_freq = s.get("state_frequency", s.get("visit_count", 1))
        type_bonus = {"category": 2.0, "reroll2": 1.0, "reroll1": 0.5}
        tb = type_bonus.get(s.get("decision_type", ""), 0)
        gap_score = 1.0 / (gap + 0.01) if gap > 0.001 else 0.5
        return gap_score * (state_freq / 1000) * (1 + tb)

    ranked = sorted(flips, key=interest_score, reverse=True)

    # Phase-diverse selection: pick top candidates from each game phase
    import math
    per_phase = math.ceil(n / 3)
    phase_groups: dict[str, list[dict]] = {"early": [], "mid": [], "late": []}
    for s in ranked:
        phase = s.get("game_phase", "late")
        if phase in phase_groups:
            phase_groups[phase].append(s)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    for phase in ["early", "mid", "late"]:
        for s in phase_groups[phase][:per_phase]:
            sid = encode_scenario_id(s)
            if sid not in seen_ids:
                selected.append(s)
                seen_ids.add(sid)

    # Fill remaining slots from overall ranking (no duplicates)
    for s in ranked:
        if len(selected) >= n:
            break
        sid = encode_scenario_id(s)
        if sid not in seen_ids:
            selected.append(s)
            seen_ids.add(sid)

    # Re-sort by interest_score
    selected.sort(key=interest_score, reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, scenario in enumerate(selected[:n]):
        scenario_id = encode_scenario_id(scenario)
        out_path = out_dir / f"scenario_card_{scenario_id}.png"
        plot_scenario_card(scenario, out_path, scenario_id=scenario_id, dpi=dpi)
        paths.append(out_path)
        dice_str = ",".join(str(d) for d in scenario["dice"])
        print(
            f"  {out_path.name} ({scenario_id}): Turn {scenario['turn']+1} "
            f"[{dice_str}] {scenario['decision_type']} "
            f"{scenario['theta_0_action']} -> {scenario['flip_action']}"
        )

    return paths
