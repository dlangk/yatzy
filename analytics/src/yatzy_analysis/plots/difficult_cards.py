"""Difficult scenario cards: composite scorecard + action EV comparison.

Generates a single PNG per scenario combining:
- Left: monospace game-state block with full scorecard
- Right top: horizontal bar chart of all action EVs
- Right bottom: action details table
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .scenario_cards import (
    CARD_HEIGHT,
    CARD_WIDTH,
    CATEGORY_NAMES,
    DIE_FACES,
    _is_scored,
    compute_score,
    encode_scenario_id,
)


def _build_state_text(scenario: dict) -> str:
    """Build compact monospace text block: game state + scorecard only.

    Info shown in the title/subtitle is NOT repeated here.
    """
    dice = scenario["dice"]
    upper_score = scenario["upper_score"]
    scored = scenario["scored_categories"]
    dtype = scenario["decision_type"]

    best_action = scenario["best_action"]
    runner_up = scenario["runner_up_action"]

    lines: list[str] = []

    # Game context (turn/decision/dice/rank/best/gap/visits are in title/subtitle)
    n_scored = bin(scored).count("1")
    lines.append(f"  Scored: {n_scored}/15 categories")
    if upper_score >= 63:
        lines.append(f"  Upper:  {upper_score}/63 (bonus!)")
    else:
        lines.append(f"  Upper:  {upper_score}/63")
    lines.append("")

    # Category table
    show_markers = dtype == "category"
    best_id = best_action.get("id", best_action.get("action_id")) if show_markers else -1
    ru_id = runner_up.get("id", runner_up.get("action_id")) if show_markers else -1

    # Box-drawing table: Category(18) | Scored(7) | Avail(7) | marker(6)
    lines.append("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u252c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
    lines.append("  \u2502 Category         "
                 "\u2502Scored "
                 "\u2502 Avail "
                 "\u2502      \u2502")
    lines.append("  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u253c"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2524")

    cat_scores = scenario.get("category_scores")
    for c in range(15):
        name = CATEGORY_NAMES[c]
        if _is_scored(scored, c):
            if cat_scores and cat_scores[c] >= 0:
                scored_str = f" {cat_scores[c]:>5} "
            else:
                scored_str = "   \u2713   "
            lines.append(
                f"  \u2502 {name:<16} \u2502{scored_str}"
                f"\u2502       \u2502      \u2502"
            )
        else:
            scr = compute_score(dice, c)
            marker = ""
            if show_markers:
                if c == best_id:
                    marker = "\u25c0 1st"
                elif c == ru_id:
                    marker = "\u25c0 2nd"
            lines.append(
                f"  \u2502 {name:<16} \u2502       "
                f"\u2502 {scr:>5} "
                f"\u2502 {marker:<5}\u2502"
            )
        if c == 5:
            lines.append(
                "  \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u2500\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                "\u253c\u2500\u2500\u2500\u2500\u2500\u2500\u2524"
            )

    lines.append("  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2534"
                 "\u2500\u2500\u2500\u2500\u2500\u2500\u2518")

    # Unique-to-panel metric
    lines.append("")
    lines.append(f"  Difficulty: {scenario['difficulty_score']:,.0f}")

    return "\n".join(lines)


def _plot_action_bars(scenario: dict, ax: plt.Axes) -> None:
    """Horizontal bar chart showing EV of all actions, best highlighted."""
    actions = scenario["all_actions"]

    # Show top 15 actions max to keep readable
    show = actions[:15]
    show = list(reversed(show))  # matplotlib barh draws bottom-up

    names = [a.get("label", a.get("action_name")) for a in show]
    evs = [a["ev"] for a in show]
    best_ev = scenario["best_action"]["ev"]
    ru_ev = scenario["runner_up_action"]["ev"]

    colors = []
    for a in show:
        if a["ev"] == best_ev:
            colors.append("#2ca02c")
        elif a["ev"] == ru_ev:
            colors.append("#ff7f0e")
        else:
            colors.append("#8fb0d4")

    y_pos = np.arange(len(show))
    bars = ax.barh(y_pos, evs, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Expected Value", fontsize=10)

    # Narrow x-range to emphasize differences
    ev_min = min(evs)
    ev_max = max(evs)
    ev_range = ev_max - ev_min
    if ev_range > 0:
        ax.set_xlim(ev_min - ev_range * 0.3, ev_max + ev_range * 0.15)
    else:
        ax.set_xlim(ev_max - 1, ev_max + 1)

    # Value labels on bars
    for bar, ev in zip(bars, evs):
        ax.text(
            bar.get_width() + ev_range * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{ev:.2f}",
            va="center", fontsize=7.5,
        )

    # Gap annotation
    gap = scenario["ev_gap"]
    ax.text(
        0.98, 0.02,
        f"Gap: {gap:.4f} EV pts",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontstyle="italic", color="#555555",
    )

    if len(actions) > 15:
        ax.text(
            0.98, 0.07,
            f"({len(actions)} total actions, showing top 15)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#999999",
        )

    ax.grid(True, axis="x", alpha=0.3)


def _plot_action_table(scenario: dict, ax: plt.Axes) -> None:
    """Table showing top actions with EVs and deltas."""
    actions = scenario["all_actions"]
    best_ev = scenario["best_action"]["ev"]

    show = actions[:10]
    ax.axis("off")

    col_labels = ["#", "Action", "EV", "Delta"]
    cell_text = []
    cell_colors = []

    for i, a in enumerate(show):
        delta = a["ev"] - best_ev
        if i == 0:
            bg = "#e8f5e9"
        elif i == 1:
            bg = "#fff3e0"
        else:
            bg = "#ffffff"
        cell_text.append([
            str(i + 1),
            a.get("label", a.get("action_name")),
            f"{a['ev']:.2f}",
            f"{delta:+.4f}" if i > 0 else "best",
        ])
        cell_colors.append([bg] * 4)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colWidths=[0.06, 0.50, 0.20, 0.24],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.2)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e0e0e0")


def plot_difficult_card(
    scenario: dict,
    out_path: Path,
    scenario_id: str = "",
    dpi: int = 150,
) -> None:
    """Generate a composite difficult-scenario card image."""
    fig = plt.figure(figsize=(CARD_WIDTH, CARD_HEIGHT), facecolor="white")

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.8, 1.4],
        height_ratios=[1.3, 1],
        hspace=0.25, wspace=0.01,
        left=0.01, right=0.98, top=0.88, bottom=0.04,
    )

    # Title
    dtype = scenario["decision_type"]
    dtype_label = {
        "reroll2": "After first roll (2 rerolls left)",
        "reroll1": "After second roll (1 reroll left)",
        "category": "Final dice \u2014 choose category",
    }.get(dtype, dtype)
    turn = scenario["turn"] + 1
    dice_str = ", ".join(str(d) for d in scenario["dice"])

    id_prefix = f"[{scenario_id}]  " if scenario_id else ""
    fig.suptitle(
        f"{id_prefix}Difficult Scenario #{scenario['rank']}: "
        f"{dtype_label} at Turn {turn}    Dice [{dice_str}]",
        fontsize=14, fontweight="bold", x=0.5, y=0.96,
    )

    best = scenario["best_action"].get("label", scenario["best_action"].get("action_name"))
    runner_up = scenario["runner_up_action"].get("label", scenario["runner_up_action"].get("action_name"))
    gap = scenario["ev_gap"]
    fig.text(
        0.5, 0.92,
        f"Best: {best}  vs  Runner-up: {runner_up}  "
        f"(gap: {gap:.4f} EV pts, {scenario['visit_count']:,} visits)",
        ha="center", fontsize=11, color="#555555",
    )

    # Left: big dice + monospace scorecard (spans both rows)
    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")

    # Large dice faces at top
    dice = scenario["dice"]
    faces = "  ".join(DIE_FACES.get(d, str(d)) for d in dice)
    ax_text.text(
        0.06, 0.97, faces,
        transform=ax_text.transAxes,
        fontsize=28, verticalalignment="top",
    )

    # Compact state text below dice
    text = _build_state_text(scenario)
    ax_text.text(
        0.03, 0.88, text,
        transform=ax_text.transAxes,
        fontfamily="monospace", fontsize=9.5,
        verticalalignment="top",
        bbox=dict(
            boxstyle="square,pad=0.6", facecolor="#f8f9fa",
            edgecolor="#dee2e6", linewidth=1,
        ),
    )

    # Right top: action EV bar chart
    ax_bars = fig.add_subplot(gs[0, 1])
    _plot_action_bars(scenario, ax_bars)

    # Right bottom: action table
    ax_table = fig.add_subplot(gs[1, 1])
    _plot_action_table(scenario, ax_table)

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_difficult_cards(
    json_path: Path,
    out_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """Generate scenario cards for all difficult scenarios in the JSON file."""
    with open(json_path) as f:
        scenarios = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for scenario in scenarios:
        scenario_id = encode_scenario_id(scenario)
        out_path = out_dir / f"scenario_{scenario_id}.png"
        plot_difficult_card(scenario, out_path, scenario_id=scenario_id, dpi=dpi)
        paths.append(out_path)

    print(f"  Generated {len(paths)} scenario cards in {out_dir}")
    return paths
