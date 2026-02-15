"""State-flow visualizations for Yatzy board-state frequency data.

Three complementary views of how 100K optimal games traverse the state space:

1. Bonus alluvial — stacked area showing upper bonus trajectory
2. State DAG — pruned layered graph of the most frequent states
3. Category streamgraph — when each category gets filled
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import PchipInterpolator

from .style import CATEGORY_NAMES as CAT_NAMES, CATEGORY_SHORT as CAT_SHORT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_GROUPS: dict[str, list[int]] = {
    "Upper low (1-3)": [0, 1, 2],
    "Upper high (4-6)": [3, 4, 5],
    "Pairs": [6, 7],
    "Trips & Quads": [8, 9],
    "Straights": [10, 11],
    "Full House": [12],
    "Chance": [13],
    "Yatzy": [14],
}

BONUS_GROUPS = ["No upper yet", "Below pace", "Above pace", "Bonus secured", "Bonus lost"]

BONUS_COLORS = {
    "Bonus secured": "#4caf50",
    "Above pace": "#42a5f5",
    "Below pace": "#ffa726",
    "No upper yet": "#bdbdbd",
    "Bonus lost": "#ef5350",
}

META_COLORS = {
    "Upper low (1-3)": "#f4a261",
    "Upper high (4-6)": "#e76f51",
    "Pairs": "#457b9d",
    "Trips & Quads": "#1d3557",
    "Straights": "#2a9d8f",
    "Full House": "#9b5de5",
    "Chance": "#adb5bd",
    "Yatzy": "#e63946",
}

# ---------------------------------------------------------------------------
# Data loading and classification
# ---------------------------------------------------------------------------


def _load_state_frequency(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "turn": int(r["turn"]),
                "scored": int(r["scored_categories"]),
                "upper": int(r["upper_score"]),
                "count": int(r["visit_count"]),
            })
    return rows


def _classify_bonus(upper: int, scored: int) -> str:
    if upper >= 63:
        return "Bonus secured"
    remaining_max = sum((i + 1) * 5 for i in range(6) if not (scored & (1 << i)))
    if upper + remaining_max < 63:
        return "Bonus lost"
    n_upper_filled = sum(1 for i in range(6) if scored & (1 << i))
    if n_upper_filled == 0:
        return "No upper yet"
    pace = 10.5 * n_upper_filled
    return "Above pace" if upper >= pace else "Below pace"


def _by_turn(rows: list[dict]) -> dict[int, list[dict]]:
    d: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        d[r["turn"]].append(r)
    return d


# ---------------------------------------------------------------------------
# Option 1: Bonus Alluvial (stacked area)
# ---------------------------------------------------------------------------


def plot_bonus_alluvial(data_path: Path, out_path: Path, dpi: int = 200) -> None:
    """Stacked area showing upper bonus trajectory across 15 turns."""
    rows = _load_state_frequency(data_path)
    bt = _by_turn(rows)
    turns = sorted(bt.keys())

    # Compute fractions per group per turn
    fractions = {g: [] for g in BONUS_GROUPS}
    for t in turns:
        total = sum(r["count"] for r in bt[t])
        gc = defaultdict(int)
        for r in bt[t]:
            gc[_classify_bonus(r["upper"], r["scored"])] += r["count"]
        for g in BONUS_GROUPS:
            fractions[g].append(gc[g] / total if total > 0 else 0.0)

    x_raw = np.array([t + 1 for t in turns], dtype=float)
    x_smooth = np.linspace(1, 15, 300)

    # PCHIP smooth interpolation, then renormalize
    smooth = {}
    for g in BONUS_GROUPS:
        interp = PchipInterpolator(x_raw, fractions[g])
        smooth[g] = np.clip(interp(x_smooth), 0, 1)
    totals = sum(smooth[g] for g in BONUS_GROUPS)
    totals = np.where(totals > 0, totals, 1)
    for g in BONUS_GROUPS:
        smooth[g] /= totals

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    # Stack order: bottom to top = lost, below, no-upper, above, secured
    stack_order = ["Bonus lost", "Below pace", "No upper yet", "Above pace", "Bonus secured"]
    bottom = np.zeros_like(x_smooth)
    for g in stack_order:
        vals = smooth[g]
        ax.fill_between(x_smooth, bottom, bottom + vals,
                        color=BONUS_COLORS[g], alpha=0.85, linewidth=0)
        ax.plot(x_smooth, bottom + vals, color="white", linewidth=0.6)
        bottom += vals

    # Annotations at turn 15
    y_accum = 0.0
    for g in stack_order:
        val = fractions[g][-1]
        if val > 0.03:
            y_mid = y_accum + val / 2
            label = f"{val:.0%}"
            ax.annotate(label, xy=(15.1, y_mid), fontsize=10, fontweight="bold",
                        color=BONUS_COLORS[g], va="center")
        y_accum += val

    ax.set_xlim(1, 15.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 16))
    ax.set_xlabel("Turn", fontsize=13)
    ax.set_ylabel("Fraction of games", fontsize=13)
    ax.set_title("Upper Bonus Trajectory — 100K Optimal Games",
                 fontsize=15, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    handles = [mpatches.Patch(color=BONUS_COLORS[g], label=g) for g in stack_order
               if max(fractions[g]) > 0.005]
    ax.legend(handles=handles[::-1], loc="upper left", fontsize=11,
              framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Option 2: Pruned State DAG
# ---------------------------------------------------------------------------


def _top_k(states: list[dict], k: int) -> tuple[list[dict], int]:
    """Return top-k states and the 'other' aggregate count."""
    ranked = sorted(states, key=lambda s: s["count"], reverse=True)
    top = ranked[:k]
    other = sum(s["count"] for s in ranked[k:])
    return top, other


def _state_label(state: dict, turn: int, total_turns: int = 15) -> str:
    """Short label for a state node."""
    scored = state["scored"]
    n = bin(scored).count("1")
    upper = state["upper"]
    if n <= 3:
        cats = [CAT_SHORT[i] for i in range(15) if scored & (1 << i)]
        return ",".join(cats) if cats else "start"
    elif n >= 12:
        missing = [CAT_SHORT[i] for i in range(15) if not (scored & (1 << i))]
        return f"−{'−'.join(missing)}"
    else:
        n_up = sum(1 for i in range(6) if scored & (1 << i))
        return f"u{upper}/{n_up}up"


def _bezier_ribbon(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha):
    """Draw a smooth ribbon (filled bezier) between two vertical extents."""
    dx = (x1 - x0) * 0.4
    verts = [
        (x0, y0_bot), (x0 + dx, y0_bot), (x1 - dx, y1_bot), (x1, y1_bot),  # bottom curve
        (x1, y1_top), (x1 - dx, y1_top), (x0 + dx, y0_top), (x0, y0_top),  # top curve (reversed)
        (x0, y0_bot),  # close
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(MplPath(verts, codes), facecolor=color,
                               edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(patch)


def plot_state_dag(data_path: Path, out_path: Path, dpi: int = 200) -> None:
    """Layered DAG showing the most frequent states per turn with transitions."""
    rows = _load_state_frequency(data_path)
    bt = _by_turn(rows)
    turns = sorted(bt.keys())

    K = 8  # top-K per turn
    node_data: dict[int, list[dict]] = {}  # turn -> list of top states
    other_counts: dict[int, int] = {}

    for t in turns:
        top, other = _top_k(bt[t], K)
        node_data[t] = top
        other_counts[t] = other

    # Compute node positions: x = turn, y = stacked within turn
    fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")

    x_spacing = 1.0
    max_bar_height = 8.0  # total y-range for stacking
    node_positions: dict[int, list[tuple[float, float, float, dict]]] = {}  # turn -> [(x, y_center, height, state)]
    other_positions: dict[int, tuple[float, float, float]] = {}  # turn -> (x, y_center, height)

    for t in turns:
        x = t * x_spacing
        total = sum(s["count"] for s in node_data[t]) + other_counts[t]
        y_cursor = 0.0

        positions = []
        for s in node_data[t]:
            h = (s["count"] / total) * max_bar_height
            y_center = y_cursor + h / 2
            positions.append((x, y_center, h, s))
            y_cursor += h + 0.03  # small gap

        # "Other" node
        h_other = (other_counts[t] / total) * max_bar_height if other_counts[t] > 0 else 0
        y_other = y_cursor + h_other / 2
        other_positions[t] = (x, y_other, h_other)
        node_positions[t] = positions

    # Draw transition ribbons between consecutive turns
    for t in turns[:-1]:
        t_next = t + 1
        if t_next not in node_data:
            continue

        src_states = node_data[t]
        src_positions = node_positions[t]
        dst_states = node_data[t_next]
        dst_positions = node_positions[t_next]

        # For each source, find compatible destinations and distribute flow
        # Track how much of each dest has been "filled" by sources
        dst_y_offsets = [0.0] * len(dst_states)
        other_y_offset = 0.0

        for si, (sx, sy, sh, ss) in enumerate(src_positions):
            src_count = ss["count"]
            # Find compatible destinations
            compat = []
            for di, ds in enumerate(dst_states):
                # Check: ss.scored is a subset of ds.scored, and upper is consistent
                if (ss["scored"] & ds["scored"]) == ss["scored"] and ds["upper"] >= ss["upper"]:
                    extra_bit = ds["scored"] & ~ss["scored"]
                    if bin(extra_bit).count("1") == 1:  # exactly one new category
                        compat.append(di)

            if not compat:
                # All flow goes to "other"
                ox, oy, oh = other_positions[t_next]
                if oh > 0:
                    ribbon_h = sh * 0.3
                    _bezier_ribbon(ax, sx + 0.15, sy - ribbon_h / 2, sy + ribbon_h / 2,
                                   ox - 0.15, oy - ribbon_h / 2, oy + ribbon_h / 2,
                                   "#cccccc", 0.15)
                continue

            # Distribute proportionally to dest counts
            total_compat = sum(dst_states[di]["count"] for di in compat)
            if total_compat == 0:
                continue

            src_y_cursor_bot = sy - sh / 2
            for di in compat:
                frac = dst_states[di]["count"] / total_compat
                ribbon_h_src = sh * frac
                ribbon_h_dst = dst_positions[di][2] * (src_count * frac / dst_states[di]["count"])
                ribbon_h_dst = min(ribbon_h_dst, dst_positions[di][2] - dst_y_offsets[di])

                if ribbon_h_src < 0.01 or ribbon_h_dst < 0.01:
                    src_y_cursor_bot += ribbon_h_src
                    continue

                dx, dy, dh, ds = dst_positions[di]
                src_top = src_y_cursor_bot + ribbon_h_src
                dst_bot = dy - dh / 2 + dst_y_offsets[di]
                dst_top = dst_bot + ribbon_h_dst

                color = BONUS_COLORS.get(_classify_bonus(ds["upper"], ds["scored"]), "#cccccc")
                _bezier_ribbon(ax, sx + 0.15, src_y_cursor_bot, src_top,
                               dx - 0.15, dst_bot, dst_top,
                               color, 0.2)

                dst_y_offsets[di] += ribbon_h_dst
                src_y_cursor_bot += ribbon_h_src

    # Draw nodes (on top of ribbons)
    for t in turns:
        for x, y, h, s in node_positions[t]:
            color = BONUS_COLORS.get(_classify_bonus(s["upper"], s["scored"]), "#cccccc")
            rect = mpatches.FancyBboxPatch(
                (x - 0.12, y - h / 2), 0.24, max(h, 0.05),
                boxstyle="round,pad=0.02", facecolor=color,
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)

        # "Other" node
        ox, oy, oh = other_positions[t]
        if oh > 0.01:
            rect = mpatches.FancyBboxPatch(
                (ox - 0.12, oy - oh / 2), 0.24, oh,
                boxstyle="round,pad=0.02", facecolor="#e0e0e0",
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)

    # Labels for early and late turns
    for t in turns:
        if t <= 3 or t >= 12:
            for x, y, h, s in node_positions[t]:
                if h > 0.2:
                    label = _state_label(s, t)
                    ax.text(x, y, label, fontsize=6, ha="center", va="center",
                            fontweight="bold", zorder=5, color="#333333")

    # Turn labels
    for t in turns:
        ax.text(t * x_spacing, -0.6, f"T{t + 1}", fontsize=9, ha="center",
                fontweight="bold", color="#555555")

    # Unique state counts
    for t in turns:
        n_states = len(bt[t])
        ax.text(t * x_spacing, -1.1, f"{n_states:,}", fontsize=7, ha="center",
                color="#999999")
    ax.text(-0.6, -1.1, "states:", fontsize=7, ha="right", color="#999999")

    # Legend
    legend_handles = [mpatches.Patch(color=BONUS_COLORS[g], label=g)
                      for g in BONUS_GROUPS if g != "Bonus lost"]
    legend_handles.append(mpatches.Patch(color="#e0e0e0", label="Other states"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlim(-0.8, 14.8)
    ax.set_ylim(-1.5, max_bar_height + 1)
    ax.set_title("State-Space DAG — Top-8 Board States per Turn",
                 fontsize=15, fontweight="bold", pad=15)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Option 3: Category Streamgraph
# ---------------------------------------------------------------------------


def plot_category_streamgraph(data_path: Path, out_path: Path, dpi: int = 200) -> None:
    """Streamgraph showing when each category meta-group gets filled."""
    rows = _load_state_frequency(data_path)
    bt = _by_turn(rows)
    turns = sorted(bt.keys())

    # Compute cumulative fill rate per category at each turn
    # frac_scored(c, t) = weighted fraction of games where category c is scored at turn t
    cum_fill = np.zeros((15, len(turns)))  # [category, turn_index]
    for ti, t in enumerate(turns):
        total = sum(r["count"] for r in bt[t])
        for r in bt[t]:
            for c in range(15):
                if r["scored"] & (1 << c):
                    cum_fill[c, ti] += r["count"]
        cum_fill[:, ti] /= total if total > 0 else 1

    # Marginal fill rate: probability category c was scored at turn t
    marginal = np.zeros((15, len(turns)))
    for ti in range(len(turns)):
        if ti == 0:
            marginal[:, ti] = cum_fill[:, ti]
        else:
            marginal[:, ti] = cum_fill[:, ti] - cum_fill[:, ti - 1]
    marginal = np.clip(marginal, 0, 1)

    # Aggregate into meta-groups
    group_names = list(META_GROUPS.keys())
    group_marginals = np.zeros((len(group_names), len(turns)))
    for gi, gname in enumerate(group_names):
        for c in META_GROUPS[gname]:
            group_marginals[gi] += marginal[c]

    # Smooth interpolation
    x_raw = np.array([t + 1 for t in turns], dtype=float)
    x_smooth = np.linspace(1, 15, 300)
    smooth_data = np.zeros((len(group_names), len(x_smooth)))
    for gi in range(len(group_names)):
        interp = PchipInterpolator(x_raw, group_marginals[gi])
        smooth_data[gi] = np.clip(interp(x_smooth), 0, 1)

    # Renormalize columns
    col_sums = smooth_data.sum(axis=0)
    col_sums = np.where(col_sums > 0, col_sums, 1)
    smooth_data /= col_sums

    # --- Plot streamgraph ---
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    colors = [META_COLORS[g] for g in group_names]
    ax.stackplot(x_smooth, smooth_data, labels=group_names, colors=colors,
                 alpha=0.85, baseline="wiggle")

    # Add stream labels at the widest point of each stream
    # Compute cumulative for label placement
    cum_bottom = np.zeros_like(x_smooth)
    # Recompute with wiggle baseline (need the actual positions)
    # stackplot returns PolyCollection; extract positions from the stacks
    ax.clear()
    polys = ax.stackplot(x_smooth, smooth_data, labels=group_names, colors=colors,
                         alpha=0.85, baseline="wiggle")

    # Place labels on right edge of each stream for clear identification
    for gi, poly in enumerate(polys):
        paths = poly.get_paths()
        if not paths:
            continue
        verts = paths[0].vertices
        n = len(x_smooth)
        top_y = verts[:n, 1]
        bot_y = verts[n:2 * n, 1][::-1] if len(verts) >= 2 * n else top_y
        center_y = (top_y + bot_y) / 2
        # Label at x=14.5 (near right edge)
        idx = np.argmin(np.abs(x_smooth - 14.5))
        ax.text(15.2, center_y[idx], group_names[gi],
                fontsize=8.5, ha="left", va="center",
                color=META_COLORS[group_names[gi]], fontweight="bold")

    ax.set_xlim(1, 17.5)
    ax.set_xticks(range(1, 16))
    ax.set_xlabel("Turn", fontsize=13)
    ax.set_title("Category Fill Timing — When Does Each Category Get Scored?",
                 fontsize=15, fontweight="bold", pad=15)

    # Clean y-axis — streamgraphs are about shape, not exact values
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: ""))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_state_flow_plots(
    data_path: Path,
    out_dir: Path,
    dpi: int = 200,
) -> list[Path]:
    """Generate all three state-flow visualizations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    p = out_dir / "state_flow_bonus_alluvial.png"
    plot_bonus_alluvial(data_path, p, dpi=dpi)
    paths.append(p)

    p = out_dir / "state_flow_dag.png"
    plot_state_dag(data_path, p, dpi=dpi)
    paths.append(p)

    p = out_dir / "state_flow_streamgraph.png"
    plot_category_streamgraph(data_path, p, dpi=dpi)
    paths.append(p)

    return paths
