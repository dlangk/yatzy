"""Reusable Yatzy scorecard renderer for terminal display.

Renders a full 15-category Scandinavian Yatzy scorecard with upper sum, bonus,
and total rows. Categories can be marked as selectable with [N] markers for
interactive use (e.g. questionnaires).

Four-column layout: Selection | Category | Score | Bonus tracker

Pure text output — no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


CATEGORY_NAMES = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "One Pair", "Two Pairs", "Three of a Kind", "Four of a Kind",
    "Small Straight", "Large Straight", "Full House", "Chance", "Yatzy",
]

# Upper section category IDs
UPPER_IDS = set(range(6))


@dataclass
class ScoreEntry:
    """A single category row in the scorecard."""

    category_id: int
    name: str
    score: int | None = None  # None = not scored yet
    selectable: bool = False
    selection_num: int | None = None  # The [N] marker number


def render_scorecard(
    entries: list[ScoreEntry],
    dice: list[int] | None = None,
    turn: int | None = None,
    header: str | None = None,
) -> str:
    """Render a full Yatzy scorecard as a string.

    Four columns: Selection | Category | Score | Bonus tracker

    - Column 0: [N] markers for selectable categories
    - Column 1: Category name
    - Column 2: Score (or - if unscored)
    - Column 3: Upper section delta from par (X = on par, +/-N = ahead/behind)
    """
    by_id: dict[int, ScoreEntry] = {e.category_id: e for e in entries}
    lines: list[str] = []

    # Header info
    if header:
        lines.append(f"  {header}")
        lines.append("")
    info_parts = []
    if turn is not None:
        info_parts.append(f"Turn {turn} of 15")
    if dice is not None:
        info_parts.append(f"Dice: [{', '.join(str(d) for d in dice)}]")
    if info_parts:
        lines.append("  " + "    ".join(info_parts))
        lines.append("")

    # Column widths
    sel_w = 5   # [10] + padding
    cat_w = 18  # category name
    score_w = 7  # score
    info_w = 5   # bonus tracker

    def sep(left: str, mid: str, right: str, fill: str = "\u2500") -> str:
        return (
            f"  {left}{fill * sel_w}{mid}{fill * cat_w}"
            f"{mid}{fill * score_w}{mid}{fill * info_w}{right}"
        )

    def row(
        sel_str: str, name: str, score_str: str, info_str: str = "",
    ) -> str:
        return (
            f"  \u2502 {sel_str:>{sel_w - 2}} "
            f"\u2502 {name:<{cat_w - 2}} "
            f"\u2502 {score_str:>{score_w - 2}} "
            f"\u2502 {info_str:>{info_w - 2}} \u2502"
        )

    def upper_delta_str(cat_id: int, score: int) -> str:
        """Format the delta from par (3x face_value) for an upper category."""
        face = cat_id + 1
        par = 3 * face
        delta = score - par
        if delta == 0:
            return "X"
        return f"{delta:+d}"

    def cat_row(cat_id: int) -> str:
        e = by_id.get(cat_id)
        name = CATEGORY_NAMES[cat_id]
        # Column 0: selection marker
        sel_str = ""
        if e is not None and e.selectable and e.selection_num is not None:
            sel_str = f"[{e.selection_num}]"
        # Column 2: score
        if e is None or e.score is None:
            score_str = "-"
        else:
            score_str = str(e.score)
        # Column 3: upper bonus tracker
        info_str = ""
        if cat_id < 6 and e is not None and e.score is not None:
            info_str = upper_delta_str(cat_id, e.score)
        return row(sel_str, name, score_str, info_str)

    # Compute upper sum, bonus delta, and reachability
    upper_sum = 0
    upper_delta = 0
    max_remaining_upper = 0
    for c in range(6):
        e = by_id.get(c)
        if e is not None and e.score is not None:
            upper_sum += e.score
            upper_delta += e.score - 3 * (c + 1)
        else:
            max_remaining_upper += 5 * (c + 1)

    if upper_sum >= 63:
        bonus_score_str = "50"
        bonus_info_str = ""
    else:
        reachable = (upper_sum + max_remaining_upper) >= 63
        bonus_score_str = "rch" if reachable else "unrch"
        bonus_info_str = f"{upper_delta:+d}" if upper_delta != 0 else "X"

    # Compute total
    total = 0
    for c in range(15):
        e = by_id.get(c)
        if e is not None and e.score is not None:
            total += e.score
    if upper_sum >= 63:
        total += 50

    # Build table
    lines.append(sep("\u250c", "\u252c", "\u2510"))
    lines.append(row("", "Category", "Score", ""))
    lines.append(sep("\u251c", "\u253c", "\u2524"))

    # Upper section
    for c in range(6):
        lines.append(cat_row(c))

    # Upper sum + bonus
    lines.append(sep("\u251c", "\u253c", "\u2524"))
    lines.append(row("", "Upper Sum", str(upper_sum), ""))
    lines.append(row("", "Bonus", bonus_score_str, bonus_info_str))
    lines.append(sep("\u251c", "\u253c", "\u2524"))

    # Lower section
    for c in range(6, 15):
        lines.append(cat_row(c))

    # Total
    lines.append(sep("\u251c", "\u253c", "\u2524"))
    lines.append(row("", "Total", str(total), ""))
    lines.append(sep("\u2514", "\u2534", "\u2518"))

    return "\n".join(lines)


def scorecard_from_scenario(
    scenario: dict,
    available_ids: list[int] | None = None,
) -> str:
    """Build a scorecard from the pivotal-scenario JSON format.

    Args:
        scenario: A scenario dict with scored_details, available, dice, turn.
        available_ids: Category IDs to mark as selectable (with [N] markers).
            If None, uses all available categories from the scenario.
    """
    entries: list[ScoreEntry] = []

    # Scored categories
    for detail in scenario.get("scored_details", []):
        entries.append(ScoreEntry(
            category_id=detail["id"],
            name=detail["name"],
            score=detail["score"],
        ))

    # Available categories (unscored)
    available = scenario.get("available", [])
    if available_ids is None:
        available_ids = [c["id"] for c in available]

    sel_num = 1
    for cat in available:
        cat_id = cat["id"]
        selectable = cat_id in available_ids
        entries.append(ScoreEntry(
            category_id=cat_id,
            name=cat["name"],
            score=None,
            selectable=selectable,
            selection_num=sel_num if selectable else None,
        ))
        if selectable:
            sel_num += 1

    return render_scorecard(
        entries,
        dice=scenario.get("dice"),
        turn=scenario.get("turn"),
    )
