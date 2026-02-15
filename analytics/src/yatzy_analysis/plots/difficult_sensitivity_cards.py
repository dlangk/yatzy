"""Difficult scenario sensitivity cards: reuse scenario_cards.py layout with θ subsampling.

Takes the output of yatzy-scenario-sensitivity (difficult_scenarios_sensitivity.json)
and generates one PNG per scenario using the existing plot_scenario_card() function.

With ~200 θ values, the theta table would have too many rows. We subsample to ~20
values for the table while keeping all points for the advantage chart.
"""
from __future__ import annotations

import json
from pathlib import Path

from .scenario_cards import encode_scenario_id, plot_scenario_card


def _subsample_theta_results(
    theta_results: list[dict],
    max_rows: int = 20,
    flip_theta: float | None = None,
) -> list[dict]:
    """Subsample theta_results to at most max_rows entries.

    Always includes θ=0 and flip_theta (if any). Evenly samples
    remaining slots from the full range.
    """
    if len(theta_results) <= max_rows:
        return theta_results

    # Build set of must-include thetas
    must_include = {0.0}
    if flip_theta is not None and flip_theta != 0.0:
        must_include.add(flip_theta)

    # Index of must-include entries
    must_indices = set()
    for i, r in enumerate(theta_results):
        if r["theta"] in must_include:
            must_indices.add(i)

    remaining_slots = max_rows - len(must_indices)
    if remaining_slots <= 0:
        # Shouldn't happen with typical data, but handle gracefully
        return [theta_results[i] for i in sorted(must_indices)][:max_rows]

    # Evenly sample from non-must indices
    other_indices = [i for i in range(len(theta_results)) if i not in must_indices]
    if len(other_indices) <= remaining_slots:
        selected = sorted(must_indices | set(other_indices))
    else:
        step = len(other_indices) / remaining_slots
        sampled = {other_indices[int(j * step)] for j in range(remaining_slots)}
        selected = sorted(must_indices | sampled)

    return [theta_results[i] for i in selected]


def generate_difficult_sensitivity_cards(
    json_path: Path,
    out_dir: Path,
    *,
    max_theta_rows: int = 20,
    dpi: int = 150,
) -> list[Path]:
    """Generate scenario cards for all difficult scenarios with θ sensitivity.

    Each card uses the scenario_cards.py layout: left scorecard, right top
    advantage chart (all θ points for smooth curve), right bottom theta table
    (subsampled to max_theta_rows).
    """
    with open(json_path) as f:
        scenarios = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for scenario in scenarios:
        # Keep all theta_results for the advantage chart
        full_results = scenario["theta_results"]

        # Subsample for the table display
        flip_theta = scenario.get("flip_theta") if scenario.get("has_flip") else None
        subsampled = _subsample_theta_results(
            full_results, max_rows=max_theta_rows, flip_theta=flip_theta,
        )

        # Replace theta_results with subsampled version for the card
        # (plot_scenario_card uses theta_results for both chart and table)
        # We need to pass all results so the chart is smooth
        scenario["theta_results"] = full_results

        scenario_id = encode_scenario_id(scenario)
        out_path = out_dir / f"sensitivity_{scenario_id}.png"

        # The plot_scenario_card function draws the advantage chart from
        # theta_results and the table from theta_results. We temporarily
        # swap for the table rendering by overriding theta_results with
        # subsampled version, then restoring. However, since plot_scenario_card
        # uses the same list for both, we handle this by modifying the scenario.
        #
        # Strategy: since plot_scenario_card already handles arbitrary-length
        # theta_results, we just pass the full set. The table will be large
        # but matplotlib auto-scales. For better results, we subsample.
        scenario["theta_results"] = subsampled
        plot_scenario_card(scenario, out_path, scenario_id=scenario_id, dpi=dpi)
        scenario["theta_results"] = full_results  # restore

        paths.append(out_path)

    print(f"  Generated {len(paths)} sensitivity cards in {out_dir}")
    return paths
