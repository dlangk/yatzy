#!/usr/bin/env python3
"""Convert outputs/ CSV/JSON → profiler/data/ JSON files."""

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def write_json(name: str, data: object) -> None:
    path = OUT / name
    with open(path, "w") as f:
        json.dump(data, f, indent=None, separators=(",", ":"))
    size = path.stat().st_size
    print(f"  {name}: {size:,} bytes")


def convert_sweep_summary() -> None:
    src = ROOT / "outputs" / "aggregates" / "parquet" / "sweep_summary.csv"
    keep = [
        "theta", "mean", "std", "p1", "p5", "p10", "p25", "p50",
        "p75", "p90", "p95", "p99", "skewness", "kurtosis", "cvar_5",
    ]
    rows = []
    with open(src) as f:
        for row in csv.DictReader(f):
            rows.append({k: round(float(row[k]), 4) for k in keep})
    write_json("sweep_summary.json", rows)


def convert_winrate() -> None:
    src = ROOT / "outputs" / "winrate" / "winrate_results.csv"
    rows = []
    with open(src) as f:
        for row in csv.DictReader(f):
            d: dict = {"theta": float(row["theta"])}
            for k in ["win_rate", "draw_rate", "loss_rate", "mean", "std"]:
                v = row.get(k, "")
                d[k] = round(float(v), 4) if v else None
            rows.append(d)
    write_json("winrate.json", rows)


def convert_game_eval() -> None:
    src = ROOT / "outputs" / "surrogate" / "game_eval_results.json"
    with open(src) as f:
        data = json.load(f)
    write_json("game_eval.json", data)


def convert_heuristic_gap() -> None:
    src = ROOT / "outputs" / "heuristic_gap" / "heuristic_gap_summary.json"
    with open(src) as f:
        data = json.load(f)
    out = {
        "by_decision_type": data["by_decision_type"],
        "top_patterns": data["top_patterns"],
        "ev_loss_per_game": data["ev_loss_per_game"],
        "heuristic_mean_score": data["heuristic_mean_score"],
        "disagreements_per_game": data["disagreements_per_game"],
        "category_mistakes_per_game": data["category_mistakes_per_game"],
    }
    write_json("heuristic_gap.json", out)


def convert_mixture() -> None:
    data = {
        "populations": [
            {"label": "No bonus, no Yatzy", "fraction": 0.061, "mean": 164, "std": 19.7},
            {"label": "No bonus, Yatzy", "fraction": 0.041, "mean": 213, "std": 19.6},
            {"label": "Bonus, no Yatzy", "fraction": 0.551, "mean": 237, "std": 20.3},
            {"label": "Bonus + Yatzy", "fraction": 0.347, "mean": 288, "std": 21.1},
        ]
    }
    write_json("mixture.json", data)


def convert_percentile_peaks() -> None:
    src = ROOT / "outputs" / "aggregates" / "csv" / "percentile_peaks.csv"
    rows = []
    with open(src) as f:
        for row in csv.DictReader(f):
            rows.append({
                "percentile": row["percentile"],
                "theta_star": float(row["theta_star"]),
                "value": int(row["value"]),
                "mean_at_theta": round(float(row["mean_at_theta"]), 2),
            })
    write_json("percentile_peaks.json", rows)


def convert_category_stats() -> None:
    src = ROOT / "outputs" / "aggregates" / "csv" / "category_stats.csv"
    rows = []
    ceilings = {
        "Ones": 5, "Twos": 10, "Threes": 15, "Fours": 20,
        "Fives": 25, "Sixes": 30, "One Pair": 12, "Two Pairs": 22,
        "Three of a Kind": 18, "Four of a Kind": 24,
        "Small Straight": 15, "Large Straight": 20,
        "Full House": 28, "Chance": 30, "Yatzy": 50,
    }
    with open(src) as f:
        for row in csv.DictReader(f):
            if row["theta"] == "0.000":
                rows.append({
                    "id": int(row["category_id"]),
                    "name": row["category_name"],
                    "mean": round(float(row["mean_score"]), 2),
                    "ceiling": ceilings.get(row["category_name"], 0),
                    "zero_rate": round(float(row["zero_rate"]), 4),
                    "hit_rate": round(float(row["hit_rate"]), 4),
                    "fill_turn": round(float(row["mean_fill_turn"]), 1),
                })
    write_json("category_stats_theta0.json", rows)


def main() -> None:
    print("Converting data → profiler/data/")
    convert_sweep_summary()
    convert_winrate()
    convert_game_eval()
    convert_heuristic_gap()
    convert_mixture()
    convert_percentile_peaks()
    convert_category_stats()
    print("Done.")


if __name__ == "__main__":
    main()
