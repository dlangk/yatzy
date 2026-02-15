"""Percentile sweep plots: curves, heatmap, frontier, and peak annotations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from .style import CMAP, setup_theme

_CORE = ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
_EXTRA = ["p1", "p999", "p9999"]
_ALL = _EXTRA[:1] + _CORE + _EXTRA[1:]  # p1, p5..p99, p999, p9999


def _plot_curves(
    df: pd.DataFrame,
    ax,
    *,
    skip: set[str] | None = None,
) -> None:
    """Draw percentile curves matching the rocket-palette style."""
    skip = skip or set()
    pct_colors = sns.color_palette("rocket", len(_CORE))

    for i, p in enumerate(_CORE):
        if p in df.columns and p not in skip:
            ax.plot(
                df["theta"], df[p],
                marker="o", markersize=4, linewidth=1.8, color=pct_colors[i],
                label=p, zorder=3,
            )

    extra_styles = {
        "p1": ("--", "tab:blue"),
        "p999": ("--", "tab:orange"),
        "p9999": ("--", "tab:red"),
    }
    for p, (ls, color) in extra_styles.items():
        if p in df.columns and p not in skip:
            ax.plot(
                df["theta"], df[p],
                linestyle=ls, marker="s", markersize=3, linewidth=1.4,
                color=color, alpha=0.8, label=p, zorder=3,
            )


def _add_peak_stars(ax, df: pd.DataFrame, peaks: pd.DataFrame, theta_range=None, skip=None) -> None:
    """Add star markers at each percentile's peak θ*."""
    pct_colors = sns.color_palette("rocket", len(_CORE))
    color_map = {p: pct_colors[i] for i, p in enumerate(_CORE)}
    extra_colors = {"p1": "tab:blue", "p999": "tab:orange", "p9999": "tab:red"}
    color_map.update(extra_colors)
    skip = skip or set()

    for _, row in peaks.iterrows():
        pct = row["percentile"]
        t = row["theta_star"]
        if pct not in color_map or pct in skip:
            continue
        if theta_range and not (theta_range[0] <= t <= theta_range[1]):
            continue
        ax.plot(
            t, row["value"],
            marker="*", markersize=14, color=color_map[pct],
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
        )


def _merge_with_coarse(sweep_df: pd.DataFrame, coarse_path: Path | None) -> pd.DataFrame:
    """Merge dense sweep with coarse summary for wider θ coverage."""
    if coarse_path is None or not coarse_path.exists():
        return sweep_df

    coarse = pd.read_parquet(coarse_path)

    # Only keep coarse rows outside the sweep range
    lo, hi = sweep_df["theta"].min(), sweep_df["theta"].max()
    outside = coarse[(coarse["theta"] < lo - 0.001) | (coarse["theta"] > hi + 0.001)]

    if len(outside) == 0:
        return sweep_df

    # Concat with all columns — missing ones become NaN, then interpolate
    merged = pd.concat([sweep_df, outside], ignore_index=True)
    merged = merged.sort_values("theta").reset_index(drop=True)
    # Interpolate NaN percentile columns so lines extend smoothly
    pct_cols = [c for c in merged.columns if c.startswith("p")]
    merged[pct_cols] = merged[pct_cols].interpolate(method="index")
    return merged


def plot_percentile_sweep(
    sweep_path: Path,
    peaks_path: Path,
    out_dir: Path,
    *,
    coarse_path: Path | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Generate all percentile sweep plots. Returns list of output paths."""
    setup_theme()
    df = pd.read_csv(sweep_path)
    peaks = pd.read_csv(peaks_path)
    df = df.sort_values("theta")

    # Merge with coarse grid for wider-range plots
    df_wide = _merge_with_coarse(df, coarse_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # --- Plot 1: Dense sweep only ---
    paths.append(_plot_full(df, peaks, out_dir, dpi=dpi))

    # --- Plot 2: Wide range (±1), merged with coarse ---
    paths.append(_plot_wide(df_wide, peaks, out_dir, dpi=dpi))

    # --- Plot 3: Mean vs theta with std band ---
    paths.append(_plot_mean_std(df, out_dir, dpi=dpi))

    # --- Plot 4: Heatmap ---
    paths.append(_plot_heatmap(df, out_dir, dpi=dpi))

    # --- Plot 5: Cost-benefit scatter ---
    paths.append(_plot_mean_cost(df, peaks, out_dir, dpi=dpi))

    # --- Plot 6: Mean–Std frontier (dense sweep + coarse extensions) ---
    paths.append(_plot_frontier(df, df_wide, peaks, out_dir, dpi=dpi))

    # --- Plot 7: Multi-percentile frontiers in (mean, p) space ---
    paths.append(_plot_percentile_frontiers(df, peaks, out_dir, dpi=dpi))

    print(f"  Generated {len(paths)} percentile sweep plots in {out_dir}")
    return paths


def _plot_full(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))

    _plot_curves(df, ax)
    _add_peak_stars(ax, df, peaks)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Score Percentiles vs Risk Parameter θ", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "percentile_sweep_curves.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_wide(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Wide range: θ ∈ [-1.0, +1.0], merging dense sweep + coarse grid."""
    wdf = df[(df["theta"] >= -1.0) & (df["theta"] <= 1.0)].copy()
    if len(wdf) == 0:
        wdf = df

    fig, ax = plt.subplots(figsize=(14, 8))

    _plot_curves(wdf, ax)
    _add_peak_stars(ax, wdf, peaks, theta_range=(-1.0, 1.0))

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(
        "Score Percentiles vs θ  (|θ| ≤ 1)",
        fontsize=15, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=9, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "percentile_sweep_wide.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_mean_std(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Mean score vs θ with ±1 std band."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        df["theta"],
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2, color="#3182ce",
    )
    ax.plot(df["theta"], df["mean"], linewidth=2.5, color="#2b6cb0", label="Mean ± 1σ")

    ev_row = df.iloc[(df["theta"].abs()).argmin()]
    ax.plot(ev_row["theta"], ev_row["mean"], marker="*", markersize=14, color="#e53e3e", zorder=5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ  (risk parameter)", fontsize=13)
    ax.set_ylabel("Mean Score", fontsize=13)
    ax.set_title("Mean Score vs θ  (risk-averse ← 0 → risk-seeking)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "percentile_sweep_mean.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Heatmap: rows = percentiles, columns = θ, cells = score values."""
    if len(df) > 40:
        step = max(1, len(df) // 40)
        sdf = df.iloc[::step].copy()
    else:
        sdf = df.copy()

    pcts_available = [p for p in _ALL if p in sdf.columns]
    matrix = sdf[pcts_available].T.values.astype(float)
    theta_labels = [f"{t:+.3f}" for t in sdf["theta"]]

    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

    ax.set_yticks(range(len(pcts_available)))
    ax.set_yticklabels(pcts_available, fontsize=10)
    ax.set_xticks(range(len(theta_labels)))
    ax.set_xticklabels(theta_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("θ", fontsize=12)
    ax.set_title("Score Percentile Heatmap across θ", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Score", fontsize=11)

    fig.tight_layout()
    path = out_dir / "percentile_sweep_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_mean_cost(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """For each percentile peak θ*, show the percentile gain vs mean cost."""
    ev_row = df.iloc[(df["theta"].abs()).argmin()]
    ev_mean = ev_row["mean"]

    pct_colors = sns.color_palette("rocket", len(_CORE))
    color_map = {p: pct_colors[i] for i, p in enumerate(_CORE)}
    color_map.update({"p1": "tab:blue", "p999": "tab:orange", "p9999": "tab:red"})

    fig, ax = plt.subplots(figsize=(10, 7))

    xs, ys, labels, colors = [], [], [], []
    for _, row in peaks.iterrows():
        pct = row["percentile"]
        if pct not in color_map or pct not in ev_row.index:
            continue
        ev_pct_val = ev_row[pct]
        gain = row["value"] - ev_pct_val
        cost = ev_mean - row["mean_at_theta"]
        xs.append(cost)
        ys.append(gain)
        labels.append(pct)
        colors.append(color_map[pct])

    ax.scatter(xs, ys, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(
            lab, (x, y),
            xytext=(8, 5), textcoords="offset points",
            fontsize=10, fontweight="bold",
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Score Cost  (EV-optimal mean − mean@θ*)", fontsize=12)
    ax.set_ylabel("Percentile Gain  (value@θ* − value@θ=0)", fontsize=12)
    ax.set_title("Cost-Benefit: Tail Optimization vs Mean Score", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "percentile_sweep_cost_benefit.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_frontier(
    df_dense: pd.DataFrame,
    df_wide: pd.DataFrame,
    peaks: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Mean–Std Pareto frontier using dense sweep for a smooth curve.

    Two layers:
    - Dense sweep (185 pts): smooth colored line, gradient by θ
    - Coarse-only extensions (|θ|>0.5): faded scatter for context

    Annotations: θ=0 star, peak-p95 diamond, select θ labels along curve,
    risk-averse / risk-seeking direction arrows.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # --- Coarse-only points (outside dense range) as faded context ---
    dense_lo, dense_hi = df_dense["theta"].min(), df_dense["theta"].max()
    coarse_only = df_wide[
        (df_wide["theta"] < dense_lo - 0.001) | (df_wide["theta"] > dense_hi + 0.001)
    ].copy()
    if len(coarse_only) > 0:
        norm_wide = mcolors.Normalize(
            vmin=df_wide["theta"].min(), vmax=df_wide["theta"].max(),
        )
        for _, row in coarse_only.iterrows():
            ax.scatter(
                row["std"], row["mean"],
                color=CMAP(norm_wide(row["theta"])),
                s=40, alpha=0.35, zorder=2,
                edgecolors="gray", linewidths=0.3,
            )

    # --- Dense sweep: colored line segments by θ ---
    ds = df_dense.sort_values("theta").reset_index(drop=True)
    norm = mcolors.Normalize(vmin=ds["theta"].min(), vmax=ds["theta"].max())

    # Draw line segments colored by θ (using LineCollection for gradient)
    stds = ds["std"].values
    means = ds["mean"].values
    thetas = ds["theta"].values

    for i in range(len(ds) - 1):
        t_mid = (thetas[i] + thetas[i + 1]) / 2
        ax.plot(
            [stds[i], stds[i + 1]], [means[i], means[i + 1]],
            color=CMAP(norm(t_mid)), linewidth=2.8, solid_capstyle="round", zorder=3,
        )

    # Scatter on top for visibility
    colors = [CMAP(norm(t)) for t in thetas]
    ax.scatter(
        stds, means,
        c=colors, s=30, zorder=4,
        edgecolors="white", linewidths=0.3,
    )

    # --- θ=0 marker ---
    ev_idx = np.abs(thetas).argmin()
    ax.scatter(
        stds[ev_idx], means[ev_idx],
        color="black", s=200, marker="*", zorder=10,
        label="θ=0 (EV-optimal)",
    )

    # --- Peak p95 marker ---
    p95_peak = peaks[peaks["percentile"] == "p95"]
    if not p95_peak.empty:
        t_star = p95_peak.iloc[0]["theta_star"]
        match = ds.iloc[(ds["theta"] - t_star).abs().argsort()[:1]]
        if len(match) > 0:
            r = match.iloc[0]
            ax.scatter(
                r["std"], r["mean"],
                color="#e53e3e", s=180, marker="D", zorder=10,
                edgecolors="black", linewidths=0.8,
                label=f"θ*_p95 = {t_star:+.2f}",
            )

    # --- Sparse θ labels along the curve ---
    label_thetas = [-0.08, -0.04, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    labeled = set()
    for t_target in label_thetas:
        idx = np.abs(thetas - t_target).argmin()
        if abs(thetas[idx] - t_target) > 0.015:
            continue
        if idx in labeled:
            continue
        labeled.add(idx)
        t_val = thetas[idx]
        label = f"θ={t_val:+.2f}" if t_val != 0.0 else "θ=0"

        # Offset direction: risk-averse labels go left, risk-seeking go right
        if t_val < -0.01:
            offset = (-12, 8)
            ha = "right"
        elif t_val > 0.01:
            offset = (8, -10)
            ha = "left"
        else:
            offset = (10, 6)
            ha = "left"

        ax.annotate(
            label,
            (stds[idx], means[idx]),
            textcoords="offset points", xytext=offset,
            fontsize=8.5, alpha=0.85, ha=ha,
            arrowprops=dict(arrowstyle="-", alpha=0.3, linewidth=0.5),
        )

    # --- Direction annotations ---
    averse_mask = thetas < -0.03
    seeking_mask = thetas > 0.15
    if averse_mask.sum() > 2:
        mid_a = np.where(averse_mask)[0][len(np.where(averse_mask)[0]) // 2]
        ax.annotate(
            "risk-averse\n(lower variance, lower mean)",
            (stds[mid_a], means[mid_a]),
            textcoords="offset points", xytext=(30, -35),
            fontsize=9.5, color="#3b4cc0", fontstyle="italic", ha="center",
            arrowprops=dict(arrowstyle="->", color="#3b4cc0", linewidth=1.2),
        )
    if seeking_mask.sum() > 2:
        mid_s = np.where(seeking_mask)[0][len(np.where(seeking_mask)[0]) // 2]
        ax.annotate(
            "risk-seeking\n(higher variance, lower mean)",
            (stds[mid_s], means[mid_s]),
            textcoords="offset points", xytext=(-60, 30),
            fontsize=9.5, color="#b40426", fontstyle="italic", ha="center",
            arrowprops=dict(arrowstyle="->", color="#b40426", linewidth=1.2),
        )

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("θ  (risk-averse ← 0 → risk-seeking)", fontsize=12)

    ax.set_xlabel("Standard Deviation", fontsize=14)
    ax.set_ylabel("Mean Score", fontsize=14)
    ax.set_title(
        "Mean–Variance Frontier across Risk Parameter θ",
        fontsize=17, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "mean_std_frontier.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_percentile_frontiers(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Mean–Std frontier with each percentile's θ* marked along the curve.

    Background: the full (std, mean) parametric curve as θ varies, colored by θ
    with opacity increasing with |θ|.

    Overlay: for each percentile's optimal θ*, mark the (std, mean) point.
    These trace a path along the frontier showing where each percentile's
    optimum lands in mean-variance space. Color by coolwarm (low pct = blue,
    high pct = red).
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    ds = df.sort_values("theta").reset_index(drop=True)
    thetas = ds["theta"].values
    means = ds["mean"].values
    stds = ds["std"].values

    # --- Background: full frontier curve, faded, with α ∝ |θ| ---
    abs_theta = np.abs(thetas)
    max_abs = abs_theta.max() if abs_theta.max() > 0 else 1.0
    alpha_arr = 0.08 + 0.35 * (abs_theta / max_abs)

    norm_theta = mcolors.Normalize(vmin=thetas.min(), vmax=thetas.max())
    for i in range(len(ds) - 1):
        t_mid = (thetas[i] + thetas[i + 1]) / 2
        a = (alpha_arr[i] + alpha_arr[i + 1]) / 2
        c = CMAP(norm_theta(t_mid))
        ax.plot(
            [stds[i], stds[i + 1]], [means[i], means[i + 1]],
            color=(*c[:3], a), linewidth=2.0, solid_capstyle="round", zorder=2,
        )

    # --- Percentile peak markers along the curve ---
    pcts = [p for p in _ALL if p in ds.columns]
    cw = plt.colormaps["coolwarm"]
    pct_norm = mcolors.Normalize(vmin=0, vmax=len(pcts) - 1)

    peak_stds, peak_means, peak_labels, peak_colors = [], [], [], []
    for j, pct in enumerate(pcts):
        row = peaks[peaks["percentile"] == pct]
        if row.empty:
            continue
        t_star = row.iloc[0]["theta_star"]
        # Find nearest θ in the sweep
        idx = np.abs(thetas - t_star).argmin()
        color = cw(pct_norm(j))
        peak_stds.append(stds[idx])
        peak_means.append(means[idx])
        peak_labels.append(pct)
        peak_colors.append(color)

    # Connect peaks with a line to show the trajectory
    if len(peak_stds) > 1:
        ax.plot(
            peak_stds, peak_means,
            color="black", linewidth=1.5, linestyle="--", alpha=0.4, zorder=4,
        )

    # Plot markers
    for i, (s, m, lab, c) in enumerate(zip(peak_stds, peak_means, peak_labels, peak_colors)):
        ax.scatter(
            s, m, color=c, s=160, zorder=6,
            edgecolors="black", linewidths=0.8, marker="o",
        )
        # Alternate label offsets to avoid overlap
        offset_x = 10 if i % 2 == 0 else -10
        ha = "left" if i % 2 == 0 else "right"
        ax.annotate(
            lab,
            (s, m),
            fontsize=9.5, fontweight="bold",
            color=c,
            xytext=(offset_x, 6), textcoords="offset points",
            ha=ha, va="bottom",
            arrowprops=dict(arrowstyle="-", alpha=0.3, linewidth=0.5),
        )

    # --- θ=0 reference marker ---
    ev_idx = np.abs(thetas).argmin()
    ax.scatter(
        stds[ev_idx], means[ev_idx],
        color="black", s=250, marker="*", zorder=10,
        label="θ=0 (EV-optimal)",
    )

    # --- Percentile colorbar ---
    sm = plt.cm.ScalarMappable(
        cmap=cw,
        norm=mcolors.Normalize(vmin=0, vmax=len(pcts) - 1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Optimized percentile (blue=low, red=high)", fontsize=12)
    cbar.set_ticks(range(len(pcts)))
    cbar.set_ticklabels(pcts)

    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.set_xlabel("Standard Deviation", fontsize=14)
    ax.set_ylabel("Mean Score", fontsize=14)
    ax.set_title(
        "Mean–Std Frontier: Where Each Percentile's Optimal θ* Lands",
        fontsize=17, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "percentile_frontiers.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
