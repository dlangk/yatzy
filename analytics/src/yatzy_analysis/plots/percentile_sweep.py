"""Percentile sweep plots: curves, heatmap, frontier, and peak annotations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
import seaborn as sns

from .style import CMAP, FONT_AXIS_LABEL, FONT_LEGEND, FONT_SUPTITLE, FONT_TICK, FONT_TITLE, GRID_ALPHA, PERCENTILES_CORE, PERCENTILES_EXTRA, setup_theme

_CORE = PERCENTILES_CORE
_EXTRA = PERCENTILES_EXTRA
_ALL = _EXTRA[:1] + _CORE + _EXTRA[1:]  # p1, p5..p99, p999, p9999


def _plot_curves(
    df: pl.DataFrame,
    ax,
    *,
    skip: set[str] | None = None,
) -> None:
    """Draw percentile curves matching the rocket-palette style."""
    skip = skip or set()
    pct_colors = sns.color_palette("rocket", len(_CORE))

    # Convert to pandas for plotting convenience (column access + matplotlib)
    pdf = df.to_pandas()

    for i, p in enumerate(_CORE):
        if p in pdf.columns and p not in skip:
            ax.plot(
                pdf["theta"], pdf[p],
                marker="o", markersize=4, linewidth=1.8, color=pct_colors[i],
                label=p, zorder=3,
            )

    extra_styles = {
        "p1": ("--", "tab:blue"),
        "p999": ("--", "tab:orange"),
        "p9999": ("--", "tab:red"),
    }
    for p, (ls, color) in extra_styles.items():
        if p in pdf.columns and p not in skip:
            ax.plot(
                pdf["theta"], pdf[p],
                linestyle=ls, marker="s", markersize=3, linewidth=1.4,
                color=color, alpha=0.8, label=p, zorder=3,
            )


def _add_peak_stars(ax, df: pl.DataFrame, peaks: pl.DataFrame, theta_range=None, skip=None) -> None:
    """Add star markers at each percentile's peak θ*."""
    pct_colors = sns.color_palette("rocket", len(_CORE))
    color_map = {p: pct_colors[i] for i, p in enumerate(_CORE)}
    extra_colors = {"p1": "tab:blue", "p999": "tab:orange", "p9999": "tab:red"}
    color_map.update(extra_colors)
    skip = skip or set()

    for row in peaks.iter_rows(named=True):
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


def _merge_with_coarse(sweep_df: pl.DataFrame, coarse_path: Path | None) -> pl.DataFrame:
    """Merge dense sweep with coarse summary for wider θ coverage."""
    if coarse_path is None or not coarse_path.exists():
        return sweep_df

    coarse = pl.read_parquet(coarse_path)

    # Only keep coarse rows outside the sweep range
    lo = sweep_df["theta"].min()
    hi = sweep_df["theta"].max()
    outside = coarse.filter(
        (pl.col("theta") < lo - 0.001) | (pl.col("theta") > hi + 0.001)
    )

    if outside.is_empty():
        return sweep_df

    # Concat, sort, then interpolate NaN percentile columns via pandas
    merged = pl.concat([sweep_df, outside], how="diagonal")
    merged = merged.sort("theta")

    # Interpolate NaN percentile columns — use pandas for index-based interpolation
    pct_cols = [c for c in merged.columns if c.startswith("p")]
    pdf = merged.to_pandas()
    pdf[pct_cols] = pdf[pct_cols].interpolate(method="index")
    return pl.from_pandas(pdf)


def plot_percentile_sweep(
    sweep_path: Path,
    peaks_path: Path,
    out_dir: Path,
    *,
    coarse_path: Path | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> list[Path]:
    """Generate all percentile sweep plots. Returns list of output paths."""
    setup_theme()
    df = pl.read_csv(sweep_path)
    peaks = pl.read_csv(peaks_path)
    df = df.sort("theta")

    # Merge with coarse grid for wider-range plots
    df_wide = _merge_with_coarse(df, coarse_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # --- Plot 1: Dense sweep only ---
    paths.append(_plot_full(df, peaks, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 2: Wide range (±1), merged with coarse ---
    paths.append(_plot_wide(df_wide, peaks, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 3: Mean vs theta with std band ---
    paths.append(_plot_mean_std(df, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 4: Heatmap ---
    paths.append(_plot_heatmap(df, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 5: Cost-benefit scatter ---
    paths.append(_plot_mean_cost(df, peaks, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 6: Mean–Std frontier (dense sweep + coarse extensions) ---
    paths.append(_plot_frontier(df, df_wide, peaks, out_dir, dpi=dpi, fmt=fmt))

    # --- Plot 7: Multi-percentile frontiers in (mean, p) space ---
    paths.append(_plot_percentile_frontiers(df, peaks, out_dir, dpi=dpi, fmt=fmt))

    print(f"  Generated {len(paths)} percentile sweep plots in {out_dir}")
    return paths


def _plot_full(
    df: pl.DataFrame,
    peaks: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))

    _plot_curves(df, ax)
    _add_peak_stars(ax, df, peaks)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score Percentiles vs Risk Parameter θ", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(loc="lower left", fontsize=FONT_LEGEND, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"percentile_sweep_curves.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_wide(
    df: pl.DataFrame,
    peaks: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Wide range: θ ∈ [-1.0, +1.0], merging dense sweep + coarse grid."""
    wdf = df.filter(
        (pl.col("theta") >= -1.0) & (pl.col("theta") <= 1.0)
    )
    if wdf.is_empty():
        wdf = df

    fig, ax = plt.subplots(figsize=(14, 8))

    _plot_curves(wdf, ax)
    _add_peak_stars(ax, wdf, peaks, theta_range=(-1.0, 1.0))

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Score Percentiles vs θ  (|θ| ≤ 1)",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=FONT_LEGEND, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"percentile_sweep_wide.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_mean_std(
    df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Mean score vs θ with ±1 std band."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract numpy arrays for plotting
    theta_arr = df["theta"].to_numpy()
    mean_arr = df["mean"].to_numpy()
    std_arr = df["std"].to_numpy()

    ax.fill_between(
        theta_arr,
        mean_arr - std_arr,
        mean_arr + std_arr,
        alpha=0.2, color="#3182ce",
    )
    ax.plot(theta_arr, mean_arr, linewidth=2.5, color="#2b6cb0", label="Mean ± 1σ")

    ev_idx = int(np.abs(theta_arr).argmin())
    ax.plot(theta_arr[ev_idx], mean_arr[ev_idx], marker="*", markersize=14, color="#e53e3e", zorder=5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("θ  (risk parameter)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Mean Score vs θ  (risk-averse ← 0 → risk-seeking)", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"percentile_sweep_mean.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_heatmap(
    df: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """Heatmap: rows = percentiles, columns = θ, cells = score values."""
    if len(df) > 40:
        step = max(1, len(df) // 40)
        sdf = df.gather_every(step)
    else:
        sdf = df

    pcts_available = [p for p in _ALL if p in sdf.columns]
    matrix = sdf.select(pcts_available).to_numpy().T.astype(float)
    theta_labels = [f"{t:+.3f}" for t in sdf["theta"].to_list()]

    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

    ax.set_yticks(range(len(pcts_available)))
    ax.set_yticklabels(pcts_available, fontsize=FONT_TICK)
    ax.set_xticks(range(len(theta_labels)))
    ax.set_xticklabels(theta_labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("θ", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Score Percentile Heatmap across θ", fontsize=FONT_TITLE, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Score", fontsize=11)

    fig.tight_layout()
    path = out_dir / f"percentile_sweep_heatmap.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_mean_cost(
    df: pl.DataFrame,
    peaks: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
) -> Path:
    """For each percentile peak θ*, show the percentile gain vs mean cost."""
    theta_arr = df["theta"].to_numpy()
    ev_idx = int(np.abs(theta_arr).argmin())
    ev_row = df.row(ev_idx, named=True)
    ev_mean = ev_row["mean"]

    pct_colors = sns.color_palette("rocket", len(_CORE))
    color_map = {p: pct_colors[i] for i, p in enumerate(_CORE)}
    color_map.update({"p1": "tab:blue", "p999": "tab:orange", "p9999": "tab:red"})

    fig, ax = plt.subplots(figsize=(10, 7))

    xs, ys, labels, colors = [], [], [], []
    for row in peaks.iter_rows(named=True):
        pct = row["percentile"]
        if pct not in color_map or pct not in ev_row:
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
    ax.set_xlabel("Mean Score Cost  (EV-optimal mean − mean@θ*)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Percentile Gain  (value@θ* − value@θ=0)", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Cost-Benefit: Tail Optimization vs Mean Score", fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"percentile_sweep_cost_benefit.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_frontier(
    df_dense: pl.DataFrame,
    df_wide: pl.DataFrame,
    peaks: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
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
    dense_lo = df_dense["theta"].min()
    dense_hi = df_dense["theta"].max()
    coarse_only = df_wide.filter(
        (pl.col("theta") < dense_lo - 0.001) | (pl.col("theta") > dense_hi + 0.001)
    )
    if len(coarse_only) > 0:
        norm_wide = mcolors.Normalize(
            vmin=df_wide["theta"].min(), vmax=df_wide["theta"].max(),
        )
        for row in coarse_only.iter_rows(named=True):
            ax.scatter(
                row["std"], row["mean"],
                color=CMAP(norm_wide(row["theta"])),
                s=40, alpha=0.35, zorder=2,
                edgecolors="gray", linewidths=0.3,
            )

    # --- Dense sweep: colored line segments by θ ---
    ds = df_dense.sort("theta")
    norm = mcolors.Normalize(vmin=ds["theta"].min(), vmax=ds["theta"].max())

    # Draw line segments colored by θ (using LineCollection for gradient)
    stds = ds["std"].to_numpy()
    means = ds["mean"].to_numpy()
    thetas = ds["theta"].to_numpy()

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
    ev_idx = int(np.abs(thetas).argmin())
    ax.scatter(
        stds[ev_idx], means[ev_idx],
        color="black", s=200, marker="*", zorder=10,
        label="θ=0 (EV-optimal)",
    )

    # --- Peak p95 marker ---
    p95_peak = peaks.filter(pl.col("percentile") == "p95")
    if not p95_peak.is_empty():
        t_star = p95_peak["theta_star"][0]
        idx = int(np.abs(thetas - t_star).argmin())
        r_std = stds[idx]
        r_mean = means[idx]
        ax.scatter(
            r_std, r_mean,
            color="#e53e3e", s=180, marker="D", zorder=10,
            edgecolors="black", linewidths=0.8,
            label=f"θ*_p95 = {t_star:+.2f}",
        )

    # --- Sparse θ labels along the curve ---
    label_thetas = [-0.08, -0.04, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    labeled = set()
    for t_target in label_thetas:
        idx = int(np.abs(thetas - t_target).argmin())
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
    cbar.set_label("θ  (risk-averse ← 0 → risk-seeking)", fontsize=FONT_AXIS_LABEL)

    ax.set_xlabel("Standard Deviation", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Mean–Variance Frontier across Risk Parameter θ",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"mean_std_frontier.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_percentile_frontiers(
    df: pl.DataFrame,
    peaks: pl.DataFrame,
    out_dir: Path,
    *,
    dpi: int = 200,
    fmt: str = "png",
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

    ds = df.sort("theta")
    thetas = ds["theta"].to_numpy()
    means = ds["mean"].to_numpy()
    stds = ds["std"].to_numpy()

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
        row = peaks.filter(pl.col("percentile") == pct)
        if row.is_empty():
            continue
        t_star = row["theta_star"][0]
        # Find nearest θ in the sweep
        idx = int(np.abs(thetas - t_star).argmin())
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
    ev_idx = int(np.abs(thetas).argmin())
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
    cbar.set_label("Optimized percentile (blue=low, red=high)", fontsize=FONT_AXIS_LABEL)
    cbar.set_ticks(range(len(pcts)))
    cbar.set_ticklabels(pcts)

    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    ax.set_xlabel("Standard Deviation", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean Score", fontsize=FONT_AXIS_LABEL)
    ax.set_title(
        "Mean–Std Frontier: Where Each Percentile's Optimal θ* Lands",
        fontsize=FONT_SUPTITLE, fontweight="bold",
    )
    ax.grid(True, alpha=GRID_ALPHA)

    fig.tight_layout()
    path = out_dir / f"percentile_frontiers.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
