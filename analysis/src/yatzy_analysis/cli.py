"""CLI entry point: yatzy-analyze extract / compute / plot / run / summary."""
from __future__ import annotations

import time
from pathlib import Path

import click


@click.group()
def cli():
    """Yatzy risk-sweep analysis pipeline."""


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
def extract(base_path: str):
    """Step 1: Read raw binaries → scores.parquet."""
    from .config import analysis_dir, discover_thetas
    from .io import fmt_theta, read_all_scores
    from .store import save_scores

    t0 = time.time()
    thetas = discover_thetas(base_path)
    if not thetas:
        click.echo("No theta directories with simulation_raw.bin found.")
        raise SystemExit(1)

    click.echo(f"Extracting scores for {len(thetas)} thetas...")
    scores = read_all_scores(thetas, base_path)
    for t in sorted(scores.keys()):
        click.echo(f"  theta={fmt_theta(t, base_path):>6s}: {len(scores[t]):>10,d} games")

    out = analysis_dir(base_path) / "scores.parquet"
    save_scores(scores, out)
    click.echo(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
def compute(base_path: str):
    """Step 2: scores.parquet → summary.parquet + kde.parquet."""
    from .compute import compute_all
    from .config import analysis_dir
    from .store import load_scores, save_kde, save_summary

    t0 = time.time()
    scores_path = analysis_dir(base_path) / "scores.parquet"
    if not scores_path.exists():
        click.echo(f"{scores_path} not found. Run 'yatzy-analyze extract' first.")
        raise SystemExit(1)

    click.echo("Loading scores.parquet...")
    scores_dict = load_scores(scores_path)
    click.echo(f"  {len(scores_dict)} thetas loaded.")

    click.echo("Computing summary stats + KDE...")
    summary_df, kde_df = compute_all(scores_dict)

    out_dir = analysis_dir(base_path)
    save_summary(summary_df, out_dir / "summary.parquet")
    save_kde(kde_df, out_dir / "kde.parquet")
    click.echo(f"Saved summary.parquet ({len(summary_df)} rows)")
    click.echo(f"Saved kde.parquet ({len(kde_df)} rows)")
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
@click.option(
    "--subset", default="all", type=click.Choice(["all", "dense", "sparse"]),
    help="Which theta subset to plot.",
)
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "svg"]))
@click.option("--dpi", default=200, type=int)
def plot(base_path: str, subset: str, fmt: str, dpi: int):
    """Step 3: summary.parquet + kde.parquet → plot PNGs/SVGs."""
    import matplotlib
    matplotlib.use("Agg")

    from .config import PLOT_SUBSETS, analysis_dir, plots_dir
    from .plots.cdf import plot_cdf, plot_tails
    from .plots.combined import plot_combined
    from .plots.density import plot_density
    from .plots.mean_std import plot_mean_vs_std
    from .plots.percentiles import plot_percentiles
    from .plots.quantile import plot_quantile
    from .store import load_kde, load_summary

    t0 = time.time()
    adir = analysis_dir(base_path)
    summary_path = adir / "summary.parquet"
    kde_path = adir / "kde.parquet"
    for p in [summary_path, kde_path]:
        if not p.exists():
            click.echo(f"{p} not found. Run 'yatzy-analyze compute' first.")
            raise SystemExit(1)

    summary_all = load_summary(summary_path)
    kde_all = load_kde(kde_path)

    if subset == "all":
        subsets_to_plot = list(PLOT_SUBSETS.items())
    elif subset == "dense":
        subsets_to_plot = [("theta_dense", PLOT_SUBSETS["theta_dense"])]
    elif subset == "sparse":
        subsets_to_plot = [("theta_sparse", PLOT_SUBSETS["theta_sparse"])]

    for subset_name, theta_list in subsets_to_plot:
        out_dir = plots_dir(base_path) / subset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter to available thetas
        available = set(summary_all["theta"].unique())
        thetas = sorted([t for t in theta_list if t in available])

        stats_df = summary_all[summary_all["theta"].isin(thetas)]
        k_df = kde_all[kde_all["theta"].isin(thetas)]
        # kde.parquet has cdf and survival columns, so it serves as cdf_df too
        cdf_df = k_df

        click.echo(f"[{subset_name}] {len(thetas)} thetas → {out_dir}/")

        plot_cdf(thetas, cdf_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  cdf_full.{fmt}")
        plot_tails(thetas, cdf_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  tails_zoomed.{fmt}")
        plot_percentiles(thetas, stats_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  percentiles_vs_theta.{fmt}")
        plot_mean_vs_std(thetas, stats_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  mean_vs_std.{fmt}")
        plot_density(thetas, k_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  density.{fmt}")
        plot_quantile(thetas, cdf_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  quantile.{fmt}")
        plot_combined(thetas, cdf_df, stats_df, k_df, out_dir, dpi=dpi, fmt=fmt)
        click.echo(f"  combined.{fmt}")

    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
def run(base_path: str):
    """Run full pipeline: extract → compute → plot."""
    ctx = click.get_current_context()
    ctx.invoke(extract, base_path=base_path)
    ctx.invoke(compute, base_path=base_path)
    ctx.invoke(plot, base_path=base_path, subset="all", fmt="png", dpi=200)


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
@click.option("--scores-bin", default=None, help="Direct path to scores.bin (old format).")
def tail(base_path: str, scores_bin: str | None):
    """Tail distribution analysis for max-policy simulations."""
    from .tail_analysis import (
        analytical_p374,
        fit_log_linear,
        load_scores_bin,
        tail_distribution,
    )

    # Load scores from either old format or new format
    scores = None
    if scores_bin is not None:
        p = Path(scores_bin)
        if not p.exists():
            click.echo(f"File not found: {scores_bin}")
            raise SystemExit(1)
        click.echo(f"Loading scores from {scores_bin} (legacy format)...")
        scores = load_scores_bin(p)
    else:
        from .io import read_scores

        raw_path = Path(base_path) / "max_policy" / "simulation_raw.bin"
        if raw_path.exists():
            click.echo(f"Loading scores from {raw_path}...")
            scores = read_scores(raw_path)
        else:
            # Try legacy fallback
            legacy = Path(base_path) / "max_policy" / "scores.bin"
            if legacy.exists():
                click.echo(f"Loading scores from {legacy} (legacy format)...")
                scores = load_scores_bin(legacy)

    click.echo()
    click.echo("=" * 65)
    click.echo("  Max-Policy Tail Distribution Analysis")
    click.echo("=" * 65)
    click.echo()

    if scores is not None and len(scores) > 0:
        # Tail distribution table
        td = tail_distribution(scores)
        click.echo(f"Loaded {td['n']:,} scores")
        click.echo(f"Mean: {td['mean']:.2f}")
        click.echo(f"Max:  {td['max']}")
        click.echo()
        click.echo("Score threshold | Count >= | Fraction     | 1/Fraction")
        click.echo("----------------|---------|--------------|------------")
        for row in td["thresholds"]:
            click.echo(
                f"  {row['threshold']:>13}  | {row['count']:>7,} | "
                f"{row['fraction']:.6e} | {row['inv_fraction']:.1e}"
            )
        click.echo()

        # Log-linear fit
        lf = fit_log_linear(scores)
        if lf["n_points"] >= 2:
            click.echo(
                f"Log-linear fit: log10(P(X >= t)) = "
                f"{lf['a']:.4f} + {lf['b']:.6f} * t"
            )
            click.echo(
                f"  Decay rate: ~10^({lf['b']:.4f}) per point "
                f"= {lf['decay_per_point']:.6f}x per point"
            )
            click.echo()
            click.echo(
                f"Extrapolated P(score >= 374) ~ "
                f"10^({lf['log10_p374']:.1f}) ~ {lf['p374']:.2e}"
            )
            click.echo(
                f"Expected games to see 374:    ~ {lf['games_needed']:.2e}"
            )
            click.echo(
                "(NOTE: this underestimates -- see analytical estimate below)"
            )
        else:
            click.echo("Insufficient data points for log-linear fit.")
        click.echo()
    else:
        click.echo("No scores found. Skipping empirical analysis.")
        click.echo()

    # Analytical estimate
    click.echo("=" * 65)
    click.echo("  Analytical Estimate: P(score = 374)")
    click.echo("=" * 65)
    click.echo()

    result = analytical_p374()
    click.echo(
        f"P(five of specific face, 3 rolls) = "
        f"{result['p_five_of_specific_face']:.6f} = "
        f"1 in {1 / result['p_five_of_specific_face']:.0f}"
    )
    click.echo()
    click.echo("Per-category probabilities (3 rolls, optimal keeping):")
    click.echo("-" * 65)
    for cat in result["categories"]:
        click.echo(
            f"  {cat['name']:<35s}  P = {cat['probability']:.6f}"
            f"  (1 in {cat['inv_probability']:,.0f})"
        )
    click.echo()
    click.echo(f"Product (assuming independence): {result['overall']:.3e}")
    click.echo(f"Expected games for 374:          {result['games_needed']:.3e}")
    click.echo()
    click.echo(
        f"At {result['games_per_sec']:,} games/sec: "
        f"~{result['years']:.2e} years"
    )


@cli.command()
@click.option("--base-path", default="results", help="Path to results directory.")
def summary(base_path: str):
    """Print summary table to console (from summary.parquet)."""
    from .config import analysis_dir
    from .store import load_summary

    summary_path = analysis_dir(base_path) / "summary.parquet"
    if not summary_path.exists():
        click.echo(f"{summary_path} not found. Run 'yatzy-analyze compute' first.")
        raise SystemExit(1)

    df = load_summary(summary_path)

    header = (
        f"{'theta':>6s} | {'n':>10s} | {'bot5':>7s} | {'min':>5s} | {'p5':>5s} | "
        f"{'p10':>5s} | {'p25':>5s} | {'p50':>5s} | {'p75':>5s} | {'p90':>5s} | "
        f"{'p95':>5s} | {'p99':>5s} | {'max':>5s} | {'top5':>7s} | {'mean':>7s} | {'std':>5s}"
    )
    click.echo("=== Score Distribution Comparison ===")
    click.echo()
    click.echo(header)
    click.echo("-" * len(header))

    for _, row in df.iterrows():
        t = row["theta"]
        tstr = f"{t:g}" if t != int(t) or abs(t) < 1 else f"{int(t)}"
        click.echo(
            f"{tstr:>6s} | {int(row['n']):>10,d} | {row['bot5_avg']:>7.1f} | "
            f"{int(row['min']):>5d} | {int(row['p5']):>5d} | {int(row['p10']):>5d} | "
            f"{int(row['p25']):>5d} | {int(row['p50']):>5d} | {int(row['p75']):>5d} | "
            f"{int(row['p90']):>5d} | {int(row['p95']):>5d} | {int(row['p99']):>5d} | "
            f"{int(row['max']):>5d} | {row['top5_avg']:>7.1f} | {row['mean']:>7.2f} | "
            f"{row['std']:>5.1f}"
        )

    click.echo()
    click.echo("=== Best theta per Metric ===")
    for metric in ["bot5_avg", "min", "p5", "p10", "p25", "p50",
                    "p75", "p90", "p95", "p99", "max", "top5_avg"]:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        val = best_row[metric]
        tstr = f"{best_row['theta']:g}"
        vstr = f"{val:.1f}" if isinstance(val, float) and val != int(val) else str(int(val))
        click.echo(f"  {metric:>8s}: theta = {tstr:>6s} (score = {vstr})")
