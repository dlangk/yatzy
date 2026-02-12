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
