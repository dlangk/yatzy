"""CLI entry point: yatzy-analyze extract / compute / plot / run / summary."""
from __future__ import annotations

import time
from pathlib import Path

import click


@click.group()
def cli():
    """Yatzy risk-sweep analysis pipeline."""


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
def extract(base_path: str):
    """[Legacy] Read raw binaries → scores.parquet. Prefer 'compute' instead."""
    from .config import analysis_dir, discover_thetas
    from .io import fmt_theta, read_all_scores
    from .store import save_scores

    t0 = time.time()
    thetas = discover_thetas(base_path)
    if not thetas:
        click.echo("No theta directories with simulation data found.")
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
@click.option("--base-path", default=".", help="Base path (repo root).")
def compute(base_path: str):
    """Read simulation binaries directly → summary.parquet + kde.parquet."""
    from .compute import compute_all
    from .config import analysis_dir, discover_thetas
    from .io import fmt_theta, read_all_scores
    from .store import save_kde, save_summary

    t0 = time.time()

    # Read scores directly from binary files (scores.bin or simulation_raw.bin)
    thetas = discover_thetas(base_path)
    if not thetas:
        click.echo("No theta directories with simulation data found.")
        raise SystemExit(1)

    click.echo(f"Reading scores for {len(thetas)} thetas...")
    scores_dict = read_all_scores(thetas, base_path)
    for t in sorted(scores_dict.keys()):
        click.echo(f"  theta={fmt_theta(t, base_path):>6s}: {len(scores_dict[t]):>10,d} games")

    click.echo("Computing summary stats + KDE...")
    summary_df, kde_df = compute_all(scores_dict)

    out_dir = analysis_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary(summary_df, out_dir / "summary.parquet")
    save_kde(kde_df, out_dir / "kde.parquet")
    click.echo(f"Saved summary.parquet ({len(summary_df)} rows)")
    click.echo(f"Saved kde.parquet ({len(kde_df)} rows)")
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
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

    theta_list = PLOT_SUBSETS.get(
        {"all": "all", "dense": "theta_dense", "sparse": "theta_sparse"}[subset],
        PLOT_SUBSETS["all"],
    )

    out_dir = plots_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to available thetas
    available = set(summary_all["theta"].unique())
    thetas = sorted([t for t in theta_list if t in available])

    stats_df = summary_all[summary_all["theta"].isin(thetas)]
    k_df = kde_all[kde_all["theta"].isin(thetas)]
    cdf_df = k_df

    click.echo(f"[{subset}] {len(thetas)} thetas → {out_dir}/")

    plot_cdf(thetas, cdf_df, out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"  cdf_full.{fmt}")
    plot_tails(thetas, cdf_df, out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"  tails_zoomed.{fmt}")
    plot_percentiles(thetas, stats_df, out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"  percentiles_vs_theta.{fmt}")
    click.echo(f"  percentiles_vs_theta_zoomed.{fmt}")
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
@click.option("--base-path", default=".", help="Base path (repo root).")
def efficiency(base_path: str):
    """Compute and display risk-seeking efficiency metrics (MER, SDVA, CVaR)."""
    from .compute import compute_all_sdva, compute_exchange_rates
    from .config import analysis_dir, plots_dir
    from .store import load_kde, load_summary

    t0 = time.time()
    adir = analysis_dir(base_path)
    summary_path = adir / "summary.parquet"
    kde_path = adir / "kde.parquet"
    for p in [summary_path, kde_path]:
        if not p.exists():
            click.echo(f"{p} not found. Run 'yatzy-analyze compute' first.")
            raise SystemExit(1)

    summary_df = load_summary(summary_path)
    kde_df = load_kde(kde_path)
    thetas = sorted(summary_df["theta"].unique().tolist())

    click.echo("Computing exchange rates...")
    mer_df = compute_exchange_rates(summary_df)

    click.echo("Computing stochastic dominance violation areas...")
    sdva_df = compute_all_sdva(kde_df, thetas)

    # Save to parquet
    out_dir = adir
    mer_df.to_parquet(out_dir / "mer.parquet", engine="pyarrow", index=False)
    sdva_df.to_parquet(out_dir / "sdva.parquet", engine="pyarrow", index=False)
    click.echo(f"Saved mer.parquet ({len(mer_df)} rows)")
    click.echo(f"Saved sdva.parquet ({len(sdva_df)} rows)")

    # Print MER table
    click.echo()
    click.echo("=== Marginal Exchange Rates (positive θ only) ===")
    click.echo(
        f"{'theta':>6s} | {'mean_cost':>9s} | {'MER_p75':>8s} | {'MER_p90':>8s} | "
        f"{'MER_p95':>8s} | {'MER_p99':>8s} | {'MER_max':>8s}"
    )
    click.echo("-" * 72)
    for _, row in mer_df[mer_df["theta"] > 0].iterrows():
        def _fmt(v: float) -> str:
            if abs(v) > 99 or v == float("inf") or v == float("-inf"):
                return "    dom"
            return f"{v:8.1f}"
        t = row["theta"]
        tstr = f"{t:g}" if t != int(t) or abs(t) < 1 else f"{int(t)}"
        click.echo(
            f"{tstr:>6s} | {row['mean_cost']:>9.1f} | {_fmt(row['mer_p75'])} | "
            f"{_fmt(row['mer_p90'])} | {_fmt(row['mer_p95'])} | "
            f"{_fmt(row['mer_p99'])} | {_fmt(row['mer_max'])}"
        )

    # Print SDVA table
    click.echo()
    click.echo("=== Stochastic Dominance Violation Areas (positive θ only) ===")
    click.echo(
        f"{'theta':>6s} | {'A_worse':>8s} | {'A_better':>8s} | {'ratio':>7s} | {'x_cross':>7s}"
    )
    click.echo("-" * 50)
    for _, row in sdva_df[(sdva_df["theta"] > 0) & (sdva_df["theta"] <= 0.5)].iterrows():
        t = row["theta"]
        tstr = f"{t:g}" if t != int(t) or abs(t) < 1 else f"{int(t)}"
        rstr = f"{row['ratio']:7.1f}" if row["ratio"] < 999 else "    inf"
        click.echo(
            f"{tstr:>6s} | {row['a_worse']:8.1f} | {row['a_better']:8.1f} | "
            f"{rstr} | {row['x_cross']:7.0f}"
        )

    # Print CVaR table
    click.echo()
    click.echo("=== CVaR Deficit (positive θ only) ===")
    click.echo(
        f"{'theta':>6s} | {'mean':>7s} | {'CVaR_1':>7s} | {'CVaR_5':>7s} | {'CVaR_10':>7s} | "
        f"{'deficit_5':>9s}"
    )
    click.echo("-" * 58)
    base = summary_df[summary_df["theta"] == 0.0]
    base_cvar5 = base.iloc[0]["cvar_5"] if not base.empty else 0.0
    for _, row in summary_df[(summary_df["theta"] >= 0) & (summary_df["theta"] <= 0.5)].iterrows():
        t = row["theta"]
        tstr = f"{t:g}" if t != int(t) or abs(t) < 1 else f"{int(t)}"
        deficit = row["cvar_5"] - base_cvar5
        click.echo(
            f"{tstr:>6s} | {row['mean']:>7.1f} | {row['cvar_1']:>7.1f} | "
            f"{row['cvar_5']:>7.1f} | {row['cvar_10']:>7.1f} | {deficit:>+9.1f}"
        )

    # Generate efficiency plot
    import matplotlib
    matplotlib.use("Agg")
    from .plots.efficiency import plot_efficiency

    pdir = plots_dir(base_path)
    pdir.mkdir(parents=True, exist_ok=True)
    plot_efficiency(thetas, summary_df, kde_df, mer_df, sdva_df, pdir)
    click.echo(f"\nSaved {pdir}/efficiency.png")

    click.echo(f"\nDone in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "svg"]))
@click.option("--dpi", default=200, type=int)
def categories(base_path: str, fmt: str, dpi: int):
    """Plot per-category statistics from category_stats.csv."""
    import matplotlib
    matplotlib.use("Agg")

    from .config import plots_dir
    from .plots.categories import plot_all_category_stats

    t0 = time.time()
    csv_path = Path(base_path) / "outputs" / "aggregates" / "csv" / "category_stats.csv"
    if not csv_path.exists():
        click.echo(f"{csv_path} not found. Run yatzy-category-sweep first.")
        raise SystemExit(1)

    out_dir = plots_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all_category_stats(csv_path, out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command("yatzy-hypothesis")
@click.option("--base-path", default=".", help="Base path (repo root).")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "svg"]))
@click.option("--dpi", default=200, type=int)
def yatzy_hypothesis(base_path: str, fmt: str, dpi: int):
    """Plot Yatzy conditional hit-rate hypothesis tests from yatzy_conditional.csv."""
    import matplotlib
    matplotlib.use("Agg")

    from .config import plots_dir
    from .plots.yatzy_hypothesis import plot_all_yatzy_hypothesis

    t0 = time.time()
    csv_path = Path(base_path) / "outputs" / "aggregates" / "csv" / "yatzy_conditional.csv"
    if not csv_path.exists():
        click.echo(f"{csv_path} not found. Run yatzy-conditional first.")
        raise SystemExit(1)

    out_dir = plots_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all_yatzy_hypothesis(csv_path, out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
def run(base_path: str):
    """Run full pipeline: compute → plot → efficiency."""
    ctx = click.get_current_context()
    ctx.invoke(compute, base_path=base_path)
    ctx.invoke(plot, base_path=base_path, subset="all", fmt="png", dpi=200)
    ctx.invoke(efficiency, base_path=base_path)


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
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

        from .config import bin_files_dir

        # Try compact format first
        compact_path = bin_files_dir(base_path) / "max_policy" / "scores.bin"
        raw_path = bin_files_dir(base_path) / "max_policy" / "simulation_raw.bin"
        if compact_path.exists():
            click.echo(f"Loading scores from {compact_path}...")
            scores = read_scores(compact_path)
        elif raw_path.exists():
            click.echo(f"Loading scores from {raw_path}...")
            scores = read_scores(raw_path)

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
@click.option("--base-path", default=".", help="Base path (repo root).")
def adaptive(base_path: str):
    """Analyze adaptive θ policy simulations and compare to fixed-θ frontier."""
    from .adaptive import (
        compute_adaptive_kde,
        compute_adaptive_summary,
        discover_adaptive_policies,
        read_adaptive_scores,
    )
    from .config import analysis_dir, plots_dir
    from .store import load_summary

    t0 = time.time()

    policies = discover_adaptive_policies(base_path)
    if not policies:
        click.echo("No adaptive policy results found in data/simulations/adaptive/.")
        click.echo("Run simulations first:")
        click.echo("  yatzy-simulate --policy bonus-adaptive --games 1000000 --output ...")
        raise SystemExit(1)

    click.echo(f"Found {len(policies)} adaptive policies: {', '.join(policies)}")

    # Read scores
    scores = read_adaptive_scores(policies, base_path)
    for name in sorted(scores.keys()):
        click.echo(f"  {name}: {len(scores[name]):>10,d} games")

    # Compute summary and KDE
    click.echo("Computing summary stats...")
    adaptive_summary = compute_adaptive_summary(scores)
    adaptive_kde = compute_adaptive_kde(scores)

    # Save
    adir = analysis_dir(base_path)
    adir.mkdir(parents=True, exist_ok=True)
    adaptive_summary.to_parquet(adir / "adaptive_summary.parquet", engine="pyarrow", index=False)
    click.echo(f"Saved adaptive_summary.parquet ({len(adaptive_summary)} rows)")

    if not adaptive_kde.empty:
        adaptive_kde.to_parquet(adir / "adaptive_kde.parquet", engine="pyarrow", index=False)
        click.echo(f"Saved adaptive_kde.parquet ({len(adaptive_kde)} rows)")

    # Print comparison table
    click.echo()
    click.echo("=== Adaptive Policy Results ===")
    header = (
        f"{'policy':>16s} | {'n':>10s} | {'mean':>7s} | {'std':>5s} | "
        f"{'p5':>5s} | {'p50':>5s} | {'p90':>5s} | {'p95':>5s} | {'p99':>5s} | "
        f"{'max':>5s} | {'bonus':>7s}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    # Also load and show the θ=0 baseline for comparison
    theta_summary_path = adir / "summary.parquet"
    if theta_summary_path.exists():
        theta_df = load_summary(theta_summary_path)
        base = theta_df[theta_df["theta"] == 0.0]
        if not base.empty:
            b = base.iloc[0]
            click.echo(
                f"{'θ=0 (baseline)':>16s} | {int(b['n']):>10,d} | {b['mean']:>7.2f} | "
                f"{b['std']:>5.1f} | {int(b['p5']):>5d} | {int(b['p50']):>5d} | "
                f"{int(b['p90']):>5d} | {int(b['p95']):>5d} | {int(b['p99']):>5d} | "
                f"{int(b['max']):>5d} |       -"
            )
        # Show best fixed-θ for p95
        if "p95" in theta_df.columns:
            peak_idx = theta_df["p95"].idxmax()
            pk = theta_df.loc[peak_idx]
            label = f"θ={pk['theta']:g} (best p95)"
            click.echo(
                f"{label:>16s} | {int(pk['n']):>10,d} | {pk['mean']:>7.2f} | "
                f"{pk['std']:>5.1f} | {int(pk['p5']):>5d} | {int(pk['p50']):>5d} | "
                f"{int(pk['p90']):>5d} | {int(pk['p95']):>5d} | {int(pk['p99']):>5d} | "
                f"{int(pk['max']):>5d} |       -"
            )
        click.echo("-" * len(header))

    for _, row in adaptive_summary.iterrows():
        # Compute bonus rate from raw scores if available
        click.echo(
            f"{row['policy']:>16s} | {int(row['n']):>10,d} | {row['mean']:>7.2f} | "
            f"{row['std']:>5.1f} | {int(row['p5']):>5d} | {int(row['p50']):>5d} | "
            f"{int(row['p90']):>5d} | {int(row['p95']):>5d} | {int(row['p99']):>5d} | "
            f"{int(row['max']):>5d} |       -"
        )

    # Generate efficiency frontier with adaptive overlay
    if theta_summary_path.exists():
        import matplotlib
        matplotlib.use("Agg")
        from .plots.efficiency import (
            plot_efficiency_adaptive_combined,
            plot_efficiency_adaptive_p99,
            plot_efficiency_adaptive_p999,
            plot_efficiency_with_adaptive,
        )

        pdir = plots_dir(base_path)
        pdir.mkdir(parents=True, exist_ok=True)
        theta_df = load_summary(theta_summary_path)
        thetas = sorted(theta_df["theta"].unique().tolist())
        plot_efficiency_with_adaptive(thetas, theta_df, adaptive_summary, pdir)
        click.echo(f"\nSaved {pdir}/efficiency_adaptive.png")
        plot_efficiency_adaptive_p99(thetas, theta_df, adaptive_summary, pdir)
        click.echo(f"Saved {pdir}/efficiency_adaptive_p99.png")
        if "p999" in theta_df.columns and "p999" in adaptive_summary.columns:
            plot_efficiency_adaptive_p999(thetas, theta_df, adaptive_summary, pdir)
            click.echo(f"Saved {pdir}/efficiency_adaptive_p999.png")
        plot_efficiency_adaptive_combined(thetas, theta_df, adaptive_summary, pdir)
        click.echo(f"Saved {pdir}/efficiency_adaptive_combined.png")

        # Density plot with adaptive overlays
        if not adaptive_kde.empty:
            from .plots.density import plot_density_with_adaptive
            from .store import load_kde

            kde_path = adir / "kde.parquet"
            if kde_path.exists():
                theta_kde = load_kde(kde_path)
                plot_density_with_adaptive(thetas, theta_kde, adaptive_kde, pdir)
                click.echo(f"Saved {pdir}/density_adaptive.png")

    click.echo(f"\nDone in {time.time() - t0:.1f}s.")


@cli.command("theta-questionnaire")
@click.option(
    "--scenarios", default="outputs/scenarios/pivotal_scenarios.json",
    help="Path to pivotal scenarios JSON.",
)
@click.option(
    "--save", "save_path", default="outputs/scenarios/answers.json",
    help="Path to save answers JSON.",
)
@click.option("--batch-size", default=15, type=int, help="Questions in first batch.")
def theta_questionnaire(scenarios: str, save_path: str, batch_size: int):
    """Interactive questionnaire to estimate your risk preference (θ)."""
    from .theta_estimator import run_questionnaire

    p = Path(scenarios)
    if not p.exists():
        click.echo(f"{p} not found. Generate scenarios first:")
        click.echo("  YATZY_BASE_PATH=. solver/target/release/pivotal-scenarios \\")
        click.echo("    --games 100000 --output outputs/scenarios/pivotal_scenarios.json")
        raise SystemExit(1)

    run_questionnaire(p, batch_size=batch_size, save_path=save_path)


@cli.command("theta-replay")
@click.option(
    "--scenarios", default="outputs/scenarios/pivotal_scenarios.json",
    help="Path to pivotal scenarios JSON.",
)
@click.option(
    "--answers", default="outputs/scenarios/answers.json",
    help="Path to saved answers JSON (scenario_id/category_id pairs or legacy indices).",
)
def theta_replay(scenarios: str, answers: str):
    """Replay saved answers and print θ estimation results."""
    from .theta_estimator import replay_answers

    for name, path in [("Scenarios", scenarios), ("Answers", answers)]:
        if not Path(path).exists():
            click.echo(f"{name} file not found: {path}")
            raise SystemExit(1)

    replay_answers(scenarios, answers)


@cli.command("theta-validate")
@click.option(
    "--scenarios", default="outputs/scenarios/pivotal_scenarios.json",
    help="Path to pivotal scenarios JSON.",
)
@click.option("--trials", default=100, type=int, help="Number of trials per (θ, β) pair.")
@click.option("--max-questions", default=30, type=int, help="Max questions per trial.")
def theta_validate(scenarios: str, trials: int, max_questions: int):
    """Validate θ estimator convergence with synthetic humans."""
    from .theta_estimator import load_scenarios
    from .theta_validation import print_validation_results, run_validation

    p = Path(scenarios)
    if not p.exists():
        click.echo(f"{p} not found. Generate scenarios first:")
        click.echo("  YATZY_BASE_PATH=. solver/target/release/pivotal-scenarios \\")
        click.echo("    --games 100000 --output outputs/scenarios/pivotal_scenarios.json")
        raise SystemExit(1)

    click.echo(f"Loading scenarios from {p}...")
    scenario_list = load_scenarios(p)
    click.echo(f"  {len(scenario_list)} scenarios loaded")
    click.echo()

    click.echo(f"Running convergence validation ({trials} trials, up to {max_questions} questions)...")
    results = run_validation(scenario_list, n_trials=trials, max_questions=max_questions)
    print_validation_results(results)


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "svg"]))
@click.option("--dpi", default=200, type=int)
def sensitivity(base_path: str, fmt: str, dpi: int):
    """Plot decision sensitivity analysis from yatzy-decision-sensitivity output."""
    import matplotlib
    matplotlib.use("Agg")

    from .config import plots_dir
    from .plots.sensitivity import plot_all_sensitivity

    t0 = time.time()
    out_dir = plots_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_all_sensitivity(base_path=base_path, out_dir=out_dir, dpi=dpi, fmt=fmt)
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command("state-flow")
@click.option("--base-path", default=".", help="Base path (repo root).")
@click.option("--dpi", default=200, type=int)
def state_flow(base_path: str, dpi: int):
    """Plot state-flow visualizations (alluvial, DAG, streamgraph) from state_frequency.csv."""
    import matplotlib
    matplotlib.use("Agg")

    from .config import plots_dir
    from .plots.state_flow import generate_state_flow_plots

    t0 = time.time()
    csv_path = Path(base_path) / "outputs" / "scenarios" / "state_frequency.csv"
    if not csv_path.exists():
        click.echo(f"{csv_path} not found. Run yatzy-decision-sensitivity first.")
        raise SystemExit(1)

    out_dir = plots_dir(base_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = generate_state_flow_plots(csv_path, out_dir, dpi=dpi)
    for p in paths:
        click.echo(f"  {p.name}")
    click.echo(f"Done in {time.time() - t0:.1f}s.")


@cli.command("state-frequency")
@click.option("--base-path", default=".", help="Base path (repo root).")
@click.option("--top-n", default=10, type=int, help="Top N states per phase to show.")
def state_frequency(base_path: str, top_n: int):
    """Analyze board-state frequency distribution from decision_sensitivity output."""
    import csv

    t0 = time.time()
    csv_path = Path(base_path) / "outputs" / "scenarios" / "state_frequency.csv"
    if not csv_path.exists():
        click.echo(f"{csv_path} not found. Run yatzy-decision-sensitivity first.")
        raise SystemExit(1)

    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "turn": int(row["turn"]),
                "scored_categories": int(row["scored_categories"]),
                "upper_score": int(row["upper_score"]),
                "visit_count": int(row["visit_count"]),
                "fraction": float(row["fraction"]),
            })

    total_visits = sum(r["visit_count"] for r in rows)

    # Per-turn summary
    turn_data: dict[int, list[dict]] = {}
    for r in rows:
        turn_data.setdefault(r["turn"], []).append(r)

    click.echo("=== Board-State Frequency by Turn ===")
    click.echo()
    click.echo(
        f"{'Turn':>4s} | {'States':>7s} | {'Visits':>10s} | {'% Total':>8s} | "
        f"{'Top-1%':>7s} | {'Top-10%':>8s}"
    )
    click.echo("-" * 60)

    for turn in sorted(turn_data.keys()):
        states = turn_data[turn]
        n_states = len(states)
        visits = sum(s["visit_count"] for s in states)
        pct = 100 * visits / total_visits if total_visits else 0

        # Concentration: what fraction of visits come from top 1% and 10% of states
        sorted_states = sorted(states, key=lambda s: s["visit_count"], reverse=True)
        top1_count = max(1, n_states // 100)
        top10_count = max(1, n_states // 10)
        top1_visits = sum(s["visit_count"] for s in sorted_states[:top1_count])
        top10_visits = sum(s["visit_count"] for s in sorted_states[:top10_count])
        top1_pct = 100 * top1_visits / visits if visits else 0
        top10_pct = 100 * top10_visits / visits if visits else 0

        click.echo(
            f"{turn:>4d} | {n_states:>7,d} | {visits:>10,d} | {pct:>7.1f}% | "
            f"{top1_pct:>6.1f}% | {top10_pct:>7.1f}%"
        )

    click.echo()
    click.echo(f"Total: {len(rows):,d} unique board states, {total_visits:,d} visits")

    # Top states per game phase
    phases = [
        ("early (turns 0-4)", lambda t: t < 5),
        ("mid (turns 5-9)", lambda t: 5 <= t < 10),
        ("late (turns 10-14)", lambda t: t >= 10),
    ]

    for label, pred in phases:
        phase_rows = [r for r in rows if pred(r["turn"])]
        if not phase_rows:
            continue
        phase_rows.sort(key=lambda r: r["visit_count"], reverse=True)
        click.echo()
        click.echo(f"=== Top {top_n} States: {label} ===")
        click.echo(
            f"{'Turn':>4s} | {'Scored':>6s} | {'Upper':>5s} | {'Visits':>10s} | {'Fraction':>10s}"
        )
        click.echo("-" * 45)
        for r in phase_rows[:top_n]:
            n_scored = bin(r["scored_categories"]).count("1")
            click.echo(
                f"{r['turn']:>4d} | {n_scored:>4d}/15 | {r['upper_score']:>5d} | "
                f"{r['visit_count']:>10,d} | {r['fraction']:>9.4f}%"
            )

    click.echo(f"\nDone in {time.time() - t0:.1f}s.")


@cli.command()
@click.option("--base-path", default=".", help="Base path (repo root).")
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
