"""Plot specifications: purpose, design, and theta-legend rendering mode."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotSpec:
    name: str  # e.g. "cdf_full"
    purpose: str  # Human: what are we communicating?
    design: str  # Human: what design decisions did we make?
    theta_legend: str  # "colorbar" | "legend" | "annotation" | "none"


# Sentinel spec that suppresses all theta legends (used when embedding in combined figures).
NO_LEGEND = PlotSpec(name="_no_legend", purpose="", design="", theta_legend="none")

PLOT_SPECS: dict[str, PlotSpec] = {
    "cdf_full": PlotSpec(
        name="cdf_full",
        purpose="Show how the full score CDF shifts with theta",
        design="One line per theta, colored by diverging cmap. Colorbar replaces legend. theta=0 thicker.",
        theta_legend="colorbar",
    ),
    "tails_zoomed": PlotSpec(
        name="tails_zoomed",
        purpose="Reveal tail behavior: left panel P(low), right panel P(high)",
        design="Two panels, same color scheme. Colorbar on right panel.",
        theta_legend="colorbar",
    ),
    "percentiles_vs_theta": PlotSpec(
        name="percentiles_vs_theta",
        purpose="Show each percentile's response to theta",
        design="X=theta, Y=score. One curve per percentile, rocket palette. Legend lists percentile names.",
        theta_legend="none",
    ),
    "percentiles_vs_theta_zoomed": PlotSpec(
        name="percentiles_vs_theta_zoomed",
        purpose="Zoomed percentiles to theta in [-0.10, 0.45] where peaks live",
        design="Same as full version, tighter x-range.",
        theta_legend="none",
    ),
    "mean_vs_std": PlotSpec(
        name="mean_vs_std",
        purpose="Visualize mean-variance tradeoff across theta",
        design="Scatter, one dot per theta. Color encodes theta via colorbar. No per-point labels.",
        theta_legend="colorbar",
    ),
    "density": PlotSpec(
        name="density",
        purpose="Show how the score PDF reshapes with theta",
        design="One density curve per theta, colored by cmap. Colorbar. theta=0 thicker.",
        theta_legend="colorbar",
    ),
    "density_adaptive": PlotSpec(
        name="density_adaptive",
        purpose="Compare fixed-theta and adaptive strategy densities",
        design="Fixed-theta as thin colored lines with colorbar. Adaptive as thick dashed with named legend.",
        theta_legend="colorbar",
    ),
    "density_3d": PlotSpec(
        name="density_3d",
        purpose="3D surface view of score x theta x density",
        design="Surface colored by theta. Wireframe slices at key thetas. Colorbar.",
        theta_legend="colorbar",
    ),
    "density_ridge": PlotSpec(
        name="density_ridge",
        purpose="Joy-division stacked density curves, one row per theta",
        design="Y-axis encodes theta via row position + tick labels. Fill colored by cmap.",
        theta_legend="none",
    ),
    "quantile": PlotSpec(
        name="quantile",
        purpose="Inverse CDF: for each probability level, how does the score change with theta",
        design="One curve per theta (x=probability, y=score). Colorbar.",
        theta_legend="colorbar",
    ),
    "combined": PlotSpec(
        name="combined",
        purpose="6-panel dashboard combining CDF, percentiles, tails, mean-std, density",
        design="Sub-plots reuse individual functions. Single shared colorbar for figure.",
        theta_legend="colorbar",
    ),
    "efficiency": PlotSpec(
        name="efficiency",
        purpose="Four-panel cost analysis: MER, frontier, CDF difference, CVaR deficit",
        design="Theta-colored scatter in frontier panel (colorbar). MER/CVaR use fixed metric colors.",
        theta_legend="colorbar",
    ),
}
