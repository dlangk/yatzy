---
paths:
  - "analytics/**/*.py"
---

# Analytics Rules

- Stack: Python + polars + numpy + scipy + matplotlib + seaborn.
- Polars is the primary DataFrame library. Convert to pandas with `.to_pandas()` only at the seaborn/matplotlib plot boundary.
- All plots use the custom diverging colormap defined in `analytics/src/yatzy_analysis/plots/style.py`:
  ```python
  CMAP = mcolors.LinearSegmentedColormap.from_list(
      "coolwarm_mid",
      ["#3b4cc0", "#8db0fe", "#F37021", "#f4987a", "#b40426"],
  )
  ```
  The center color is `#F37021` (orange). Always use `CMAP` from `plots.style` for theta-indexed visualizations.
- Use `uv` for package management, never pip directly.
- CLI entry point: `analytics/.venv/bin/yatzy-analyze` (Click framework).
- Binary data is read via `io.py` (numpy-vectorized I/O for scores.bin and simulation_raw.bin).
- Parquet I/O via `store.py`. CSV output via polars.
- Use `save_fig()` from `plots.style` for consistent output.
- When adding a new plot, follow the pattern in existing `plots/*.py` files.
