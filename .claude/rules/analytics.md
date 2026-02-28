---
paths:
  - "analytics/**/*.py"
---

# Analytics Rules

- All theta-indexed plots use `CMAP` from `plots.style` — center is `#F37021` (orange).
- Polars is the primary DataFrame library. Convert to pandas only at the plot boundary.
- Use `uv` for package management, never pip directly.
- Use `save_fig()` from `plots.style` for consistent output.
