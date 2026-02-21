Run an experiment following the scientific method: $ARGUMENTS

This command runs a rigorous experiment and produces a lab report. Follow this protocol EXACTLY:

## 1. Pre-Registration (BEFORE running anything)

Write the following to `theory/<YYYY-MM-DD>_<experiment-name>.md`:

### Hypothesis
State a specific, falsifiable hypothesis.

### Predictions
State concrete, quantitative predictions BEFORE seeing any data:
- What metric will you measure?
- What direction do you expect?
- What magnitude would be surprising?
- What would falsify the hypothesis?

### Method
- What simulation/sweep will you run?
- What parameters (θ, games, seed)?
- How many iterations?
- What statistical tests will you use?

## 2. Run the Experiment

Execute the simulation/sweep as specified. Save raw data to appropriate location under `data/` or `outputs/`.

## 3. Analysis

Run the pre-specified analyses. Generate plots using the custom colormap (`CMAP` from `analytics/src/yatzy_analysis/plots/style.py`).

## 4. Lab Report

Complete the theory document with Results and Discussion sections:
- Results: clinical reporting. Numbers, tables, plot references. No interpretation.
- Discussion: interpret results. Did predictions hold? What was surprising? Limitations? Follow-ups?

## 5. Update Documentation

If the experiment reveals something about the system worth remembering, update the appropriate file in `theory/` (see `theory/README.md` for the directory index).
