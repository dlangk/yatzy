Run a parameter sweep: $ARGUMENTS

1. Parse the arguments to identify: theta values (or grid name), game count, output location.
2. Ensure strategy tables exist for all requested θ values:
   ```bash
   YATZY_BASE_PATH=. solver/target/release/yatzy-precompute --theta <T>
   ```
3. Run the sweep (resumable — safe to re-run):
   ```bash
   # Full grid (37 θ values from configs/theta_grid.toml):
   YATZY_BASE_PATH=. solver/target/release/yatzy-sweep --grid all --games <N>

   # Specific thetas:
   YATZY_BASE_PATH=. solver/target/release/yatzy-sweep --thetas <t1>,<t2> --games <N>

   # Check inventory:
   YATZY_BASE_PATH=. solver/target/release/yatzy-sweep --list
   ```
4. After sweep completes, compute summary statistics:
   ```bash
   analytics/.venv/bin/yatzy-analyze compute --csv
   ```
5. Report: number of θ values completed, total games simulated, output locations.
