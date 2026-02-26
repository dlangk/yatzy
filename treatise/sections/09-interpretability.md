:::section{#part-viii}

:::part-title
Part VIII
:::

## Interpretability and the Human Player

The 82-point gap between human and optimal play is not random noise. It has
structure. Understanding where it concentrates turns a dice game into a lens
on bounded rationality.

To visualize what the solver "knows," we embed game states and keep-multiset
actions into a shared latent space using a two-tower network, then project with
::concept[UMAP]{umap}.
The 462 unique keeps organize into clean semantic clusters --pairs,
triples, straight draws --without supervision. The joint manifold reveals
that the solver has independently discovered the same decision categories
humans name intuitively, but applies them with far greater contextual precision.

:::html
<div class="chart-container" id="chart-umap-manifold"><div id="chart-umap-manifold-svg"></div></div>
:::

The gap concentrates in two regions. First, near the upper bonus threshold
(upper scores 40–62), where a single assignment swings the 50-point bonus
and over half of all surrogate errors occur. Humans treat the bonus as binary
("on track" or "lost") rather than computing the marginal value of each point.
Second, in multi-category tradeoff states where dice could reasonably go to
Threes, Full House, or Chance --choices that require combinatorial
assessment humans approximate with crude heuristics.

To measure individual players, we built a cognitive profiling system grounded
in ::concept[resource rationality]{resource-rationality}.
Four parameters capture distinct departure modes from optimality: &theta; (risk
attitude), &beta; (decision precision), &gamma; (myopia), and <var>d</var>
(search depth). Rather than requiring 50–100 observed games, estimation
compresses into 30 diagnostic scenarios where the optimal action shifts with
the player's profile. Q-values are pre-computed across 108 parameter
combinations (6&theta; × 6&gamma; × 3<var>d</var>), and a
client-side Nelder-Mead optimizer fits all four parameters from the quiz
answers alone.

The player card grid extends profiling into prediction: 648 parameter
combinations simulated at 10,000 games each map any cognitive profile to a
full score distribution. A player receives not just parameter estimates but
expected mean, variance, bonus rate, and comparison to the optimal solver.
The card makes the invisible gap visible.

The 82-point gap is not about intelligence. The solver evaluates every
successor state; humans pattern-match. The solver tracks marginal upper-section
value; humans feel the bonus as binary. These are signatures of a mind evolved
for fast, adequate decisions with limited working memory. The gap is a window
into how we think.

:::depth-2

The cognitive model defines action probabilities via a softmax over modified
Q-values:

:::equation
P(<var>a</var> | <var>s</var>) &propto;
exp(&beta; &middot; <var>Q</var><sub>&theta;,&gamma;,<var>d</var></sub>(<var>a</var>, <var>s</var>))
:::

The parameter &theta; selects which pre-computed strategy table (state values)
backs the Q-value computation. &gamma; &lt; 1 discounts future state values,
making the player myopic. <var>d</var> &isin; {8, 20, 999} truncates lookahead
depth by adding noise to Q-values computed from shallower search. The estimator
runs 24 independent Nelder-Mead optimizations (8 start points × 3
<var>d</var> values) and selects the maximum-likelihood fit, with a weak
log-normal prior on &beta; centered at 3 to prevent drift in flat regions.

Scenario generation uses diversity-constrained stratified sampling across
36 semantic buckets (3 game phases × 3 decision types × 4 tension
types). From a master pool of ~595 candidates, 30 are selected subject to
action-fatigue constraints (no action label in top-2 more than 4 times) and
fingerprint deduplication that blocks functionally equivalent scenarios with
identical top-3 actions and EVs.

The player card grid covers 648 parameter combinations. Each combination
runs 10,000 Monte Carlo games, producing distributions of mean score, standard
deviation, upper bonus rate, and percentile markers. The grid is stored as a
single JSON file (`blog/data/player_card_grid.json`) that the
client-side profiler loads to render player cards without any server
computation.

:::

:::depth-3

The two-tower Q-network architecture embeds 56-dimensional state features and
6-dimensional action features (keep-multiset face counts) into a shared 16D
latent space via bilinear interaction:

```python
class TwoTowerQ(nn.Module):
    def __init__(self, n_features=56, n_action_dim=6, embed_dim=16):
        self.state_tower  = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, embed_dim))
        self.action_tower = nn.Sequential(
            nn.Linear(n_action_dim, 32), nn.ReLU(),
            nn.Linear(32, embed_dim))
        self.state_bias  = nn.Linear(n_features, 1)
        self.action_bias = nn.Linear(n_action_dim, 1)

    def forward(self, state, action):
        z_s = self.state_tower(state)
        z_a = self.action_tower(action)
        return (z_s * z_a).sum(-1) + self.state_bias(state) + self.action_bias(action)
```

Joint UMAP projects the concatenated state (25K samples) and action (462
keep-multisets) embeddings with cosine metric, `n_neighbors=15`,
`min_dist=0.1`. The resulting visualization reveals 12 emergent
action clusters (Keep All, Reroll All, Yatzy, Four of a Kind, Full House,
Triple, Two Pair, High Pair, Low Pair, Straight Draw, Partial, Single)
that arise purely from the Q-value structure without label supervision.

The Q-grid for cognitive profiling uses 108 parameter combinations:

```python
theta_grid = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.1]    # 6 values
gamma_grid = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]         # 6 values
depth_grid = [8, 20, 999]                                # 3 values
# Total: 6 x 6 x 3 = 108 combos

# Player card grid
beta_grid  = [0.5, 1, 2, 3, 5, 10]                     # 6 values
# Total cards: 6 x 6 x 6 x 3 = 648 combos x 10K games
```

Critical implementation note: Q-grid keys use Rust float formatting (e.g.,
`"0"` not `"0.0"`) for parameter values. The JavaScript
estimator must match this formatting exactly when looking up Q-values, or
key misses will silently produce undefined likelihoods.

:::

:::
