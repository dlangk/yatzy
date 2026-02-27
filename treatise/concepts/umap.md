# UMAP

**UMAP** (Uniform Manifold Approximation and Projection) is a dimensionality reduction algorithm that projects high-dimensional data into 2D or 3D for visualization while preserving local neighborhood structure.

## How It Works

UMAP operates in two phases:

1. **Build a graph in high dimensions**: for each data point, find its nearest neighbors and construct a weighted graph where edge weights reflect similarity. UMAP uses a fuzzy simplicial set framework: nearby points are connected with high weight, distant points with low weight. The "fuzzy" part means each point has a local notion of distance, adapting to the data's varying density.

2. **Optimize a low-dimensional layout**: find 2D coordinates that preserve the graph structure as faithfully as possible. Points that are neighbors in high dimensions should be neighbors in 2D; points that are distant should remain distant. The optimization minimizes the cross-entropy between the high-dimensional and low-dimensional fuzzy graphs using stochastic gradient descent.

The mathematical foundation comes from algebraic topology: UMAP treats the data as samples from a Riemannian manifold and seeks a low-dimensional manifold with equivalent topological structure.

## Comparison with t-SNE

UMAP and t-SNE solve similar problems but differ in important ways:

- **Speed**: UMAP is significantly faster, especially on large datasets. For the ~1.43M Yatzy states, UMAP completes in minutes where t-SNE would take hours.
- **Global structure**: UMAP better preserves relationships *between* clusters, not just within them. Clusters that are close in high dimensions tend to be close in the 2D projection.
- **Reproducibility**: UMAP with a fixed random seed produces deterministic layouts. t-SNE can produce visually different layouts across runs.
- **Scalability**: UMAP's approximate nearest-neighbor search (via NN-descent) scales gracefully to millions of points.

## Application to Yatzy

In the Yatzy project, UMAP visualizes **policy decision clusters**. Each game state is a point in a high-dimensional feature space:

- Dice face counts (6 dimensions)
- Upper section progress (1 dimension)
- Scored categories (15 binary dimensions)
- Rerolls remaining (1 dimension)

The optimal action at each state assigns it a label (e.g., "score Sixes" or "keep the pair"). Projecting these labeled states into 2D with UMAP reveals:

- **Coherent clusters**: states with the same optimal action group together, confirming that the policy has learnable structure. If decisions were essentially random, no clusters would form.
- **Decision boundaries**: the borders between clusters show where the optimal action changes. These often align with upper-bonus thresholds, confirming the bonus's dominant strategic role.
- **Outlier states**: isolated points where the optimal action is counterintuitive. These are valuable for constructing quiz scenarios that test strategic understanding.

## Interpretation Caveats

UMAP projections preserve local neighborhoods but can distort global distances. Two clusters that appear far apart in 2D may not be far apart in the original feature space. Cluster sizes are also unreliable -- UMAP tends to normalize cluster densities, making large and small clusters appear similar in size.

Always interpret UMAP plots as showing *topology* (which points are near which) rather than *geometry* (exact distances or densities). The relative positions of clusters are informative; their absolute positions and sizes are not.

## Key Hyperparameters

UMAP's behavior is controlled by two main parameters:

- **n_neighbors**: how many nearest neighbors to consider when building the high-dimensional graph. Small values (5--15) emphasize very local structure; larger values (50--200) capture broader patterns. For Yatzy policy visualization, 15--30 neighbors typically gives good results.
- **min_dist**: how tightly points are allowed to cluster in the 2D projection. Small values (0.0--0.1) produce tight, separated clusters; larger values (0.5--1.0) produce a more uniform spread. For categorical decision labels, small min_dist highlights cluster boundaries.

Unlike PCA, UMAP is non-linear: it can unfold complex manifold structures that linear methods would crush into overlapping blobs. This is essential for Yatzy policy visualization, where the decision boundaries in the 23-dimensional feature space are highly non-linear, shaped by the bonus threshold, category interactions, and reroll dynamics.

## Practical Workflow

A typical UMAP analysis of Yatzy policies follows these steps:

1. **Sample states**: extract 50,000--100,000 game states from simulation runs, recording the feature vector and optimal action for each
2. **Normalize features**: standardize each dimension to zero mean and unit variance, preventing high-range features (like upper_total, 0--63) from dominating low-range features (like dice counts, 0--5)
3. **Run UMAP**: project to 2D with n_neighbors=20, min_dist=0.05, and a fixed random seed for reproducibility
4. **Color by action**: assign each point a color based on its optimal action label (e.g., red for "score Sixes," blue for "score Full House")
5. **Inspect boundaries**: examine where colors change to identify the state-space regions where decisions are most sensitive

The resulting plots are among the most informative visualizations of the policy structure, revealing at a glance which actions dominate, where boundaries lie, and which states are ambiguous.
