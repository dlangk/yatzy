:::html
<header class="post-header">
  <h1>Can You Be Skilled At Playing Yatzy?</h1>
  <p class="post-subtitle">
    Computing the optimal strategy for Scandinavian Yatzy.
  </p>
  <p class="post-meta">
    Daniel Langkilde &middot; 2025 &middot;
    <a href="https://github.com/dlangk/yatzy">Source code</a>
  </p>

  <figure class="hero-cover">
    <video autoplay muted loop playsinline
           aria-label="Exact Yatzy score distribution as the solver's risk parameter sweeps from neutral to bold and back">
      <source src="media/hero-density.webm" type="video/webm">
      <source src="media/hero-density.mp4" type="video/mp4">
      <img src="media/hero-density.gif" alt="Animated Yatzy score distribution sweeping the risk parameter">
    </video>
    <figcaption>The score distribution as the solver's risk-dial &theta; sweeps from neutral to risk-seeking.</figcaption>
  </figure>

  <section class="abstract" aria-label="Abstract">
    <h2 class="abstract-label">Abstract</h2>
    <p><strong>Can you be skilled at playing Yatzy? The answer is yes.</strong> To arrive at this answer, we reduce the state space of Scandinavian Yatzy to about 1.43 million reachable states. This is possible if you ignore dice order, discard history, prune the impossible, and then decompose each turn into a reroll widget. We then use backward induction to solve for the optimal policy. <strong>Optimal play averages 248.4 points with a standard deviation of 38.5 points</strong>. We add a risk dial, &theta;, that lets us slide the policy along the mean-variance frontier (&part;E/&part;V = &minus;&theta;/2), trading average points for rare high scores. We also show that there is a second-mover advantage when playing head-to-head, but that even a clairvoyant player maxes out at 55.3% win probability. Yatzy rewards skill, but luck dominates.</p>
  </section>

  <!-- TOC -->
</header>
:::
