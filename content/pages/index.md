<div class="hero">

# Grokking, Visualized

A small transformer learned to add numbers on a 113-hour clock. For thousands of steps it seemed only to have memorised; meanwhile it was building a general algorithm, and then, suddenly, it understood. This site shows — with the real network's weights, activations, gradients and outputs at every moment of training — *what* it learned and *how* it got there, explained step by step from scratch.

<div class="cta"><a class="primary" href="chapters/00-overview.html">Learn step by step →</a><a href="dashboard.html">Open the dashboard</a></div>

</div>

{{fig:loss-curves}}

**What you will find here**

- **22 chapters** that build everything from “what is a remainder?” to the paper's progress measures, defining every term and symbol before it is used, with worked examples and exact (short) quotes from the sources.
- **One big dashboard** with every visualization linked together: drag through training and watch the network's embeddings turn into circles, its neurons into waves, and its memorisation melt away.
- **The real thing:** the paper's own published training run {{cite:paperckpt}}, re-analysed with code that reproduces {{num:rchecks.n_match}} of the {{num:rchecks.n_rows}} reported numbers and claims we checked, plus 22 networks we retrained ourselves to check the claims with preregistered statistics.

Based on Nanda, Chan, Lieberum, Smith and Steinhardt, *Progress measures for grokking via mechanistic interpretability* (ICLR 2023) {{cite:nanda2023}}, the ARENA course chapter on grokking {{cite:arena152}}, and the videos by Neel Nanda with Jess Smith and Lawrence Chan, and by Welch Labs {{cite:video_replication}} {{cite:video_paper}} {{cite:video_welch}}. Built by Libor Burian. [Code and data on GitHub](https://github.com/BurnyCoder/visualizing-grokking-modular-addition).
