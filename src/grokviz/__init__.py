"""grokviz — pipeline that trains, analyses and exports data for the "Grokking, Visualized" website.

Global context: this package re-implements (with source links) the analysis of Nanda et al. 2023,
"Progress measures for grokking via mechanistic interpretability" (https://arxiv.org/abs/2301.05217),
for the one-layer transformer that groks addition mod 113, and exports the results for the static site.
The single entry point is `grokviz.pipeline` (phases are added milestone by milestone).
"""

# Package version string; kept in sync with pyproject.toml's [project].version by hand.
__version__ = "0.1.0"
