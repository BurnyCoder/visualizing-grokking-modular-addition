To claim “the network computes X” we must say *how well* X describes it. This chapter collects the yardsticks the paper and ARENA use.

## Mean, variance and how much an approximation explains

The [[dfn:mean]] of a list is its average. The [[dfn:variance]] is the average squared distance from the mean — how spread out the numbers are.

::: example
For y = (1, 2, 3, 6): mean = 12/4 = 3; variance = ((1−3)² + (2−3)² + (3−3)² + (6−3)²)/4 = (4 + 1 + 0 + 9)/4 = **3.5**.
:::

If we approximate data y by some prediction ŷ, the [[dfn:residual]] is what is left, y − ŷ. The [[dfn:fve|fraction of variance explained]] compares the leftover with the original spread:

$$\text{FVE} = 1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}.$$

FVE = 1 means the approximation is perfect; FVE = 0 means it is no better than always predicting the mean.

## Least squares

[[dfn:least-squares|Least squares]] chooses the free coefficients of an approximation to make the sum of squared residuals as small as possible. The paper fits the logits this way (chapter 15), ARENA fits neuron activations this way (chapter 14).

::: example
**Fitting a line ŷ = α + βx** to y = (1, 2, 3, 6) at x = (0, 1, 2, 3). Least squares gives α = 0.6, β = 1.6, predictions (0.6, 2.2, 3.8, 5.4), residuals (0.4, −0.2, −0.8, 0.6). Sum of squared residuals 1.2, total sum of squares around the mean 14, so FVE = 1 − 1.2/14 ≈ **0.914**: a line explains 91 % of the variance.
:::

## How many directions does a matrix use? SVD and PCA

The [[dfn:svd|singular value decomposition]] writes any matrix as rotation × stretching × rotation, $M = U\,\mathrm{diag}(s_1, s_2, \ldots)\,V^\top$ {{cite:wiki_svd}}. The stretch factors $s_1 \ge s_2 \ge \ldots \ge 0$ are the singular values; the number of non-zero ones is the rank (chapter 3). Many tiny singular values mean the matrix effectively uses only a few directions.

::: example
$\begin{pmatrix}3&0\\0&1\end{pmatrix}$ has singular values 3 and 1. $\begin{pmatrix}1&2\\2&4\end{pmatrix}$ has singular values 5 and 0 — its second row is twice its first, so it only ever outputs multiples of one direction (rank 1).
:::

[[dfn:pca|Principal component analysis]] applies the same idea to a cloud of points: it finds the directions along which the points vary most, so we can draw a 128-dimensional cloud faithfully in 2D {{cite:wiki_pca}}. ARENA uses the SVD of the embedding matrix to see how many directions it uses during training {{cite:arena152}}:

{{fig:we-svals}}

The same singular values at every checkpoint of training (largest at the bottom): by the end, about ten of them stand above the rest.

{{fig:svals-over-training}}

{{fig:embed-pca}}

## How concentrated is a list? The Gini coefficient

To measure sparsity — “are only a few Fourier components large?” — the paper uses the [[dfn:gini|Gini coefficient]] of the list of component sizes {{cite:nanda2023}}, citing Hurley and Rickard, who show it satisfies all six intuitive requirements of a sparsity measure they list {{cite:hurley2009}}. With the N values sorted from smallest to largest, $c_{(1)} \le \ldots \le c_{(N)}$:

$$G(c) = 1 - 2\sum_{k=1}^{N}\frac{c_{(k)}}{\lVert c\rVert_1}\cdot\frac{N - k + \tfrac12}{N}, \qquad \lVert c\rVert_1 = \sum_k |c_k|.$$

The paper does not print the formula; we use Hurley and Rickard's definition, which gives final values consistent with the paper's Fig. 7 (chapter 17).

::: example
**Gini of four values.** Equal values (1, 1, 1, 1): G = **0**. All mass in one (0, 0, 0, 4): G = 1 − 1/4 = **0.75** (the maximum for N = 4 is 1 − 1/N). Steadily increasing (1, 2, 3, 4): G = **0.25**. Scaling all values by the same factor never changes G.
:::

{{widget:gini-toy}}
