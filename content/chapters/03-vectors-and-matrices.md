The network is nothing but arithmetic on lists of numbers. This chapter introduces the handful of operations it uses, each with a tiny example.

## Vectors

A [[dfn:vector]] is an ordered list of numbers, written in parentheses or as a column. Its [[dfn:dimension]] is how many numbers it has. The network keeps a vector of dimension 128 at each input position.

::: example
**Two vectors in dimension 2:** u = (1, 2) and v = (3, 4). You can picture each as an arrow from the origin to the point with those coordinates.
:::

The [[dfn:dot-product]] of two vectors multiplies matching entries and adds the results:

$$u \cdot v = \sum_i u_i v_i, \qquad (1, 2)\cdot(3, 4) = 1\cdot 3 + 2\cdot 4 = 11.$$

The [[dfn:norm]] (length) of a vector is $\lVert u\rVert = \sqrt{u\cdot u}$; for example $\lVert (3,4)\rVert = \sqrt{9+16} = 5$. Dot products measure alignment: $u \cdot v = \lVert u\rVert\,\lVert v\rVert \cos(\text{angle between them})$ — positive when the arrows point roughly the same way, zero when they are perpendicular, negative when they point apart. (The cosine is explained properly in chapter 9; for now: it is 1 for “same direction”, 0 for “perpendicular”, −1 for “opposite”.)

{{widget:dot-product}}

## Matrices

A [[dfn:matrix]] is a rectangular table of numbers. A matrix with m rows and n columns is called “m × n”. Its main use is to turn one vector into another: the [[dfn:matrix-vector-product]] $Mv$ has, as its i-th entry, the dot product of row i of M with v.

::: example
**A 2 × 3 matrix times a vector of dimension 3:**

$$\begin{pmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{pmatrix}\begin{pmatrix} 4 \\ 5 \\ 6\end{pmatrix} = \begin{pmatrix} 1\cdot4 + 0\cdot5 + 2\cdot6 \\ -1\cdot4 + 3\cdot5 + 1\cdot6\end{pmatrix} = \begin{pmatrix} 16 \\ 17 \end{pmatrix}.$$

The 3-dimensional input became a 2-dimensional output.
:::

The [[dfn:matrix-product]] $MN$ of two matrices applies one after the other: $(MN)v = M(Nv)$. Its entry in row i, column j is the dot product of row i of M with column j of N. The [[dfn:transpose]] $M^\top$ turns rows into columns; it lets us write a dot product as a product: $u\cdot v = u^\top v$.

::: note
**Two conventions.** The paper writes vectors as columns and multiplies matrices on the left ($W_E\,t$), so the embedding matrix is 128 × 113 {{cite:nanda2023}}. Code libraries such as TransformerLens and the ARENA course use rows and multiply on the right ($t\,W_E$, with a 113 × 128 matrix) {{cite:arena152}}. The two are transposes of each other and give the same numbers; this site follows the paper.
:::

## One-hot vectors: how tokens enter the network

A [[dfn:one-hot]] vector is all zeros except for a single 1. It is the standard way to turn a choice (“which token?”) into a vector. Multiplying a matrix by a one-hot vector simply **picks out one column**:

::: example
**Picking a column.** With 4 tokens and vectors of dimension 2:

$$\begin{pmatrix} 7 & 1 & -2 & 0 \\ 3 & 5 & 4 & 9\end{pmatrix}\begin{pmatrix} 0\\0\\1\\0\end{pmatrix} = \begin{pmatrix} -2 \\ 4\end{pmatrix} \quad\text{(column 2, counting from 0).}$$

The network does exactly this: token 100 is the one-hot vector with a 1 in entry 100, and $W_E$ times it is column 100 of $W_E$ — the 128 numbers that represent “100”.
:::

## Bases, orthonormal bases and projections

A [[dfn:basis]] is a set of vectors from which every vector can be built as a unique weighted sum. The everyday basis of 2D space is (1, 0) and (0, 1): the vector (3, 4) is 3·(1, 0) + 4·(0, 1). An [[dfn:orthonormal-basis]] is one whose vectors have length 1 and are perpendicular to each other. Its great convenience: **the weights are just dot products**. If $e_1, \ldots, e_n$ is orthonormal, then every v equals $\sum_i (v\cdot e_i)\, e_i$.

::: example
**A rotated basis of 2D space:** $e_1 = (1, 1)/\sqrt2$ and $e_2 = (-1, 1)/\sqrt2$ both have length 1 and $e_1\cdot e_2 = 0$. For v = (3, 4): $v\cdot e_1 = 7/\sqrt2$ and $v\cdot e_2 = 1/\sqrt2$, and indeed $\tfrac{7}{\sqrt2}e_1 + \tfrac{1}{\sqrt2}e_2 = (3, 4)$.
:::

The [[dfn:projection]] of v onto a direction e (of length 1) is the part of v along e: $(v\cdot e)\,e$. All weighted sums of some chosen vectors form a [[dfn:subspace]] (a line, a plane, …), and the projection onto a subspace with orthonormal basis $e_1, \ldots, e_m$ is $\sum_{i=1}^m (v\cdot e_i)\,e_i$ — the sum of the single-direction projections {{cite:arena152}}. Chapter 10 will use exactly this with a basis made of waves.

Finally, the [[dfn:rank]] of a matrix is the number of independent directions its outputs can point in. A 113 × 512 matrix could have rank up to 113; we will see that the trained network's most important matrix has rank about 10.
