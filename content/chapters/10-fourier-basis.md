The network's weights and activations are lists of 113 numbers — one per input number. This chapter introduces a way to describe *any* such list as a mix of waves. It is the lens through which the paper reads the whole network.

## A basis made of waves

Chapter 3 said that in an orthonormal basis, the weights of a vector are just dot products. The [[dfn:fourier-basis]] of 113-dimensional space is an orthonormal basis made of waves {{cite:wiki_dft}}: the constant vector, and for each frequency k = 1, …, 56 a cosine and a sine wave,

$$\text{Const}(x) = \tfrac{1}{\sqrt{113}}, \qquad \cos_k(x) = \sqrt{\tfrac{2}{113}}\cos(w_k x), \qquad \sin_k(x) = \sqrt{\tfrac{2}{113}}\sin(w_k x),$$

for x = 0, …, 112. That is 1 + 2·56 = 113 vectors, as many as the dimension. The factors $\sqrt{1/113}$ and $\sqrt{2/113}$ give every vector length 1; one can check that any two different ones have dot product 0. We stack them as the rows of a 113 × 113 matrix [[symdef:F]], in the order Const, cos 1, sin 1, cos 2, sin 2, …, sin 56 — the order used in the paper's code {{cite:paperrepo}}.

Writing a list of 113 numbers in this basis — computing its dot product with each wave — is the [[dfn:dft|discrete Fourier transform]]; each dot product is a [[dfn:fourier-coefficient]]. Because F is orthonormal, the coefficients are $F v$ and the original is recovered as $F^\top (F v) = v$.

::: example
**The whole thing for p = 5.** The basis has five vectors (rounded to 3 decimals):

| | x = 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Const | 0.447 | 0.447 | 0.447 | 0.447 | 0.447 |
| cos 1 | 0.632 | 0.195 | −0.512 | −0.512 | 0.195 |
| sin 1 | 0 | 0.602 | 0.372 | −0.372 | −0.602 |
| cos 2 | 0.632 | −0.512 | 0.195 | 0.195 | −0.512 |
| sin 2 | 0 | 0.372 | −0.602 | 0.602 | −0.372 |

Take $v = (3, 1, 0, 0, 1)$. Its coefficients are the dot products with the rows: Const 2.236, cos 1 **2.288**, sin 1 **0**, cos 2 0.874, sin 2 **0**. The sine coefficients vanish because v is symmetric around x = 0 (v(1) = v(4), v(2) = v(3)), like the cosines. Check: $2.236^2 + 2.288^2 + 0.874^2 = 11.0 = 3^2 + 1^2 + 1^2$ — the squared length is the same in both bases, as it must be for an orthonormal basis.
:::

{{widget:dft-p5}}

## Periodic means sparse

A vector is [[dfn:sparse]] if only a few of its entries are large. The key intuition, as the ARENA course puts it, is that a function that looks periodic in the ordinary basis is sparse in the Fourier basis {{cite:arena152}}: a wave of frequency 14 is *one* basis vector (well, a mix of cos 14 and sin 14), so its transform has only two non-zero coefficients, while a random-looking list spreads over all 113.

{{widget:periodic-sparse}}

This is why the Fourier basis is the right lens: if the network's weights turn out to be sparse in it, the network is using a few specific waves — and we can name them.

## Two inputs: the 2D Fourier basis

Activations depend on two numbers, a and b: they form a 113 × 113 grid. The [[dfn:fourier-2d|2D Fourier basis]] consists of products of a wave in a with a wave in b, $F_i(a)\,F_j(b)$, for all 113 × 113 pairs (i, j). Transforming a grid G into it means $C = F\,G\,F^\top$: transform along a, then along b.

Three kinds of 2D basis functions will matter:

- **Constant:** Const(a)·Const(b) — the same value everywhere.
- A [[dfn:linear-term]]: a wave in one input only, e.g. $\cos(w_k a)\cdot\text{Const}(b)$ — vertical or horizontal stripes.
- A [[dfn:quadratic-term]]: a product of waves in both inputs, e.g. $\cos(w_k a)\cos(w_k b)$ — a checkerboard of blobs.

{{widget:basis-2d}}

The combination that matters most is diagonal stripes. By the angle-addition identity of chapter 9,

$$\cos(w_k(a+b)) = \cos(w_k a)\cos(w_k b) - \sin(w_k a)\sin(w_k b),$$

so a grid that depends only on $a + b$ at frequency k is the difference of two quadratic terms — in the normalised basis it is the unit vector $\big(\cos_k\!\otimes\cos_k - \sin_k\!\otimes\sin_k\big)/\sqrt2$, and likewise $\sin(w_k(a+b))$ is $\big(\sin_k\!\otimes\cos_k + \cos_k\!\otimes\sin_k\big)/\sqrt2$ (here ⊗ means “product of a wave in a and a wave in b”) {{cite:paperrepo}} {{cite:arena152}}. These two **cos/sin(w_k(a+b)) directions** are the main characters of the rest of the story.

::: note
**Our convention.** cos k has index 2k − 1 and sin k index 2k in F, as in the paper's own code and ARENA's code. (One of the notebooks in this project's submodule uses the opposite order; we ported everything to the paper's order.)
:::
