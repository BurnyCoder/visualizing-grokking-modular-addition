## The question as a function

A [[dfn:function]] is a rule that gives exactly one output for each input. The rule the network must learn is

$$f(a, b) = (a + b) \bmod 113, \qquad a, b \in \{0, 1, \ldots, 112\}.$$

There are 113 choices for a and 113 for b, so there are 113 × 113 = **12,769** possible questions — few enough that we can check the network on every single one.

## Examples, labels, and the [[preview:training|training]]/test split

A [[dfn:dataset]] is a collection of examples: each example is an input together with its correct output, called its [[dfn:label]]. Here the example “(100, 50)” has label 37.

The paper's setup shows the network only a random 30 % of the questions while it learns {{cite:nanda2023}}. Those form the [[dfn:training-set]]: {{num:task.n_train}} questions. The remaining {{num:task.n_test}} questions form the [[dfn:test-set]]: they are never used to adjust the network, only to check it.

{{fig:split-grid}}

The split is made by shuffling all 12,769 pairs with a fixed [[preview:seed|random seed]] and taking the first 30 % (0.3 × 12,769 = 3,830.7, rounded down) {{cite:paperrepo}}. Hover the picture: each cell is one question (a = row, b = column); yellow cells are training questions.

::: example
**Is (100, 50) a training question?** In the paper's split it is a *test* question — the network never sees its answer during training. (By contrast (5, 71) *is* a training question.) That is why (100, 50) is our running example throughout the site: when the trained network gets it right, it is not because it saw it.
:::

## Two ways to get training questions right

A network can answer its training questions correctly in two very different ways:

- [[dfn:memorization]]: store each of the 3,830 question–answer pairs separately, like a lookup table. This says nothing about the other 8,939 questions.
- [[dfn:generalization]]: capture the rule itself, so that unseen questions are answered correctly too.

The test set tells the two apart: memorisation scores ~0 % on it (about 1 in 113 right by luck), generalisation scores ~100 %.

## Questions as token sequences

The network reads its input as a short sequence of symbols called [[dfn:token|tokens]], each written as an integer id. The question “100 + 50 =” becomes the three tokens

$$(\,100,\; 50,\; 113\,)$$

where ids 0–112 stand for the numbers and the extra id 113 stands for “=” {{cite:arena152}}. The set of all tokens the network knows is its [[dfn:vocabulary]] — 114 tokens here. The [[dfn:position]] of a token is its place in the sequence: position 0 holds a, position 1 holds b, position 2 holds “=”.

Note that the “+” is never shown: every input is just “a b =”, and the network has to discover that the task is addition from the examples alone.

::: nuance
The network outputs a score for each of 114 tokens (including “=”), but only the 113 number tokens are possible answers. Like the paper's analysis code, our analyses drop the “=” output {{cite:paperrepo}}; training uses all 114, exactly as the paper's training code did.
:::
