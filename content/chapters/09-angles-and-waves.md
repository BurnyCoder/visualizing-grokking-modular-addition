The trained network represents numbers as *angles*. This chapter builds the trigonometry it uses, from scratch.

## Angles in radians

An [[dfn:angle]] measures an amount of turning. We measure it in [[dfn:radian|radians]]: a full turn is $2\pi \approx 6.283$ radians, half a turn is $\pi \approx 3.1416$, a quarter turn is $\pi/2 \approx 1.571$. Angles that differ by whole turns point the same way: $\theta$ and $\theta + 2\pi$ are the same direction.

## The unit circle, cosine and sine

The [[dfn:unit-circle]] is the circle of radius 1 around the origin. Start at the point (1, 0) and turn counter-clockwise by an angle θ: you arrive at a point whose horizontal coordinate is the [[dfn:cosine]] $\cos\theta$ and whose vertical coordinate is the [[dfn:sine]] $\sin\theta$:

$$\text{point at angle } \theta = (\cos\theta,\ \sin\theta).$$

::: example
**Values to remember.** $\theta = 0$: point (1, 0), so cos 0 = 1, sin 0 = 0. $\theta = \pi/2$ (quarter turn): (0, 1). $\theta = \pi$: (−1, 0). $\theta = 3\pi/2$: (0, −1). And for any θ, $\cos^2\theta + \sin^2\theta = 1$ (the point is at distance 1 from the origin).
:::

Because $\theta$ and $\theta + 2\pi$ are the same point, cos and sin are [[dfn:periodic]]: they repeat every $2\pi$. The [[dfn:frequency]] of a wave is how many times it repeats over a range. On our 113 numbers we use the waves

$$\cos\!\Big(\frac{2\pi k}{113}\,x\Big), \qquad \sin\!\Big(\frac{2\pi k}{113}\,x\Big), \qquad x = 0, 1, \ldots, 112,$$

which complete exactly k full cycles as x runs from 0 to 113. The angle per unit step is [[sym:wk]] $= 2\pi k/113$; for k = 14 it is ≈ 0.7785 radians.

A hand turning around the circle draws both waves: its height is the sine and its horizontal position the cosine. Press ▶ to turn it twice around.

{{widget:phasor-wave}}

## Numbers as angles: the clock

Placing the number x at angle $w_k x$ on the unit circle turns the 113 numbers into 113 points on a circle — a clock face with 113 hours, visited k steps at a time. Because $w_k \cdot 113 = 2\pi k$ is a whole number of turns, x and x + 113 land on the same point: **wrapping around modulo 113 happens automatically**.

::: example
**The running example at frequency k = 14.** The point for a = 100 is at angle $w_{14}\cdot 100$; since $14\cdot100 = 1400$ and $1400 \bmod 113 = 44$, this is the same point as $44/113$ of a full turn. For b = 50: $14\cdot 50 = 700$, $700 \bmod 113 = 22$, i.e. $22/113$ of a turn. The correct answer c* = 37 sits at $14 \cdot 37 = 518$, $518 \bmod 113 = 66$ — exactly $44 + 22$. **Adding the angles of a and b lands on the angle of a + b.**
:::

Press ▶ next to a to walk a around the clock and watch a + b follow it.

{{widget:unit-circle}}

## Rotation, and the angle-addition identities

A [[dfn:rotation]] turns every point around the origin by the same angle. Rotating by x and then by y is the same as rotating by x + y. Rotating the point $(\cos y, \sin y)$ by an angle x can be done with a matrix–vector product (chapter 3):

$$\begin{pmatrix}\cos x & -\sin x\\ \sin x & \cos x\end{pmatrix}\begin{pmatrix}\cos y\\ \sin y\end{pmatrix} = \begin{pmatrix}\cos x\cos y - \sin x \sin y\\ \sin x \cos y + \cos x \sin y\end{pmatrix}.$$

The result must be the point at angle $x + y$, i.e. $(\cos(x+y), \sin(x+y))$. Comparing coordinates gives the [[dfn:angle-addition]] identities {{cite:wiki_trig}}:

$$\cos(x + y) = \cos x \cos y - \sin x \sin y, \qquad \sin(x + y) = \sin x \cos y + \cos x \sin y.$$

Replacing y by −y (and using cos(−y) = cos y, sin(−y) = −sin y) gives the subtraction version

$$\cos(x - y) = \cos x \cos y + \sin x \sin y .$$

::: example
**Numerical check** with x = 0.5 and y = 0.3:
$\cos 0.5 \cos 0.3 - \sin 0.5 \sin 0.3 = 0.8776\cdot0.9553 - 0.4794\cdot0.2955 = 0.8384 - 0.1417 = 0.6967$, and indeed $\cos 0.8 = 0.6967$.
:::

{{widget:angle-addition}}

These two lines are the whole trick of the network. If you know $\cos$ and $\sin$ of $w a$ and of $w b$ — four numbers — you can compute $\cos$ and $\sin$ of $w(a+b)$ using only **multiplications and additions**. Chapter 12 shows the network doing exactly that.

::: nuance
**Why several frequencies?** For a single k, a cosine wave is largest at one point but also *nearly* largest at others (e.g. $\cos(w_{14}\cdot 8) = 0.998$ is almost 1, the paper's App. B example {{cite:nanda2023}}). That is why the network will combine several frequencies — see “[[preview:constructive-interference|constructive interference]]” in chapter 12.
:::
