## Whole numbers and remainders

An [[dfn:integer]] is a whole number: 0, 1, 2, 3, … (and also −1, −2, …). When we divide one integer by another, whatever does not fit a whole number of times is the [[dfn:remainder]].

::: example
**14 divided by 12.** 12 fits into 14 once (1 × 12 = 12), and 14 − 12 = **2** is left over. So the remainder is 2.

**150 divided by 113.** 113 fits once, 150 − 113 = **37** is left over.

**113 divided by 113.** It fits exactly once; the remainder is **0**.
:::

## “mod”: keeping only the remainder

The operation [[dfn:modulo]] — written “x mod p” — throws away everything except the remainder. So 14 mod 12 = 2, 150 mod 113 = 37, and 113 mod 113 = 0. The result is always one of 0, 1, …, p − 1.

[[dfn:modular-addition|Modular addition]] means: add normally, then take the result mod p. A clock does this with p = 12 (if we call 12 o'clock “0”): four hours after 9 o'clock it is 1 o'clock, because (9 + 4) mod 12 = 13 mod 12 = 1 {{cite:wiki_modular}}. The number p where counting wraps around is the [[dfn:modulus]].

::: example
**Clock with 113 hours.** With p = 113:

- (5 + 71) mod 113 = 76 mod 113 = **76** (no wrap-around needed),
- (100 + 50) mod 113 = 150 mod 113 = **37** (wraps once),
- (112 + 1) mod 113 = 113 mod 113 = **0** (lands exactly on “midnight”).

Note that the answer to *every* question is one of only 113 possibilities, 0 … 112.
:::

{{widget:clock}}

## Why 113?

The paper uses p = 113, which is a [[dfn:prime]] — its only divisors are 1 and 113 {{cite:nanda2023}}. The paper simply states that p is prime; one convenient consequence (our remark, not a claim of the paper) is that for any k between 1 and 112, the numbers k·0, k·1, …, k·112, each taken mod 113, are all different. Later we will place the numbers around a circle in steps of k; with a prime modulus every such step size visits all 113 positions before repeating, so no two numbers ever collide.

::: example
**Why primes avoid collisions.** With p = 12 (not prime) and step k = 4, the positions 4·a mod 12 for a = 0, 1, 2, 3 are 0, 4, 8, 0 — the number 3 lands on the same spot as 0. With p = 113 and k = 14 (one of the step sizes the trained network actually uses), the positions 14·a mod 113 for a = 0, 1, …, 112 are all different: 0, 14, 28, …, 112, 13, 27, …
:::

Two more facts we will use:

- The order does not matter: (a + b) mod p = (b + a) mod p.
- Wrapping around is “free”: adding p to anything does not change the result mod p, e.g. (37 + 113) mod 113 = 37.
