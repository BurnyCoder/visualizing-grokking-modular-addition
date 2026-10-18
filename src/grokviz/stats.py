"""Statistics helpers: confidence intervals, exact tests, multiple-comparison correction, power.

Local explanation: small, well-known procedures (t-interval, bootstrap percentile interval, exact sign test,
Fisher's exact test, Holm's step-down correction, power of the sign test and of the one-sample t-test).
Global context: used by `hypotheses.py` to evaluate the preregistered hypotheses H1–H5 across seeds and by
`patching.py` for bootstrap intervals over sampled inputs; `power_table` is quoted in docs/preregistration.md.
Sources: scipy.stats (https://docs.scipy.org/doc/scipy/reference/stats.html) — `binomtest`, `fisher_exact`,
`t`, `nct`; Holm 1979 step-down (https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method); bootstrap
percentile interval (Efron & Tibshirani 1993; https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
"""

from __future__ import annotations  # postponed annotations

import math  # sqrt

import numpy as np  # arrays, RNG
from scipy import stats  # distributions and exact tests


def t_interval(values, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) Student-t confidence interval for the mean of a small sample (e.g. seeds)."""
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:  # no spread information
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.5 + level / 2, df=len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))  # t quantile × standard error
    return mean, mean - half, mean + half


def bootstrap_interval(values, level: float = 0.95, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) percentile-bootstrap interval for the mean (resample with replacement)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)  # n_boot resampled means
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(x.mean()), float(low), float(high)


def sign_test(successes: int, n: int) -> float:
    """One-sided exact sign (binomial) test p-value for H1: P(success) > 1/2."""
    return float(stats.binomtest(successes, n, 0.5, alternative="greater").pvalue)


def fisher_one_sided(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """One-sided Fisher exact test that group A's success rate exceeds group B's."""
    table = [[a_success, a_total - a_success], [b_success, b_total - b_success]]
    return float(stats.fisher_exact(table, alternative="greater").pvalue)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down correction: sort p ascending, compare p_(i) with α/(m−i); stop at the first failure."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out, still_rejecting = {}, True
    for i, name in enumerate(order):
        threshold = alpha / (m - i)
        reject = still_rejecting and pvalues[name] <= threshold
        still_rejecting = reject  # once one test fails, all larger p-values fail too
        adjusted = min(1.0, max(pvalues[o] * (m - j) for j, o in enumerate(order[: i + 1])))  # monotone adjusted p
        out[name] = {"p": pvalues[name], "threshold": threshold, "p_holm": adjusted, "reject": reject}
    return out


def sign_test_power(n: int, true_rate: float, alpha: float) -> float:
    """Power of the one-sided exact sign test with n seeds when each seed agrees with probability true_rate."""
    crit = next((k for k in range(n + 1) if stats.binom.sf(k - 1, n, 0.5) <= alpha), None)  # smallest k with P(X ≥ k) ≤ α
    if crit is None:  # even n/n agreeing seeds cannot reach significance (e.g. n = 5 at α = 0.01)
        return 0.0
    return float(stats.binom.sf(crit - 1, n, true_rate))


def t_test_power(n: int, effect_size: float, alpha: float) -> float:
    """Power of a one-sided one-sample t-test (effect size d = mean/sd) via the noncentral t distribution."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(crit, df, effect_size * math.sqrt(n)))


def power_table(n: int, alpha: float, n_tests: int) -> dict:
    """Numbers quoted in the preregistration: sign-test feasibility and t-test detectable effect at Holm's strictest level."""
    strict = alpha / n_tests
    detectable = next(d / 100 for d in range(10, 400) if t_test_power(n, d / 100, strict) >= 0.8)
    return {
        "n_seeds": n,
        "alpha": alpha,
        "holm_strictest_alpha": strict,
        "p_all_agree": 0.5**n,
        "p_one_disagrees": float(stats.binom.sf(n - 2, n, 0.5)),
        "sign_test_power_if_rate_0.95": sign_test_power(n, 0.95, strict),
        "t_test_min_effect_for_80pct_power": detectable,
    }
