"""Statistical helpers used across the detectors and evaluation."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(successes, trials, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion.

    Used instead of a bare rate everywhere in this project. A conversion rate of
    3/900 and one of 14/5600 look different until you put intervals on them --
    then they overlap almost entirely. Never report a rate on a small
    denominator without this.
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(trials > 0, successes / trials, 0.0)
        denom = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denom
        margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom

    lo = np.clip(centre - margin, 0, 1)
    hi = np.clip(centre + margin, 0, 1)
    lo = np.where(trials > 0, lo, 0.0)
    hi = np.where(trials > 0, hi, 1.0)
    return lo, hi


def shannon_entropy(counts: np.ndarray, axis: int = 1) -> np.ndarray:
    """Normalised Shannon entropy (0 = all mass in one bin, 1 = uniform).

    Applied to the 24-bin hour-of-day histogram. Humans sleep, so real traffic
    concentrates in waking hours and scores below 1. Server-side bots run flat
    around the clock and score near 1. Entities at either extreme are unusual.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        logp = np.where(p > 0, np.log(p), 0.0)
    ent = -(p * logp).sum(axis=axis)
    n_bins = counts.shape[axis]
    return ent / np.log(n_bins)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Population Stability Index between a baseline and each row of `actual`.

    PSI = sum( (a - e) * ln(a / e) ) over bins.
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift.
    Here the baseline is the global population and each row is one entity, so a
    high PSI means "this entity does not behave like the population".
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if actual.ndim == 1:
        actual = actual[None, :]

    e = expected / max(expected.sum(), eps)
    a = actual / np.maximum(actual.sum(axis=1, keepdims=True), eps)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return ((a - e) * np.log(a / e)).sum(axis=1)


def ks_statistic(baseline_hist: np.ndarray, entity_hist: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic on binned distributions (max CDF gap).

    Cross-population comparison: how far does this entity's hour-of-day CDF
    diverge from the population CDF? Range 0-1, higher = more divergent.
    """
    baseline_hist = np.asarray(baseline_hist, dtype=float)
    entity_hist = np.asarray(entity_hist, dtype=float)
    if entity_hist.ndim == 1:
        entity_hist = entity_hist[None, :]

    base_cdf = np.cumsum(baseline_hist / max(baseline_hist.sum(), 1e-9))
    totals = np.maximum(entity_hist.sum(axis=1, keepdims=True), 1e-9)
    ent_cdf = np.cumsum(entity_hist / totals, axis=1)
    return np.abs(ent_cdf - base_cdf[None, :]).max(axis=1)


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Map an arbitrary score to [0, 1] by rank.

    Detector scores live on incomparable scales (a PSI, a distance, a
    coefficient of variation). Rank-normalising before the weighted blend stops
    whichever detector happens to have the widest raw range from dominating.
    """
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) <= 1:
        return np.zeros_like(x)
    ranks = stats.rankdata(x, method="average")
    return (ranks - 1) / (len(x) - 1)


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05):
    """Benjamini-Hochberg FDR correction.

    We run one hypothesis test per entity -- thousands of them. At alpha=0.01
    across 10,000 entities you expect ~100 false discoveries by chance alone.
    Controlling the FALSE DISCOVERY RATE instead of the per-test error rate is
    the correct treatment, and it is the difference between "I ran a test" and
    "I ran a defensible screen".

    Returns (rejected_mask, adjusted_pvalues).
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty(n)
    out[order] = adjusted
    return out <= q, out


def binomial_deficit_pvalue(successes: np.ndarray, trials: np.ndarray, baseline: float):
    """One-sided binomial test: is this entity converting FAR BELOW baseline?

    Used to build the ground-truth proxy for evaluation (see evaluate.py).
    Small p = the entity's near-zero conversion rate is very unlikely to be
    chance given how many clicks it produced.
    """
    successes = np.asarray(successes, dtype=int)
    trials = np.asarray(trials, dtype=int)
    return stats.binom.cdf(successes, trials, baseline)
