"""Paired significance tests for WER differences and intervals for proportions.

All tests compare two systems on the same clips (paired), with corpus-level (micro) WER:
``WER = sum(word edits) / sum(reference words)``; resampling recomputes this ratio, never an average
of per-clip WERs.

Resampling units (``paired_bootstrap``):

* clips (default): i.i.d. clips;
* one-way clusters (``groups``): whole speakers or sentences are resampled, so clips of one speaker
  stay together;
* two-way crossed clusters (``groups`` and ``groups2``): read speech crosses speakers with
  sentences; speakers and sentences are resampled independently and each clip is weighted by the
  product of its speaker's and sentence's resampling counts (the pigeonhole bootstrap, Owen 2007,
  Ann. Appl. Stat. 1(2): 386-411), which is conservative for crossed random effects.

The p-value is the two-sided bootstrap p-value of H0: difference = 0 (twice the smaller tail mass of
the resampled differences); ``holm`` adjusts a family of them (Holm 1979).
"""

from __future__ import annotations

import math

import numpy as np
from rapidfuzz.distance import Levenshtein

from qxp_asr.metrics import official_normalize


def edits(refs: list[str], hyps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip word edit counts and reference word counts (official normalization)."""
    r = [official_normalize(t).split() for t in refs]
    e = [
        Levenshtein.distance(x, official_normalize(h or "").split())
        for x, h in zip(r, hyps, strict=True)
    ]
    return np.array(e, dtype=float), np.array([len(x) for x in r], dtype=float)


def _codes(groups) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    return inv, int(inv.max()) + 1


def _weights(rng: np.random.Generator, n_res: int, groups, groups2) -> np.ndarray:
    """Resampling weights per clip, shape (n_res, n_clips)."""
    codes, k = _codes(groups)
    counts = rng.multinomial(k, np.full(k, 1 / k), size=n_res)  # (n_res, k)
    w = counts[:, codes].astype(float)
    if groups2 is not None:
        codes2, k2 = _codes(groups2)
        counts2 = rng.multinomial(k2, np.full(k2, 1 / k2), size=n_res)
        w *= counts2[:, codes2]
    return w


def bootstrap_diffs(
    ea: np.ndarray,
    eb: np.ndarray,
    n: np.ndarray,
    groups=None,
    groups2=None,
    resamples: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Resampled WER(a) - WER(b) in points (one value per resample)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(n))
    w = _weights(rng, resamples, groups, groups2)
    num, den = w @ (ea - eb), w @ n
    ok = den > 0
    return 100 * num[ok] / den[ok]


def summarize(diffs: np.ndarray) -> tuple[float, float, float]:
    """95 % percentile interval and two-sided bootstrap p-value of resampled differences."""
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return float(lo), float(hi), float(p)


def paired_bootstrap(
    ea: np.ndarray,
    eb: np.ndarray,
    n: np.ndarray,
    groups=None,
    groups2=None,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """WER(a) - WER(b) in points, its 95 % percentile interval and two-sided bootstrap p-value.

    ``ea``/``eb`` are per-clip edit counts of the two systems (they may be averages over several
    runs, e.g. weight-search seeds), ``n`` the reference word counts."""
    if groups is None:
        groups = np.arange(len(n))
    diff = bootstrap_diffs(ea, eb, n, groups, groups2, resamples, seed)
    lo, hi, p = summarize(diff)
    return {
        "diff": 100 * float((ea - eb).sum() / n.sum()),
        "lo": lo,
        "hi": hi,
        "p": p,
        "wer_a": 100 * float(ea.sum() / n.sum()),
        "wer_b": 100 * float(eb.sum() / n.sum()),
        "units": len(np.unique(np.asarray(groups, dtype=object).astype(str))),
    }


def sign_test(ea: np.ndarray, eb: np.ndarray, n: np.ndarray, groups) -> dict:
    """Exact two-sided sign test over clusters (e.g. speakers): does a have lower WER than b in more
    clusters than chance? For very few clusters (4-6 speakers) a cluster bootstrap is unreliable;
    this test is exact but cannot reach p < 0.05 with fewer than 6 clusters."""
    codes, k = _codes(groups)
    da = np.bincount(codes, weights=ea, minlength=k)
    db = np.bincount(codes, weights=eb, minlength=k)
    nn = np.bincount(codes, weights=n, minlength=k)
    d = (da - db) / np.maximum(nn, 1)
    better, worse = int((d < 0).sum()), int((d > 0).sum())
    m = better + worse
    tail = sum(math.comb(m, i) for i in range(min(better, worse) + 1)) / 2**m if m else 1.0
    return {"better": better, "worse": worse, "units": k, "p": min(1.0, 2 * tail)}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test for paired binary outcomes (e.g. "source contains a better
    transcript" for two rollout sources on the same clips)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    m = only_a + only_b
    tail = sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2**m if m else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval of a binomial proportion k / n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)
