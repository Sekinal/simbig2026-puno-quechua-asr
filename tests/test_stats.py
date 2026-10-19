import numpy as np
import pytest

from qxp_asr.stats import edits, holm, mcnemar, paired_bootstrap, sign_test, wilson


def test_edits_use_official_normalization():
    e, n = edits(["Mana, allinchu?", "ari"], ["mana allinchu", None])
    assert e.tolist() == [0.0, 1.0]
    assert n.tolist() == [2.0, 1.0]


def test_identical_systems_have_zero_difference():
    rng = np.random.default_rng(0)
    e = rng.integers(0, 4, 200).astype(float)
    n = rng.integers(3, 12, 200).astype(float)
    r = paired_bootstrap(e, e, n, resamples=500)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 and r["p"] == 1.0


def test_clear_improvement_is_significant():
    rng = np.random.default_rng(1)
    n = np.full(300, 8.0)
    eb = rng.integers(1, 4, 300).astype(float)
    ea = np.maximum(eb - 1, 0)
    r = paired_bootstrap(ea, eb, n, resamples=2000)
    assert r["hi"] < 0 and r["p"] < 0.01
    assert r["diff"] == pytest.approx(100 * (ea - eb).sum() / n.sum())


def test_speaker_clusters_widen_the_interval_when_errors_are_correlated():
    rng = np.random.default_rng(2)
    spk = np.repeat(np.arange(6), 50)
    effect = rng.normal(0, 1.5, 6)[spk]  # the difference depends on the speaker
    n = np.full(300, 10.0)
    eb = np.full(300, 2.0)
    ea = eb + effect + rng.normal(0, 0.3, 300)
    clip = paired_bootstrap(ea, eb, n, resamples=3000)
    clus = paired_bootstrap(ea, eb, n, groups=spk, resamples=3000)
    assert (clus["hi"] - clus["lo"]) > 3 * (clip["hi"] - clip["lo"])
    assert clus["units"] == 6


def test_two_way_interval_at_least_as_wide_as_one_way():
    rng = np.random.default_rng(3)
    spk, sent = np.arange(400) % 20, np.arange(400) // 2
    n = np.full(400, 8.0)
    eb = np.full(400, 2.0)
    ea = eb + rng.normal(0, 0.5, 20)[spk] + rng.normal(0, 0.5, 200)[sent]
    one = paired_bootstrap(ea, eb, n, groups=sent, resamples=3000)
    two = paired_bootstrap(ea, eb, n, groups=spk, groups2=sent, resamples=3000)
    assert (two["hi"] - two["lo"]) >= (one["hi"] - one["lo"])


def test_null_coverage_close_to_nominal():
    rng = np.random.default_rng(4)
    covered = 0
    for _ in range(200):
        n = np.full(150, 8.0)
        e = rng.integers(0, 4, (2, 150)).astype(float)  # exchangeable systems
        r = paired_bootstrap(e[0], e[1], n, resamples=1000, seed=int(rng.integers(1e9)))
        covered += r["lo"] <= 0 <= r["hi"]
    assert 0.90 <= covered / 200 <= 0.99


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_wilson_interval():
    lo, hi = wilson(9, 150)
    assert lo == pytest.approx(0.0319, abs=1e-3) and hi == pytest.approx(0.1098, abs=1e-3)
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan


def test_sign_test_exact_values():
    spk = np.repeat(np.arange(6), 10)
    n = np.full(60, 5.0)
    eb = np.full(60, 2.0)
    r = sign_test(eb - 1, eb, n, spk)  # a better for all 6 speakers
    assert (r["better"], r["worse"]) == (6, 0) and r["p"] == pytest.approx(2 / 64)
    r4 = sign_test(eb[:40] - 1, eb[:40], n[:40], spk[:40])  # 4 speakers: best possible p
    assert r4["p"] == pytest.approx(0.125)


def test_mcnemar_exact():
    a = np.array([1] * 10 + [0] * 90, bool)
    b = np.zeros(100, bool)
    assert mcnemar(a, b)["p"] == pytest.approx(2 / 1024)  # 10 discordant pairs, all one way
    assert mcnemar(a, a)["p"] == 1.0
