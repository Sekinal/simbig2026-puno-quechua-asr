import pytest

from qxp_asr.metrics import official_normalize, wer_cer


def test_official_normalize_keeps_apostrophe_and_enye():
    assert official_normalize("¿K'anka, Ñuqa?") == "k'anka ñuqa"


def test_wer_cer_micro():
    out = wer_cer(["a b c d", "e f"], ["a b x d", "e f"])
    assert out["wer"] == pytest.approx(100 / 6)
    assert out["n"] == 2


def test_empty_reference_dropped():
    assert wer_cer(["", "a"], ["x", "a"])["n"] == 1
