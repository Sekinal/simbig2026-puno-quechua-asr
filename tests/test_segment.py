import numpy as np

from qxp_asr.paths import SAMPLE_RATE
from qxp_asr.segment import cut_points


def test_short_audio_is_not_cut():
    wav = np.ones(10 * SAMPLE_RATE, dtype=np.float32)
    assert cut_points(wav, 5, 20) == [0, len(wav)]


def test_cut_lands_in_the_pause():
    rng = np.random.default_rng(0)
    speech = rng.standard_normal(30 * SAMPLE_RATE).astype(np.float32)
    speech[17 * SAMPLE_RATE : 18 * SAMPLE_RATE] *= 1e-4  # one-second pause at 17-18 s
    bounds = cut_points(speech, min_s=10, max_s=25)
    assert len(bounds) == 3
    assert 17 * SAMPLE_RATE <= bounds[1] <= 18 * SAMPLE_RATE
