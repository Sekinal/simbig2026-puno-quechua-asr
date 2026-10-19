"""Waveform augmentations targeted at the train→test shifts found in EDA (reports/eda/FINDINGS.md).

* ``random_trim``: test clips are silence-trimmed (median leading silence 0.01 s vs 0.55 s in
  train).
* ``speed_perturb``: Ko et al., 2015 ("Audio augmentation for speech recognition", Interspeech) —
  0.9/1.0/1.1 speed perturbation; also shifts pitch/formants, which helps with the gender and age
  mismatch (test has 1/3 male scripted speech and only elderly spontaneous speakers).
* ``add_noise``: low-level noise mixing; test audio was recorded on uncontrolled personal devices.

Gain changes are pointless: the encoder layer-normalizes every waveform.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as taf

from qxp_asr.features import HOP, frame_energy_db
from qxp_asr.paths import SAMPLE_RATE


def random_trim(
    wav: np.ndarray, rng: np.random.Generator, max_margin_s: float = 0.2, floor_db: float = 35.0
) -> np.ndarray:
    """Cut leading/trailing non-speech, leaving a random margin in [0, max_margin_s]."""
    energy = frame_energy_db(wav)
    active = np.flatnonzero(energy > energy.max() - floor_db)
    if active.size == 0:
        return wav
    margin = int(max_margin_s * SAMPLE_RATE)
    start = max(0, active[0] * HOP - int(rng.integers(0, margin + 1)))
    end = min(len(wav), (active[-1] + 1) * HOP + 400 + int(rng.integers(0, margin + 1)))
    return wav[start:end] if end - start > SAMPLE_RATE // 4 else wav


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """Sox-style speed change: duration scales by 1/factor, pitch by factor."""
    if abs(factor - 1.0) < 1e-6:
        return wav
    orig = round(factor * 100)
    out = taf.resample(torch.from_numpy(wav), orig_freq=orig, new_freq=100)
    return out.numpy()


def add_noise(wav: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Mix white/pink-ish Gaussian noise at ``snr_db`` relative to the signal power."""
    power = float(np.mean(wav**2)) + 1e-10
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    if rng.random() < 0.5:  # crude pink-ish colouring via a one-pole low-pass
        noise = np.asarray(taf.lowpass_biquad(torch.from_numpy(noise), SAMPLE_RATE, 2000.0))
    noise *= np.sqrt(power / (10 ** (snr_db / 10)) / (np.mean(noise**2) + 1e-10))
    return (wav + noise).astype(np.float32)


def add_reverb(wav: np.ndarray, rng: np.random.Generator, rt60: float) -> np.ndarray:
    """Convolve with a synthetic room impulse response: a direct path plus exponentially decaying
    noise (60 dB decay after ``rt60`` seconds), mixed so the output keeps the input's power."""
    n = int(rt60 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct-to-reverberant ratio varies
    out = np.convolve(wav, rir.astype(np.float32))[: len(wav)]
    return (out * np.sqrt(np.mean(wav**2) / (np.mean(out**2) + 1e-10))).astype(np.float32)


def band_limit(wav: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Band-pass a channel (telephone 300-3400 Hz up to a mild low-pass)."""
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    x = taf.highpass_biquad(x, SAMPLE_RATE, low_hz)
    x = taf.lowpass_biquad(x, SAMPLE_RATE, high_hz)
    return np.asarray(x, dtype=np.float32)


class Augmenter:
    """Stochastic augmentation chain applied per training example.

    ``profile="mild"`` is the default used by every model so far. ``profile="strong"`` is a
    SimCLR-style distortion set for robustness to channel and room: continuous speed 0.8-1.25,
    noise on half the clips at 5-30 dB, synthetic reverberation and random band-limiting."""

    def __init__(
        self,
        seed: int,
        p_trim: float = 0.7,
        speeds: tuple[float, ...] = (0.9, 1.0, 1.1),
        p_speed: float = 0.6,
        p_noise: float = 0.2,
        snr_range: tuple[float, float] = (15.0, 35.0),
        profile: str = "mild",
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.p_trim, self.speeds, self.p_speed = p_trim, speeds, p_speed
        self.p_noise, self.snr_range = p_noise, snr_range
        self.strong = profile == "strong"
        if self.strong:
            self.p_speed, self.p_noise, self.snr_range = 0.7, 0.5, (5.0, 30.0)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() < self.p_trim:
            wav = random_trim(wav, rng)
        if rng.random() < self.p_speed:
            factor = rng.uniform(0.8, 1.25) if self.strong else rng.choice(self.speeds)
            wav = speed_perturb(wav, float(factor))
        if self.strong and rng.random() < 0.4:
            wav = add_reverb(wav, rng, float(rng.uniform(0.2, 0.8)))
        if self.strong and rng.random() < 0.3:
            wav = band_limit(wav, float(rng.uniform(100, 400)), float(rng.uniform(3000, 7000)))
        if rng.random() < self.p_noise:
            wav = add_noise(wav, rng, float(rng.uniform(*self.snr_range)))
        return wav


# Perturbed views used as RL rollout sources ("aug" exploration; scripts/pass_at_k.py): greedy
# decodes of slowed / sped-up / noisy copies find a better transcript than the beam top-1 far more
# often than beam search does (reports/RESULTS.md, exploration sweep). (speed, SNR dB or None)
VIEWS: list[tuple[float, float | None]] = [
    (1.0, None), (0.85, None), (0.9, None), (0.95, None),
    (1.05, None), (1.1, None), (1.0, 20.0), (0.9, 15.0),
    # extension for the scaling curve (views 7-12)
    (0.8, None), (1.15, None), (0.95, 25.0), (1.05, 15.0), (0.85, 20.0), (1.1, 20.0),
]  # fmt: skip


def view(wav: np.ndarray, speed: float, snr: float | None, rng: np.random.Generator) -> np.ndarray:
    out = speed_perturb(wav, speed)
    return add_noise(out, rng, snr) if snr is not None else out
