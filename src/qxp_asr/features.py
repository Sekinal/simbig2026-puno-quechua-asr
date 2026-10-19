"""Cheap clip-level acoustic descriptors for EDA and data filtering.

Everything is computed from 25 ms / 10 ms frame energies, so ~33k clips take a minute on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from qxp_asr.audio import load

FRAME = 400  # 25 ms @ 16 kHz
HOP = 160  # 10 ms
EPS = 1e-10


@dataclass(frozen=True, slots=True)
class ClipStats:
    rms_dbfs: float  # overall loudness
    peak: float  # max |x|
    clip_frac: float  # fraction of samples with |x| >= 0.999 (hard clipping)
    snr_db: float  # 95th - 10th percentile frame energy (dB): crude SNR proxy
    speech_frac: float  # fraction of frames within 30 dB of the loudest frames
    lead_sil_s: float  # leading silence (s) before the first speech frame
    trail_sil_s: float  # trailing silence (s) after the last speech frame
    dc_offset: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def frame_energy_db(wav: np.ndarray) -> np.ndarray:
    if len(wav) < FRAME:
        wav = np.pad(wav, (0, FRAME - len(wav)))
    n = 1 + (len(wav) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = wav[idx]
    return 10 * np.log10(np.mean(frames**2, axis=1) + EPS)


def clip_stats(path: str) -> dict[str, float]:
    wav = load(path)
    energy = frame_energy_db(wav)
    loud = np.percentile(energy, 95)
    speech = energy > loud - 30
    active = np.flatnonzero(speech)
    first, last = (active[0], active[-1]) if active.size else (0, len(energy) - 1)
    return ClipStats(
        rms_dbfs=float(10 * np.log10(np.mean(wav**2) + EPS)),
        peak=float(np.max(np.abs(wav))) if wav.size else 0.0,
        clip_frac=float(np.mean(np.abs(wav) >= 0.999)) if wav.size else 0.0,
        snr_db=float(loud - np.percentile(energy, 10)),
        speech_frac=float(speech.mean()),
        lead_sil_s=float(first * HOP / 16_000),
        trail_sil_s=float((len(energy) - 1 - last) * HOP / 16_000),
        dc_offset=float(np.mean(wav)) if wav.size else 0.0,
    ).as_dict()
