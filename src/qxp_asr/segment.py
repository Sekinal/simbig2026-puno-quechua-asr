"""Pause-based cutting of long audio (shared by long-clip segmentation and LLM inference).

Cuts are placed at the quietest 300 ms window between ``min_s`` and ``max_s`` after the previous
cut, so they fall in pauses rather than inside words (fixed-length chunking made the fine-tuned
LLM-ASR hallucinate on mid-word tails: 4x more insertions on > 39 s clips).
"""

from __future__ import annotations

import numpy as np

from qxp_asr.features import HOP, frame_energy_db
from qxp_asr.paths import SAMPLE_RATE

WIN = 30  # frames (300 ms) for the pause search


def cut_points(wav: np.ndarray, min_s: float, max_s: float) -> list[int]:
    """Sample indices of segment boundaries, including 0 and ``len(wav)``."""
    if len(wav) <= max_s * SAMPLE_RATE:
        return [0, len(wav)]
    energy = frame_energy_db(wav)
    smooth = np.convolve(energy, np.ones(WIN) / WIN, mode="same")
    fps = SAMPLE_RATE // HOP
    cuts, start, n = [0], 0, len(energy)
    while n - start > max_s * fps:
        lo, hi = start + int(min_s * fps), min(n, start + int(max_s * fps))
        cut = lo + int(np.argmin(smooth[lo:hi]))
        cuts.append(cut)
        start = cut
    return [c * HOP for c in cuts] + [len(wav)]
