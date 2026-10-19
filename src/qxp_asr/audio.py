"""Audio decoding helpers (mp3 -> 16 kHz mono 16-bit WAV via ffmpeg)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from qxp_asr.paths import SAMPLE_RATE


def to_wav(src: Path, dst: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Decode ``src`` with ffmpeg into a mono PCM WAV at ``sample_rate``. Idempotent."""
    if dst.exists() and dst.stat().st_size > 44:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    tmp.rename(dst)
    return dst


def load(path: Path | str) -> np.ndarray:
    """Load a WAV as float32 in [-1, 1]."""
    wav, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        msg = f"{path}: expected {SAMPLE_RATE} Hz, got {sr}"
        raise ValueError(msg)
    return wav
