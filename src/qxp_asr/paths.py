"""Filesystem layout. Everything hangs off ``QXP_ROOT`` (default: /workspace/simbig)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QXP_ROOT", "/workspace/simbig"))
DATA = ROOT / "data"
RAW = DATA / "raw"
AUDIO = DATA / "audio"  # 16 kHz mono PCM WAVs, one per clip id
EXTERNAL = DATA / "external"  # organizer repo artefacts (curated text, cv26 splits)
MANIFESTS = DATA / "manifests"
ARTIFACTS = ROOT / "artifacts"  # checkpoints, LMs, predictions
REPORTS = ROOT / "reports"  # EDA figures / tables

SCRIPTED_DIR = RAW / "scripted" / "cv-corpus-27.0-2026-09-11" / "qxp"
SPONTANEOUS_DIR = RAW / "spontaneous" / "sps-corpus-5.0-2026-09-11-qxp"

AUDIO_OOD = DATA / "audio_ood"  # organizer radio-drama clips (out-of-domain evaluation only)

SAMPLE_RATE = 16_000
# Shared-task test set. Covered by a data-use agreement: never copy it off the GPU box.
TEST_DIR = DATA / "test_task1" / "qxp_test_dataset"
