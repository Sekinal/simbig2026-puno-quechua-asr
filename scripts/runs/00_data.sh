#!/usr/bin/env bash
# Experiment 00: data preparation (run once on the GPU box, in /workspace/simbig/repo).
# Prerequisites: CV Scripted 27.0 + Spontaneous 5.0 qxp tarballs extracted under data/raw/
# (scripted/, spontaneous/), organizer repo artefacts under data/external/, Task-1 test set
# under data/test_task1/ (data-use agreement), organizer OOD radio clips resampled to
# data/audio_ood/ (16 kHz mono).
set -euo pipefail
cd /workspace/simbig/repo
uv run python scripts/prepare_audio.py          # mp3 -> 16 kHz mono wav
uv run python scripts/build_manifest.py         # data/manifests/all.parquet (+ roles)
uv run python scripts/eda.py                    # reports/eda
uv run python scripts/build_text_corpus.py download && uv run python scripts/build_text_corpus.py build
uv run python scripts/build_lm.py --name dev_v2 --mode dev
