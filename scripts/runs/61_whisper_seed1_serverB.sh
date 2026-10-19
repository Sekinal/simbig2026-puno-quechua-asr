#!/usr/bin/env bash
# Experiment 61 (server B): second training seed of the Whisper decoder-only and encoder-only LoRA
# (exp55), so the transfer result (encoder 27.7 vs decoder 34.1 itest unseen) rests on two seeds.
set -euo pipefail
cd /workspace/simbig/repo
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/simbig/repo/src
PY=/workspace/simbig/envs/bench_hf/.venv/bin/python; R=/workspace/simbig/repo/scripts
P=/workspace/simbig/artifacts/pseudo
DATA="--pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet"
for v in enc:lora:frozen dec:frozen:lora; do
  IFS=: read -r name e d <<< "$v"
  $PY $R/train_whisper.py --run whisper_lora_${name}_s1 --encoder $e --decoder $d --seed 1 $DATA
  echo "EXP61 $name DONE"
done
echo "EXP61 DONE"
