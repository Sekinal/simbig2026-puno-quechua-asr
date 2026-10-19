#!/usr/bin/env bash
# Experiment 55 (server B, after exp54): transfer of "adapt the encoder, not the decoder" to another
# model family: Whisper large-v3 (no Quechua in pretraining) with rank-16 LoRA on the decoder only,
# the encoder only, or both; same data as the omniASR runs; plus zero-shot. A 4-step smoke run first.
set -euo pipefail
until grep -q "EXP54 DONE" /workspace/simbig/logs/exp54.log; do sleep 120; done
cd /workspace/simbig/repo
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/simbig/repo/src
PY=/workspace/simbig/envs/bench_hf/.venv/bin/python; R=/workspace/simbig/repo/scripts
P=/workspace/simbig/artifacts/pseudo
DATA="--pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet"
$PY $R/train_whisper.py --run tmp_whisper_smoke --encoder lora --decoder lora --steps 4 --warmup 2 \
    $DATA --eval-batch 64 > /workspace/simbig/logs/exp55_smoke.log 2>&1 || { echo SMOKE FAILED; exit 1; }
rm -rf /workspace/simbig/artifacts/runs/tmp_whisper_smoke /workspace/simbig/artifacts/preds/tmp_whisper_smoke.parquet
echo "SMOKE OK"
$PY $R/train_whisper.py --run whisper_zs --steps 0
for v in dec:frozen:lora enc:lora:frozen encdec:lora:lora; do
  IFS=: read -r name e d <<< "$v"
  $PY $R/train_whisper.py --run whisper_lora_$name --encoder $e --decoder $d $DATA
  echo "EXP55 $name DONE"
done
echo "EXP55 DONE"
