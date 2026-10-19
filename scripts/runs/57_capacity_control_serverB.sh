#!/usr/bin/env bash
# Experiment 57 (server B, after exp55): is the scale effect of exp54 only LoRA capacity? Rank-16
# encoder LoRA has ~75M parameters on the 7B encoder but ~7M on the 300M one (128x2048 vs 24x1024
# layers). Encoder-only LoRA on the 300M and 1B encoders at a rank that matches the 7B adapter's
# parameter count (7B rank 16: 75M; 300M rank 160 and 1B rank 64: 71M each), same data, steps and lr.
set -euo pipefail
until grep -q "EXP55 DONE" /workspace/simbig/logs/exp55.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
DATA="--sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1
  --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
  --steps 1500 --warmup 100 --eval-every 300 --eval-clips 150 --workers 8"
for v in 300M:160 1B:64; do
  size=${v%%:*}; rank=${v#*:}; card=omniASR_LLM_${size}_v2; tag=$(echo $size | tr 'A-Z' 'a-z')
  uv run python $R/train_llm.py --run llm${tag}_lora_enc_r$rank --card $card --encoder lora \
      --decoder frozen --rank $rank --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 $DATA
  uv run python $R/infer_llm.py --name llm${tag}_lora_enc_r$rank --card $card --encoder-lora $rank \
      --weights $A/runs/llm${tag}_lora_enc_r$rank/best_trainable.pt
  echo "EXP57 $size DONE"
done
echo "EXP57 DONE"
