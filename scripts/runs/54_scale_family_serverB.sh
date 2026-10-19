#!/usr/bin/env bash
# Experiment 54 (server B): does "adapt the encoder, not the decoder" hold across model scale?
# omniASR-LLM-1B-v2 and -300M-v2: zero-shot, decoder-only LoRA (v2 recipe), encoder-only LoRA and
# enc+dec LoRA (exp34b/44b recipe), same data (v2 sources), 1,500 steps each.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
DATA="--sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1
  --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
  --steps 1500 --warmup 100 --eval-every 300 --eval-clips 150 --workers 8 --rank 16"
for size in 1B 300M; do
  card=omniASR_LLM_${size}_v2; tag=$(echo $size | tr 'A-Z' 'a-z')
  uv run python $R/infer_llm.py --name llm${tag}_zs --card $card
  uv run python $R/train_llm.py --run llm${tag}_lora_dec --card $card --encoder frozen \
      --decoder lora --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 $DATA
  uv run python $R/infer_llm.py --name llm${tag}_lora_dec --card $card --decoder-lora 16 \
      --weights $A/runs/llm${tag}_lora_dec/best_trainable.pt
  uv run python $R/train_llm.py --run llm${tag}_lora_enc --card $card --encoder lora \
      --decoder frozen --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 $DATA
  uv run python $R/infer_llm.py --name llm${tag}_lora_enc --card $card --encoder-lora 16 \
      --weights $A/runs/llm${tag}_lora_enc/best_trainable.pt
  uv run python $R/train_llm.py --run llm${tag}_lora_encdec --card $card --encoder lora \
      --decoder lora --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 $DATA
  uv run python $R/infer_llm.py --name llm${tag}_lora_encdec --card $card --encoder-lora 16 \
      --decoder-lora 16 --weights $A/runs/llm${tag}_lora_encdec/best_trainable.pt
  echo "EXP54 $size DONE"
done
echo "EXP54 DONE"
