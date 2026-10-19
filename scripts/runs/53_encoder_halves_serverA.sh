#!/usr/bin/env bash
# Experiment 53 (server A): where in the 7B encoder (128 layers) must the adaptation happen?
# Encoder-only LoRA (decoder frozen; exp44b recipe) restricted to the bottom half (layers 0-63) and
# to the top half (64-127). Reference: encoder-only LoRA on all layers, itest unseen 16.4.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
for half in bottom:0:64 top:64:128; do
  name=${half%%:*}; range=${half#*:}
  uv run python $R/train_llm.py --run llm7b_lora_enc_$name --card omniASR_LLM_7B_v2 \
      --encoder lora --decoder frozen --rank 16 --encoder-layers $range \
      --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
      --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
      --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
      --eval-every 300 --eval-clips 150 --workers 8
  uv run python $R/infer_llm.py --name llm7b_lora_enc_$name --card omniASR_LLM_7B_v2 \
      --encoder-lora 16 --weights $A/runs/llm7b_lora_enc_$name/best_trainable.pt
  echo "EXP53 $name DONE"
done
echo "EXP53 DONE"
