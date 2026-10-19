#!/usr/bin/env bash
# Experiment 08: LLM-7B with LoRA on encoder AND decoder (acoustic adaptation to new speakers and
# elderly voices; §g.1/§g.6), activation checkpointing, same data as exp06.
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp10.log 2>/dev/null; do sleep 60; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/simbig/artifacts/pseudo
uv run python /workspace/simbig/repo/scripts/train_llm.py --run llm7b_lora_encdec \
    --card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python /workspace/simbig/repo/scripts/infer_llm.py --name llm7b_lora_encdec \
    --card omniASR_LLM_7B_v2 --decoder-lora 16 --encoder-lora 16 \
    --weights /workspace/simbig/artifacts/runs/llm7b_lora_encdec/best_trainable.pt
echo "ALL DONE"
