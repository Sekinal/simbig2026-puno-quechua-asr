#!/usr/bin/env bash
# Experiment 03: omniASR-LLM-7B-v2, frozen encoder + LoRA (r=16) on all LLaMA-decoder projections.
# Generalisation-first choice (docs/LITERATURE.md §g.6): LoRA forgets less than full FT, the frozen
# 7B encoder keeps its 1,600-language acoustic robustness; qxp_Latn language token always passed.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python /workspace/simbig/repo/scripts/train_llm.py --run llm7b_lora_v1 \
    --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,spont_silver:1 \
    --steps 1200 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
