#!/usr/bin/env bash
# Experiment 09: like-for-like comparison with the organizer baseline. Train in "org" mode (every
# scripted sentence seen; dev/itest speakers unseen = the organizers' cv26 training condition) and
# evaluate on itest_scripted (= the organizers' cv26 test split). Recipe = exp06 (LoRA v2).
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp08.log; do sleep 60; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/simbig/artifacts/pseudo
uv run python /workspace/simbig/repo/scripts/train_llm.py --run llm7b_lora_org --mode org \
    --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 500 --eval-clips 150 --workers 8
uv run python /workspace/simbig/repo/scripts/infer_llm.py --name llm7b_lora_org --decoder-lora 16 \
    --card omniASR_LLM_7B_v2 --weights /workspace/simbig/artifacts/runs/llm7b_lora_org/best_trainable.pt
echo "ALL DONE"
