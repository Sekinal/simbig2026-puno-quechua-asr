#!/usr/bin/env bash
# Experiment 06: agreement-filtered pseudo-labels for the spontaneous pool, then LLM-7B LoRA v2
# trained from the base model (not from v1, to limit confirmation bias; §g.5) with older speakers
# (50+) up-weighted, since every spontaneous test speaker is 60+.
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp05.log; do sleep 30; done
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python scripts/pseudo_label.py --name pseudo_v1 \
    --pred pool_llm7b_lora_v1:1.5 --pred pool_ctc1b_v2_wise0.85_greedy:1 --pred silver:1 \
    --max-disagreement 0.15
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/simbig/artifacts/pseudo
uv run python /workspace/simbig/repo/scripts/train_llm.py --run llm7b_lora_v2 \
    --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python /workspace/simbig/repo/scripts/infer_llm.py --name llm7b_lora_v2 --decoder-lora 16 \
    --card omniASR_LLM_7B_v2 --weights /workspace/simbig/artifacts/runs/llm7b_lora_v2/best_trainable.pt
echo "ALL DONE"
