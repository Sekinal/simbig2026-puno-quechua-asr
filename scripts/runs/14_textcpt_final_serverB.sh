#!/usr/bin/env bash
# Experiment 14 (server B, after exp13): final-mode decoder text-CPT adapter for the final hybrid
# rescoring pool (text-CPT improved OOD radio WER 15.1 -> 14.3 in dev mode).
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp13.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_final_tc --mode final --card omniASR_LLM_7B_v2 \
    --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 500 --eval-clips 150 --workers 8 \
    --text-corpus $A/lm/final_v1/all.txt --text-weight 0.3 --text-batch 16 --grad-ckpt
echo "ALL DONE"
