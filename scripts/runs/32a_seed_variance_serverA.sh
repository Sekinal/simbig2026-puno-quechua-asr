#!/usr/bin/env bash
# Experiment 32a (server A): training-seed variance. The LoRA v2 recipe (exp06) retrained with
# seed 2; with the original (seed 0) this gives the spread due to training alone, against which the
# single-model differences in the paper are judged.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_v2_s2 --seed 2 \
    --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_v2_s2 --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v2_s2/best_trainable.pt
echo "EXP32 DONE"
