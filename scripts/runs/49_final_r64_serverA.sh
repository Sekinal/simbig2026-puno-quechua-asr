#!/usr/bin/env bash
# Experiment 49 (server A): final-mode rank-64 enc+dec LoRA, started in parallel with its dev-mode
# scorer test (exp48) so it is ready in time if that test is positive. Same recipe as exp39 at rank 64.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_final_encdec_r64 --mode final \
    --card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 64 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 500 --eval-clips 150 --workers 8
echo "EXP49 DONE"
