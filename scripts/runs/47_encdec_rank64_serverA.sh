#!/usr/bin/env bash
# Experiment 47 (server A): the encoder is the adaptation site (exp44b: encoder-only LoRA ~ enc+dec);
# does it want more capacity? enc+dec recipe (exp34b) with LoRA rank 64 instead of 16 (alpha 2r).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_encdec_r64 --card omniASR_LLM_7B_v2 \
    --encoder lora --decoder lora --rank 64 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_encdec_r64 --card omniASR_LLM_7B_v2 \
    --decoder-lora 64 --encoder-lora 64 --weights $A/runs/llm7b_lora_encdec_r64/best_trainable.pt
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a llm7b_lora_encdec_r64 \
    --b llm7b_lora_encdec
echo "EXP47 DONE"
