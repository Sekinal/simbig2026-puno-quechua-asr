#!/usr/bin/env bash
# Experiment 42 (server B): the encoder+decoder LoRA (exp34b) retrained with seed 1, so the paper's
# new headline gain (itest unseen 16.0 vs 20.2 for LoRA v2) is shown not to be one lucky run.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_encdec_s1 --seed 1 \
    --card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_encdec_s1 --card omniASR_LLM_7B_v2 \
    --decoder-lora 16 --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec_s1/best_trainable.pt
echo "EXP42 DONE"
