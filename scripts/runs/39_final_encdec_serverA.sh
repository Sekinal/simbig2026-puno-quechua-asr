#!/usr/bin/env bash
# Experiment 39 (server A): final-mode encoder+decoder LoRA for the submission. In dev mode (exp34) it
# beats LoRA v2 (same data) on held-out sentences: itest unseen 16.0 vs 20.2 (-4.2 [-7.3,-1.5],
# speakers x sentences resampled), neutral on spontaneous and OOD. Same recipe in final mode (all
# labelled data except dev; exp10 pseudo-labels); last checkpoint, as for the other final adapters
# (dev-mode best step 900 vs last 1500: 12.94 vs 12.98).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_final_encdec --mode final \
    --card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 500 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_final_encdec --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --encoder-lora 16 --weights $A/runs/llm7b_final_encdec/last_trainable.pt \
    --roles dev_scripted,dev_spont,ood,test
echo "EXP39 DONE"
