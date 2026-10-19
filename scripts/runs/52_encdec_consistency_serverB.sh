#!/usr/bin/env bash
# Experiment 52 (server B): enc+dec LoRA with two-view consistency regularization (SimCLR-style
# invariance for ASR; R-Drop / CR-CTC applied to the LLM-ASR decoder): every batch holds two strong
# augmentations of each clip; loss = CE on both + 0.5 x symmetric KL of their token distributions.
# 30 s x 8 accumulation keeps the unique audio per update of exp34b (60 s x 4); 1,000 steps (each
# costs ~2x; exp34b's best dev step was 900). A 6-step smoke run guards the new loss first.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
COMMON="--card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 16 --augment strong
  --consistency 0.5 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1
  --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
  --warmup 100 --lr-lora 2e-4 --max-batch-seconds 30 --grad-accum 8 --workers 8"
uv run python $R/train_llm.py --run tmp_smoke_cons $COMMON --steps 6 --eval-every 1000 --eval-clips 4
rm -rf $A/runs/tmp_smoke_cons
echo "SMOKE OK"
uv run python $R/train_llm.py --run llm7b_lora_encdec_cons $COMMON --steps 1000 --eval-every 250 \
    --eval-clips 150
uv run python $R/infer_llm.py --name llm7b_lora_encdec_cons --card omniASR_LLM_7B_v2 \
    --decoder-lora 16 --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec_cons/best_trainable.pt
echo "EXP52 DONE"
