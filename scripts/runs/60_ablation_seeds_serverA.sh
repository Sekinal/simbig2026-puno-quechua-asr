#!/usr/bin/env bash
# Experiment 60 (server A): do exp56's layer results replicate on other adapters? The two enc+dec
# seeds (exp34b, exp42; decoder LoRA kept) with the encoder update kept only in layers 0-95 (top
# quarter removed), 32-127 (bottom quarter removed) and 32-63 (the OOD block).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
for run in llm7b_lora_encdec llm7b_lora_encdec_s1; do
  for range in 0:96 32:128 32:64; do
    uv run python $R/infer_llm.py --name ${run}_keep${range/:/_} --card omniASR_LLM_7B_v2 \
        --encoder-lora 16 --decoder-lora 16 --weights $A/runs/$run/best_trainable.pt \
        --roles dev_scripted,dev_spont,itest_scripted,itest_spont,ood --keep-encoder-layers $range
    echo "EXP60 $run $range DONE"
  done
done
echo "EXP60 DONE"
