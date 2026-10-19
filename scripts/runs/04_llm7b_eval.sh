#!/usr/bin/env bash
# Experiment 04: zero-shot vs LoRA-fine-tuned omniASR-LLM-7B on all evaluation roles (+ test).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts
while pgrep -f "[t]rain_llm.py --run llm7b_lora_v1" >/dev/null; do sleep 30; done
uv run python $R/infer_llm.py --name zs_llm7b_all --card omniASR_LLM_7B_v2
uv run python $R/infer_llm.py --name llm7b_lora_v1 --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights /workspace/simbig/artifacts/runs/llm7b_lora_v1/best_trainable.pt
echo "ALL DONE"
